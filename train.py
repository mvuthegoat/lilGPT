import os
import numpy as np
import time
import math
import pickle
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from minivnGPT.model import minivnGPT, ModelArguments
from data.prepare import preprocess


# Hyperparameters
@dataclass
class TrainingArguments:
    batch_size: int = 2
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50
    learning_rate: float = 3e-4
    gradient_accumulation_steps: int = 12  # used to simulate larger batch sizes
    always_save_checkpoint: bool = (
        False  # if True, always save a checkpoint after each eval
    )
    init_from: str = "scratch"  # mode of training -- 'scratch','resume'
    out_dir: str = "llm-checkpoints"

    # System
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16' (float16 will auto implement a GradScaler)

    # DDP Settings
    backend = "nccl"  # 'nccl', 'gloo', etc.

    # Learning rate decay settings
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 10000
    min_lr: float = 3e-5

    # Optimizer settings
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 0.0
    eps: float = 1e-5

    # Tokenizer
    tokenizer_path: str = "meta-llama/Llama-2-7b-chat-hf"

    # Wandb log
    wandb_log = True  # disabled by default
    wandb_project = "mvuthegoat"
    wandb_run_name = "minivnGPT"  # 'run' + str(time.time())


def train():
    training_args = TrainingArguments()

    # DDP
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"World size is {world_size}")
    ddp = world_size != 1
    if ddp:
        init_process_group(backend=training_args.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        training_args.device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(training_args.device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed

        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        if training_args.gradient_accumulation_steps != 1:
            assert training_args.gradient_accumulation_steps % ddp_world_size == 0
            training_args.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    if master_process:
        os.makedirs(training_args.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in training_args.device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[training_args.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model_args = ModelArguments()
    # Tokenize data
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=training_args.tokenizer_path,
        model_max_length=model_args.context_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    model_args.vocab_size = tokenizer.vocab_size
    train_data, val_data = preprocess(tokenizer)
    print(f"Train data has {len(train_data)} examples")
    print(f"Eval data has {len(val_data)} examples")
    tokens_per_iter = (
        training_args.gradient_accumulation_steps
        * ddp_world_size
        * training_args.batch_size
        * model_args.context_length
    )
    print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    # init these up here, can override if init_from='resume' (i.e. resume training from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    if training_args.init_from == "scratch":
        print("Initialize a new model from scratch")
        model = minivnGPT(model_args)
    elif training_args.init_from == "resume":
        # resume training from checkpoint
        print(f"Resume training from {training_args.out_dir}")
        model_args_dict = {}
        ckpt_path = os.path.join(training_args.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=training_args.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in [
            "n_layer",
            "n_head",
            "d_model",
            "context_length",
            "bias",
            "vocab_size",
        ]:
            model_args_dict[k] = getattr(checkpoint_model_args, k)
        # create the model
        model_args = ModelArguments(**model_args_dict)
        model = minivnGPT(model_args)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        # Load state dict to model
        model.load_state_dict(state_dict)

        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    model.to(training_args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # An example of a model's parameter and its dtype
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Dtype: {param.dtype}")
        break

    # Randomly sample batch
    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(
            high=len(data) - model_args.context_length, size=(training_args.batch_size,)
        )  # Randomize the starting indices for each training example in a batch
        # Stack training examples in a batch
        x = torch.stack(
            [
                torch.from_numpy(
                    (data[i : i + model_args.context_length]).astype(np.int64)
                )
                for i in ix
            ]
        )
        # Target label for each training exampl in a batch. Each label is the next word in the sequence
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + model_args.context_length]).astype(np.int64)
                )
                for i in ix
            ]
        )

        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(
                training_args.device, non_blocking=True
            ), y.pin_memory().to(training_args.device, non_blocking=True)
        else:
            x, y = x.to(training_args.device), y.to(training_args.device)
        return x, y

    # Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(training_args.dtype == "float16"))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=training_args.weight_decay,
    )
    if training_args.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # Free memory

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(training_args.eval_iters)
            for k in range(training_args.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Learning rate decay scheduler (cosine with warmup) (from Karpathy's nanoGPT)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < training_args.warmup_iters:
            return training_args.learning_rate * it / training_args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > training_args.lr_decay_iters:
            return training_args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - training_args.warmup_iters) / (
            training_args.lr_decay_iters - training_args.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return training_args.min_lr + coeff * (
            training_args.learning_rate - training_args.min_lr
        )

    # logging
    if training_args.wandb_log and master_process:
        import wandb

        wandb.init(
            project=training_args.wandb_project, name=training_args.wandb_run_name
        )

    # training loop
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    raw_model = model.module if ddp else model  # unwrap DDP container if needed

    # Training loop
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if training_args.decay_lr else training_args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % training_args.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if training_args.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                    }
                )
            # Save checkpoint after every eval_interval (same as epoch) if the loss is minimum or always save checkpoint in on
            if losses["val"] < best_val_loss or training_args.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    # checkpoint contains these params
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "val_loss": losses["val"],
                        "best_val_loss": best_val_loss,
                    }
                    print(f"saving checkpoint to {training_args.out_dir}")
                    torch.save(
                        checkpoint, os.path.join(training_args.out_dir, "ckpt.pt")
                    )

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(training_args.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == training_args.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = (
                    loss / training_args.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if training_args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % training_args.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * training_args.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1

        # termination conditions
        if iter_num > training_args.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    # Check CUDA
    print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")

    train()
