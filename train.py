import os
import numpy as np
import time
import math
import re
import glob
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm
from functools import partial

import torch
from model import Transformer, ModelArgs
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task


# Hyperparameters
@dataclass
class TrainingArgs:
    # Data
    max_batch_size: int = 2
    max_seq_len: int = 2048
    vocab_source: str = (
        "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
    )
    vocab_size: int = 32000  # the Llama 2 tokenizer has 32K tokens

    # I/O
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50
    save_total_limit: int = 1  # total number of checkpoints to save
    save_best_checkpoint: bool = True  # whether to save the best val checkpoint
    init_from: str = "scratch"  # mode of training -- 'scratch','resume'
    out_dir: str = "little-checkpoints"

    # Model
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    multiple_of: int = 32
    dropout: float = 0.0

    # System
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16' (float16 will auto implement a GradScaler)
    compile: bool = False

    # DDP Settings
    backend: str = "nccl"  # 'nccl', 'gloo', etc.

    # Learning rate decay settings
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 3e-5

    # Optimizer settings
    gradient_accumulation_steps: int = 12  # used to simulate larger batch sizes
    learning_rate: float = 3e-4
    max_iters: int = 100000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 0.0

    # Wandb log
    wandb_log: bool = True  # disabled by default
    wandb_project: str = "llm"
    wandb_run_name: str = "lilGPT"  # 'run' + str(time.time())


def process_checkpoints(output_dir=None, checkpoint_prefix="ckpt", save_total_limit=0):
    ordering_and_checkpoint_path = []

    glob_checkpoints = [
        str(x)
        for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")
        if os.path.exists(x)
    ]
    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

    if save_total_limit <= 0 or len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    # Deleting checkpoints
    for checkpoint in checkpoints_to_be_deleted:
        # the checkpoint is a single file, not a directory -> use os.remove() instead of shutil.rmtree()
        os.remove(checkpoint)


def train():
    training_args = TrainingArgs()

    assert training_args.vocab_source in ["llama2", "custom"]
    assert (
        training_args.vocab_source == "custom" or training_args.vocab_size == 32000
    ), "The vocab from Meta has 32K tokens"

    # -----------------------------------------------------------------------------
    training_args_fields = [
        field.name for field in TrainingArgs.__dataclass_fields__.values()
    ]
    config_keys = [
        k
        for k in training_args_fields
        if isinstance(getattr(training_args, k), (int, float, bool, str))
    ]
    exec(open("configurator.py").read())  # overrides from command line or config file
    config = {key: getattr(training_args, key) for key in config_keys}
    # -----------------------------------------------------------------------------

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
        assert training_args.gradient_accumulation_steps % ddp_world_size == 0
        training_args.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = (
        training_args.gradient_accumulation_steps
        * ddp_world_size
        * training_args.max_batch_size
        * training_args.max_seq_len
    )
    if master_process:
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")
        print(
            f"breaks down as: {training_args.gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {training_args.max_batch_size} batch size * {training_args.max_seq_len} max seq len"
        )

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

    # task-specific setup
    iter_batches = partial(
        Task.iter_batches,
        batch_size=training_args.max_batch_size,
        max_seq_len=training_args.max_seq_len,
        vocab_size=training_args.vocab_size,
        vocab_source=training_args.vocab_source,
        device=training_args.device,
        num_workers=0,
    )

    # init these up here, can override if init_from='resume' (i.e. resume training from a checkpoint)
    iter_num_start = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(
        dim=training_args.dim,
        n_layers=training_args.n_layers,
        n_heads=training_args.n_heads,
        n_kv_heads=training_args.n_kv_heads,
        vocab_size=training_args.vocab_size,
        multiple_of=training_args.multiple_of,
        max_seq_len=training_args.max_seq_len,
        dropout=training_args.dropout,
        max_batch_size=training_args.max_batch_size,
    )  # start with model_args from command line

    if training_args.init_from == "scratch":
        print("Initialize a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif training_args.init_from == "resume":
        # resume training from checkpoint
        print(f"Resume training from {training_args.out_dir}")
        ckpt_path = os.path.join(training_args.out_dir, "ckpt-31500.pt")
        checkpoint = torch.load(ckpt_path, map_location=training_args.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in [
            "dim",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "vocab_size",
            "multiple_of",
            "max_seq_len",
            "max_batch_size",
        ]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num_start = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    model.to(training_args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # An example of a model's parameter and its dtype
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Dtype: {param.dtype}")
        break

    # Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(training_args.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        training_args.weight_decay,
        training_args.learning_rate,
        (training_args.beta1, training_args.beta2),
        device_type,
    )
    if training_args.init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if training_args.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            batch_iter = iter_batches(split=split)
            losses = torch.zeros(training_args.eval_iters)  # keep on CPU
            for k in range(training_args.eval_iters):
                X, Y = next(batch_iter)
                with ctx:
                    logits = model(X, Y)
                    loss = raw_model.last_loss
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
            project=training_args.wandb_project,
            name=training_args.wandb_run_name,
            config=config,
        )

    train_batch_iter = iter_batches(split="train")
    X, Y = next(train_batch_iter)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0

    # Training loop
    for iter_num in tqdm(range(iter_num_start, training_args.max_iters)):
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
                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * tokens_per_iter,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        },
                        step=iter_num,
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")

            # Save checkpoint after every eval_interval
            best_ckpt = None
            if training_args.save_best_checkpoint and losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_ckpt = True
            if iter_num > 0:
                # checkpoint contains these params
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "val_loss": losses["val"],
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                if best_ckpt:
                    print(f"Saving best checkpoint to {training_args.out_dir}")
                    torch.save(
                        checkpoint,
                        os.path.join(training_args.out_dir, f"ckpt-best-{iter_num}.pt"),
                    )
                    process_checkpoints(
                        output_dir=training_args.out_dir,
                        checkpoint_prefix="ckpt-best",
                        save_total_limit=1,
                    )
                else:
                    print(f"Saving checkpoint to {training_args.out_dir}")
                    torch.save(
                        checkpoint,
                        os.path.join(training_args.out_dir, f"ckpt-{iter_num}.pt"),
                    )
                    process_checkpoints(
                        output_dir=training_args.out_dir,
                        checkpoint_prefix="ckpt",
                        save_total_limit=training_args.save_total_limit,
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
                logits = model(X, Y)
                loss = raw_model.last_loss
                loss = (
                    loss / training_args.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(train_batch_iter)
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
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item() * training_args.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    training_args.max_batch_size
                    * training_args.gradient_accumulation_steps,
                    dt,
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )

        local_iter_num += 1

        # termination conditions
        if iter_num > training_args.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    # Check CUDA availability
    print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")

    train()
