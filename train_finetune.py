# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from tokenizer import Tokenizer

# Training
import os
import time

from model import Transformer, ModelArgs
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from train import TrainingArgs, process_checkpoints
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime
from tqdm import tqdm
from functools import partial

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
# IGNORE_TOKEN_ID = 0


@dataclass
class ModelArguments:
    checkpoint: Optional[str] = field(default="little-checkpoints/ckpt-best-98500.pt")


@dataclass
class DataArgs:
    data_path: str = field(
        default="data/clean_alpaca_data.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=2048,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, data_path, split, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()

        rank0_print("Formatting inputs...")
        raw_data = json.load(open(data_path, "r"))
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        self.split = split

    def __iter__(self):
        # train/test split
        # np.random.seed(0)
        # perm = np.random.permutation(len(self.input_ids))
        perm = np.arange(len(self.input_ids))
        split = int(len(perm) * 0.98)
        train_indices = perm[:split]
        eval_indices = perm[split:]

        indices = train_indices if self.split == "train" else eval_indices

        while True:
            for i in indices:
                x = self.input_ids[i]
                y = self.labels[i]

                yield x, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        rank0_print("Loading data...")
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


def train():
    training_args = TrainingArgs(
        out_dir="finetune_ckpt", wandb_log=False, init_from="resume"
    )
    data_args = DataArgs()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
        model_max_length=training_args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

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
        device=training_args.device,
        num_workers=0,
        data_path=data_args.data_path,
        tokenizer=tokenizer,
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

        # init from a model saved in a specific directory
        checkpoint = "little-checkpoints/ckpt-best-98500.pt"
        checkpoint_dict = torch.load(checkpoint, map_location=training_args.device)
        gptconf = ModelArgs(**checkpoint_dict["model_args"])
        model = Transformer(gptconf)
        state_dict = checkpoint_dict["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    elif training_args.init_from == "resume":
        # resume training from checkpoint
        print(f"Resume training from {training_args.out_dir}")
        ckpt_path = os.path.join(training_args.out_dir, "ckpt-best-1000.pt")
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
