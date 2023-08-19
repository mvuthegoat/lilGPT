import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    functional as F,
)  # can be directly applied to input tensors, don't need to create a module to use it

from datasets import load_dataset
from transformers import AutoTokenizer
from minivnGPT.model import minivnGPT, minivnGPTConfig, minivnGPTForQA
from torch.utils.data import DataLoader

# Load dataset
dataset = load_dataset("mlqa", "mlqa.vi.vi")
dataset["train"] = dataset["test"]  # change the dict key test to training
del dataset["test"]
dataset["train"] = dataset["train"].select([i for i in range(3000)])

train_data = dataset["train"]
val_data = dataset["validation"]

# print(dataset)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
print(tokenizer.is_fast)
print(tokenizer.vocab_size)


# Pre-processing train dataset
def train_feature_preprocess(examples):
    """generate start and end indexes of the answers (in token form, not char) in context"""

    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        padding="max_length",
        max_length=256,  # same as model context_length
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    # If returning overflowing tokens, we need to return a mapping from the features idx to the original example idx
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # offset_mapping refers to start index char and end index char of each token with respect to whole text.
    offset_mapping = tokenized_examples.pop("offset_mapping")
    # initialize the starting_positions and ending_positions list (these are the labels we use to train)
    starting_positions = []
    ending_positions = []

    for i, mapping_pos in enumerate(offset_mapping):
        # get the input_ids of every example
        input_ids = tokenized_examples["input_ids"][i]
        # sequence_ids returns a list mapping the tokens to the id of their original sentences (None for special tokens, 0 for tokens in 1st sequence, 1 for tokens in 2nd sequence)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # use sample mapping to get the sample idx from examples where the input_ids belong
        sample_idx = sample_mapping[i]
        # get the answer for this sample
        answer = examples["answers"][sample_idx]
        # get the character position of answer_start and answer_end
        answer_start_char = answer["answer_start"][0]
        answer_end_char = answer_start_char + len(answer["text"][0]) - 1

        # if there's no answer in the sample --> let starting_position and ending_position be 0
        if len(answer["text"]) == 0:
            starting_positions.append(0)
            ending_positions.append(0)
        else:
            # find context_start_char and context_end_char using sequence_ids
            context_start_token = 0
            while sequence_ids[context_start_token] != 1:
                context_start_token += 1
            context_end_token = context_start_token
            while sequence_ids[context_end_token] == 1:
                context_end_token += 1
            context_end_token -= 1

            # get the character position of context_start and context_end
            context_start_char = mapping_pos[context_start_token][0]
            context_end_char = mapping_pos[context_end_token][1]

            # check if the answer is within the context. If not, append 0 to starting and ending positions
            if (
                context_start_char > answer_start_char
                or context_end_char < answer_end_char
            ):
                starting_positions.append(0)
                ending_positions.append(0)
            else:
                # find token positions of answer_start and answer_end
                answer_start_token = 0
                while (
                    answer_start_token < len(mapping_pos)
                    and mapping_pos[answer_start_token][0] <= answer_start_char
                ):
                    answer_start_token += 1
                # append answer_start_token
                starting_positions.append(answer_start_token - 1)

                answer_end_token = context_end_token
                while (
                    answer_end_token >= 0
                    and mapping_pos[answer_end_token][1] >= answer_end_char
                ):
                    answer_end_token -= 1
                ending_positions.append(answer_end_token + 1)

    tokenized_examples["start_positions"] = starting_positions
    tokenized_examples["end_positions"] = ending_positions

    return tokenized_examples


train_dataset = train_data.map(
    train_feature_preprocess, batched=True, remove_columns=train_data.column_names
)
val_dataset = val_data.map(
    train_feature_preprocess, batched=True, remove_columns=train_data.column_names
)

print(train_dataset)
print(val_dataset)

# Hyperparameters
eval_interval = 100
eval_iters = 10
learning_rate = 3e-4
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # mode of training -- 'scratch','resume'
out_dir = "/home/minhvn/workspace/llm/my_model/checkpoints"  ## the directory storing checkpoints
# batch size, epoch, and iteration
batch_size = 16
max_iters = 1000
num_epochs = (int)(max_iters / (len(train_dataset) / batch_size))
print(f"Number of epochs to train: {num_epochs}")

device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

# Convert Huggingface dataset to torch dataset format
train_dataset = train_dataset.with_format(type="torch", device=device)
# eval_dataset = eval_dataset.set_format(type='torch', device=device)
val_dataset = val_dataset.with_format(type="torch", device=device)


print(f"train_dataset has start: {(train_dataset['start_positions'][999])}")
print(f"train_dataset has end: {(train_dataset['end_positions'][999])}")


# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# init these up here, can override if init_from='resume' (i.e. resume training from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# Initialize models
model_args = minivnGPTConfig()
model = minivnGPT(model_args)
pre_trained_ckpt = torch.load(
    "/home/minhvn/workspace/llm/my_model/checkpoints/ckpt-2000.pt"
)
model.load_state_dict(pre_trained_ckpt["model"])
print(f"LOSS FROM PRE-TRAINED: {pre_trained_ckpt['val_loss']}")
qa_model = minivnGPTForQA(model, model_args)

if init_from == "resume":
    # resume training from checkpoint
    print(f"Resume training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt-qa.pt")  # specify checkpoint's path
    checkpoint = torch.load(ckpt_path)
    # TO DO: force the model params to be the same as checkpoint params
    train_args = model_args
    qa_model = minivnGPTForQA(train_args)
    # load state dict for model
    state_dict = checkpoint["model"]
    qa_model.load_state_dict(state_dict)

    # override these variables for training
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["val_loss"]

model.to(device)
qa_model.to(device)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    qa_model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        eval_iter_num = 0
        data = train_loader if split == "train" else val_loader

        for batch in data:
            input_ids = batch["input_ids"]
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]

            outputs = qa_model(input_ids, start_positions, end_positions)
            loss = outputs.loss
            losses[eval_iter_num] = loss.item()
            eval_iter_num += 1
            if eval_iter_num >= eval_iters:
                break

        out[split] = losses.mean()
    qa_model.train()
    return out


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(qa_model.parameters(), lr=learning_rate)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])

checkpoint = None  # Free memory

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    # checkpoint contains these params
                    checkpoint = {
                        "model": qa_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "val_loss": losses["val"],
                    }
                    print(f"saving checkpoint to {out_dir}")
                    NEW_CKPT_PATH = f"ckpt-qa-{iter_num}.pt"
                    torch.save(checkpoint, os.path.join(out_dir, NEW_CKPT_PATH))

        # Train a batch
        input_ids = batch["input_ids"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]

        outputs = qa_model(input_ids, start_positions, end_positions)
        # evaluate the loss
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # increment iter_num until reach max_iters
        iter_num += 1
        if iter_num > max_iters:
            break
