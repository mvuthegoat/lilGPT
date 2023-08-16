import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F # can be directly applied to input tensors, don't need to create a module to use it

from datasets import load_dataset
from transformers import AutoTokenizer
from minivnGPT.model import minivnGPT, minivnGPTConfig

## dataset used: "Libosa2707/vietnamese-poem" from huggingface dataset
# Load dataset + process
dataset = load_dataset("Libosa2707/vietnamese-poem")
#print(dataset)

dataset = dataset["train"].train_test_split(test_size=0.1)
dataset['validation'] = dataset['test']  # change the dict key test to valid
del dataset['test']

## sample some dataset so that we can reduce training time.
dataset["train"] = dataset["train"].select([i for i in range(8000)])
dataset["validation"] = dataset["validation"].select([i for i in range(2000)])

# Tokenizer part
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
#print(tokenizer.is_fast)
#print(tokenizer.vocab_size)
train_data = tokenizer.encode(str(dataset["train"]["content"]))
val_data = tokenizer.encode(str(dataset["validation"]["content"]))


# Hyperparameters
torch.manual_seed(1337)
batch_size = 16
max_iters = 2000
eval_interval = 100
eval_iters = 150
learning_rate = 3e-4
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch'  # mode of training -- 'scratch','resume'
out_dir = '/home/minhvn/workspace/llm/my_model/checkpoints' ## the directory storing checkpoints
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# Check CUDA
if torch.cuda.is_available():
    # CUDA is available
    print("Running on GPU")
else:
    # CUDA is not available
    print("Running on CPU")


# init these up here, can override if init_from='resume' (i.e. resume training from a checkpoint)
iter_num = 0
best_val_loss = 1e9

model_args = minivnGPTConfig()
model = minivnGPT(model_args)

if init_from == 'resume':
  # resume training from checkpoint
  print (f"Resume training from {out_dir}")
  ckpt_path = os.path.join(out_dir, 'ckpt-999.pt')  # specify checkpoint's path
  checkpoint = torch.load(ckpt_path, map_location=device)
  # TO DO: force the model params to be the same as checkpoint params
  train_args = model_args
  model = minivnGPT(train_args)
  # load state dict for model
  state_dict = checkpoint['model']
  model.load_state_dict(state_dict)

  # override these variables for training
  iter_num = checkpoint["iter_num"]
  best_val_loss = checkpoint["val_loss"]

model.to(device)
print(sum(p.numel() for p in model.parameters()), 'Model parameters')

# Randomly sample batch
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(high=len(data) - model_args.context_length, size=(batch_size,)) # Randomize the starting indices for each training example in a batch
  x = torch.stack([torch.tensor((data[i:i+model_args.context_length])) for i in ix])  # Stack training examples in a batch
  y = torch.stack([torch.tensor((data[i+1:i+1+model_args.context_length])) for i in ix])  # Target label for each training exampl in a batch. Each label is the next word in the sequence
  
  if device_type == 'cuda':
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
  else:
    x, y = x.to(device), y.to(device)
  return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if init_from == "resume":
  optimizer.load_state_dict(checkpoint["optimizer"])

checkpoint = None   # Free memory

# Training loop
while True:
  if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Save checkpoint after every eval_interval (same as epoch) if the loss is minimum or always save checkpoint in on
    if losses["val"] < best_val_loss or always_save_checkpoint:
      best_val_loss = losses['val']
      if iter_num > 0:
        # checkpoint contains these params
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'val_loss': losses["val"],
        }
        print(f"saving checkpoint to {out_dir}")
        NEW_CKPT_PATH = f'ckpt-{iter_num}.pt'
        torch.save(checkpoint, os.path.join(out_dir, NEW_CKPT_PATH))

    # sample a batch of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  # increment iter_num until reach max_iters
  iter_num += 1
  if iter_num > max_iters:
    break


"""
#generation:

input = "Sóng bắt đầu từ gió"
enc_input = tokenizer.encode(input)

enc_input = torch.tensor(enc_input).type(torch.long).view(1, -1)
print(enc_input.shape)

output_sentence = tokenizer.decode(model.generate(enc_input, 500)[0].tolist(), skip_special_tokens=True)

out = output_sentence
out = out.replace(r'\ n', '\n')

print(out)

"""