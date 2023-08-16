import torch
import torch.nn as nn
from torch.nn import functional as F # can be directly applied to input tensors, don't need to create a module to use it

#from minivnGPT_old import Transformer
import numpy as np
import pandas as pd

# Hyperparameters
batch_size = 16
context_length = 32
max_iters = 1000
eval_interval = 100
eval_iters = 200
learning_rate = 1e-3 
d_model = 64
n_head = 4
n_layer = 4
dropout_rate = 0.2
vocab_size = 64000 # 64000 for PhoBERT

def get_angles(pos, k, d_model):
  """
  Get the angles for the positional encoding
  
  Arguments:
      pos -- Column vector containing the positions [[0], [1], ...,[N-1]]. Shape: (pos, 1)
      k --   Row vector containing the dimension span [[0, 1, 2, ..., d_model-1]]. Shape (d_model, 1)
      d_model(integer) -- embedding dim
  
  Returns:
      angles -- (pos, d_model) numpy array 
  """
  i = k // 2     # (1, d_model)
  angles = pos / np.power(10000, 2 * i / d_model)    # (pos, d_model)
  return angles

def positional_encoding(positions, d_model):
  """
  Precomputes a matrix with all the positional encodings 
  
  Arguments:
      positions (int) -- Maximum number of positions to be encoded = context_length = number of time steps
      d_model (int) -- embedding dim (encoding size) 
  
  Returns:
      pos_encoding -- (1, position, d_model) A matrix with the positional encodings
  """
  # Get the angles
  angle_rads = get_angles(np.arange(positions)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
  # Use sine and cosine of different functions
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) 
  
  pos_encoding = angle_rads[np.newaxis, ...] # (1, T, C)
  return torch.from_numpy(pos_encoding).type(torch.float32) # (position, d_model) = (T, C)


'''
pos_enc = positional_encoding(context_length, d_model)
print(pos_enc.shape)

a = torch.randn(16, 32, 64)
res = a + pos_enc
print(res)
'''


torch.manual_seed(1337)

class Head(nn.Module):
  """ Single head of self-attention"""
  def __init__(self, d_k):
    # d_k is the dimension for Q, K, V
    # d_model is the embedding dimension
    super().__init__()
    self.query = nn.Linear(d_model, d_k, bias=False)
    self.key = nn.Linear(d_model, d_k, bias=False)
    self.value = nn.Linear(d_model, d_k, bias=False)

    self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))  # Save trill as the model's attribute
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    
    B, T, C = q.shape
    # Here, C is dk (dimension of key,value,query)
    # compute attention scores. @ is batch multiply
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T); k.transpose(-2,-1) transposes k in the last 2 dimensions
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T). Apply look ahead mask
    wei = F.softmax(wei, dim=-1) # (B, T, T). Apply softmax to get the weights before aggregate v (values)
    wei = self.dropout(wei)  # Apply dropout

    attention_scores = wei @ v # Aggregate v. (B, T, T) @ (B, T, dk) -> (B, T, dk) = (B, T, C)
    return attention_scores

class MultiHeadAttention(nn.Module):
  """ Multi-head Attention"""
  def __init__(self, n_head, d_k):
    super().__init__()
    self.heads = nn.ModuleList([Head(d_k) for _ in range(n_head)])
    self.proj = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout_rate)
    
  def forward(self, x):
    # Concatenate all heads from the self.heads list along the channel dimension (last dimension)
    concat = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, d_k * n_head) = (B, T, d_model)
    # Apply projection/ linear transformation to this concat
    out = self.proj(concat) # (B, T, d_model)
    out = self.dropout(out) # (B, T, d_model)
    return out

class FeedForward(nn.Module):
  """ Design a simple feedforward linear layer followed by a non-linearity """
  def __init__(self, d_model):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(d_model, 4 * d_model), # Use the out_features as 4 * d_model as the paper uses
        nn.ReLU(),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(dropout_rate)
    )
  
  def forward(self, x):
    return self.net(x)

class DecoderLayer(nn.Module):
  """ A single Decoder layer / block """
  def __init__(self, d_model, n_head):
    super().__init__()
    self.mha = MultiHeadAttention(n_head, d_model // n_head)
    self.ffn = nn.Linear(d_model, d_model)
    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):
    # Slight deviation from original paper: layernorm is applied before going through mha or ffn
    x = x + self.mha(self.layernorm1(x))  # We add the previous term because of residual/skip connection -> optimize training
    x = x + self.ffn(self.layernorm2(x))
    x = self.dropout(x)
    return x

class Transformer(nn.Module):
  """ Decoder-only transformer """
  def __init__(self):   # Suggested parameters: (self, n_layer, d_model, n_head, context_length)
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    #self.pos_encoding = positional_encoding(context_length, d_model) (context_length specified here might be different from the input's context length)
    self.decoder_layers = nn.Sequential(*[DecoderLayer(d_model, n_head) for _ in range(n_layer)])
    self.layernorm_final = nn.LayerNorm(d_model)
    self.linear_final = nn.Linear(d_model, vocab_size)

  def forward(self, x, targets = None):
    # Initially, x has shape (B, T)
    # encode both the embedding and position
    token_emb = self.embedding(x)
    position_emb = positional_encoding(x.shape[1], d_model) # x.shape[1] is the context_length of input
    x = token_emb + position_emb # pos_encoding has shape (T, d_model). self.embedding(x) has shape (B, T, d_model). So, pos_encoding will be broadcasted
    # x now has shape (B, T, d_model)

    x = self.decoder_layers(x) # (B, T, d_model)
    x = self.layernorm_final(x) # (B, T, d_model)

    logits = self.linear_final(x) # (B, T, vocab_size)

    # Similar to the Bigram model
    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(-1)

      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, x, max_new_tokens):
    # Generate each new token until max_new_token is reached
    for _ in range(max_new_tokens):
      # length of input must be equal to context_length -> truncate
      x_trunc = x[:, -context_length:]
      # Call the forward function to get the logits / predictions
      logits, loss = self(x_trunc) 
      # Get the element from the last time-step
      logits = logits[:, -1, :] # (B, C)
      # Apply softmax to get probabilities for all classes along the first dimension (class dimension, not the batch dimension)
      probs = F.softmax(logits, dim=1) # (B, C)
      x_next = torch.multinomial(input=probs, num_samples=1) # (B, 1). bc num_samples=1
      # Concat x_next to the input x along the 1st dimension (the time-step dimension)
      x = torch.cat((x, x_next), dim=1) # (B, T+1).  Token at the T+1 time-step

    return x