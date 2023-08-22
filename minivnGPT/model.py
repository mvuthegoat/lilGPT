import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    functional as F,
)  # can be directly applied to input tensors, don't need to create a module to use it

from dataclasses import dataclass
from typing import Optional, Union, Tuple


def precompute_freqs_cis(d_model: int, pos: int, theta: float = 10000.0):
    """Compute frequencies in complex form (cis(x) = cos(x) + isin(x) = e^ix)"""

    # Faster way to calculate the original position embedding function
    freqs = 1.0 / (
        theta ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model)
    )
    t = torch.arange(pos, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # same as pos/10000^(2i/d_model)

    # Convert to complex form using the cis(x) formula
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Ensure freqs_cis has the proper shape for broadcasting when perform element-wise operations"""

    ndim = x.ndim
    assert 0 <= 1 < ndim
    # x has shape (bs, context_length, d_model), freqs_cis has shape (context_length, d_model)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]  # only preserve some dimensions, others are set to 1
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary Position Embedding"""

    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, q_)
    # flatten the tensor in the 2 dimension (the imaginary part)
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(2)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(2)
    return q_out.type_as(q), k_out.type_as(k)


class Head(nn.Module):
    """Single head of self-attention"""

    def __init__(self, config):
        # d_k is the dimension for Q, K, V
        # d_model is the embedding dimension
        super().__init__()
        d_k = config.d_model // config.n_head
        self.query = nn.Linear(config.d_model, d_k, bias=False)
        self.key = nn.Linear(config.d_model, d_k, bias=False)
        self.value = nn.Linear(config.d_model, d_k, bias=False)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.context_length, config.context_length))
        )  # Save trill as the model's attribute
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        B, T, C = q.shape
        # Here, C is dk (dimension of key,value,query)
        # compute attention scores. @ is batch multiply
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, C) @ (B, C, T) -> (B, T, T); k.transpose(-2,-1) transposes k in the last 2 dimensions
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T). Apply look ahead mask
        wei = F.softmax(
            wei, dim=-1
        )  # (B, T, T). Apply softmax to get the weights before aggregate v (values)
        wei = self.dropout(wei)  # Apply dropout

        attention_scores = (
            wei @ v
        )  # Aggregate v. (B, T, T) @ (B, T, dk) -> (B, T, dk) = (B, T, C)
        return attention_scores


class MultiHeadAttention(nn.Module):
    """Multi-head Attention"""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        # Concatenate all heads from the self.heads list along the channel dimension (last dimension)
        concat = torch.cat(
            [h.forward(x, freqs_cis) for h in self.heads], dim=-1
        )  # (B, T, d_k * n_head) = (B, T, d_model)
        # Apply projection/ linear transformation to this concat
        out = self.proj(concat)  # (B, T, d_model)
        out = self.dropout(out)  # (B, T, d_model)
        return out


class FeedForward(nn.Module):
    """Design a simple feedforward linear layer followed by a non-linearity"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                config.d_model, 4 * config.d_model
            ),  # Use the out_features as 4 * d_model as the paper uses
            nn.ReLU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A single Decoder layer / block"""

    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ffn = nn.Linear(config.d_model, config.d_model)
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.layernorm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        # Slight deviation from original paper: layernorm is applied before going through mha or ffn
        x = x + self.mha.forward(
            self.layernorm1(x), freqs_cis
        )  # We add the previous term because of residual/skip connection -> optimize training
        x = x + self.ffn(self.layernorm2(x))
        x = self.dropout(x)
        return x


@dataclass
class ModelArguments:
    d_model: int = 768
    n_head: int = 12
    n_layer: int = 12
    context_length: int = 2048
    dropout_rate: float = 0.0
    bias: bool = False
    vocab_size: int = -1  # defined later by tokenizer
    num_labels: int = 2  # Number of labels to use in the last layer added to the model, typically for a classification task (2 for Q&A fine tuning)


class minivnGPT(nn.Module):
    """Decoder-only transformer"""

    def __init__(
        self, config
    ):  # Suggested parameters: (self, n_layer, d_model, n_head, context_length)
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.freqs_cis = precompute_freqs_cis(
            self.config.d_model // self.config.n_head, self.config.context_length
        )
        # self.pos_encoding = positional_encoding(context_length, d_model) (context_length specified here might be different from the input's context length)
        self.decoder_layers = nn.ModuleList()
        for _ in range(self.config.n_layer):
            self.decoder_layers.append(Block(self.config))

        self.layernorm_final = nn.LayerNorm(config.d_model)
        self.linear_final = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x: torch.Tensor, targets=None):
        # Initially, x has shape (B, T)
        # encode both the token and position embeddings
        device = x.device
        x = self.embedding(x)
        self.freqs_cis = self.freqs_cis.to(x.device)

        for layer in self.decoder_layers:
            x = layer(x, self.freqs_cis)  # (B, T, d_model)

        x = self.layernorm_final(x)  # (B, T, d_model)

        logits = self.linear_final(x)  # (B, T, vocab_size)

        # Similar to the Bigram model
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(
        self,
        x,
        max_new_tokens,
    ):
        # Generate each new token until max_new_token is reached
        for _ in range(max_new_tokens):
            # length of input must be equal to context_length -> truncate
            x_trunc = x[:, -self.config.context_length :]
            # Call the forward function to get the logits / predictions
            logits, loss = self(x_trunc)
            # Get the element from the last time-step
            logits = logits[:, -1, :]  # (B, C)
            # Apply softmax to get probabilities for all classes along the first dimension (class dimension, not the batch dimension)
            probs = F.softmax(logits, dim=1)  # (B, C)
            x_next = torch.multinomial(
                input=probs, num_samples=1
            )  # (B, 1). bc num_samples=1
            # Concat x_next to the input x along the 1st dimension (the time-step dimension)
            x = torch.cat((x, x_next), dim=1)  # (B, T+1).  Token at the T+1 time-step

        return x


@dataclass
class QAModelOutput:
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None


class minivnGPTForQA(nn.Module):
    def __init__(self, pre_trained_model, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(
            config.vocab_size, config.num_labels
        )  # last linear layer will generate output of 2 dimensions
        self.minivngpt = pre_trained_model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ):
        # get the outputs from the last layer of the pre-trained model
        outputs = self.minivngpt(input_ids)
        # outputs = torch.tensor(outputs)
        # print(f"outputs after minivngpt: {type(outputs[0])}") #### TUPLE??
        # print(outputs)

        # then, pass this output to the new layer qa_outputs specific for QA task
        logits = self.qa_outputs(outputs[0])

        # print(f"outputs after qa_outputs: {type(logits)}")
        # get the start_logits and end_logits by splitting logits into 2 parts
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # compute total loss
        total_loss = None

        if start_positions is not None and end_positions is not None:
            # if starting and ending positions are out of range (context_length), clamp them within context_length
            ignored_index = start_logits.shape[
                1
            ]  # start_logits.shape[1] is the context_length
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # loss functions
            start_loss = F.cross_entropy(
                start_logits, start_positions, ignore_index=ignored_index
            )
            end_loss = F.cross_entropy(
                end_logits, end_positions, ignore_index=ignored_index
            )
            total_loss = (start_loss + end_loss) / 2

        # return object under the format of QAModelOutput
        return QAModelOutput(
            loss=total_loss, start_logits=start_logits, end_logits=end_logits
        )

    def generate(self, x):
        outputs = self(input_ids=x)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # pick the highest probability index --> use argmax
        answer_start = torch.argmax(start_scores, dim=1)
        answer_end = torch.argmax(end_scores, dim=1)

        return answer_start, answer_end
