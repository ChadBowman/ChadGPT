import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ A head of self-attention """
    def __init__(self, *, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        # normalize
        weights = q @ k.transpose(-2, -1) * C**-0.5
        # Tril to create a "moving average" of T
        # Also, we are only allowing the past to communicate with any node,
        # making this a decoder block
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        # softmax to normalize to probability
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
