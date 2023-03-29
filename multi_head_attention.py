import torch
import torch.nn as nn
from head import Head


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, *, num_heads, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(
            n_embed=n_embed,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout
        ) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
