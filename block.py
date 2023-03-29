import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(
            num_heads=n_heads, n_embed=n_embed, head_size=head_size, block_size=block_size, dropout=dropout
        )
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
