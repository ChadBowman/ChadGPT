import torch
import torch.nn as nn
from .block import Block
from torch.nn import functional as F


class Transformer(nn.Module):
    def __init__(self, *, vocab_size, block_size, n_heads, n_embed, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, batch, targets=None):
        B, T = batch.shape
        # batch and targets are both (B,T) tensor
        token_emb = self.token_embedding_table(batch)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = token_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # B is the batch dimension
        # T is is the time dimension (a training vector, sequence of tokens)
        # C is the channel (the score of each next token, the prediction)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, batch, max_new_tokens):
        """
        Takes a batch of samples in the shape (B,T) where:
            B (batch) samples to run in parallel
            T (time) sequence of tokens to be used as context
                to predict the next token

        Concatenates the result to the sequence T, max_new_tokens times;
            to be used in the next iteration.
        """
        for _ in range(max_new_tokens):
            # truncate batch to the last block_size tokens
            batch_truncated = batch[:, -self.block_size:]
            # feed forward
            logits, loss = self(batch_truncated)
            # grab the last token in each sequence (for now)
            logits = logits[:, -1, :]  # (B,C)
            # apply softmax to get probabilities for each token
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            predictions = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled prediction to the sequence,
            # creating a new batch for the next iteration
            batch = torch.cat((batch, predictions), dim=1)  # (B,T+1)
        return batch
