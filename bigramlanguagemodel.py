import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, batch, targets=None):
        # batch and targets are both (B,T) tensor
        logits = self.token_embedding_table(batch)  # (B,T,C)
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
            # feed forward
            logits, loss = self(batch)
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
