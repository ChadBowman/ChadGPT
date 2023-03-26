from char_tokenizer import CharTokenizer
from bigramlanguagemodel import BigramLanguageModel
import torch

text = None
with open("data/shakespeare.txt", "r") as f:
    text = f.read()

tokenizer = CharTokenizer(text=text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 32  # how many samples we will process in parallel
block_size = 8  # the maximum context length for predictions


def generate_batch(split):
    data = train_data if split == "train" else val_data
    # get batch_size number of random integers to sample from
    idxs = torch.randint(len(data) - block_size, (batch_size,))
    # stack a batch of training vectors
    x = torch.stack([data[i:i + block_size] for i in idxs])
    # stack a batch of validation vectors (supervised learning)
    # this is the next token to predict
    y = torch.stack([data[i + 1: i + block_size + 1] for i in idxs])
    return x, y


xb, yb = generate_batch("train")

m = BigramLanguageModel(tokenizer.vocab_n)
logits, loss = m(xb, yb)
# we expect a loss of -ln(1/vocab_n) = -ln(1/65) = 4.174

# start with 0 (probably newline character)
idx = torch.zeros((1, 1), dtype=torch.long)
#result = m.generate(idx, max_new_tokens=100)
#print(tokenizer.decode(result[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(1000):
    xb, yb = generate_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

result = m.generate(idx, max_new_tokens=500)
print(tokenizer.decode(result[0].tolist()))
print(loss.item())
