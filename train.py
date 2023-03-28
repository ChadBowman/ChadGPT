import logging
import torch
from char_tokenizer import CharTokenizer
from lm import BigramLanguageModel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Train:
    def __init__(self, model, tokenizer, *,
                 batch_size, block_size, max_iters,
                 eval_interval, eval_iters, learning_rate):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.learning_rate = learning_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_data(self, text, *, split_pro=0.9):
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(split_pro * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        model = self.model.to(self.device)
        log.info(f"training model on {next(model.parameters()).device}")

        for step in range(self.max_iters):
            # evaluate losses on train and validation set periodically
            if step % self.eval_interval == 0:
                losses = self._estimate_loss()
                log.info(f"step {step}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

            # sample a batch
            xb, yb = self._generate_batch("train")

            # evaluate loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return model

    def _generate_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        # get batch_size number of random integers to sample from
        idxs = torch.randint(len(data) - self.block_size, (self.batch_size,))
        # stack a batch of training vectors
        x = torch.stack([data[i:i + self.block_size] for i in idxs])
        # stack a batch of validation vectors (supervised learning)
        # this is the next token to predict
        y = torch.stack([data[i + 1: i + self.block_size + 1] for i in idxs])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def _estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self._generate_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out


text = None
with open("data/shakespeare.txt", "r") as f:
    text = f.read()

block_size = 8
tokenizer = CharTokenizer(text=text)
model = BigramLanguageModel(vocab_size=tokenizer.vocab_n,
                            block_size=block_size,
                            n_embed=32)
train = Train(model, tokenizer,
              batch_size=32,
              block_size=block_size,
              max_iters=50000,
              eval_interval=500,
              eval_iters=200,
              learning_rate=1e-3)

# train the model
train.set_data(text)
model = train.train()

# generate from model
# start with 0 (probably newline character)
seed_batch = torch.zeros((1, 1), dtype=torch.long, device=train.device)
result = model.generate(seed_batch, max_new_tokens=500)[0].tolist()
result = tokenizer.decode(result)
print(result)
