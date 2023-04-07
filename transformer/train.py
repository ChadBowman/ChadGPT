import logging
import torch

log = logging.getLogger(__name__)


class Train:
    """Primary training object"""
    def __init__(self, tokenizer, model, *, batch_size, max_iters,
                 learning_rate, eval_interval, eval_iters, device):
        """Parameters:
            tokenizer (Tokenizer): tokenizer to use
            model (Transformer): language model to train
            batch_size (int): How many examples each head processes in parellel
            max_iters (int): The number of training iterations
            learning_rate (float): How agressive backpropagation should run
            eval_interval (int): How many training iterations should run between each preformance check
            eval_iters (int): How many iterations to run in the preformance check
            device (str): Device to use (cpu or cuda)
        """
        self.tokenizer = tokenizer
        self.model = model
        self.block_size = model.block_size
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.device = device

    def set_data(self, text, *, split_pro=0.9):
        """Set dataset to train on

        Parameters:
            text (str): text to train on
            split_pro (float): proportion of training set vs validation set
        """
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(split_pro * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def train(self):
        """Train model"""
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
        """Generates batch of training data"""
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
