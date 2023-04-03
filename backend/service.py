import os
import torch
from transformer.tokenizers.char_tokenizer import CharTokenizer
from transformer.train import Train
from transformer.transformer import Transformer

shake_vocab = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
shake_tokenizer = CharTokenizer(vocab=[c for c in shake_vocab])
model_params = ["vocab_size", "block_size", "n_heads", "n_embed", "n_layer", "dropout", "device"]
train_params = ["batch_size", "max_iters", "learning_rate", "eval_interval", "eval_iters", "device"]


def model_path(name: str):
    return os.path.join("models", f"{name}")


def load_model(model_name: str, hyperparams: dict):
    state = torch.load(model_path(model_name))
    hyperparams = {key: hyperparams[key] for key in model_params if key in hyperparams}
    model = Transformer(**hyperparams)
    model.load_state_dict(state)
    return model


def save_model(model: Transformer, model_name: str):
    torch.save(model.state_dict(), model_path(model_name))


def build_model(hyperparams: dict):
    hyperparams = {key: hyperparams[key] for key in model_params if key in hyperparams}
    model = Transformer(**hyperparams)
    return model


def build_trainer(dataset_name, tokenizer_name, model, hyperparams):
    # TODO actually use dataset_name, custom tokenizer
    dataset_path = os.path.join("datasets", "input", "shakespeare.txt")
    with open(dataset_path, "r") as file:
        data = file.read()
    tokenizer = shake_tokenizer

    params = {key: hyperparams[key] for key in train_params if key in hyperparams}
    trainer = Train(tokenizer, model, **params)
    trainer.set_data(data, split_pro=0.9)

    return trainer
