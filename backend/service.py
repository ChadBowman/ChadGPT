import csv
import os
import torch
from transformer.tokenizers.char_tokenizer import CharTokenizer
from transformer.train import Train
from transformer.transformer import Transformer

shake_vocab = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
shake_tokenizer = CharTokenizer(vocab=[c for c in shake_vocab])
model_params = ["vocab_size", "block_size", "n_heads", "n_embed", "n_layer", "dropout", "device"]
train_params = ["batch_size", "max_iters", "learning_rate", "eval_interval", "eval_iters", "device"]
csv_path = os.path.join("models", "hyperparams.csv")
param_map = {}


def model_path(name: str):
    return os.path.join("models", f"{name}")


def save_params():
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", *model_params])
        writer.writeheader()
        for name, params in param_map.items():
            writer.writerow({"name": name, **params})


def load_params():
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row["name"]
            del row["name"]
            ints = {k: int(v) for k, v in row.items() if "." not in v and "cpu" not in v}
            floats = {k: float(v) for k, v in row.items() if "." in v}
            param_map[name] = {**ints, **floats, "device": row["device"]}


def load_model(model_name: str):
    state = torch.load(model_path(model_name))
    load_params()
    hyperparams = param_map[model_name]
    hyperparams = {key: hyperparams[key] for key in model_params if key in hyperparams}
    model = Transformer(**hyperparams)
    model.load_state_dict(state)
    return model


def save_model(model: Transformer, model_name: str, hyperparams: dict):
    torch.save(model.state_dict(), model_path(model_name))
    hyperparams = {key: hyperparams[key] for key in model_params if key in hyperparams}
    param_map[model_name] = hyperparams
    save_params()


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
