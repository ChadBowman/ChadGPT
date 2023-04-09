import csv
import os
import torch
import logging
from transformer.tokenizer import TikTokenTokenizer, CharTokenizer, WordTokenizer
from transformer.train import Train
from transformer.transformer import Transformer


def get_text():
    with open("datasets/input/shakespeare.txt", "r") as file:
        return file.read()


log = logging.getLogger(__name__)
shake_vocab = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
tokenizers = {
    "tiktoken": TikTokenTokenizer(encoding="r50k_base"),
    "character": CharTokenizer(vocab=[c for c in shake_vocab]),
    "word": WordTokenizer(text=get_text())
}
model_params = ["vocab_size", "block_size", "n_heads", "n_embed", "n_layer", "dropout", "device"]
train_params = ["batch_size", "max_iters", "learning_rate", "eval_interval", "eval_iters", "device"]
csv_path = os.path.join("models", "hyperparams.csv")
param_map = {}  # TODO yuck


def model_path(name: str):
    """Get the models path location"""
    return os.path.join("models", f"{name}")


def save_params():
    """Save model parameters to file"""
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "tokenizer", *model_params])
        writer.writeheader()
        for name, params in param_map.items():
            writer.writerow({"name": name, **params})


def load_params():
    """Load model parameters from file"""
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row["name"]
            del row["name"]
            # this is terrible
            ints = {k: int(v) for k, v in row.items()
                    if "." not in v and "c" not in v and "t" not in v}
            floats = {k: float(v) for k, v in row.items() if "." in v}
            # TODO fine a better way to manage tokenizer
            param_map[name] = {
                **ints, **floats, "device": row["device"], "tokenizer": row["tokenizer"]
            }


def load_model(model_name: str):
    """Load model from file"""
    local_device_setting = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(model_path(model_name), map_location=torch.device(local_device_setting))
    load_params()
    hyperparams = param_map[model_name]
    hyperparams = {key: hyperparams[key] for key in model_params if key in hyperparams}
    hyperparams["device"] = local_device_setting  # override whatever is in the file
    model = Transformer(**hyperparams)
    model.load_state_dict(state)
    return model


def save_model(model: Transformer, model_name: str, hyperparams: dict, tokenizer_name: str):
    """Save model to file"""
    torch.save(model.state_dict(), model_path(model_name))
    hyperparams = {key: hyperparams[key] for key in model_params if key in hyperparams}
    hyperparams["tokenizer"] = tokenizer_name
    param_map[model_name] = hyperparams
    save_params()


def build_model(hyperparams: dict):
    """Build new model"""
    hyperparams = {key: hyperparams[key] for key in model_params if key in hyperparams}
    model = Transformer(**hyperparams)
    return model


def build_trainer(dataset_name, tokenizer_name, model, hyperparams):
    """Build new training object"""
    # TODO actually use dataset_name
    dataset_path = os.path.join("datasets", "input", "shakespeare.txt")
    with open(dataset_path, "r") as file:
        data = file.read()

    tokenizer = tokenizers[tokenizer_name]

    params = {key: hyperparams[key] for key in train_params if key in hyperparams}
    trainer = Train(tokenizer, model, **params)
    trainer.set_data(data, split_pro=0.9)

    log.info(f"trainer built using dataset {dataset_name}, tokenizer {tokenizer_name}, model {model}")

    return trainer
