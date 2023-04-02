import os
import torch
from transformer.tokenizers.char_tokenizer import CharTokenizer
from transformer.train import Train
from transformer.transformer import Transformer

model_cache = {}


def load_model(model_name: str):
    cache_result = model_cache.get(model_name)
    if cache_result:
        return cache_result
    model_path = os.path.join("models", f"{model_name}.pt")
    model = torch.load(model_path)
    model_cache[model_name] = model
    return model


def build_trainer(dataset_name, tokenizer_name, model_name, hyperparams):
    dataset_path = os.path.join("datasets", "inputs", "shakespeare.txt")
    with open(dataset_path, "r") as file:
        data = file.read()

    tokenizer = CharTokenizer(text=data)
    model = load_model(model_name)
    trainer = Train(model, tokenizer, **hyperparams)
    trainer.set_data(data)
    return trainer
