import logging
import os
import torch
from .service import build_trainer, build_model, save_model, tokenizers, load_model, param_map
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
dataset = APIRouter(prefix="/ds")
lang_model = APIRouter(prefix="/lm")
model_cache = {}


@dataset.get("", tags=["dataset"])
def get_datasets():
    """Returns a collection of dataset names"""
    path = os.path.join("datasets", "input")
    return JSONResponse(content=os.listdir(path))


@dataset.post("/upload", tags=["dataset"])
def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset file

    Parameters:
    file (File): file to upload
    """
    file_path = os.path.join("datasets", "input", file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return response(f"{file.filename} dataset saved")


@lang_model.get("", tags=["model"])
def get_models():
    """Returns a collection of saved language model names"""
    models = list(filter(lambda item: ".csv" not in item, os.listdir("models")))
    return JSONResponse(content=models)


@lang_model.post("/{name}/train", tags=["model"])
async def train_model(name: str, body: dict):
    """Trains language model, saves model to file.

    Parameters:
    name (str): name of model
    body (str): JSON object, expected format example:

        {
            "dataset": "shakespeare.txt",
            "tokenizer": "character",
            "hyperparameters": {
                "vocab_size": 65,
                "block_size": 32,
                "n_heads": 3,
                "n_embed": 96,
                "n_layer": 3,
                "dropout": 0.2,
                "batch_size": 4,
                "max_iters": 100,
                "learning_rate": 1e-4,
                "eval_interval": 500,
                "eval_iters": 1000
            }
        }
    """
    log.info(f"training {name}")
    dataset = body.get("dataset", "shakespeare")
    tokenizer_name = body.get("tokenizer", "character")
    hyperparams = body.get("hyperparameters")
    hyperparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tokenizers[tokenizer_name]
    hyperparams["vocab_size"] = tokenizer.n_vocab

    model = build_model(hyperparams)
    trainer = build_trainer(dataset, tokenizer_name, model, hyperparams)
    trainer.train()

    save_model(model, name, hyperparams, tokenizer_name)
    model_cache[name] = model

    return response(f"{name} trained")


@lang_model.get("/{name}/eval", tags=["model"])
def eval_model(name: str, tokens: int = 1000, split_newlines=False):
    """Evaluates language model

    Parameters:
    name (str): name of model
    token (int): number of tokens to generate
    split_newlines (bool): returns a newline split array if true

    Returns:
    The model's output up to {tokens} length.
    """
    log.info(f"generating {tokens} tokens using the {name} model")
    model = model_cache.get(name)
    if not model:
        model = load_model(name)
    seed = torch.zeros((1, 1), dtype=torch.long, device=model.device)
    result = model.generate(seed, max_new_tokens=tokens)[0].tolist()
    tokenizer = param_map[name]["tokenizer"]
    result = tokenizers[tokenizer].decode(result)
    if split_newlines:
        result = result.split("\n")
    return JSONResponse(content={"eval": result})


def response(message):
    return JSONResponse(content={"message": message})
