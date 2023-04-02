import os
import torch
from .service import build_trainer, load_model
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from transformer.tokenizers.char_tokenizer import BASIC
from transformer.transformer import Transformer

dataset = APIRouter(prefix="/ds")
lang_model = APIRouter(prefix="/lm")


def response(message):
    return JSONResponse(content={"message": message})


@dataset.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    file_path = os.path.join("datasets", "input", file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return response(f"{file.filename} dataset saved")


@lang_model.post("/{name}")
def create_model(name: str, body: dict):
    """
    hyperparameters
    version
    """
    hyperparams = body.get("hyperparameters")
    if not hyperparams:
        raise Exception("hyperparameters required")

    version = body.get("version", "0.1.0")
    name = f"{name}V{version}.pt"
    file_path = os.path.join("models", name)
    model = Transformer(**hyperparams)
    torch.save(model.state_dict(), file_path)

    return response(f"{name} created")


@lang_model.post("/{name}/train")
async def train_model(name: str, body: dict):
    """
    dataset
    tokenizer
    model
    hyperparameters
        batch_size
        block_size
        max_iters
        eval_interval
        eval_iters
        learning_rate
    """
    dataset = body.get("dataset", "shakespeare")
    tokenizer = body.get("tokenizer", "character")
    model = body.get("model")
    hyperparams = body.get("hyperparameters")
    if not hyperparams:
        raise Exception("hyperparameters required")
    trainer = build_trainer(dataset, tokenizer, model, hyperparams)
    trainer.train()


@lang_model.get("/{name}/eval")
def eval_model(name: str, tokens: int = 1000):
    model = load_model(name)
    seed = torch.zeros((1, 1), dtype=torch.long, device=model.device)
    result = model.generate(seed, max_new_tokens=tokens)[0].tolist()
    result = BASIC.decode(result)
    return response(result)
