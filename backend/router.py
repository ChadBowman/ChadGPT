import os
import torch
from .service import build_trainer, build_model, save_model, shake_tokenizer, load_model
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

dataset = APIRouter(prefix="/ds")
lang_model = APIRouter(prefix="/lm")
model_cache = {}


def response(message):
    return JSONResponse(content={"message": message})


@dataset.get("")
def get_datasets():
    path = os.path.join("datasets", "input")
    return JSONResponse(content=os.listdir(path))


@dataset.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    file_path = os.path.join("datasets", "input", file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return response(f"{file.filename} dataset saved")


@lang_model.get("")
def get_models():
    return JSONResponse(content=os.listdir("models"))


@lang_model.post("/{name}/train")
async def train_model(name: str, body: dict):
    """
    dataset
    tokenizer
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
    hyperparams = body.get("hyperparameters")
    hyperparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(hyperparams)
    trainer = build_trainer(dataset, tokenizer, model, hyperparams)
    trainer.train()

    save_model(model, name, hyperparams)
    model_cache[name] = model

    return response(f"{name} trained")


@lang_model.get("/{name}/eval")
def eval_model(name: str, tokens: int = 1000, split_newlines=False):
    model = model_cache.get(name)
    if not model:
        model = load_model(name)
    seed = torch.zeros((1, 1), dtype=torch.long, device=model.device)
    result = model.generate(seed, max_new_tokens=tokens)[0].tolist()
    result = shake_tokenizer.decode(result)
    if split_newlines:
        result = result.split("\n")
    return JSONResponse(content={"eval": result})
