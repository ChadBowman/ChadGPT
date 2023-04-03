import logging
from fastapi import FastAPI
from .router import dataset, lang_model

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.include_router(dataset)
app.include_router(lang_model)


@app.get("/")
async def root():
    return "Hello, World!"
