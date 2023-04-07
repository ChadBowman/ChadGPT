# ChadGPT 🦾🤖

![ChadGPT](https://github.com/ChadBowman/ChadGPT/blob/master/assets/chadgpt.png)

This project was created as a way for me to learn and get some hands-on experience with a transformer and the self-attention mechanism. I relied on Andrej Karpathy's ([github](https://github.com/karpathy), [twitter](https://twitter.com/karpathy)) _[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6050s&ab_channel=AndrejKarpathy)_ tutorial on YouTube to build out the bulk of this project. I highly recommend watching his video if you have a few hours to spare.

Andrej's tutorial is largely based off of the original transformer paper from 2017, [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf?).

This is a decoder-only transformer which means you cannot give it a prompt or input aside from a "seed" token. It will predict the next token so acts as a document generator. I would love to use what I've learned to add in the encoder half at some point.

## Installation
### Docker
If you just want to get something to work quickly or want to interact with one of the pre-trained models, Docker is the best option. Currently CUDA is not supported with this route, so if you want to train large models you will want to build the project from source. The front-end is optional of course. This was my first React project so keep expectations in line 🤓.
```
docker pull chadbowman0/chadgpt:latest
docker pull chadbowman0/chadgpt-frontend:latest
```

### Local
This is the preferred method if you want to train large models with CUDA support, assuming you have a CUDA-capable GPU.
```
git clone git@github.com:ChadBowman/chadgpt.git ~/chadgpt
```

Create a virtual environment:
```
python -m venv ~/chadgpt/venv && source ~/chadgpt/venv/bin/activate
```

Install CUDA if you haven't already:
* [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
* [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

Install dependencies:
```
pip install -r ~/chadgpt/requirements.txt
```

## Execution
### Docker
```
docker run -p 8000:8000 chadbowman0/chadgpt:latest
```

Optional front-end:
```
docker run -p 3000:3000 chadbowman0/chadgpt-frontend:latest
```

Visit the [/docs](localhost:8000/docs) endpoint to see the API in full. Current operations are:
* GET `/ds` get datasets
* POST `/ds/upload` upload new dataset
* GET `/lm` get language models
* POST `/lm/{name}/train` train model
* GET `/lm/{name}/eval` evaluate model (generate text!)

### Local
```
python -m uvicorn backend.api:app
```

Optional front-end:
```
cd frontend
npm build run
npm install -g serve
serve -s build
```

### Tokeizers
Currently only the character tokenizer is available.

### Datasets
Currently only the complete works of Shakespeare dataset is included.

### Hyperparameters

The hyperparameters available for training are:
* `n_layer`: The number of sequential attention layers
* `n_embed`: The size of the embedding dimension
* `n_head`: The number of attention "heads" working in parallel
* `block_size`: The span of sequential tokens to use in each example (AKA context size)
* `dropout`: The proportion of nodes to "dropout" during training
* `learning_rate`: How agressive backpropagation should run
* `max_iters`: The number of training iterations
* `batch_size`: How many examples each head processes in parellel
* `vocab_size`: The number of different tokens in the model's vocabulary
* `eval_interval`: How many training iterations should run between each preformance check
* `eval_iters`: How many iterations to run in the preformance check

## Requests / UI
The [requests](https://github.com/ChadBowman/ChadGPT/blob/master/requests) directory contains a couple curl commands for convenience. They also contain reasonable hyperparameter values for various devices. Alternatively, you can use the front-end:

![ui](https://github.com/ChadBowman/ChadGPT/blob/master/assets/ui.png)

## Pre-trained models
Current models you can interact with are:
* Shakespeare (comes from Andrej's tutorial)
