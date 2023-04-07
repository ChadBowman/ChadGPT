# ChadGPT 🦾🤖

![ChadGPT](https://github.com/ChadBowman/ChadGPT/blob/master/assets/chadgpt.png)

My decoder-only transformer.

This project was created as a way for me to learn how transformers, specifically the self-attention mechanism, works. I relied on Andrej Karpathy's ([github](https://github.com/karpathy), [twitter](https://twitter.com/karpathy)) "[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6050s&ab_channel=AndrejKarpathy)" tutorial on YouTube to build out the bulk of this project. I highly recommend watching his video if you have a few hours to spare.

Andrej's tutorial is largely based off of the original transformer paper from 2017, [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf?).

## Installation
### Docker
The easiest way to install and run this project is with docker. Currently CUDA is not supported with this route, so if you want to train large models you will want to build the project from source. The front-end is optional of course. This is my first React project so keep expectations in line 🤓.
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

Visit the `/docs` endpoint to see the API in full. Current operations are:
* GET `/ds` get datasets
* POST `/ds/upload` upload new dataset
* GET `/lm` get language models
* POST `/lm/{name}/train` train model
* GET `/lm/{name}/eval evaluate model

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
