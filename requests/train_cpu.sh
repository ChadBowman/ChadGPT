curl -X POST localhost:8000/lm/shakespeare_small/train \
    -H "Content-Type: application/json" \
    -d '{
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
            "max_iters": 3000,
            "learning_rate": 1e-3,
            "eval_interval": 500,
            "eval_iters": 200
        }
    }'
