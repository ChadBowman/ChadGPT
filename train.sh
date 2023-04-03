http POST :8000/lm/small/train \
    dataset=shakespeare.txt \
    tokenizer=char \
    hyperparameters:='{
        "vocab_size": 65,
        "block_size": 32,
        "n_heads": 3,
        "n_embed": 96,
        "n_layer": 3,
        "dropout": 0.2,
        "batch_size": 4,
        "max_iters": 3000,
        "learning_rate": 1e-4,
        "eval_interval": 500,
        "eval_iters": 200
    }'
