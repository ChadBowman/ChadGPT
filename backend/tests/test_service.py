from backend.service import build_model

test_model_params = {
    "vocab_size": 65,
    "block_size": 32,
    "n_heads": 6,
    "n_embed": 512,
    "n_layer": 6,
    "dropout": 0.2,
    "device": "cuda"
}


def test_build_model_param_filter():
    params = {"junk": "cat", **test_model_params}
    assert build_model(params)
