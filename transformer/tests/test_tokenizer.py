from pytest import raises
from transformer.tokenizer import CharTokenizer, TokenizerException


def test_missing_char_encode():
    tok = CharTokenizer(vocab=["a", "b", "c"])
    with raises(TokenizerException):
        tok.encode("d")


def test_missing_char_decode():
    tok = CharTokenizer(vocab=["a", "b", "c"])
    with raises(TokenizerException):
        tok.decode([5])
