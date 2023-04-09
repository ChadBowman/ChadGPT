import tiktoken


class CharTokenizer:
    """Tokenizes text by character"""

    def __init__(self, *, vocab=None, text=None):
        self.vocab = vocab or self._parse_vocab(text)
        self.n_vocab = len(self.vocab)
        self.ch_to_token = {ch: i for i, ch in enumerate(self.vocab)}
        self.token_to_ch = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, string):
        try:
            return [self.ch_to_token[c] for c in string]
        except KeyError as e:
            raise TokenizerException(f"Unable to encode string, unknown character: {e}")

    def decode(self, tokens):
        try:
            return "".join([self.token_to_ch[token] for token in tokens])
        except KeyError as e:
            raise TokenizerException(f"Unable to decode tokens, unknown character for {e}")

    def _parse_vocab(self, text):
        if text is None:
            raise Exception("No text available to generate tokenizer")
        return sorted(list(set(text)))


class TikTokenTokenizer:
    def __init__(self, encoding="r50k_base"):
        self.encoding = tiktoken.get_encoding(encoding)
        self.n_vocab = self.encoding.n_vocab

    def encode(self, string):
        return self.encoding.encode(string)

    def decode(self, tokens):
        return self.encoding.decode(tokens)


class TokenizerException(Exception):
    pass
