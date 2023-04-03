class CharTokenizer:
    def __init__(self, *, vocab=None, text=None):
        self.vocab = vocab or self._parse_vocab(text)
        self.vocab_n = len(self.vocab)
        self.ch_to_token = {ch: i for i, ch in enumerate(self.vocab)}
        self.token_to_ch = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, string):
        return [self.ch_to_token[c] for c in string]

    def decode(self, tokens):
        return "".join([self.token_to_ch[token] for token in tokens])

    def _parse_vocab(self, text):
        if text is None:
            raise Exception("No text available to genearte tokenizer")
        return sorted(list(set(text)))
