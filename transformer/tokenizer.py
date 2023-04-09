import tiktoken


class SimpleTokenizer:
    """Tokenizes text by character"""
    CHAR = "char"
    WORD = "word"

    def __init__(self, tokenizer_type, *, vocab=None, text=None):
        self.tokenizer_type = tokenizer_type
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

        if self.tokenizer_type == SimpleTokenizer.CHAR:
            vocab_set = set(text)
        elif self.tokenizer_type == SimpleTokenizer.WORD:
            special = "\n !$&',-.3:;?"
            vocab_set = set()
            for line in text.split("\n"):
                for word in line.split(" "):
                    for char in special:
                        word = word.replace(char, "")
                    vocab_set.add(word)
            for char in special:
                vocab_set.add(char)

        return sorted(list(vocab_set))


class CharTokenizer(SimpleTokenizer):
    def __init__(self, *, vocab=None, text=None):
        super().__init__(SimpleTokenizer.CHAR, vocab=vocab, text=text)


class WordTokenizer(SimpleTokenizer):
    def __init__(self, *, vocab=None, text=None):
        self.vocab = vocab or self._parse_vocab(text)
        print(" ".join(self.vocab))
        self.n_vocab = len(self.vocab)
        print(self.n_vocab)
        self.ch_to_token = {ch: i for i, ch in enumerate(self.vocab)}
        self.token_to_ch = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, string):
        try:
            return [self.ch_to_token[c] for c in self.gen_word(string)]
        except KeyError as e:
            raise TokenizerException(f"Unable to encode string, unknown character: {e}")

    def decode(self, tokens):
        try:
            return "".join([self.token_to_ch[token] for token in tokens])
        except KeyError as e:
            raise TokenizerException(f"Unable to decode tokens, unknown character for {e}")

    def gen_word(self, text):
        special = [ch for ch in "\n !$&',-.3:;?"]
        for line in text.split("\n"):
            for word in line.split(" "):
                punct = []
                while len(word) > 1 and word[-1] in special:
                    punct = [word[-1], *punct]
                    word = word[0:-1]
                yield word
                for p in punct:
                    yield p
                yield " "
            yield "\n"

    def _parse_vocab(self, text):
        if text is None:
            raise Exception("No text available to generate tokenizer")
        vocab_set = {word for word in self.gen_word(text)}
        return sorted(list(vocab_set))


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
