from char_tokenizer import CharTokenizer

text = None
with open("data/shakespeare.txt", "r") as f:
    text = f.read()

tokenizer = CharTokenizer(text=text)

