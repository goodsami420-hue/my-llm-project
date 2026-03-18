import os
import json
import tiktoken

PAD_TOKEN  = "<|pad|>"
BOS_TOKEN  = "<|user|>"
EOS_TOKEN  = "<|assistant|>"
SEP_TOKEN  = "<|sep|>"
END_TOKEN  = "<|end|>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, END_TOKEN]


class BPETokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")
        self._special = {}
        start_id = self._enc.n_vocab
        for i, tok in enumerate(SPECIAL_TOKENS):
            self._special[tok] = start_id + i
        self._id_to_special = {v: k for k, v in self._special.items()}

    @property
    def vocab_size(self):
        return self._enc.n_vocab + len(SPECIAL_TOKENS)

    @property
    def pad_id(self):  return self._special[PAD_TOKEN]
    @property
    def bos_id(self):  return self._special[BOS_TOKEN]
    @property
    def eos_id(self):  return self._special[EOS_TOKEN]
    @property
    def sep_id(self):  return self._special[SEP_TOKEN]
    @property
    def end_id(self):  return self._special[END_TOKEN]

    @property
    def id_to_token(self):
        return self._id_to_special

    def encode(self, text):
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids):
        normal = [i for i in ids if i < self._enc.n_vocab]
        if not normal:
            return ""
        return self._enc.decode(normal)

    def encode_conversation(self, user_msg):
        return [self.bos_id] + self.encode(user_msg) + [self.sep_id]

    def train(self, texts, vocab_size=50262, min_freq=2):
        print(f"Using tiktoken GPT-2 tokenizer | vocab size: {self.vocab_size}")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"type": "tiktoken", "encoding": "gpt2"}, f)
        print(f"Tokenizer saved -> {path}")

    def load(self, path):
        print(f"Tokenizer loaded | vocab size: {self.vocab_size}")