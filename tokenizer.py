import os
import json
import tiktoken

# Special tokens — must NOT be split into subwords
SPECIAL_TOKENS = {
    "<|user|>":      50257,
    "<|assistant|>": 50258,
    "<|eos|>":       50259,
    "<|pad|>":       50260,
    "<|end|>":       50261,
}

VOCAB_SIZE = 50262  # 50257 (gpt2) + 5 special tokens


class BPETokenizer:
    def __init__(self):
        # Register special tokens so they are NEVER split into subwords
        self._enc = tiktoken.get_encoding("gpt2")
        self._special = SPECIAL_TOKENS
        self._id_to_token = {v: k for k, v in SPECIAL_TOKENS.items()}

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    @property
    def pad_id(self):       return SPECIAL_TOKENS["<|pad|>"]
    @property
    def bos_id(self):       return SPECIAL_TOKENS["<|user|>"]
    @property
    def eos_id(self):       return SPECIAL_TOKENS["<|eos|>"]
    @property
    def assistant_id(self): return SPECIAL_TOKENS["<|assistant|>"]
    @property
    def end_id(self):       return SPECIAL_TOKENS["<|end|>"]

    @property
    def id_to_token(self):
        return self._id_to_token

    def encode(self, text):
        """Encode text, keeping special tokens intact."""
        # Split on special tokens first so they are never broken into subwords
        import re
        pattern = "(" + "|".join(re.escape(k) for k in SPECIAL_TOKENS.keys()) + ")"
        parts   = re.split(pattern, text)
        ids     = []
        for part in parts:
            if part in SPECIAL_TOKENS:
                ids.append(SPECIAL_TOKENS[part])
            elif part:
                ids.extend(self._enc.encode(part))
        return ids

    def decode(self, ids):
        normal = [i for i in ids if i < self._enc.n_vocab]
        if not normal:
            return ""
        return self._enc.decode(normal)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"type": "tiktoken", "encoding": "gpt2", "special_tokens": SPECIAL_TOKENS}, f)
        print(f"Tokenizer saved -> {path}")

    def load(self, path):
        print(f"Tokenizer loaded | vocab size: {self.vocab_size}")
