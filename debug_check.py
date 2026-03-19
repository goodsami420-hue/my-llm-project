"""
debug_check.py — training data সঠিক কিনা verify করো
python debug_check.py চালাও pretokenize এর পরে
"""
import json
from tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.load("tokenizer/tokenizer.json")

with open("data/tokenized_data.json", "r") as f:
    samples = json.load(f)

print(f"Total samples: {len(samples)}")
print()

# first 3 samples check
for i in range(min(3, len(samples))):
    s     = samples[i]
    ids   = s["ids"]
    mask  = s["mask"]
    asst_count = sum(1 for m in mask if m == 1)
    user_count = sum(1 for m in mask if m == -1)

    print(f"Sample {i}:")
    print(f"  Length    : {len(ids)}")
    print(f"  User tokens   : {user_count}")
    print(f"  Asst tokens   : {asst_count}")
    print(f"  Decoded (first 80 chars): {tokenizer.decode(ids[:40])!r}")
    print()

# check special tokens
print(f"Special token IDs:")
print(f"  <|user|>      = {tokenizer.bos_id}")
print(f"  <|assistant|> = {tokenizer.assistant_id}")
print(f"  <|eos|>       = {tokenizer.eos_id}")
print()

# verify first sample starts with user token
first_ids = samples[0]["ids"]
if first_ids[0] == tokenizer.bos_id:
    print("GOOD: First token is <|user|>")
else:
    print(f"WARNING: First token is {first_ids[0]}, expected {tokenizer.bos_id}")
