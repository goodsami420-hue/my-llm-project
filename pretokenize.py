import os
import json
from tokenizer import BPETokenizer
from tqdm import tqdm

BLOCK_SIZE = 128  # must match config.py block_size

print("Pre-tokenizing all data...")

os.makedirs("tokenizer", exist_ok=True)
os.makedirs("data", exist_ok=True)

tokenizer = BPETokenizer()
tokenizer.save("tokenizer/tokenizer.json")

with open("data/training_data.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)

all_samples = []
skipped = 0

for conv in tqdm(conversations):
    turns = conv.get("conversations", [])
    ids   = []
    mask  = []

    for i in range(0, len(turns) - 1, 2):
        if i + 1 >= len(turns):
            break

        user = turns[i]
        asst = turns[i + 1]

        if user.get("role") != "user" or asst.get("role") != "assistant":
            skipped += 1
            continue

        # Format: <|user|> text \n <|assistant|> text <|eos|>
        user_text = "<|user|> " + user.get("content", "").strip()
        asst_text = "<|assistant|> " + asst.get("content", "").strip()

        user_ids = tokenizer.encode(user_text)
        asst_ids = tokenizer.encode(asst_text) + [tokenizer.eos_id]

        # user tokens: masked (-1), assistant tokens: trained (1)
        ids.extend(user_ids + asst_ids)
        mask.extend([-1] * len(user_ids) + [1] * len(asst_ids))

    if len(ids) < 4:
        skipped += 1
        continue

    # Truncation: sliding window chunks of block_size
    for start in range(0, len(ids) - BLOCK_SIZE, BLOCK_SIZE // 2):
        chunk      = ids[start: start + BLOCK_SIZE + 1]
        chunk_mask = mask[start: start + BLOCK_SIZE + 1]
        if len(chunk) < BLOCK_SIZE + 1:
            break
        # only keep chunks that have at least some assistant tokens
        if sum(1 for m in chunk_mask if m == 1) < 2:
            continue
        all_samples.append({"ids": chunk, "mask": chunk_mask})

with open("data/tokenized_data.json", "w", encoding="utf-8") as f:
    json.dump(all_samples, f)

print(f"Done! {len(all_samples)} samples saved. Skipped: {skipped}")
print(f"Sample 0 decoded: {tokenizer.decode(all_samples[0]['ids'][:50]) if all_samples else 'N/A'}")
