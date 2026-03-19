import os
import json
from tokenizer import BPETokenizer
from tqdm import tqdm

print("Pre-tokenizing all data...")

os.makedirs("tokenizer", exist_ok=True)
os.makedirs("data", exist_ok=True)

tokenizer = BPETokenizer()
tokenizer.save("tokenizer/tokenizer.json")

with open("data/training_data.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)

all_samples = []

for conv in tqdm(conversations):
    turns = conv.get("conversations", [])
    ids   = []
    mask  = []

    for i in range(0, len(turns) - 1, 2):
        user = turns[i]
        asst = turns[i + 1]

        if user.get("role") != "user" or asst.get("role") != "assistant":
            continue

        # role tokens as text — model learns structure this way
        user_text = "<|user|> " + user.get("content", "").strip()
        asst_text = "<|assistant|> " + asst.get("content", "").strip()

        user_ids = tokenizer.encode(user_text)
        asst_ids = tokenizer.encode(asst_text) + [tokenizer.eos_id]

        ids.extend(user_ids + asst_ids)
        mask.extend([-1] * len(user_ids) + [1] * len(asst_ids))

    if len(ids) > 0:
        all_samples.append({"ids": ids, "mask": mask})

with open("data/tokenized_data.json", "w", encoding="utf-8") as f:
    json.dump(all_samples, f)

print(f"Done! {len(all_samples)} samples saved.")
