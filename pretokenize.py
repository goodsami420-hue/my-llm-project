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
    ids = []
    mask = []
    for i in range(0, len(turns) - 1, 2):
        if i + 1 >= len(turns):
            break
        user = turns[i]
        asst = turns[i + 1]
        if user["role"] != "user" or asst["role"] != "assistant":
            continue
        user_ids = tokenizer.encode_conversation(user["content"])
        asst_ids = ([tokenizer.eos_id]
                    + tokenizer.encode(asst["content"])
                    + [tokenizer.end_id])
        ids += user_ids + asst_ids
        mask += [-1] * len(user_ids) + [1] * len(asst_ids)
    if ids:
        all_samples.append({"ids": ids, "mask": mask})

with open("data/tokenized_data.json", "w") as f:
    json.dump(all_samples, f)

print(f"Done! {len(all_samples)} samples saved.")