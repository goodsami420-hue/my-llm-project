import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from tokenizer import BPETokenizer


def load_json_data(path):
    tokenized_path = "data/tokenized_data.json"
    if os.path.exists(tokenized_path):
        print("Loading pre-tokenized data...")
        with open(tokenized_path, "r") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    conversations = []
    for item in data:
        turns = item.get("conversations", [])
        if len(turns) >= 2:
            conversations.append(turns)
    print(f"Loaded {len(conversations)} conversations from {path}")
    return conversations


def extract_texts_for_tokenizer(conversations):
    texts = []
    for conv in conversations:
        for turn in conv:
            texts.append(turn["content"])
    return texts


class ConversationDataset(Dataset):
    def __init__(self, data, tokenizer, block_size, split="train", val_ratio=0.02):
        self.block_size = block_size
        self.samples = []

        if data and isinstance(data[0], dict) and "ids" in data[0]:
            random.shuffle(data)
            val_n = max(1, int(len(data) * val_ratio))
            items = data[val_n:] if split == "train" else data[:val_n]
            print(f"Processing {len(items)} pre-tokenized samples for {split} split...")
            for item in items:
                ids  = item["ids"]
                mask = item["mask"]
                for start in range(0, len(ids) - block_size, block_size // 2):
                    chunk      = ids[start: start + block_size + 1]
                    chunk_mask = mask[start: start + block_size + 1]
                    if len(chunk) < block_size + 1:
                        break
                    self.samples.append((chunk, chunk_mask))
        else:
            random.shuffle(data)
            val_n = max(1, int(len(data) * val_ratio))
            convs = data[val_n:] if split == "train" else data[:val_n]
            print(f"Processing {len(convs)} conversations for {split} split...")
            self._build_samples(convs, tokenizer, block_size)

        print(f"{split} samples: {len(self.samples)}")

    def _build_samples(self, convs, tokenizer, block_size):
        for conv in convs:
            ids = []
            loss_mask = []
            for i in range(0, len(conv) - 1, 2):
                if i + 1 >= len(conv):
                    break
                user_turn = conv[i]
                asst_turn = conv[i + 1]
                if user_turn["role"] != "user" or asst_turn["role"] != "assistant":
                    continue
                user_ids = tokenizer.encode_conversation(user_turn["content"])
                asst_ids = ([tokenizer.eos_id]
                            + tokenizer.encode(asst_turn["content"])
                            + [tokenizer.end_id])
                ids += user_ids + asst_ids
                loss_mask += [-1] * len(user_ids) + [1] * len(asst_ids)
            for start in range(0, len(ids) - block_size, block_size // 2):
                chunk      = ids[start: start + block_size + 1]
                chunk_mask = loss_mask[start: start + block_size + 1]
                if len(chunk) < block_size + 1:
                    break
                self.samples.append((chunk, chunk_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, mask = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.long)
        y = torch.where(m == 1, y, torch.tensor(-1, dtype=torch.long))
        return x, y