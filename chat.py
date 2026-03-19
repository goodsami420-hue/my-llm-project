import os
import sys
import torch
from config import ModelConfig, ChatConfig
from model import GPT
from tokenizer import BPETokenizer


def load_model(cfg):
    checkpoint_path = cfg.get_checkpoint_path()
    if checkpoint_path is None:
        print("No checkpoint found! Run train.py first.")
        sys.exit(1)
    print(f"Loading: {checkpoint_path}")

    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    mcfg  = ModelConfig()
    saved = ckpt.get("model_cfg", {})
    for k, v in saved.items():
        setattr(mcfg, k, v)
    mcfg.vocab_size = ckpt["model_state"]["lm_head.weight"].shape[0]

    model = GPT(mcfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = BPETokenizer()
    tokenizer.load(cfg.tokenizer_path)

    print(f"Model loaded | params: {model.count_params()/1e6:.1f}M | block_size: {mcfg.block_size}")
    return model, tokenizer


def generate(model, tokenizer, prompt_ids, max_new_tokens=100, temperature=0.7, top_k=40):
    block_size = model.cfg.block_size
    idx        = torch.tensor([prompt_ids[-block_size:]], dtype=torch.long)
    generated  = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            if idx_cond.size(1) == 0:
                break

            logits, _ = model(idx_cond)
            if logits.size(0) == 0 or logits.size(1) == 0:
                break

            logits = logits[:, -1, :]

            # repetition penalty
            for tid in set(generated[-30:]):
                if 0 <= tid < logits.size(-1):
                    logits[0, tid] /= 1.3

            logits = logits / max(temperature, 1e-8)

            top_k_actual = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k_actual)
            logits[logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                break

            next_token = torch.multinomial(probs, num_samples=1)
            token_id   = next_token.item()

            # stop at EOS
            if token_id == tokenizer.eos_id:
                break

            # skip special tokens in output
            skip_ids = {tokenizer.pad_id, tokenizer.bos_id,
                        tokenizer.assistant_id, tokenizer.end_id}
            if token_id not in skip_ids:
                generated.append(token_id)

            idx = torch.cat((idx, next_token), dim=1)

    return generated


def chat():
    cfg        = ChatConfig()
    model, tokenizer = load_model(cfg)
    block_size = model.cfg.block_size

    print("\n" + "="*50)
    print("  Chat with your LLM!")
    print("  'quit' to exit | 'clear' to reset history")
    print("="*50 + "\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            history = []
            print("[History cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})

        # EXACT match with training format
        prompt_ids = []
        for turn in history:
            if turn["role"] == "user":
                text = "<|user|> " + turn["content"].strip()
                prompt_ids += tokenizer.encode(text)
            else:
                text = "<|assistant|> " + turn["content"].strip()
                prompt_ids += tokenizer.encode(text) + [tokenizer.eos_id]

        # signal model: generate assistant response
        prompt_ids += tokenizer.encode("<|assistant|>")

        max_prompt = block_size - cfg.max_new_tokens
        if max_prompt < 1:
            max_prompt = block_size // 2
        prompt_ids = prompt_ids[-max_prompt:]

        print("AI: ", end="", flush=True)

        new_ids  = generate(model, tokenizer, prompt_ids,
                            max_new_tokens=cfg.max_new_tokens,
                            temperature=cfg.temperature,
                            top_k=cfg.top_k)

        response = tokenizer.decode(new_ids).strip()
        if not response:
            response = "..."

        print(response)
        print()

        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat()
