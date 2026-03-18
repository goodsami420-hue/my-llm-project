import os
import time
import math
import torch

from config import ModelConfig, TrainConfig
from model import GPT
from tokenizer import BPETokenizer
from dataset import ConversationDataset, load_json_data
from torch.utils.data import DataLoader


def get_device():
    try:
        import torch_directml
        print("Intel Arc DirectML detected")
        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        print("CUDA GPU detected")
        return torch.device("cuda")
    print("CPU mode")
    return torch.device("cpu")


def get_lr(step, cfg):
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / cfg.warmup_iters
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_dl, device, eval_iters):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_dl):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def save_checkpoint(model, optimizer, step, loss, cfg, tag="latest"):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"{tag}_model.pt")
    torch.save({
        "step":        step,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "loss":        loss,
        "model_cfg":   vars(ModelConfig()),
    }, path)
    print(f"  Checkpoint saved -> {path}")


def load_checkpoint(model, optimizer, cfg):
    path = os.path.join(cfg.checkpoint_dir, "latest_model.pt")
    if not os.path.exists(path):
        return 0, float('inf')
    # always load to CPU first to avoid device mismatch
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    print(f"  Resumed from step {ckpt['step']}")
    return ckpt["step"], ckpt["loss"]


def train():
    tcfg = TrainConfig()
    mcfg = ModelConfig()

    device = get_device()

    # tokenizer
    tokenizer = BPETokenizer()
    if os.path.exists(tcfg.tokenizer_path):
        print("[1/4] Loading existing tokenizer...")
        tokenizer.load(tcfg.tokenizer_path)
    else:
        print("[1/4] Saving tokenizer...")
        os.makedirs("tokenizer", exist_ok=True)
        tokenizer.save(tcfg.tokenizer_path)

    mcfg.vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {mcfg.vocab_size}")

    # dataset
    print("[2/4] Building dataset...")
    data     = load_json_data(tcfg.data_path)
    train_ds = ConversationDataset(data, tokenizer, mcfg.block_size, split="train")
    val_ds   = ConversationDataset(data, tokenizer, mcfg.block_size, split="val")
    train_dl = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=tcfg.batch_size, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # model — create on CPU first
    print("[3/4] Building model...")
    model     = GPT(mcfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        betas=(tcfg.beta1, tcfg.beta2),
    )

    # load checkpoint (on CPU) then move to device
    start_step, best_loss = load_checkpoint(model, optimizer, tcfg)
    model = model.to(device)

    # training loop
    print("[4/4] Training started...\n")
    print(f"{'Step':>8} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10} | {'ms/step':>8}")
    print("-" * 60)

    model.train()
    step       = start_step
    train_iter = iter(train_dl)
    t0         = time.time()
    accum_loss = 0.0

    while step < tcfg.max_iters:
        lr = get_lr(step, tcfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        for _ in range(tcfg.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            if loss is None:
                continue
            (loss / tcfg.grad_accum).backward()
            accum_loss += (loss / tcfg.grad_accum).item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        step += 1

        if step % tcfg.log_interval == 0:
            dt = (time.time() - t0) / tcfg.log_interval
            print(f"{step:>8} | {accum_loss:>10.4f} | {'---':>10} | {lr:>10.2e} | {dt*1000:>7.1f}ms")
            accum_loss = 0.0
            t0 = time.time()

        if step % tcfg.eval_interval == 0:
            val_loss = estimate_loss(model, val_dl, device, tcfg.eval_iters)
            print(f"{'>>> EVAL':>8} | {'---':>10} | {val_loss:>10.4f} |")
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, tcfg, tag="best")

        if step % tcfg.save_interval == 0:
            save_checkpoint(model, optimizer, step, accum_loss, tcfg, tag="latest")

    print("\nTraining complete!")
    save_checkpoint(model, optimizer, step, best_loss, tcfg, tag="final")


if __name__ == "__main__":
    train()