import argparse
import os
import math, json, urllib.request
import tempfile
import torch
from model import MiniGPT

# Prefer NVIDIA CUDA, then Apple Silicon MPS, and fall back to CPU.
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

parser = argparse.ArgumentParser(description="Train MiniGPT from scratch.")
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Check device, forward/backward, checkpoint save/load, and next-token generation without downloading training data.",
)
ARGS = parser.parse_args()

BLOCK_SIZE = 256
BATCH_SIZE = 32 if DEVICE == "cuda" else 8
STEPS = 3000
LR = 3e-4


def run_smoke_test():
    """Run a tiny end-to-end check without downloading the training corpus."""
    torch.manual_seed(42)
    config = {
        "vocab_size": 32,
        "block_size": 16,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
    }
    model = MiniGPT(**config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    token_ids = torch.randint(0, config["vocab_size"], (2, 8), device=DEVICE)

    logits, loss = model(token_ids, token_ids)
    expected_shape = (2, 8, config["vocab_size"])
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError("Forward pass produced a missing or non-finite loss.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Backward pass did not produce gradients.")
    optimizer.step()

    with tempfile.TemporaryDirectory(prefix="minigpt-smoke-") as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "smoke-checkpoint.pt")
        torch.save(model.state_dict(), checkpoint_path)
        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
        restored_model = MiniGPT(**config).to(DEVICE)
        restored_model.load_state_dict(state_dict)
        restored_model.eval()

        with torch.no_grad():
            restored_logits, _ = restored_model(token_ids)
            next_tokens = torch.argmax(restored_logits[:, -1, :], dim=-1)

        if not torch.isfinite(restored_logits).all().item():
            raise RuntimeError("Restored model produced non-finite logits.")
        if tuple(next_tokens.shape) != (token_ids.shape[0],):
            raise RuntimeError(f"Unexpected next-token shape: {tuple(next_tokens.shape)}")

    print(f"SMOKE TEST PASSED | device={DEVICE}")
    print("Checks: forward/loss, backward/gradients, optimizer step, checkpoint save/load, next-token generation")
    print(f"Logits shape: {expected_shape} | Next-token batch shape: {tuple(next_tokens.shape)}")


if ARGS.smoke_test:
    run_smoke_test()
    raise SystemExit(0)

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "from_scratch/data.txt"

if not os.path.exists(DATA_FILE):
    os.makedirs("from_scratch", exist_ok=True)
    print("Downloading public training text...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

with open(DATA_FILE, "r", encoding="utf-8") as data_file:
    text = data_file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def get_batch(source):
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


model = MiniGPT(
    vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=6,
    n_head=6,
    n_embd=384,
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
params = sum(p.numel() for p in model.parameters())
print(f"Device: {DEVICE} | Parameters: {params:,} | Vocab: {vocab_size}")

for step in range(STEPS):
    model.train()
    xb, yb = get_batch(train_data)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_data)
            _, vloss = model(xv, yv)
        print(f"step {step:4d} | train {loss.item():.4f} | val {vloss.item():.4f}")

os.makedirs("from_scratch/checkpoint", exist_ok=True)
torch.save(
    {
        "model": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "vocab_size": vocab_size,
            "block_size": BLOCK_SIZE,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
        },
    },
    "from_scratch/checkpoint/minigpt.pt",
)
print("Saved: from_scratch/checkpoint/minigpt.pt")

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
with torch.no_grad():
    for _ in range(300):
        idx = context[:, -BLOCK_SIZE:]
        logits, _ = model(idx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, 1)
        context = torch.cat((context, nxt), dim=1)
print("\n--- SAMPLE ---\n" + decode(context[0].tolist()))
