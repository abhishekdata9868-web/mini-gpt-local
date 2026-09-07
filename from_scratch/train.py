import os, math, json, urllib.request
import torch
from model import MiniGPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 256
BATCH_SIZE = 32 if DEVICE == "cuda" else 8
STEPS = 3000
LR = 3e-4

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "from_scratch/data.txt"

if not os.path.exists(DATA_FILE):
    os.makedirs("from_scratch", exist_ok=True)
    print("Downloading public training text...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

text = open(DATA_FILE, "r", encoding="utf-8").read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

def get_batch(source):
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model = MiniGPT(vocab_size, block_size=BLOCK_SIZE, n_layer=6, n_head=6, n_embd=384).to(DEVICE)
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
torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos, "config": {"vocab_size": vocab_size, "block_size": BLOCK_SIZE, "n_layer": 6, "n_head": 6, "n_embd": 384}}, "from_scratch/checkpoint/minigpt.pt")
print("Saved: from_scratch/checkpoint/minigpt.pt")

model.eval()
context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
with torch.no_grad():
    for _ in range(300):
        idx = context[:, -BLOCK_SIZE:]
        logits, _ = model(idx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, 1)
        context = torch.cat((context, nxt), dim=1)
print("\n--- SAMPLE ---\n" + decode(context[0].tolist()))
