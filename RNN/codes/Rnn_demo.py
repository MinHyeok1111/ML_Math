import math
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm

torch.manual_seed(42)

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("device:", device)

ds = load_dataset("wikitext", "wikitext-2-raw-v1")
train_lines = [s for s in ds["train"]["text"] if s and not s.isspace()]
val_lines   = [s for s in ds["validation"]["text"] if s and not s.isspace()]

def tokenize(line: str):
    return line.strip().split()

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = SPECIALS

counter = Counter()
for line in train_lines:
    counter.update(tokenize(line))

max_vocab = 12000
most_common = counter.most_common(max_vocab - len(SPECIALS))
itos = SPECIALS + [w for w, _ in most_common]
stoi = {w: i for i, w in enumerate(itos)}

def encode_lines(lines):
    ids = []
    for line in lines:
        ids.append(stoi[BOS])
        for w in tokenize(line):
            ids.append(stoi.get(w, stoi[UNK]))
        ids.append(stoi[EOS])
    return torch.tensor(ids, dtype=torch.long)

train_ids = encode_lines(train_lines)
val_ids   = encode_lines(val_lines)

vocab_size = len(itos)
print("vocab_size:", vocab_size)
print("train tokens:", len(train_ids), "val tokens:", len(val_ids))

class StreamDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.x = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.x) - (self.seq_len + 1))

    def __getitem__(self, idx):
        x = self.x[idx : idx + self.seq_len]
        y = self.x[idx + 1 : idx + 1 + self.seq_len]
        return x, y

SEQ_LEN = 32
BATCH   = 32

train_ds = StreamDataset(train_ids, SEQ_LEN)
val_ds   = StreamDataset(val_ids, SEQ_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, drop_last=True, num_workers=0)

class GRULM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=128, hid_dim=256, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, h=None):
        e = self.emb(x)
        out, h = self.rnn(e, h)
        logits = self.fc(out)
        return logits, h

model = GRULM(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

def ppl_from_loss(loss_val: float):
    return math.exp(min(20.0, loss_val))

def run_epoch(dl, train: bool, epoch_idx: int):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dl, desc=f"{'TRAIN' if train else 'VAL  '} epoch {epoch_idx}", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), y.reshape(B*T))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * (B*T)
        total_tokens += (B*T)

        avg_loss = total_loss / max(1, total_tokens)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{ppl_from_loss(avg_loss):.2f}")

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, ppl_from_loss(avg_loss)

EPOCHS = 2
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_ppl = run_epoch(train_dl, train=True, epoch_idx=ep)
    va_loss, va_ppl = run_epoch(val_dl, train=False, epoch_idx=ep)
    print(f"epoch {ep:02d} | train loss {tr_loss:.4f} ppl {tr_ppl:.2f} | val loss {va_loss:.4f} ppl {va_ppl:.2f}")

@torch.no_grad()
def generate(prompt: str, max_new=30, temperature=1.0, top_k=30):
    model.eval()

    def sample_from_logits(logits_1d):
        logits_1d = logits_1d / max(1e-6, temperature)
        if top_k is not None and top_k > 0:
            v, idx = torch.topk(logits_1d, k=min(top_k, logits_1d.numel()))
            probs = torch.softmax(v, dim=-1)
            pick = torch.multinomial(probs, 1)
            return idx[pick].item()
        probs = torch.softmax(logits_1d, dim=-1)
        return torch.multinomial(probs, 1).item()

    words = tokenize(prompt)
    ids = [stoi[BOS]] + [stoi.get(w, stoi[UNK]) for w in words]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    h = None
    _, h = model(x, h)

    out_words = words[:]
    last_id = x[0, -1].view(1, 1)

    for _ in range(max_new):
        logits, h = model(last_id, h)
        next_id = sample_from_logits(logits[0, -1])
        if next_id == stoi[EOS]:
            break
        out_words.append(itos[next_id])
        last_id = torch.tensor([[next_id]], device=device)

    return " ".join(out_words)

print(generate("The meaning of life", max_new=30, temperature=0.9, top_k=30))