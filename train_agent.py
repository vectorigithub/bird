import torch
import re
from functools import lru_cache
from torch.utils.data import DataLoader
from torch import optim, nn

from configure_agent import DEVICE, BATCH_SIZE, LR, EPOCHS, CHECKPOINT_FILE, DATA_DIR, FILE_PATTERNS
from handler_agent import load_corpus, CharVocab, TextDataset, match_any_pattern
from model import TinyTransformerLM


@lru_cache(maxsize=1)
def load_vocab_and_model():
    ckpt = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    chars = ckpt.get("chars") or list(ckpt["vocab"].keys())
    vocab = CharVocab(chars)

    model = TinyTransformerLM(vocab_size=len(vocab.stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return vocab, model


@lru_cache(maxsize=1)
def _searchable_corpus():
    files = []
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file() and match_any_pattern(p, FILE_PATTERNS):
            try:
                files.append((p, p.read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                continue
    return files


def regex_search(pattern: str):
    matches = []
    compiled = re.compile(pattern)
    for _, text in _searchable_corpus():
        for m in compiled.finditer(text):
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            matches.append(text[start:end])
    return matches


def generate_example(prompt: str, max_new_tokens: int = 200):
    vocab, model = load_vocab_and_model()
    encoded = torch.tensor([vocab.encode(prompt)], dtype=torch.long).to(DEVICE)
    out = model.generate(encoded, max_new_tokens=max_new_tokens)
    return vocab.decode(out[0].tolist())

def train():
    text = load_corpus()
    if not text.strip():
        raise ValueError("No training text found under data/raw")

    vocab = CharVocab(text)
    dataset = TextDataset(text, vocab)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE == "cuda",
        drop_last=True,
    )

    model = TinyTransformerLM(vocab_size=len(vocab.stoi)).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if i % 50 == 0:
                print(f"epoch {epoch} step {i} loss {loss.item():.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "chars": list(vocab.stoi.keys()),
            "block_size": dataset.block_size,
        },
        CHECKPOINT_FILE,
    )

if __name__ == "__main__":
    train()
