import torch
import re
from functools import lru_cache
from torch.utils.data import DataLoader
from torch import optim, nn
import platform
import psutil

from runtimes.configure_agent import DEVICE, BATCH_SIZE, LR, EPOCHS, CHECKPOINT_FILE, DATA_DIR, FILE_PATTERNS, validate_config
from runtimes.handler_agent import load_corpus, CharVocab, TextDataset, match_any_pattern
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


def display_system_specs():
    """Display system specifications and training configuration."""
    print("\n" + "="*60)
    print("SYSTEM SPECIFICATIONS & CONFIGURATION")
    print("="*60)
    
    # System info
    print(f"\nOS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
    print(f"\nCPU: {cpu_count} cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    # GPU info
    print(f"\nDevice: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Training config
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Checkpoint File: {CHECKPOINT_FILE}")
    print("="*60 + "\n")


def train():
    # Validate configuration first
    config_errors = validate_config()
    if config_errors:
        print("\n" + "="*60)
        print("CONFIGURATION ERRORS - SETUP REQUIRED")
        print("="*60)
        for error in config_errors:
            print(f"❌ {error}")
        print("\n⚠️  Please fix the above errors in config.json before running again.")
        print("="*60 + "\n")
        raise ValueError("Configuration validation failed.")
    
    # Display system specifications
    display_system_specs()
    
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


def run_agent():
    while True:
        try:
            q = input("\n[you] > ")
        except EOFError:
            break

        if q.strip().lower() in {"exit", "quit"}:
            break

        if not q.strip():
            continue

        pattern = q  # you can map natural language → regex later
        snippets = regex_search(pattern)
        context = "\n\n".join(snippets[:3])

        prompt = f"Context:\n{context}\n\nTask:\n{q}\n\nAnswer:\n"
        out = generate_example(prompt, max_new_tokens=256)
        print("\n[agent] >")
        print(out)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bird.py [train|run]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "train":
        train()
    elif command == "run":
        run_agent()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python bird.py [train|run]")
        sys.exit(1)
