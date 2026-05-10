import torch
import re
import os
import sys
import argparse
from functools import lru_cache
from torch.utils.data import DataLoader
from torch import optim, nn
import platform
import psutil
from pathlib import Path

from runtimes.handler_agent import load_corpus, CharVocab, TextDataset, match_any_pattern
from runtimes.train_handler import TrainingHandler, can_resume_training, load_checkpoint
from model import TinyTransformerLM


DEVICE = "cpu"
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 5
CHECKPOINT_FILE = Path("model/phoenix/checkpoints/checkpoint.pt")
DATA_DIR = Path("dataset")
FILE_PATTERNS = []
validate_config = lambda: []
get_gpu_info = lambda: (None, None, None)


def initialize_runtime(compute_target: str):
    """Initialize runtime configuration after compute target is selected."""
    global DEVICE, BATCH_SIZE, LR, EPOCHS, CHECKPOINT_FILE, DATA_DIR, FILE_PATTERNS, validate_config, get_gpu_info

    os.environ["BIRD_COMPUTE"] = compute_target

    from runtimes import configure_agent as cfg

    DEVICE = cfg.DEVICE
    BATCH_SIZE = cfg.BATCH_SIZE
    LR = cfg.LR
    EPOCHS = cfg.EPOCHS
    CHECKPOINT_FILE = cfg.CHECKPOINT_FILE
    DATA_DIR = cfg.DATA_DIR
    FILE_PATTERNS = cfg.FILE_PATTERNS
    validate_config = cfg.validate_config
    get_gpu_info = cfg.get_gpu_info


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

    # GPU info - using centralized function from configure_agent
    print(f"\nDevice: {DEVICE.upper()}")
    if DEVICE.startswith("cuda"):
        gpu_name, gpu_backend, gpu_memory = get_gpu_info()
        print(f"GPU: {gpu_name}")
        print(f"Backend: {gpu_backend}")
        if gpu_memory is not None:
            print(f"GPU Memory: {gpu_memory:.2f} GB")

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
        pin_memory=DEVICE.startswith("cuda"),
        drop_last=True,
    )

    model = TinyTransformerLM(vocab_size=len(vocab.stoi)).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # Check for existing checkpoint to resume training
    start_epoch = 0
    start_step = 0
    resumed = False

    if can_resume_training(CHECKPOINT_FILE):
        response = input("\n📁 Found existing checkpoint. Resume training? (y/n): ").strip().lower()
        if response == 'y':
            checkpoint = load_checkpoint(CHECKPOINT_FILE, DEVICE)
            if checkpoint:
                model.load_state_dict(checkpoint["model"])
                if "optimizer" in checkpoint:
                    opt.load_state_dict(checkpoint["optimizer"])
                start_epoch = checkpoint.get("epoch", 0)
                start_step = checkpoint.get("step", 0)
                resumed = True
                print(f"✓ Resumed from epoch {start_epoch}, step {start_step}")

    # Set up training handler for graceful interrupt handling
    with TrainingHandler(CHECKPOINT_FILE) as handler:
        try:
            for epoch in range(start_epoch, EPOCHS):
                if handler.is_interrupted():
                    # Save checkpoint before potentially exiting
                    handler.save_checkpoint(
                        model.state_dict(),
                        opt.state_dict(),
                        epoch,
                        start_step if epoch == start_epoch else 0,
                        vocab,
                        dataset.block_size,
                    )
                    handler.wait_for_save_completion()

                    # Ask user if they want to continue or exit
                    if not resumed:
                        response = input("\n🔄 Resume training from last checkpoint? (y/n): ").strip().lower()
                        if response != 'y':
                            print("\n✓ Training stopped. Checkpoint saved.")
                            return
                    handler.reset_interrupt()
                    resumed = False

                model.train()
                for i, (x, y) in enumerate(loader):
                    # Check for interrupt at each step
                    if handler.is_interrupted():
                        handler.save_checkpoint(
                            model.state_dict(),
                            opt.state_dict(),
                            epoch,
                            i,
                            vocab,
                            dataset.block_size,
                            loss.item() if 'loss' in locals() else None,
                        )
                        handler.wait_for_save_completion()

                        # Ask user if they want to continue or exit
                        response = input("\n🔄 Continue training after this step? (y/n): ").strip().lower()
                        if response != 'y':
                            print("\n✓ Training paused. Checkpoint saved.")
                            return
                        handler.reset_interrupt()

                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()

                    if i % 50 == 0:
                        print(f"epoch {epoch} step {i} loss {loss.item():.4f}")

                    start_step = i + 1

                # Reset step counter for next epoch
                start_step = 0

        except KeyboardInterrupt:
            # This shouldn't normally be reached due to our signal handler,
            # but handle it just in case
            print("\n\n⚠️  Unexpected interrupt. Saving checkpoint...")
            handler.save_checkpoint(
                model.state_dict(),
                opt.state_dict(),
                epoch if 'epoch' in locals() else 0,
                i if 'i' in locals() else 0,
                vocab,
                dataset.block_size,
                loss.item() if 'loss' in locals() else None,
            )
            handler.wait_for_save_completion()
            print("\n✓ Training interrupted. Checkpoint saved.")

    # Final save
    torch.save(
        {
            "model": model.state_dict(),
            "chars": list(vocab.stoi.keys()),
            "block_size": dataset.block_size,
        },
        CHECKPOINT_FILE,
    )
    print(f"\n✓ Training complete! Model saved to: {CHECKPOINT_FILE}")


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
    parser = argparse.ArgumentParser(description="bird runtime")
    parser.add_argument("command", choices=["train", "run"], help="command to execute")
    parser.add_argument("--compute", required=True, help="compute target in gpuN format (e.g., gpu0)")
    args = parser.parse_args()

    if not re.fullmatch(r"gpu\d+", args.compute.lower()):
        print("Invalid --compute value. Expected format: gpu0, gpu1, ...", file=sys.stderr)
        sys.exit(1)

    try:
        initialize_runtime(args.compute.lower())
    except Exception as e:
        print(f"Failed to initialize runtime: {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "train":
        train()
    elif args.command == "run":
        run_agent()