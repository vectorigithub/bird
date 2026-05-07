from pathlib import Path
import json
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
CHECKPOINT_FILE = BASE_DIR / "checkpoint.pt"
SETTINGS_FILE = BASE_DIR / "settings.json"


def _load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}

    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


_settings = _load_settings()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# regex patterns for files to include
FILE_PATTERNS = [
    r".*\.txt$",
    r".*\.py$",
]

# basic LM hyperparams (tiny on purpose)
BLOCK_SIZE = 128
BATCH_SIZE = 32
EMBED_DIM = 128
N_HEAD = 4
N_LAYER = 2
LR = 3e-4
EPOCHS = 5

DATA_DIR = Path(_settings.get("DATA_DIR", DATA_DIR))
CHECKPOINT_FILE = Path(_settings.get("CHECKPOINT_FILE", CHECKPOINT_FILE))
DEVICE = _settings.get("DEVICE", DEVICE)
FILE_PATTERNS = _settings.get("FILE_PATTERNS", FILE_PATTERNS)
BLOCK_SIZE = int(_settings.get("BLOCK_SIZE", BLOCK_SIZE))
BATCH_SIZE = int(_settings.get("BATCH_SIZE", BATCH_SIZE))
EMBED_DIM = int(_settings.get("EMBED_DIM", EMBED_DIM))
N_HEAD = int(_settings.get("N_HEAD", N_HEAD))
N_LAYER = int(_settings.get("N_LAYER", N_LAYER))
LR = float(_settings.get("LR", LR))
EPOCHS = int(_settings.get("EPOCHS", EPOCHS))