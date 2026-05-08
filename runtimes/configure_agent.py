from pathlib import Path
import json
import torch

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR.parent / "config.json"


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_FILE}")

    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load config file: {e}")


_config = _load_config()
def _get(section: str, key: str, default=None):
    """Helper to read settings from a sectioned config with flat-key fallback."""
    # prefer sectioned config
    sec = _config.get(section, {}) if isinstance(_config.get(section, {}), dict) else {}
    if key in sec:
        return sec.get(key)
    # fallback to top-level key for compatibility
    return _config.get(key, default)


# Determine device based on config (supports runtime->DEVICE or top-level DEVICE)
def _get_device() -> str:
    device_config = str(_get("runtime", "DEVICE", "auto")).lower()
    if device_config == "cpu":
        return "cpu"
    elif device_config in {"nvidia", "amd"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:  # "auto" or default
        return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _get_device()

# Load settings from config.json (sectioned or flat)
DATA_DIR = Path(_get("handler", "DATA_DIR", ""))
CHECKPOINT_FILE = Path(_get("trainer", "CHECKPOINT_FILE", ""))
FILE_PATTERNS = _get("handler", "FILE_PATTERNS", [])
BLOCK_SIZE = int(_get("trainer", "BLOCK_SIZE", 128))
BATCH_SIZE = int(_get("trainer", "BATCH_SIZE", 32))
EMBED_DIM = int(_get("model", "EMBED_DIM", 128))
N_HEAD = int(_get("model", "N_HEAD", 4))
N_LAYER = int(_get("model", "N_LAYER", 2))
LR = float(_get("trainer", "LR", 3e-4))
EPOCHS = int(_get("trainer", "EPOCHS", 5))


def validate_config() -> list[str]:
    """
    Validate configuration settings.
    Returns a list of error messages. Empty list means config is valid.
    """
    errors = []
    
    # Required settings that should not be empty (supporting sectioned config)
    data_dir = _get("handler", "DATA_DIR", _config.get("DATA_DIR", ""))
    ckpt = _get("trainer", "CHECKPOINT_FILE", _config.get("CHECKPOINT_FILE", ""))

    if not str(data_dir).strip():
        errors.append("Missing or empty 'DATA_DIR' (handler.DATA_DIR): Path to training data directory")
    if not str(ckpt).strip():
        errors.append("Missing or empty 'CHECKPOINT_FILE' (trainer.CHECKPOINT_FILE): Path where to save the trained model checkpoint")

    # Check for unknown top-level sections
    allowed_top = {"runtime", "handler", "trainer", "model"}
    for key in _config.keys():
        if key not in allowed_top:
            errors.append(f"Unknown top-level setting in config: '{key}'")

    # Validate DATA_DIR exists
    data_path = Path(data_dir) if data_dir else None
    if data_path and not data_path.exists():
        errors.append(f"DATA_DIR does not exist: {data_path}")

    # Validate CHECKPOINT_FILE parent directory exists
    ckpt_path = Path(ckpt) if ckpt else None
    if ckpt_path and not ckpt_path.parent.exists():
        errors.append(f"CHECKPOINT_FILE parent directory does not exist: {ckpt_path.parent}")
    
    return errors