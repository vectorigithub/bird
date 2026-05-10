from pathlib import Path
import json
import torch
import sys
import os
import re


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR.parent / "config.json"

def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        print(f"Warning: Config file not found at {CONFIG_FILE}", file=sys.stderr)
        return {}

    try:
        content = CONFIG_FILE.read_text(encoding="utf-8")
        if not content.strip():
            print(f"Warning: Config file is empty: {CONFIG_FILE}", file=sys.stderr)
            return {}
        return json.loads(content)
    except OSError as e:
        print(f"Warning: Failed to read config file: {e}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Config file contains invalid JSON: {e}", file=sys.stderr)
        return {}


_config = _load_config()
def _get(section: str, key: str, default=None):
    # prefer sectioned config
    sec = _config.get(section, {}) if isinstance(_config.get(section, {}), dict) else {}
    if key in sec:
        return sec.get(key)
    # fallback to top-level key for compatibility
    return _config.get(key, default)


def _prompt_user_for_value(prompt_text: str, required: bool = True) -> str:
    try:
        suffix = " (required)" if required else " (optional, press Enter to skip)"
        user_input = input(f"{prompt_text}{suffix}: ").strip()

        if not user_input and required:
            return ""
        return user_input
    except EOFError:
        # Handle case where stdin is closed or not available
        return ""
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)


def _is_gpu_available() -> bool:
    """Check if GPU is available (supports both NVIDIA CUDA and AMD ROCm)."""
    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        return True
    # Check for AMD ROCm (PyTorch with ROCm still uses 'cuda' device type)
    # ROCm availability is indicated by torch.version.hip
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        # Try to initialize a tensor on CUDA to verify ROCm GPU access
        try:
            torch.tensor([1.0], device='cuda')
            return True
        except (RuntimeError, AssertionError):
            pass
    return False


def _get_selected_gpu_index_from_env() -> int | None:
    """Parse BIRD_COMPUTE env var in format gpuN and return N."""
    raw = os.getenv("BIRD_COMPUTE", "").strip().lower()
    if not raw:
        return None

    match = re.fullmatch(r"gpu(\d+)", raw)
    if not match:
        raise ValueError(f"Invalid BIRD_COMPUTE value '{raw}'. Expected format: gpu0, gpu1, ...")

    return int(match.group(1))


# Determine device based on config (supports runtime->DEVICE or top-level DEVICE)
def _get_device() -> str:
    selected_gpu_index = _get_selected_gpu_index_from_env()
    if selected_gpu_index is not None:
        if not _is_gpu_available():
            raise RuntimeError("GPU was requested via BIRD_COMPUTE, but no GPU is available to PyTorch.")

        gpu_count = torch.cuda.device_count()
        if selected_gpu_index >= gpu_count:
            raise RuntimeError(
                f"Requested gpu{selected_gpu_index}, but only {gpu_count} GPU(s) are available."
            )

        return f"cuda:{selected_gpu_index}"

    device_config = str(_get("runtime", "DEVICE", "auto")).lower()
    if device_config == "cpu":
        return "cpu"
    elif device_config in {"gpu", "nvidia", "amd"}:
        # "gpu", "nvidia", or "amd" all request GPU acceleration
        return "cuda:0" if _is_gpu_available() else "cpu"
    else:  # "auto" or default
        return "cuda:0" if _is_gpu_available() else "cpu"


def get_gpu_info():
    """Get GPU information supporting both NVIDIA CUDA and AMD ROCm.

    Returns:
        tuple: (gpu_name, gpu_backend, gpu_memory_gb) or (None, None, None) if no GPU
    """
    device = _get_device()
    if not device.startswith("cuda"):
        return None, None, None

    gpu_index = 0
    if ":" in device:
        try:
            gpu_index = int(device.split(":", 1)[1])
        except ValueError:
            gpu_index = 0

    gpu_name = "Unknown"
    gpu_backend = "CUDA"
    gpu_memory = None

    try:
        gpu_name = torch.cuda.get_device_name(gpu_index)
    except Exception:
        pass

    # Check for AMD ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        gpu_backend = f"ROCm ({torch.version.hip})"
    else:
        # NVIDIA CUDA
        gpu_backend = f"CUDA ({torch.version.cuda or 'Unknown'})"

    try:
        gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
    except Exception:
        pass

    return gpu_name, gpu_backend, gpu_memory


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


def ensure_config_valid() -> bool:
    errors = validate_config()

    if not errors:
        return True

    print("\n=== Configuration Issues Detected ===")
    print("Some required configuration values are missing or invalid.")
    print("Please provide the missing information:\n")

    # Get current values
    data_dir = _get("handler", "DATA_DIR", _config.get("DATA_DIR", ""))
    ckpt = _get("trainer", "CHECKPOINT_FILE", _config.get("CHECKPOINT_FILE", ""))

    # Prompt for missing DATA_DIR
    needs_data_dir = not str(data_dir).strip()
    if needs_data_dir:
        user_input = _prompt_user_for_value("Enter path to training data directory (DATA_DIR)", required=True)
        if not user_input:
            print("\nNo input was given, incomplete needs to run.", file=sys.stderr)
            sys.exit(1)
        data_dir = user_input
        print(f"✓ DATA_DIR set to: {data_dir}")

    # Prompt for missing CHECKPOINT_FILE
    needs_ckpt = not str(ckpt).strip()
    if needs_ckpt:
        user_input = _prompt_user_for_value("Enter path to save model checkpoint (CHECKPOINT_FILE)", required=True)
        if not user_input:
            print("\nNo input was given, incomplete needs to run.", file=sys.stderr)
            sys.exit(1)
        ckpt = user_input
        print(f"✓ CHECKPOINT_FILE set to: {ckpt}")

    # Validate provided paths
    if data_dir:
        data_path = Path(data_dir)
        if not data_path.exists():
            create_dir = input(f"\nDirectory '{data_path}' does not exist. Create it? (y/n): ").strip().lower()
            if create_dir == 'y':
                try:
                    data_path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ Created directory: {data_path}")
                except Exception as e:
                    print(f"✗ Failed to create directory: {e}", file=sys.stderr)
                    print("\nNo input was given, incomplete needs to run.", file=sys.stderr)
                    sys.exit(1)
            else:
                print("\nNo input was given, incomplete needs to run.", file=sys.stderr)
                sys.exit(1)

    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.parent.exists():
            create_parent = input(f"\nParent directory '{ckpt_path.parent}' does not exist. Create it? (y/n): ").strip().lower()
            if create_parent == 'y':
                try:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"✓ Created directory: {ckpt_path.parent}")
                except Exception as e:
                    print(f"✗ Failed to create directory: {e}", file=sys.stderr)
                    print("\nNo input was given, incomplete needs to run.", file=sys.stderr)
                    sys.exit(1)
            else:
                print("\nNo input was given, incomplete needs to run.", file=sys.stderr)
                sys.exit(1)

    print("\n=== Configuration Complete ===\n")
    return True


def validate_config() -> list[str]:
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