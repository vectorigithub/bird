#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
ROCM_HOME="/opt/rocm"

info() { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*"; }

usage() {
  cat <<'EOF'
Usage: ./n.bash <train|run> [--compute gpuN]

Examples:
  ./n.bash train --compute gpu0
  ./n.bash run --compute gpu1

Notes:
  - If --compute is omitted in an interactive shell, you will be prompted.
  - In non-interactive mode, --compute is required.
EOF
}

validate_compute_format() {
  local value="$1"
  [[ "$value" =~ ^gpu[0-9]+$ ]]
}

list_torch_gpus() {
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(0)

count = torch.cuda.device_count()
for idx in range(count):
    try:
        name = torch.cuda.get_device_name(idx)
    except Exception:
        name = "Unknown GPU"
    print(f"gpu{idx}|{name}")
PY
}

list_rocm_smi_gpus() {
  if ! command -v rocm-smi >/dev/null 2>&1; then
    return 0
  fi

  rocm-smi --showid 2>/dev/null | awk '
    /GPU\[[0-9]+\][[:space:]]*:[[:space:]]*Device Name:/ {
      idx = $0
      sub(/^.*GPU\[/, "", idx)
      sub(/\].*$/, "", idx)

      name = $0
      sub(/^.*Device Name:[[:space:]]*/, "", name)

      if (idx ~ /^[0-9]+$/ && length(name) > 0) {
        print "gpu" idx "|" name
      }
    }
  '
}

list_nvidia_smi_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits 2>/dev/null | awk -F',' '
    {
      idx=$1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx)
      name=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      if (idx ~ /^[0-9]+$/ && length(name) > 0) {
        print "gpu" idx "|" name
      }
    }
  '
}

compute_exists_in_list() {
  local target="$1"
  local available="$2"
  while IFS='|' read -r gpu_id _; do
    [[ -z "$gpu_id" ]] && continue
    if [[ "$gpu_id" == "$target" ]]; then
      return 0
    fi
  done <<< "$available"
  return 1
}

COMMAND=""
COMPUTE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    train|run)
      if [[ -n "$COMMAND" ]]; then
        err "Multiple commands provided. Use only one of: train, run"
        usage
        exit 1
      fi
      COMMAND="$1"
      shift
      ;;
    --compute)
      if [[ -z "${2:-}" ]]; then
        err "--compute requires a value (gpu0, gpu1, ...)"
        exit 1
      fi
      COMPUTE="${2,,}"
      shift 2
      ;;
    --compute=*)
      COMPUTE="${1#*=}"
      COMPUTE="${COMPUTE,,}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$COMMAND" ]]; then
  err "No command specified"
  usage
  exit 1
fi

if [[ -n "$COMPUTE" ]] && ! validate_compute_format "$COMPUTE"; then
  err "Invalid --compute value '$COMPUTE'. Expected format: gpu0, gpu1, ..."
  exit 1
fi

# Check if already in virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # Not in venv, activate it
  if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    err "Virtual environment not found at ${VENV_DIR}"
    err "Run './f.bash' first to set up the environment."
    exit 1
  fi
  source "${VENV_DIR}/bin/activate"
  info "Activated virtual environment"
else
  info "Already in virtual environment: $VIRTUAL_ENV"
fi

# Check ROCm installation
info "Checking ROCm stack support..."
if [ ! -d "$ROCM_HOME" ]; then
  warn "ROCm not found at $ROCM_HOME. GPU acceleration may not be available."
else
  info "ROCm installation found at $ROCM_HOME"
fi

# Check rocm-smi availability
if command -v rocm-smi >/dev/null 2>&1; then
  info "rocm-smi detected. Checking GPU..."
  rocm-smi --showid 2>/dev/null || warn "Could not query GPU with rocm-smi"
else
  warn "rocm-smi not found. GPU detection may not work."
fi

# Check user groups for GPU device access
USER_GROUPS=$(id -nG)
need_groups=()
if ! echo " $USER_GROUPS " | grep -qw "render"; then
  need_groups+=(render)
fi
if ! echo " $USER_GROUPS " | grep -qw "video"; then
  need_groups+=(video)
fi
if [ ${#need_groups[@]} -ne 0 ]; then
  warn "You are missing recommended groups for GPU access: ${need_groups[*]}"
  warn "To add your user to them (requires sudo):"
  warn "  sudo usermod -a -G ${need_groups[*]} $(whoami)"
  warn "Then log out and log back in for group changes to take effect."
else
  info "User is in required GPU groups (render/video)"
fi

# Check for ROCm HIP / OpenCL SDKs (Arch Linux - filesystem/binaries checks, no package manager)
info "Checking for ROCm SDK components (hip / OpenCL)..."
HIP_SDK=false
OPENCL_SDK=false

# HIP detection: look for hipcc, hipconfig, or hip dir under /opt/rocm
if command -v hipcc >/dev/null 2>&1 || command -v hipconfig >/dev/null 2>&1; then
  HIP_SDK=true
fi
if [ -d "/opt/rocm/hip" ] || [ -d "/opt/rocm/hip-sdk" ] || [ -d "/opt/rocm/bin" ] && ls /opt/rocm/bin/hip* >/dev/null 2>&1; then
  HIP_SDK=true
fi

# OpenCL detection: clinfo or opencl dir under /opt/rocm
if command -v clinfo >/dev/null 2>&1; then
  OPENCL_SDK=true
fi
if [ -d "/opt/rocm/opencl" ] || [ -d "/opt/rocm/ocl" ]; then
  OPENCL_SDK=true
fi

if $HIP_SDK; then
  info "ROCm HIP SDK: detected"
else
  warn "ROCm HIP SDK: not detected (hipcc/hipconfig or /opt/rocm/hip not found)"
  warn "If you installed ROCm via AUR or other packages, ensure /opt/rocm is present and hip binaries are on PATH."
fi

if $OPENCL_SDK; then
  info "OpenCL SDK (ROCm OpenCL): detected"
else
  warn "OpenCL SDK: not detected (clinfo or /opt/rocm/opencl not found)"
  warn "If you need OpenCL, install rocm-opencl-sdk or ensure clinfo is available."
fi

# Set ROCm environment variables
export ROCM_HOME=$ROCM_HOME
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ROCM_HOME/lib
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#:}"
export HIP_PLATFORM=amd
export GPU_DEVICE_ORDINAL=0

# Resolve compute target from CLI or interactive prompt
AVAILABLE_GPUS=$(list_torch_gpus || true)
if [[ -z "$AVAILABLE_GPUS" ]]; then
  AVAILABLE_GPUS=$(list_rocm_smi_gpus || true)
fi
if [[ -z "$AVAILABLE_GPUS" ]]; then
  AVAILABLE_GPUS=$(list_nvidia_smi_gpus || true)
fi

if [[ -z "$COMPUTE" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    if [[ -n "$AVAILABLE_GPUS" ]]; then
      info "Available compute targets:"
      while IFS='|' read -r gpu_id gpu_name; do
        [[ -z "$gpu_id" ]] && continue
        printf "  - %s (%s)\n" "$gpu_id" "$gpu_name"
      done <<< "$AVAILABLE_GPUS"
    else
      warn "No GPUs were auto-detected by torch/rocm-smi/nvidia-smi right now."
      warn "You can still enter a target manually (example: gpu0)."
    fi

    while true; do
      read -r -p "Select compute target (gpu0, gpu1, ...): " COMPUTE
      COMPUTE="${COMPUTE,,}"

      if ! validate_compute_format "$COMPUTE"; then
        err "Invalid compute target '$COMPUTE'. Expected gpuN format."
        continue
      fi

      if [[ -n "$AVAILABLE_GPUS" ]] && ! compute_exists_in_list "$COMPUTE" "$AVAILABLE_GPUS"; then
        err "Requested compute target '$COMPUTE' is not in the detected GPU list."
        continue
      fi

      break
    done
  else
    err "Missing required --compute gpuN in non-interactive mode."
    exit 1
  fi
fi

if [[ -n "$AVAILABLE_GPUS" ]] && ! compute_exists_in_list "$COMPUTE" "$AVAILABLE_GPUS"; then
  err "Requested compute target '$COMPUTE' is not available."
  info "Detected targets:"
  while IFS='|' read -r gpu_id gpu_name; do
    [[ -z "$gpu_id" ]] && continue
    printf "  - %s (%s)\n" "$gpu_id" "$gpu_name"
  done <<< "$AVAILABLE_GPUS"
  exit 1
fi

export BIRD_COMPUTE="$COMPUTE"
info "Selected compute target: $BIRD_COMPUTE"

# Check PyTorch ROCm support in venv
info "Checking PyTorch ROCm support..."
PYTORCH_CHECK=$(python -c "
import torch
has_cuda = torch.cuda.is_available()
has_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
print(f'{has_cuda},{has_hip}')
" 2>/dev/null || echo "false,false")

IFS=',' read -r CUDA_AVAIL HIP_AVAIL <<< "$PYTORCH_CHECK"

if [ "$CUDA_AVAIL" = "True" ]; then
  info "PyTorch CUDA support: AVAILABLE"
  GPU_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown GPU")
  info "GPU Device: $GPU_INFO"
elif [ "$HIP_AVAIL" = "True" ]; then
  info "PyTorch HIP (ROCm) support: AVAILABLE"
  GPU_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown GPU")
  info "GPU Device: $GPU_INFO"
else
  warn "PyTorch GPU support NOT available. Will use CPU."
  warn "Ensure ROCm is installed and PyTorch was built with ROCm support."
fi

info "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Unknown')"

# Run bird.py with parameters
info "Running: python bird.py $COMMAND --compute $BIRD_COMPUTE"
python bird.py "$COMMAND" --compute "$BIRD_COMPUTE"
