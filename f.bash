#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
VERBOSE=false

info() { printf "\n\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\n\033[1;31m[ERROR]\033[0m %s\n" "$*"; }
vinfo() { if $VERBOSE; then printf "\n\033[1;36m[VERBOSE]\033[0m %s\n" "$*"; fi; }

run_cmd() {
  if $VERBOSE; then
    vinfo "Running: $*"
  fi
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      warn "Ignoring unknown argument: $1"
      shift
      ;;
  esac
done

if $VERBOSE; then
  set -x
  vinfo "Verbose mode enabled"
fi

info "Ensuring virtual environment at ${VENV_DIR}"
if [ ! -d "${VENV_DIR}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
  else
    err "No python interpreter found on PATH."
    exit 1
  fi

  vinfo "Creating venv with ${PY_BIN}"
  run_cmd "${PY_BIN}" -m venv "${VENV_DIR}"
  info "Virtual environment created at ${VENV_DIR}"
else
  info "Virtual environment already exists at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

info "Upgrading pip inside venv"
run_cmd python -m pip install --upgrade pip setuptools wheel

if [ -f "${REQUIREMENTS_FILE}" ] && [ -s "${REQUIREMENTS_FILE}" ]; then
  info "Installing dependencies from ${REQUIREMENTS_FILE}"
  if $VERBOSE; then
    vinfo "Requirements to install:"
    sed 's/^/  - /' "${REQUIREMENTS_FILE}"
  fi
  run_cmd python -m pip install -r "${REQUIREMENTS_FILE}"
else
  warn "${REQUIREMENTS_FILE} is missing or empty; skipping dependency installation."
fi

info "Bootstrap complete. Virtual environment: ${VENV_DIR}"
info "To activate the venv in your shell: source ${VENV_DIR}/bin/activate"
info "To run the agent script: python agent_ingest.py"
info "Requirements file: ${REQUIREMENTS_FILE}"

info "Done."
