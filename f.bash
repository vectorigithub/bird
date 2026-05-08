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

info "=========================================="
info "SETUP INSTRUCTIONS"
info "=========================================="
info ""
info "1. DATASET PREPARATION"
info "   - Create a 'dataset/' directory in the project root"
info "   - Place text-based files inside dataset/:"
info "     * Code files (.py, .js, .ts, .java, etc.)"
info "     * Documentation (.md, .txt, .rst, etc.)"
info "     * Config files (.json, .yaml, .xml, etc.)"
info "     * Any character/text-based files"
info "   - All text files in dataset/ will be read by default"
info ""
info "2. CONFIGURE handler_agent (config.json)"
info "   Edit config.json to customize file loading:"
info "   "
info "   Example handler section:"
info "   {"
info "     \"handler\": {"
info "       \"DATA_DIR\": \"dataset/\",\"
info "       \"_include_extensions\": [\".py\", \".md\", \".txt\"],"
info "       \"_exclude_extensions\": [\".log\", \".tmp\"],"
info "       \"_include_patterns\": [\".*\\\\.py$\"],"
info "       \"_exclude_patterns\": [\"test.*\", \".*/tests/.*\"]"
info "     }"
info "   }"
info "   "
info "   Options:"
info "   - DATA_DIR: Directory to scan (default: dataset/)"
info "   - include_extensions: Whitelist specific extensions"
info "   - exclude_extensions: Blacklist specific extensions"
info "   - include_patterns: Regex patterns for files to include"
info "   - exclude_patterns: Regex patterns for files to exclude"
info "   (Prefix with _ to keep as commented examples)"
info ""
info "3. CONFIGURE WEB SEARCH (queries.web.json)"
info "   Edit runtimes/leechers/queries.web.json:"
info "   "
info "   {"
info "     \"queries\": ["
info "       \"your search query 1\","
info "       \"your search query 2\""
info "     ],"
info "     \"_settings\": {"
info "       \"max_results_per_query\": 10,"
info "       \"delay_between_queries_ms\": 1000"
info "     }"
info "   }"
info "   "
info "   To run web searches:"
info "   cd runtimes/leechers"
info "   npm install node-fetch"
info "   node leech.web.js --batch --output results.json"
info ""
info "4. VERIFY SETUP"
info "   - Ensure dataset/ contains text files"
info "   - Check config.json has valid JSON syntax"
info "   - Verify queries.web.json if using batch search"
info ""
info "=========================================="

info "Done."