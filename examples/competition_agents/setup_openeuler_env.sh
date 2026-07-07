#!/usr/bin/env bash
set -Eeuo pipefail

# One-command environment bootstrap for the competition_agents demo on openEuler/Linux.
# Usage from repository root:
#   bash examples/competition_agents/setup_openeuler_env.sh
#
# Optional environment variables:
#   ENV_NAME=competition-agents
#   PYTHON_VERSION=3.11
#   CONDA_HOME=$HOME/miniconda3
#   HF_ENDPOINT=https://hf-mirror.com
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   NPM_REGISTRY=https://registry.npmmirror.com
#   DOWNLOAD_MODELS=1
#   INSTALL_SYSTEM_PACKAGES=1

ENV_NAME="${ENV_NAME:-competition-agents}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_BIN="$CONDA_HOME/bin/conda"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
CONDA_CHANNEL_ALIAS="${CONDA_CHANNEL_ALIAS:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}"
NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmmirror.com}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"

QWEN_MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)
EMBEDDING_MODELS=(
  "sentence-transformers/all-MiniLM-L6-v2"
)

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/../.." >/dev/null 2>&1
  pwd
}

install_system_packages() {
  if [[ "$INSTALL_SYSTEM_PACKAGES" != "1" ]]; then
    return
  fi
  if ! command -v dnf >/dev/null 2>&1; then
    log "dnf not found, skip system package installation."
    return
  fi
  local sudo_cmd=()
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo_cmd=(sudo)
    else
      log "sudo not found, skip dnf install. Install git/curl/wget/tar/bzip2 manually if missing."
      return
    fi
  fi
  log "Installing basic system packages with dnf."
  "${sudo_cmd[@]}" dnf install -y git curl wget tar gzip bzip2 xz which ca-certificates || \
    log "dnf install failed; continuing because packages may already exist."
}

install_miniconda_if_needed() {
  if [[ -x "$CONDA_BIN" ]]; then
    log "Using existing conda: $CONDA_BIN"
    return
  fi

  local arch installer_name installer_url tmp_installer
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) installer_name="Miniconda3-latest-Linux-x86_64.sh" ;;
    aarch64|arm64) installer_name="Miniconda3-latest-Linux-aarch64.sh" ;;
    *) fail "Unsupported CPU architecture: $arch" ;;
  esac

  installer_url="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/${installer_name}"
  tmp_installer="/tmp/${installer_name}"
  log "Downloading Miniconda from Tsinghua mirror: $installer_url"
  curl -L --retry 5 --connect-timeout 20 -o "$tmp_installer" "$installer_url"
  log "Installing Miniconda to $CONDA_HOME"
  bash "$tmp_installer" -b -p "$CONDA_HOME"
}

configure_conda() {
  log "Configuring conda mirrors."
  "$CONDA_BIN" config --set show_channel_urls yes
  "$CONDA_BIN" config --set channel_alias "$CONDA_CHANNEL_ALIAS"
  "$CONDA_BIN" config --remove-key channels >/dev/null 2>&1 || true
  "$CONDA_BIN" config --remove-key default_channels >/dev/null 2>&1 || true
  "$CONDA_BIN" config --add default_channels "$CONDA_CHANNEL_ALIAS/pkgs/main"
  "$CONDA_BIN" config --add default_channels "$CONDA_CHANNEL_ALIAS/pkgs/r"
  "$CONDA_BIN" config --add channels defaults
  "$CONDA_BIN" config --add channels conda-forge
  "$CONDA_BIN" config --set custom_channels.conda-forge "$CONDA_CHANNEL_ALIAS/cloud"
  "$CONDA_BIN" config --set custom_channels.pytorch "$CONDA_CHANNEL_ALIAS/cloud"
}

create_or_update_env() {
  if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Conda env '$ENV_NAME' already exists; updating core packages."
  else
    log "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION."
    "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi

  log "Installing Node.js into conda env."
  "$CONDA_BIN" install -y -n "$ENV_NAME" -c conda-forge "nodejs>=20,<23"
}

conda_run() {
  "$CONDA_BIN" run -n "$ENV_NAME" "$@"
}

install_python_packages() {
  log "Configuring pip mirror."
  conda_run python -m pip config set global.index-url "$PIP_INDEX_URL"
  conda_run python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

  log "Installing Python runtime packages."
  conda_run python -m pip install -U pip setuptools wheel
  conda_run python -m pip install \
    "torch>=2.5" \
    "transformers>=4.45,<5" \
    "accelerate>=0.34" \
    "sentence-transformers>=3.0" \
    "huggingface-hub>=0.25" \
    "safetensors>=0.4" \
    "tokenizers>=0.20" \
    "numpy>=1.26" \
    "requests>=2.32" \
    "pytest>=8"

  log "Installing local LangGraph packages from this repository."
  conda_run python -m pip install -e libs/checkpoint -e libs/prebuilt -e libs/langgraph
}

install_frontend_packages() {
  log "Installing frontend packages with npm mirror."
  pushd examples/competition_agents/frontend >/dev/null
  conda_run npm config set registry "$NPM_REGISTRY"
  conda_run npm install
  popd >/dev/null
}

download_models() {
  if [[ "$DOWNLOAD_MODELS" != "1" ]]; then
    log "DOWNLOAD_MODELS=$DOWNLOAD_MODELS, skip model download."
    return
  fi

  log "Downloading Qwen and embedding models through HF_ENDPOINT=$HF_ENDPOINT"
  export HF_ENDPOINT
  export HF_HUB_ENABLE_HF_TRANSFER=0

  local model_args=()
  for model in "${QWEN_MODELS[@]}"; do
    model_args+=("--qwen" "$model")
  done
  for model in "${EMBEDDING_MODELS[@]}"; do
    model_args+=("--embedding" "$model")
  done

  conda_run python - "${model_args[@]}" <<'PY'
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--qwen", action="append", default=[])
parser.add_argument("--embedding", action="append", default=[])
args = parser.parse_args()

print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT", ""))
print("HF_HOME =", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

from huggingface_hub import snapshot_download

for model_name in args.qwen:
    print(f"\nDownloading causal LM: {model_name}")
    snapshot_download(
        repo_id=model_name,
        resume_download=True,
        local_files_only=False,
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    print(f"Ready: {model_name}")

for model_name in args.embedding:
    print(f"\nDownloading embedding model: {model_name}")
    snapshot_download(
        repo_id=model_name,
        resume_download=True,
        local_files_only=False,
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
    )
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(model_name)
    print(f"Ready: {model_name}")
PY
}

write_env_file() {
  local env_file="examples/competition_agents/.env.local"
  if [[ -f "$env_file" ]]; then
    log "$env_file already exists; keeping it."
    return
  fi
  log "Writing default local .env file."
  cat > "$env_file" <<EOF
LLM_PROVIDER=local_hf
LOCAL_HIDDEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_HIDDEN_POOLED_DIM=128
LOCAL_LLM_PLANNER_MAX_NEW_TOKENS=48
LOCAL_LLM_REASONER_MAX_NEW_TOKENS=72
EMBEDDING_PROVIDER=local
HF_ENDPOINT=$HF_ENDPOINT
EOF
}

verify_installation() {
  log "Verifying installation."
  conda_run python - <<'PY'
import importlib
mods = ["torch", "transformers", "sentence_transformers", "huggingface_hub", "langgraph"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'ok')}")
PY
  conda_run npm --version
  conda_run node --version
}

print_next_steps() {
  cat <<EOF

Environment setup finished.

Activate:
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"

Run backend:
  cd "$(pwd)/examples/competition_agents"
  python runner_server.py

Run frontend in another terminal:
  cd "$(pwd)/examples/competition_agents/frontend"
  npm run dev -- --host 0.0.0.0 --port 5173

Open:
  http://<server-ip>:5173

If model download failed because the mirror is unreachable, rerun only that step:
  DOWNLOAD_MODELS=1 bash examples/competition_agents/setup_openeuler_env.sh
EOF
}

main() {
  local root
  root="$(repo_root)"
  cd "$root"
  log "Repository root: $root"

  install_system_packages
  install_miniconda_if_needed
  configure_conda
  create_or_update_env
  install_python_packages
  install_frontend_packages
  download_models
  write_env_file
  verify_installation
  print_next_steps
}

main "$@"
