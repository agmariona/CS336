#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GIT_REPO_URL="git@github.com:USER/REPO.git" ./setup_runpod.sh
# or:
#   GIT_REPO_URL="https://github.com/USER/REPO.git" ./setup_runpod.sh

REPO_URL="https://github.com/agmariona/CS336.git"
WORKDIR="/workspace"
REPO_DIR="CS336"
PROJECT_DIR="assignment2-systems"

echo "==> Updating apt packages"
apt-get update
apt-get install -y \
  git \
  curl \
  ca-certificates \
  build-essential \
  python3 \
  python3-pip \
  python3-venv \
  pkg-config

echo "==> Installing uv if needed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Checking Nsight Systems"
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found; installing Nsight Systems..."
  apt-get install -y nsight-systems-2025.6.3
fi

echo "==> nsys version"
if command -v nsys >/dev/null 2>&1; then
  nsys --version || true
else
  echo "nsys still unavailable."
fi

echo "==> Preparing workspace"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "Repo already exists; pulling latest"
  cd "$REPO_DIR"
  git pull --ff-only
else
  echo "Cloning repo"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

echo "==> Installing project dependencies"
cd "$WORKDIR/$REPO_DIR/$PROJECT_DIR"
uv sync --link-mode=copy

echo "==> Basic sanity checks"
uv run python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count())"
uv run python -c "import cs336_basics, cs336_systems; print('imports ok')"

echo "==> Done"
echo "Repo: $WORKDIR/$REPO_DIR"
