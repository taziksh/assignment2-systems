#!/usr/bin/env bash
set -euo pipefail
trap 'echo "✗ failed at line $LINENO" >&2' ERR

NSYS="2026.1.3"
NSYS_DIR="/opt/nvidia/nsight-systems/${NSYS}/target-linux-x64"

echo "[1/4] system packages"
apt-get update -qq
apt-get install -y -qq tmux wget gnupg

echo "[2/4] nsight systems ${NSYS}"
if [ ! -x "${NSYS_DIR}/nsys" ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  apt-get install -y -qq "nsight-systems-${NSYS}"
fi
export PATH="${NSYS_DIR}:${PATH}"
grep -q "${NSYS_DIR}" ~/.bashrc 2>/dev/null || echo "export PATH=\"${NSYS_DIR}:\$PATH\"" >> ~/.bashrc

echo "[3/4] python env"
pip install -q uv
uv sync

echo "[4/4] verify"
nsys --version
uv run python -c "import cs336_basics, os; print('cs336_basics ->', os.path.dirname(cs336_basics.__file__))"

echo "✓ done"
