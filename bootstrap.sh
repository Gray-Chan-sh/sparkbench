#!/usr/bin/env bash
# Runs on the rented box, before anything else. Idempotent.
set -euo pipefail

ARCH="$(uname -m)"
echo "[bootstrap] arch=${ARCH} python=$(python -V 2>&1)"

if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[bootstrap] no CUDA torch found -> installing"
  case "${ARCH}" in
    aarch64|arm64) IDX="https://download.pytorch.org/whl/cu130" ;;
    *)             IDX="https://download.pytorch.org/whl/cu128" ;;
  esac
  pip install --no-cache-dir --index-url "${IDX}" torch
fi

pip install --no-cache-dir -r "$(dirname "$0")/requirements.txt"

python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"[bootstrap] torch={torch.__version__} cuda={torch.version.cuda} "
      f"cc={p.major}.{p.minor} vram={p.total_memory/2**30:.1f}GiB sm={p.multi_processor_count}")
PY

if command -v ffmpeg >/dev/null; then
  echo "[bootstrap] ffmpeg hwaccels: $(ffmpeg -hide_banner -hwaccels 2>/dev/null | tr '\n' ' ')"
else
  echo "[bootstrap] WARNING: no ffmpeg on PATH"
fi

echo "[bootstrap] done -> python -m sparkbench smoke --profile ${SPARKBENCH_PROFILE:-gb10}"
