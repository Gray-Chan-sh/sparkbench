#!/usr/bin/env bash
# Runs on the rented box, before anything else. Idempotent.
#
# Two things here are worth more than they look:
#   * the torch index is tried in order and each attempt is verified with a real
#     cuda.is_available(), because a metered box that "installed fine" and cannot
#     see the GPU is money spent for nothing;
#   * PEP 668. Ubuntu 24.04 marks the system python as externally managed, so a
#     bare `pip install` fails there. The Dockerfile passes
#     --break-system-packages; this script has to do the same or it dies on the
#     exact image it ships with.
set -uo pipefail

ARCH="$(uname -m)"
echo "[bootstrap] arch=${ARCH} python=$(python3 -V 2>&1)"

# ---------------------------------------------------------------- pip flags
PIP_FLAGS=""
if python3 -c 'import os,sysconfig,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")) else 1)' 2>/dev/null; then
  PIP_FLAGS="--break-system-packages"
  echo "[bootstrap] system python is externally managed -> ${PIP_FLAGS}"
fi

pip_install() { python3 -m pip install --no-cache-dir ${PIP_FLAGS} "$@"; }

# ---------------------------------------------------------------- torch
# Two separate questions, and they are not the same question:
#   can it see the GPU?  -- a cu124 build says yes on a Blackwell card
#   does it have sm_121 SASS? -- without it every kernel JITs from PTX at first
#   call. The matmul still succeeds, the timings are fiction.
# The PyTorch (Vast) template on this offer ships exactly that: working, wrong.
torch_state() {
  python3 - <<'PY'
import sys
try:
    import torch
except Exception:
    print("none"); sys.exit(0)
if not torch.cuda.is_available():
    print("no-cuda"); sys.exit(0)
p = torch.cuda.get_device_properties(0)
want = f"sm_{p.major}{p.minor}"
archs = torch.cuda.get_arch_list()
print(("native" if want in archs else "no-sass") + f" want={want} archs={','.join(archs) or '-'}")
PY
}

torch_native() {
  case "$(torch_state)" in
    native*) return 0 ;;
    *)       return 1 ;;
  esac
}

install_torch() {
  local candidates
  case "${ARCH}" in
    aarch64|arm64) candidates="cu130 cu128 pypi" ;;
    *)             candidates="cu128 cu130 pypi" ;;
  esac
  local c
  for c in ${candidates}; do
    echo "[bootstrap] torch: trying ${c}"
    case "${c}" in
      pypi) pip_install torch ;;
      *)    pip_install --index-url "https://download.pytorch.org/whl/${c}" torch ;;
    esac
    if torch_native; then
      echo "[bootstrap] torch: ok via ${c} ($(torch_state))"
      return 0
    fi
    echo "[bootstrap] torch: ${c} -> $(torch_state)"
  done
  return 1
}

state="$(torch_state)"
case "${state}" in
  native*)
    echo "[bootstrap] torch already has native kernels: ${state}"
    ;;
  no-sass*)
    if [ "${SPARKBENCH_ALLOW_JIT:-0}" = "1" ]; then
      echo "[bootstrap] WARNING: ${state} -> PTX JIT; continuing because SPARKBENCH_ALLOW_JIT=1"
      echo "[bootstrap] WARNING: these numbers are not comparable to a native build."
    else
      echo "[bootstrap] torch sees the GPU but has no matching SASS: ${state}"
      echo "[bootstrap] every kernel would JIT-compile; reinstalling"
      install_torch || {
        echo "[bootstrap] FATAL: no torch build here has native kernels for this GPU." >&2
        echo "[bootstrap]        set SPARKBENCH_ALLOW_JIT=1 to measure anyway (numbers will not be comparable)." >&2
        exit 1
      }
    fi
    ;;
  *)
    install_torch || {
      echo "[bootstrap] FATAL: no torch build here can see the GPU." >&2
      echo "[bootstrap]        stop the instance; do not pay for a broken box." >&2
      exit 1
    }
    ;;
esac

pip_install -r "$(dirname "$0")/requirements.txt" || {
  echo "[bootstrap] WARNING: some requirements failed to install" >&2
}

# ---------------------------------------------------------------- tools
if ! command -v tmux >/dev/null; then
  # run.sh survives an SSH drop with tmux; nohup is the fallback if apt is unavailable.
  (apt-get update -qq && apt-get install -y -qq --no-install-recommends tmux) >/dev/null 2>&1 \
    && echo "[bootstrap] tmux installed" \
    || echo "[bootstrap] no tmux (apt failed); run.sh will fall back to nohup"
fi

# ---------------------------------------------------------------- report
python3 - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"[bootstrap] torch={torch.__version__} cuda={torch.version.cuda} "
      f"cc={p.major}.{p.minor} vram={p.total_memory/2**30:.1f}GiB sm={p.multi_processor_count}")
print(f"[bootstrap] arch_list={torch.cuda.get_arch_list()}")
PY

if command -v ffmpeg >/dev/null; then
  echo "[bootstrap] ffmpeg hwaccels: $(ffmpeg -hide_banner -hwaccels 2>/dev/null | tr '\n' ' ')"
else
  echo "[bootstrap] WARNING: no ffmpeg on PATH"
fi

echo "[bootstrap] done -> python3 -m sparkbench smoke --profile ${SPARKBENCH_PROFILE:-gb10}"
