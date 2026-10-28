# Reproducible arm64 image for GB10 / sm_121.
#
# Two supported bases:
#   nvcr.io/nvidia/pytorch:*  -> has sm_121 kernels and python preinstalled (needs an NGC key)
#   nvidia/cuda:*-devel-*     -> public, no login; torch is installed from the cu130 index
#
# Build on any x86 machine (emulated) or in CI:
#   docker buildx build --platform linux/arm64 -t sparkbench:latest --build-arg BASE=<img> .
ARG BASE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/sparkbench \
    SPARKBENCH_OUT=/workspace/results

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CUDA base images ship no python; NGC images ship python + torch.
RUN if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends \
            python3 python3-pip python3-venv \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN python3 -c "import torch" 2>/dev/null || \
    pip3 install --no-cache-dir --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu130 torch

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Hard gate: a torch build without Blackwell SASS is useless here. PTX-only would
# "work" but JIT-compiles every kernel, which silently ruins all timings.
RUN python3 - <<'PY'
import sys
import torch

archs = []
try:
    archs = torch.cuda.get_arch_list()
except Exception as exc:  # noqa: BLE001
    print("warn: arch list unavailable:", exc)

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("archs:", archs)

if not archs:
    sys.exit("ERROR: torch reports no compiled CUDA archs (CPU-only build?)")
if not any(a in archs for a in ("sm_100", "sm_103", "sm_110", "sm_120", "sm_121", "sm_130")):
    sys.exit(f"ERROR: no Blackwell arch in {archs} -> this image cannot run on GB10")
PY

COPY . /opt/sparkbench
WORKDIR /workspace
CMD ["sleep", "infinity"]
