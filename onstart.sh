#!/usr/bin/env bash
# vast.ai onstart: fetched and executed by the host when the instance boots.
# Keep it non-interactive and make it survive a missing network by retrying.
set -uo pipefail

REPO_URL="${SPARKBENCH_REPO:-}"          # e.g. https://github.com/you/sparkbench.git
PROFILE="${SPARKBENCH_PROFILE:-gb10}"
WORK="${SPARKBENCH_WORK:-/workspace/sparkbench}"

echo "[onstart] $(date -u) arch=$(uname -m)"

mkdir -p /workspace
if [ -n "${REPO_URL}" ]; then
  for i in 1 2 3; do
    git clone --depth 1 "${REPO_URL}" "${WORK}" && break
    echo "[onstart] clone failed, retry ${i}"; sleep 5
  done
fi

# Prebuilt image already carries the code at /opt/sparkbench.
if [ ! -d "${WORK}" ] && [ -d /opt/sparkbench ]; then
  WORK=/opt/sparkbench
fi

cd "${WORK}" || { echo "[onstart] no code at ${WORK}"; sleep infinity; }
bash bootstrap.sh || echo "[onstart] bootstrap failed, instance left up for debugging"

export SPARKBENCH_OUT=/workspace/results
python -m sparkbench smoke --profile "${PROFILE}" && \
  python -m sparkbench all --profile "${PROFILE}" || \
  echo "[onstart] bench exited non-zero"

echo "[onstart] finished $(date -u); keeping container alive"
sleep infinity
