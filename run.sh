#!/usr/bin/env bash
# Full run: smoke gate -> A..E -> report -> audit -> push.
# Survives SSH drops (tmux). Each section is resumable, so a re-run skips finished work.
#
#   bash run.sh                  # full ~3h
#   bash run.sh --phases a,b     # partial
#   bash run.sh --skip-smoke     # only if you already ran the gate by hand
set -uo pipefail
cd "$(dirname "$0")" || exit 1
SELF="$PWD/$(basename "$0")"

PHASES=""
SKIP_SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --phases)     PHASES="$2"; shift ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
  shift
done

# results/ must exist before tee runs inside the detached session, not after.
mkdir -p results

if [ -z "${TMUX:-}" ] && [ "${SB_NO_TMUX:-0}" != "1" ]; then
  if command -v tmux >/dev/null; then
    tmux kill-session -t sparkbench 2>/dev/null || true
    quoted=$(printf ' %q' "$@")
    tmux new-session -d -s sparkbench "SB_NO_TMUX=1 bash '$SELF'$quoted 2>&1 | tee -a results/run.log"
    echo "started in tmux:  tmux attach -t sparkbench   (or tail -f results/run.log)"
  else
    # No tmux (fresh container): nohup still survives the SSH drop.
    SB_NO_TMUX=1 nohup bash "$SELF" "$@" >> results/run.log 2>&1 &
    echo "no tmux; started with nohup (pid $!) -> tail -f results/run.log"
  fi
  exit 0
fi
echo "=== sparkbench start $(date -u) ===" | tee -a results/run.log

bash bootstrap.sh 2>&1 | tee -a results/run.log

if [ "$SKIP_SMOKE" -eq 0 ]; then
  echo "--- smoke gate $(date -u)" | tee -a results/run.log
  if ! python3 -m sparkbench smoke --profile "${SB_PROFILE:-gb10}" 2>&1 | tee -a results/run.log; then
    echo "=== smoke FAILED — stopping before spending the booking ===" | tee -a results/run.log
    exit 1
  fi
fi

if [ -n "$PHASES" ]; then
  IFS=',' read -ra list <<< "$PHASES"
  for p in "${list[@]}"; do
    echo "--- section $p $(date -u)" | tee -a results/run.log
    python3 -m sparkbench "$p" --profile "${SB_PROFILE:-gb10}" 2>&1 | tee -a results/run.log
  done
else
  python3 -m sparkbench all --profile "${SB_PROFILE:-gb10}" 2>&1 | tee -a results/run.log
fi

echo "--- report $(date -u)" | tee -a results/run.log
python3 tools/report.py --profile "${SB_PROFILE:-gb10}" 2>&1 | tee -a results/run.log

echo "--- audit $(date -u)" | tee -a results/run.log
python3 -m sparkbench audit 2>&1 | tee -a results/run.log

echo "--- push $(date -u)" | tee -a results/run.log
python3 tools/push.py --git --webhook 2>&1 | tee -a results/run.log

date -u > results/DONE
echo "=== done $(date -u) — 实例仍在计费，确认结果已推送后销毁 ===" | tee -a results/run.log
