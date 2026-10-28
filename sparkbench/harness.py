"""Shared plumbing: env capture, timing, JSONL resume, power/clock sampling."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SPARKBENCH_OUT", ROOT / "results"))
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "results.jsonl"


def sh(cmd: str, timeout: int = 30) -> str:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def load_profile(name: str) -> dict:
    path = ROOT / "profiles" / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"unknown profile: {name} (looked in {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def env_info() -> dict:
    import torch

    info = {
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "hostname": platform.node(),
        "kernel": platform.release(),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            device=p.name,
            cc=f"{p.major}.{p.minor}",
            sm_count=p.multi_processor_count,
            vram_gib=round(p.total_memory / 2**30, 1),
        )
    else:
        info["device"] = None
    return info


def _done_keys() -> set[str]:
    keys: set[str] = set()
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:  # noqa: BLE001
                continue
    return keys


DONE = _done_keys()


def emit(test: str, key: str, payload: dict) -> None:
    rec = {"key": f"{test}:{key}", "test": test, "ts": time.time(), **payload}
    with JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    DONE.add(rec["key"])


def skip(test: str, key: str, resume: bool) -> bool:
    return resume and f"{test}:{key}" in DONE


def bench(fn, warmup: int = 3, iters: int = 5, torch=None) -> dict:
    """Median/p95 wall time of fn(), with a warmup pass and a sync per iteration."""
    for _ in range(warmup):
        fn()
    if torch is not None:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch is not None:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "n": iters,
        "median_ms": round(samples[len(samples) // 2] * 1e3, 3),
        "p95_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))] * 1e3, 3),
        "min_ms": round(samples[0] * 1e3, 3),
    }


SAMPLER_FIELDS = "timestamp,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu"


class Sampler:
    """Background `nvidia-smi -lms` writer. Silently no-ops without nvidia-smi."""

    def __init__(self, tag: str, interval_ms: int = 500):
        self.path = OUT / f"smi-{tag}.csv"
        self.proc = None
        self.interval_ms = interval_ms

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            fh = self.path.open("w", encoding="utf-8")
            try:
                self.proc = subprocess.Popen(
                    ["nvidia-smi", f"--query-gpu={SAMPLER_FIELDS}",
                     "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                    stdout=fh, stderr=subprocess.DEVNULL,
                )
            except Exception:  # noqa: BLE001
                self.proc = None
            self._fh = fh
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        fh = getattr(self, "_fh", None)
        if fh:
            fh.close()
        return False

    def summary(self) -> dict:
        if not self.path.exists():
            return {}
        rows = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    "util": float(parts[1]),
                    "mem_mib": float(parts[2]),
                    "clock_mhz": float(parts[3]),
                    "power_w": float(parts[4]),
                    "temp_c": float(parts[5]),
                })
            except ValueError:
                continue
        if not rows:
            return {}
        n = len(rows)
        head = rows[: max(1, n // 6)]
        tail = rows[-max(1, n // 6):]

        def med(rs, k):
            v = sorted(r[k] for r in rs)
            return round(v[len(v) // 2], 1)

        return {
            "samples": n,
            "start": {k: med(head, k) for k in ("clock_mhz", "power_w", "temp_c")},
            "end": {k: med(tail, k) for k in ("clock_mhz", "power_w", "temp_c")},
            "peak_temp_c": round(max(r["temp_c"] for r in rows), 1),
            "peak_power_w": round(max(r["power_w"] for r in rows), 1),
            "mean_util_pct": round(sum(r["util"] for r in rows) / n, 1),
        }
