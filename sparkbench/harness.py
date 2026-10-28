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


def _cgroup_limit_bytes() -> int | None:
    """Container memory cap. On GB10 this caps the *unified* pool, not just RAM:
    a 64 GiB cap makes a 128 GiB machine behave like a 64 GiB machine, and the
    failure surfaces as a CUDA OOM that looks like a VRAM problem."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            continue
        if v == "max":
            return None
        if v.isdigit():
            n = int(v)
            return None if n > 2**60 else n   # cgroup v1 "unlimited" sentinel
    return None


def _meminfo() -> dict:
    try:
        out = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(":")
            out[k.strip()] = int(v.strip().split()[0]) * 1024
        return out
    except Exception:  # noqa: BLE001
        return {}


def memory_facts(vram_gib: float | None = None) -> dict:
    """What this process can actually reach, versus what the GPU claims."""
    gib = lambda b: round(b / 2**30, 1) if b else None  # noqa: E731
    mem = _meminfo()
    total, avail = mem.get("MemTotal"), mem.get("MemAvailable")
    cap = _cgroup_limit_bytes()
    container = Path("/.dockerenv").exists() or "docker" in sh("head -c 4000 /proc/1/cgroup 2>/dev/null")
    vram_b = vram_gib * 2**30 if vram_gib else None
    unified = bool(vram_gib and total and abs(vram_gib - total / 2**30) / (total / 2**30) < 0.15)
    # The "effective pool" only means something when CPU and GPU share one pool.
    # On a discrete card the cgroup cap limits host RAM, not VRAM.
    candidates = [x for x in (total, cap) if x] + ([vram_b] if unified and vram_b else [])
    return {
        "container": bool(container),
        "mem_total_gib": gib(total),
        "mem_available_gib": gib(avail),
        "cgroup_cap_gib": gib(cap),
        "vram_gib": vram_gib,
        "effective_pool_gib": gib(min(candidates)) if unified and candidates else None,
        "unified": unified,
    }


def _git_info() -> dict:
    """Which code produced these numbers. List form: the repo path may contain spaces."""
    def run(*args):
        try:
            p = subprocess.run(["git", "-C", str(ROOT), *args],
                               capture_output=True, text=True, timeout=10)
            return p.returncode, p.stdout.strip()
        except Exception:  # noqa: BLE001
            return 1, ""

    rc, commit = run("rev-parse", "--short", "HEAD")
    if rc != 0 or not commit:
        return {"commit": None, "dirty": None}
    rc2, dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(dirty) if rc2 == 0 else None}


def write_lockfile() -> Path | None:
    """One version string does not pin transitive deps."""
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:  # noqa: BLE001
        return None
    if not out.strip():
        return None
    p = OUT / "requirements.lock"
    p.write_text(out, encoding="utf-8")
    return p


def write_manifest(extra: dict | None = None) -> Path:
    """Per-file sha256 so "complete" is verifiable rather than claimed."""
    import hashlib

    files = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name == "manifest.json":
            continue
        rel = str(p.relative_to(OUT))
        size = p.stat().st_size
        if size > 32 * 2**20:
            files.append({"path": rel, "size": size, "sha256": None, "unhashed": "large"})
            continue
        files.append({"path": rel, "size": size,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})

    doc = {
        "generated": now(),
        "version": _version(),
        "git": _git_info(),
        "revisions": dict(_REVS),
        "count": len(files),
        "bytes": sum(f["size"] for f in files),
        "files": files,
        **(extra or {}),
    }
    path = OUT / "manifest.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


_REVS: dict[str, str | None] = {}


def _cache_revision(repo_id: str) -> str | None:
    base = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    d = base / ("models--" + repo_id.replace("/", "--"))
    ref = d / "refs" / "main"
    if ref.exists():
        return ref.read_text(encoding="utf-8").strip()
    snaps = d / "snapshots"
    if snaps.exists():
        found = sorted(p.name for p in snaps.iterdir() if p.is_dir())
        if found:
            return found[-1]
    return None


def model_revision(repo_id: str) -> str | None:
    """Pin weights: HF repos are updated in place under the same id."""
    if repo_id in _REVS:
        return _REVS[repo_id]
    sha = None
    revfile = OUT / "revisions.json"
    if revfile.exists():
        try:
            sha = json.loads(revfile.read_text(encoding="utf-8")).get(repo_id)
        except Exception:  # noqa: BLE001
            sha = None
    if not sha:
        try:
            from huggingface_hub import model_info

            sha = model_info(repo_id, timeout=20).sha
        except Exception:  # noqa: BLE001
            sha = None
    if not sha:
        sha = _cache_revision(repo_id)
    _REVS[repo_id] = sha
    return sha


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


def env_info() -> dict:
    import torch

    info = {
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "hostname": platform.node(),
        "kernel": platform.release(),
        "version": _version(),
        "git": _git_info(),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        try:
            archs = torch.cuda.get_arch_list()
        except Exception:  # noqa: BLE001
            archs = []
        info.update(
            device=p.name,
            cc=f"{p.major}.{p.minor}",
            sm_count=p.multi_processor_count,
            vram_gib=round(p.total_memory / 2**30, 1),
            # SASS present in this build. If sm_121 is missing, every kernel is
            # JIT-compiled from PTX at first call: it runs, and every timing is wrong.
            torch_archs=archs,
        )
    else:
        info["device"] = None
    info["memory"] = memory_facts(info.get("vram_gib"))
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
