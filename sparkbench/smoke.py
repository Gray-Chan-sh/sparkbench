"""Smoke gate: run first, 3 minutes. If this fails, do not spend the rest of the booking."""
from __future__ import annotations

import subprocess
import sys

from .harness import env_info, load_profile, sh

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


def _row(name, status, detail=""):
    return (name, status, detail)


def run(profile_name: str) -> int:
    prof = load_profile(profile_name)
    expect = prof.get("expect", {})
    rows = []

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: torch not importable: {exc}")
        return 2

    info = env_info()
    rows.append(_row("architecture", OK if info["arch"] in ("aarch64", "arm64") else WARN, info["arch"]))

    cuda = torch.cuda.is_available()
    rows.append(_row("cuda available", OK if cuda else FAIL, str(cuda)))
    if not cuda:
        _print(rows)
        return 1

    rows.append(_row("device", OK, str(info.get("device"))))
    rows.append(_row("torch/cuda", OK, f"{info['torch']} / {info['torch_cuda']}"))
    rows.append(_row("driver", OK, info.get("driver", "")))

    cc = info.get("cc")
    want_cc = expect.get("cc")
    rows.append(_row("compute capability", OK if cc == want_cc else FAIL, f"{cc} (expected {want_cc})"))

    vram = info.get("vram_gib") or 0
    want_vram = expect.get("vram_gib")
    if want_vram:
        ratio = vram / want_vram
        status = OK if 0.88 <= ratio <= 1.02 else (WARN if ratio > 0.75 else FAIL)
        rows.append(_row("vram", status, f"{vram} GiB (expected ~{want_vram})"))
    else:
        rows.append(_row("vram", OK, f"{vram} GiB"))

    # --- what the process can actually reach ------------------------------------
    # On a unified-memory box a container memory cap silently truncates the GPU pool,
    # and the resulting CUDA OOM reads like "not enough VRAM". Catch it before the run.
    mem = info.get("memory") or {}
    rows.append(_row("runtime", OK if not mem.get("container") else WARN,
                     "container" if mem.get("container") else "bare metal"))
    if mem.get("unified") is not None:
        if mem.get("unified"):
            rows.append(_row("unified memory", OK,
                             f"yes (cpu {mem.get('mem_total_gib')} GiB == gpu {vram} GiB)"))
        else:
            rows.append(_row("unified memory", OK,
                             f"no (cpu {mem.get('mem_total_gib')} GiB, gpu {vram} GiB)"))

    cap = mem.get("cgroup_cap_gib")
    if mem.get("unified") and cap and vram and cap < vram * 0.9:
        rows.append(_row("cgroup cap vs pool", FAIL,
                         f"cap {cap} GiB < pool {vram} GiB -> you cannot use the full card"))
    else:
        rows.append(_row("cgroup cap vs pool", OK,
                         f"cap {cap} GiB" if cap else "no cap"))

    avail = mem.get("mem_available_gib")
    if avail is not None and vram:
        rows.append(_row("free memory at start", OK if avail > vram * 0.5 else WARN,
                         f"{avail} GiB available"))

    # Real matmul, not just an import: catches missing sm_XX kernels.
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        rows.append(_row("bf16 matmul", OK, f"sum={float(c[0, 0]):.2f}"))
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        rows.append(_row("bf16 matmul", FAIL, f"{type(exc).__name__}: {str(exc)[:120]}"))

    # Decode path: a CPU-only ffmpeg on an ARM box silently invalidates every video number.
    hw = sh("ffmpeg -hide_banner -hwaccels 2>&1 | tail -n +2")
    has_nvdec = any(k in hw.lower() for k in ("cuda", "nvdec"))
    rows.append(_row("ffmpeg hwaccel", OK if has_nvdec else WARN, hw.replace("\n", " ")[:80]))

    try:
        import av  # noqa: F401
        rows.append(_row("pyav", OK, "importable"))
    except Exception as exc:  # noqa: BLE001
        rows.append(_row("pyav", WARN, str(exc)[:60]))

    rows.append(_row("nvidia-smi", OK if sh("which nvidia-smi") else WARN, ""))
    rows.append(_row("hostname", OK, info["hostname"]))

    _print(rows)
    return 0 if all(r[1] != FAIL for r in rows) else 1


def _print(rows):
    width = max(len(r[0]) for r in rows) + 2
    print("\n" + "=" * 62)
    for name, status, detail in rows:
        mark = {OK: " OK ", WARN: "WARN", FAIL: "FAIL"}[status]
        print(f"[{mark}] {name.ljust(width)} {detail}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else "gb10"))
