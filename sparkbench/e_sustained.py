"""E. Sustained load: burst numbers are marketing, steady-state is reality.

Runs a matmul loop for N seconds while sampling clocks/power/temp, then compares
the first window against the last window to quantify thermal/power throttling.
"""
from __future__ import annotations

import time

import torch

from .harness import Sampler, emit, skip


def run(profile: dict, resume: bool = True) -> None:
    cfg = profile.get("sustained", {})
    duration = float(cfg.get("seconds", 600))
    n = int(cfg.get("matmul_size", 8192))
    key = f"{int(duration)}s-{n}"

    if skip("e_sustained", key, resume):
        return

    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    flops = 2 * n**3

    windows: list[float] = []
    t_end = time.perf_counter() + duration
    with Sampler("e-sustained", interval_ms=500) as smp:
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            for _ in range(5):
                torch.mm(a, b)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            windows.append(round(5 * flops / dt / 1e12, 1))
            print(f"[E] {len(windows) * 5 * dt:>6.0f}s  {windows[-1]:>6.1f} TFLOP/s", end="\r")

    head = windows[: max(1, len(windows) // 6)]
    tail = windows[-max(1, len(windows) // 6):]
    head_med = sorted(head)[len(head) // 2]
    tail_med = sorted(tail)[len(tail) // 2]
    decay = round((1 - tail_med / head_med) * 100, 1) if head_med else None

    out = {
        "seconds": duration,
        "size": n,
        "head_tflops": head_med,
        "tail_tflops": tail_med,
        "decay_pct": decay,
        "samples": len(windows),
        "smi": smp.summary(),
    }
    emit("e_sustained", key, out)
    print(" " * 60, end="\r")
    print(f"[E] {duration:.0f}s soak: {head_med} -> {tail_med} TFLOP/s (decay {decay}%)")
    del a, b
    torch.cuda.empty_cache()
