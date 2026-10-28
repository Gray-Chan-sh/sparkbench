"""C. Capacity ceiling: how much of the unified pool can the GPU actually take,
and what happens at the edge (OOM kill vs swap cliff). Unified memory makes this
non-obvious: the machine may not OOM at all, it may just fall off a performance cliff.
"""
from __future__ import annotations

import time

import torch

from .harness import Sampler, emit, skip


def run(profile: dict, resume: bool = True) -> None:
    if skip("c_capacity", "alloc", resume):
        return

    step = profile["capacity"]["step_gib"]
    cap = profile["capacity"]["max_gib"]
    cliff_factor = profile["capacity"].get("cliff_factor", 3.0)
    elems = step * 2**30 // 4  # float32 per step

    blocks: list[torch.Tensor] = []
    fill_ms: list[float] = []
    reached = 0
    cliff_gib = None
    error = None

    with Sampler("c-capacity") as smp:
        while reached < cap:
            try:
                b = torch.empty(elems, dtype=torch.float32, device="cuda")
                t0 = time.perf_counter()
                b.fill_(1.0)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1e3
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                break
            blocks.append(b)
            fill_ms.append(round(dt, 1))
            reached += step
            if len(fill_ms) >= 4:
                baseline = sorted(fill_ms[:3])[1]
                if cliff_gib is None and dt > baseline * cliff_factor:
                    cliff_gib = reached
                    print(f"[C] swap cliff at ~{reached} GiB ({dt:.0f} ms vs {baseline:.0f} ms baseline)")
            print(f"[C] {reached:>4} GiB allocated, fill {dt:>7.1f} ms", end="\r")

    print(" " * 70, end="\r")
    summary = {
        "allocated_gib": reached,
        "cliff_gib": cliff_gib,
        "fill_ms": fill_ms,
        "declared_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "error": error,
        "smi": smp.summary(),
    }
    emit("c_capacity", "alloc", summary)
    print(f"[C] reached {reached} GiB (declared {summary['declared_vram_gib']} GiB), cliff={cliff_gib}")
    if error:
        print(f"[C] stopped by: {error[:160]}")

    del blocks
    torch.cuda.empty_cache()
