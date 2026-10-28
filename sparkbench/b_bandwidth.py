"""B. Memory bandwidth: the number that actually decides decode speed on GB10."""
from __future__ import annotations

import torch

from .harness import Sampler, bench, emit, skip


def run(profile: dict, resume: bool = True) -> None:
    gib = profile["bandwidth"]["buffer_gib"]
    n = gib * 2**30 // 4  # float32 elements

    cases = {}

    if not skip("b_bandwidth", "copy", resume):
        x = torch.empty(n, dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)
        x.fill_(1.0)
        with Sampler("b-copy") as smp:
            r = bench(lambda: y.copy_(x), warmup=2, iters=5, torch=torch)
        gbs = round(2 * gib / (r["median_ms"] / 1e3), 1)  # read + write
        cases["copy"] = gbs
        emit("b_bandwidth", "copy", {"gib": gib, "gbs": gbs, "pattern": "read+write", **r, "smi": smp.summary()})
        print(f"[B] copy      {gib} GiB -> {gbs:>6.1f} GB/s")
        del x, y
        torch.cuda.empty_cache()

    if not skip("b_bandwidth", "read", resume):
        x = torch.empty(n, dtype=torch.float32, device="cuda")
        x.fill_(1.0)
        with Sampler("b-read") as smp:
            r = bench(lambda: x.sum(), warmup=2, iters=5, torch=torch)
        gbs = round(gib / (r["median_ms"] / 1e3), 1)
        cases["read"] = gbs
        emit("b_bandwidth", "read", {"gib": gib, "gbs": gbs, "pattern": "read", **r, "smi": smp.summary()})
        print(f"[B] read      {gib} GiB -> {gbs:>6.1f} GB/s")
        del x
        torch.cuda.empty_cache()

    if not skip("b_bandwidth", "write", resume):
        y = torch.empty(n, dtype=torch.float32, device="cuda")
        with Sampler("b-write") as smp:
            r = bench(lambda: y.fill_(2.0), warmup=2, iters=5, torch=torch)
        gbs = round(gib / (r["median_ms"] / 1e3), 1)
        cases["write"] = gbs
        emit("b_bandwidth", "write", {"gib": gib, "gbs": gbs, "pattern": "write", **r, "smi": smp.summary()})
        print(f"[B] write     {gib} GiB -> {gbs:>6.1f} GB/s")
        del y
        torch.cuda.empty_cache()

    # Decode ceiling: bytes/sec over model size. This is the projection that matters
    # for "how big a model can this box serve at a usable rate".
    bw = max(cases.values()) if cases else None
    if bw:
        proj = {f"{gb}GB-model": round(bw / gb, 2) for gb in (8, 35, 70, 120)}
        emit("b_bandwidth", "decode_ceiling", {"gbs": bw, "tok_s_ceiling": proj})
        print("[B] decode ceiling (tok/s, bandwidth-bound, 1 pass/token):",
              ", ".join(f"{k}={v}" for k, v in proj.items()))
