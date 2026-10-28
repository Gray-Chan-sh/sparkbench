"""A. Compute ceiling: achieved TFLOP/s vs the marketing number."""
from __future__ import annotations

import torch

from .harness import Sampler, bench, emit, skip

TF = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _tflops(flops: float, ms: float) -> float:
    return round(flops / (ms / 1e3) / 1e12, 1)


def _fp8_probe(size: int) -> dict:
    """FP8 needs _scaled_mm; NVFP4 needs TensorRT-LLM/ModelOpt, neither is in core torch."""
    out = {"status": "unavailable"}
    try:
        a = torch.randn(size, size, device="cuda", dtype=torch.float8_e4m3fn)
        b = torch.randn(size, size, device="cuda", dtype=torch.float8_e4m3fn).t().contiguous().t()
        sa = torch.tensor(1.0, device="cuda")
        sb = torch.tensor(1.0, device="cuda")
        variants = [
            lambda: torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16),
            lambda: torch._scaled_mm(a, b.T, sa, sb, out_dtype=torch.bfloat16),
        ]
        for fn in variants:
            try:
                r = bench(fn, warmup=3, iters=5, torch=torch)
                out = {"status": "ok", "size": size, "tflops": _tflops(2 * size**3, r["median_ms"]), **r}
                break
            except Exception as exc:  # noqa: BLE001
                out = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"[:300]}
    except Exception as exc:  # noqa: BLE001
        out = {"status": "unavailable", "detail": f"{type(exc).__name__}: {exc}"[:300]}
    return out


def _fp4_probe() -> dict:
    for mod in ("transformer_engine", "modelopt", "flashinfer"):
        try:
            __import__(mod)
            return {"status": "importable", "module": mod,
                    "note": "NVFP4 path present but not exercised here; bench via TRT-LLM/ModelOpt separately"}
        except Exception:  # noqa: BLE001
            continue
    return {"status": "unavailable", "note": "no NVFP4 runtime found; FP4 is an ecosystem question, not a hardware question"}


def run(profile: dict, resume: bool = True) -> None:
    cfg = profile["compute"]
    for dtype_name in cfg["dtypes"]:
        dt = TF[dtype_name]
        for n in cfg["sizes"]:
            key = f"{dtype_name}-{n}"
            if skip("a_compute", key, resume):
                continue
            a = torch.randn(n, n, device="cuda", dtype=dt)
            b = torch.randn(n, n, device="cuda", dtype=dt)
            with Sampler(f"a-{key}") as smp:
                r = bench(lambda: torch.mm(a, b), warmup=3, iters=5, torch=torch)
            tflops = _tflops(2 * n**3, r["median_ms"])
            emit("a_compute", key, {"dtype": dtype_name, "size": n, "tflops": tflops, **r, "smi": smp.summary()})
            print(f"[A] {dtype_name} {n:>5} -> {tflops:>7.1f} TFLOP/s  (median {r['median_ms']:.1f} ms)")
            del a, b
            torch.cuda.empty_cache()

    if not skip("a_compute", "fp8", resume):
        out = _fp8_probe(cfg.get("fp8_size", 4096))
        emit("a_compute", "fp8", out)
        print(f"[A] fp8 -> {out.get('status')} {out.get('tflops', '')}")

    if not skip("a_compute", "fp4", resume):
        out = _fp4_probe()
        emit("a_compute", "fp4", out)
        print(f"[A] nvfp4 -> {out.get('status')}")
