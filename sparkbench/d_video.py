"""D. Real workload: video decode throughput + long-video VLM latency curve.

D1 decode-only: always runs, tells you whether NVDEC/CPU decode is the real bottleneck.
D2 model sweep: one or more VLMs from the profile, each over a frame-count ladder.
   The point is the *shape* of the curve (frames -> prompt tokens -> TTFT/peak mem),
   not a single number, so the ladder is deliberately coarse.
"""
from __future__ import annotations

import subprocess
import time

import torch

from .harness import Sampler, emit, skip

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _decode_pyav(path: str, frames: int) -> dict:
    import av  # type: ignore

    container = av.open(path)
    stream = container.streams.video[0]
    t0 = time.perf_counter()
    n = 0
    for _frame in container.decode(stream):
        n += 1
        if n >= frames:
            break
    dt = time.perf_counter() - t0
    container.close()
    return {"decoder": "pyav", "frames": n, "seconds": round(dt, 3), "fps": round(n / dt, 2)}


def _decode_ffmpeg(path: str, frames: int) -> dict:
    t0 = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(frames), "-f", "null", "-"],
        check=True, capture_output=True,
    )
    dt = time.perf_counter() - t0
    return {"decoder": "ffmpeg", "frames": frames, "seconds": round(dt, 3), "fps": round(frames / dt, 2)}


def _decode(path: str, frames: int) -> dict:
    try:
        return _decode_pyav(path, frames)
    except Exception:  # noqa: BLE001
        try:
            return _decode_ffmpeg(path, frames)
        except Exception as exc:  # noqa: BLE001
            return {"decoder": "none", "error": f"{type(exc).__name__}: {exc}"[:200]}


def _frames(path: str, n: int, size: int = 448):
    """Uniform sample of n frames as PIL images."""
    import av  # type: ignore

    container = av.open(path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    out = []
    if total:
        want = {int(i * (total - 1) / max(1, n - 1)) for i in range(n)}
        for idx, frame in enumerate(container.decode(stream)):
            if idx in want:
                out.append(frame.to_image().resize((size, size)))
                if len(out) >= n:
                    break
    else:
        for frame in container.decode(stream):
            out.append(frame.to_image().resize((size, size)))
            if len(out) >= n:
                break
    container.close()
    return out


def _model_bench(model_id: str, images, question: str, max_new_tokens: int, dtype: str) -> dict:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = DTYPES.get(dtype, torch.bfloat16)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch_dtype, device_map="cuda"
        ).eval()
    except TypeError:  # older transformers only knows torch_dtype
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="cuda"
        ).eval()
    load_s = time.perf_counter() - t0

    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    total = time.perf_counter() - t1

    new_tokens = int(out.shape[-1] - inputs["input_ids"].shape[-1])
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    del model, inputs, out
    torch.cuda.empty_cache()
    return {
        "model": model_id,
        "dtype": dtype,
        "load_s": round(load_s, 1),
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "total_s": round(total, 2),
        "decode_tok_s": round(new_tokens / total, 2) if total else None,
        "peak_gib": peak,
    }


def _model_list(cfg: dict) -> list[dict]:
    if cfg.get("models"):
        return cfg["models"]
    if cfg.get("model"):
        return [{"label": "default", "id": cfg["model"],
                 "dtype": cfg.get("dtype", "bfloat16"), "frames": cfg.get("frames", [16, 64])}]
    return []


def run(profile: dict, resume: bool = True, only_model: str | None = None) -> None:
    cfg = profile.get("video", {})
    video = cfg.get("video")

    if video and not skip("d_video", "decode", resume):
        with Sampler("d-decode") as smp:
            r = _decode(video, int(cfg.get("decode_frames", 300)))
        emit("d_video", "decode", {**r, "video": video, "smi": smp.summary()})
        print(f"[D1] decode {r.get('fps')} fps via {r.get('decoder')}")

    models = _model_list(cfg)
    if not video:
        print("[D2] skipped (set video.video in the profile)")
        return
    if not models:
        print("[D2] skipped (no models configured)")
        return

    question = cfg.get("question", "Describe what happens in this video, in order.")
    max_new = int(cfg.get("max_new_tokens", 128))
    size = int(cfg.get("frame_size", 448))

    for spec in models:
        label = spec.get("label") or spec["id"].split("/")[-1]
        if only_model and only_model not in (label, spec["id"]):
            continue
        for n in spec.get("frames", [16, 64]):
            key = f"{label}-{n}f"
            if skip("d_video", key, resume):
                continue
            images = _frames(video, n, size)
            with Sampler(f"d-{label}-{n}") as smp:
                try:
                    r = _model_bench(spec["id"], images, question, max_new,
                                     spec.get("dtype", "bfloat16"))
                except Exception as exc:  # noqa: BLE001
                    r = {"model": spec["id"], "error": f"{type(exc).__name__}: {exc}"[:400]}
            r.update({"label": label, "frames": n, "smi": smp.summary()})
            emit("d_video", key, r)
            if "error" in r:
                print(f"[D2] {label:<8} {n:>3}f -> ERROR {r['error'][:110]}")
            else:
                print(f"[D2] {label:<8} {n:>3}f -> {r['prompt_tokens']:>6} tok prompt, "
                      f"{r['decode_tok_s']:>6} tok/s, {r['total_s']:>6}s, peak {r['peak_gib']} GiB, "
                      f"load {r['load_s']}s")
            del images
