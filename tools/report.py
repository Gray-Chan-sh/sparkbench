"""Turn results.jsonl into the one-page verdict: achieved vs expected, and where the wall is."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"no results at {path}")
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rows


def pick(rows, test, key=None):
    hits = [r for r in rows if r["test"] == test and (key is None or r["key"].endswith(key))]
    return hits[-1] if hits else None


def ratio(actual, expected):
    if actual is None or not expected:
        return None
    return actual / expected


def bar(pct):
    if pct is None:
        return ""
    filled = int(round(pct * 20))
    return "#" * max(0, min(20, filled)) + "." * (20 - max(0, min(20, filled))) + f" {pct * 100:5.1f}%"


def _render(args) -> int:
    prof = json.loads((ROOT / "profiles" / f"{args.profile}.json").read_text(encoding="utf-8"))
    expect = prof.get("expect", {})
    path = Path(args.results) if args.results else ROOT / "results" / "results.jsonl"
    rows = load(path)
    if not rows:
        return 1

    env = pick(rows, "env", "start") or {}
    info = env.get("env", {})
    print(f"# {prof['name']} — {info.get('device', '?')} "
          f"({info.get('arch', '?')}, cc {info.get('cc', '?')}, torch {info.get('torch', '?')})\n")

    best = {}
    for r in rows:
        if r["test"] == "a_compute" and r.get("dtype") and "tflops" in r:
            best[r["dtype"]] = max(best.get(r["dtype"], 0), r["tflops"])
    for r in rows:
        if r["test"] == "a_compute" and r["key"].endswith(":fp8") and r.get("status") == "ok":
            best["fp8"] = r.get("tflops")

    print("## Compute")
    print("| dtype | achieved TFLOP/s | expected | of peak |")
    print("|---|---|---|---|")
    for dt in ("bf16", "fp16", "fp8"):
        actual = best.get(dt)
        exp = expect.get(f"{dt}_tflops")
        pct = ratio(actual, exp)
        print(f"| {dt} | {actual if actual is not None else 'n/a'} | {exp or 'n/a'} | {bar(pct)} |")

    bw = pick(rows, "b_bandwidth", ":copy")
    if bw:
        pct = ratio(bw.get("gbs"), expect.get("bandwidth_gbs"))
        print("\n## Memory")
        print(f"- copy bandwidth: **{bw['gbs']} GB/s** vs {expect.get('bandwidth_gbs')} spec — {bar(pct)}")
        ceil = pick(rows, "b_bandwidth", ":decode_ceiling")
        if ceil:
            print("- decode ceiling (tok/s): " + ", ".join(f"{k}={v}" for k, v in ceil["tok_s_ceiling"].items()))

    cap = pick(rows, "c_capacity", ":alloc")
    if cap:
        print("\n## Capacity")
        print(f"- declared VRAM: {cap['declared_vram_gib']} GiB")
        print(f"- allocated before stop: **{cap['allocated_gib']} GiB**")
        print(f"- swap/perf cliff: {cap['cliff_gib'] or 'none detected'} GiB")
        if cap.get("error"):
            print(f"- stopped by: `{cap['error'][:120]}`")

    dec = pick(rows, "d_video", ":decode")
    if dec:
        print(f"\n## Video decode\n- {dec.get('fps')} fps ({dec.get('decoder')}) on `{Path(dec.get('video', '')).name}`")

    vid = [r for r in rows if r["test"] == "d_video" and "frames" in r]
    if vid:
        print("\n## Long video")
        print("| model | frames | prompt tok | decode tok/s | total s | peak GiB |")
        print("|---|---|---|---|---|---|")
        for r in sorted(vid, key=lambda x: (x.get("label", ""), x["frames"])):
            label = r.get("label", r.get("model", "?"))
            if "error" in r:
                print(f"| {label} | {r['frames']} | error | {r['error'][:40]} | | |")
            else:
                print(f"| {label} | {r['frames']} | {r['prompt_tokens']} | {r['decode_tok_s']} | "
                      f"{r['total_s']} | {r['peak_gib']} |")

    sus = pick(rows, "e_sustained")
    if sus:
        smi = sus.get("smi", {})
        print("\n## Sustained")
        print(f"- {sus['seconds']:.0f}s soak: {sus['head_tflops']} -> {sus['tail_tflops']} TFLOP/s "
              f"(**decay {sus['decay_pct']}%**)")
        if smi:
            print(f"- clock {smi['start'].get('clock_mhz')} -> {smi['end'].get('clock_mhz')} MHz, "
                  f"power {smi['start'].get('power_w')} -> {smi['end'].get('power_w')} W, "
                  f"peak {smi['peak_temp_c']}°C")

    print("\n## Verdict")
    bits = []
    if best.get("bf16"):
        pct = ratio(best["bf16"], expect.get("bf16_tflops"))
        bits.append(f"BF16 达到标称 {pct * 100:.0f}%" if pct else "BF16 未知")
    if bw:
        bits.append(f"带宽 {bw['gbs']}/{expect.get('bandwidth_gbs')} GB/s")
    if vid:
        ok = [r for r in vid if "error" not in r]
        if ok:
            top = max(ok, key=lambda r: r["frames"])
            bits.append(f"{top['frames']} 帧 {top['total_s']}s / {top['peak_gib']} GiB")
    if sus and sus.get("decay_pct") is not None:
        bits.append(f"长跑衰减 {sus['decay_pct']}%")
    print("- " + "; ".join(bits))
    return 0


def _summary(rows: list[dict], expect: dict, info: dict) -> dict:
    def last(test, key=None):
        hits = [r for r in rows if r["test"] == test and (key is None or r["key"].endswith(key))]
        return hits[-1] if hits else None

    best = {}
    for r in rows:
        if r["test"] == "a_compute" and r.get("dtype") and "tflops" in r:
            best[r["dtype"]] = max(best.get(r["dtype"], 0), r["tflops"])

    bw = last("b_bandwidth", ":copy")
    cap = last("c_capacity", ":alloc")
    sus = last("e_sustained")
    bf16 = best.get("bf16")
    exp = expect.get("bf16_tflops")

    return {
        "env": info,
        "bf16_tflops": bf16,
        "bf16_pct": round(bf16 / exp * 100, 1) if bf16 and exp else None,
        "fp8_tflops": best.get("fp8"),
        "bandwidth_gbs": (bw or {}).get("gbs"),
        "vram_peak_gib": (cap or {}).get("allocated_gib"),
        "vram_usable_gib": (cap or {}).get("declared_vram_gib"),
        "swap_cliff_gib": (cap or {}).get("cliff_gib"),
        "soak_decay_pct": (sus or {}).get("decay_pct"),
        "counts": {t: sum(1 for r in rows if r["test"] == t) for t in
                   ("a_compute", "b_bandwidth", "c_capacity", "d_video", "e_sustained")},
    }


def _publish(args, text: str, rows: list[dict], expect: dict, info: dict) -> None:
    import csv

    out = Path(os.environ.get("SPARKBENCH_OUT", ROOT / "results"))
    pub = out / "publish"
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "report.md").write_text(text, encoding="utf-8")

    summary = _summary(rows, expect, info)
    (pub / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    keys = ["key", "test", "ts", "tflops", "gbs", "allocated_gib", "peak_gib",
            "seconds", "decay_pct", "frames", "status"]
    extra = ["dtype", "size", "label", "model", "median_ms", "p95_ms", "n"]
    with (pub / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys + extra + ["error"])
        for r in rows:
            w.writerow([r.get(k) for k in keys] + [r.get(k) for k in extra] + [r.get("error", "")])
    print(f"\nwritten -> {pub}")


def main(argv=None) -> int:
    import contextlib
    import io

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="gb10")
    ap.add_argument("--results", default=None)
    args = ap.parse_args(argv)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _render(args)

    text = buf.getvalue()
    print(text, end="")

    prof = json.loads((ROOT / "profiles" / f"{args.profile}.json").read_text(encoding="utf-8"))
    path = Path(args.results) if args.results else ROOT / "results" / "results.jsonl"
    rows = load(path)
    env = pick(rows, "env", "start") or {}
    _publish(args, text, rows, prof.get("expect", {}), env.get("env", {}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
