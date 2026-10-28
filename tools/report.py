"""Turn results.jsonl into the one-page verdict: achieved vs expected, and where the wall is."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no results at {path}")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="gb10")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    prof = json.loads((ROOT / "profiles" / f"{args.profile}.json").read_text(encoding="utf-8"))
    expect = prof.get("expect", {})
    path = Path(args.results) if args.results else ROOT / "results" / "results.jsonl"
    rows = load(path)

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


if __name__ == "__main__":
    raise SystemExit(main())
