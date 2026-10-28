"""CLI: python -m sparkbench <smoke|a|b|c|d|e|all> --profile gb10

Bench modules import torch at module level, so they are loaded lazily: --help and
`smoke` must still work on a box where the CUDA stack is broken (that is exactly
when you need them).
"""
from __future__ import annotations

import argparse
import sys
import traceback

from .harness import JSONL, emit, env_info, load_profile

NAMES = {
    "a": "compute",
    "b": "bandwidth",
    "c": "capacity",
    "d": "video",
    "e": "sustained",
}


def _suite():
    from . import a_compute, b_bandwidth, c_capacity, d_video, e_sustained

    return {
        "a": a_compute.run,
        "b": b_bandwidth.run,
        "c": c_capacity.run,
        "d": d_video.run,
        "e": e_sustained.run,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sparkbench")
    ap.add_argument("what", choices=["smoke", "a", "b", "c", "d", "e", "all"])
    ap.add_argument("--profile", default="gb10")
    ap.add_argument("--model", default=None, help="only run this one from video.models (label or HF id)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)

    if args.what == "smoke":
        from .smoke import run as smoke
        return smoke(args.profile)

    suite = _suite()

    if not JSONL.exists():
        info = env_info()
        emit("env", "start", {"profile": args.profile, "env": info})
        print(f"[env] {info}")

    order = list(suite) if args.what == "all" else [args.what]
    failed = []
    for key in order:
        print(f"\n--- {NAMES[key]} ---")
        try:
            if key == "d":
                suite[key](prof, resume=not args.no_resume, only_model=args.model)
            else:
                suite[key](prof, resume=not args.no_resume)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{NAMES[key]}: {exc}")
            traceback.print_exc()
    print(f"\nresults -> {JSONL}")
    if failed:
        print("failed sections: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
