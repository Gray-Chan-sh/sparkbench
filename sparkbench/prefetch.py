"""Warm the D-group model cache before the clock starts.

The long-video phase pulls Qwen3-VL checkpoints; on a metered instance every minute
spent downloading is a minute you paid for and did not measure.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .harness import OUT, load_profile


def models(profile_name: str) -> list[dict]:
    prof = load_profile(profile_name)
    return prof.get("video", {}).get("models") or []


def run(profile_name: str = "gb10", dry_run: bool = False) -> int:
    items = models(profile_name)
    if not items:
        print("no video.models in this profile; nothing to prefetch")
        return 0

    print(f"{'model':<52}{'est GiB':>9}")
    print("-" * 62)
    for m in items:
        print(f"{m['id'][:51]:<52}{'?':>9}")
    print("\n(parameter counts are not known ahead of download)")

    if dry_run:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub missing: pip install huggingface_hub", file=sys.stderr)
        return 1

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"\nendpoint: {endpoint}")
    revisions: dict[str, str | None] = {}
    failed = []
    for m in items:
        print(f"\n>> {m['id']}", flush=True)
        try:
            path = snapshot_download(repo_id=m["id"], endpoint=endpoint,
                                     max_workers=4, resume_download=True)
            revisions[m["id"]] = Path(path).name if "snapshots" in Path(path).parts else None
            print(f"   cached -> {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(m["id"])

    if revisions:
        p = OUT / "revisions.json"
        p.write_text(json.dumps(revisions, indent=2), encoding="utf-8")
        print(f"\nrevisions pinned -> {p}")

    if failed:
        print(f"\n{len(failed)} failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0
