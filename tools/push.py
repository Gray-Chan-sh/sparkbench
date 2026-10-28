"""Publish results: push the small stuff to a git repo and/or a webhook.

Only report.md / summary.json / metrics.csv / results.jsonl travel. Raw captures and
large files stay on the box; pushing gigabytes from a metered instance is the
easiest way to waste money.

  CGBENCH_RESULTS_REMOTE   https://github.com/you/sparkbench-results.git
  CGBENCH_GIT_TOKEN        fine-grained PAT (contents:write on that repo)
  CGBENCH_WEBHOOK          feishu/wecom/slack incoming webhook
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SPARKBENCH_OUT", ROOT / "results"))
PUBLISH = OUT / "publish"


def summary() -> str:
    f = PUBLISH / "summary.json"
    if not f.exists():
        return "sparkbench: no results yet"
    d = json.loads(f.read_text(encoding="utf-8"))
    env = d.get("env", {})
    return "\n".join([
        f"DGX Spark 硬件画像 — {env.get('device', '?')} (cc {env.get('cc')})",
        f"BF16: {d.get('bf16_tflops')} TFLOP/s ({d.get('bf16_pct')}% of 125)",
        f"带宽: {d.get('bandwidth_gbs')} GB/s of 273",
        f"显存: 峰值 {d.get('vram_peak_gib')} GiB，可用 {d.get('vram_usable_gib')} GiB",
        f"长跑: {d.get('soak_decay_pct')}% 衰减",
    ])


def _git(*args, cwd=PUBLISH) -> bool:
    p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        print(f"git {' '.join(args)} -> {p.stderr[:300]}")
    return p.returncode == 0


def push_git() -> bool:
    remote = os.environ.get("CGBENCH_RESULTS_REMOTE")
    if not remote:
        print("CGBENCH_RESULTS_REMOTE not set; skipping git push")
        return False
    for name in ("results.jsonl", "manifest.json", "audit.json"):
        src = OUT / name
        if src.exists():
            shutil.copy2(src, PUBLISH / name)
    if not (PUBLISH / ".git").exists():
        _git("init", "-b", "main")
        _git("remote", "add", "origin", remote)
    _git("add", "-A")
    _git("-c", "user.name=sparkbench", "-c", "user.email=sparkbench@local",
         "commit", "-m", f"results: {summary().splitlines()[0]}")

    args = ["push", "origin", "HEAD:main"]
    token = os.environ.get("CGBENCH_GIT_TOKEN")
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        args = ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}", *args]
    ok = _git(*args)
    print("git push:", "ok" if ok else "failed")
    return ok


def push_webhook() -> bool:
    url = os.environ.get("CGBENCH_WEBHOOK")
    if not url:
        print("CGBENCH_WEBHOOK not set; skipping webhook")
        return False
    text = summary()
    fmt = os.environ.get("CGBENCH_WEBHOOK_FORMAT", "feishu")
    body = ({"msg_type": "text", "content": {"text": text}} if fmt == "feishu"
            else {"msgtype": "text", "text": {"content": text}} if fmt == "wecom"
            else {"text": text})
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            print("webhook:", resp.status)
        return True
    except Exception as exc:  # noqa: BLE001
        print("webhook failed:", exc)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", action="store_true")
    ap.add_argument("--webhook", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary or not (a.git or a.webhook):
        print(summary())
        return 0
    ok = True
    if a.git:
        ok &= push_git()
    if a.webhook:
        ok &= push_webhook()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
