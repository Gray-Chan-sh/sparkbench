"""Pull results from the rented box without disturbing the run.

  * never touch a file that is still being written (mp4 is mtime-guarded)
  * keep each pull to a few MB, so a weight download keeps its bandwidth
  * tolerate a half-written last JSONL line
  * remote side is a read-only `tar`
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

SMALL = [
    "results/results.jsonl",
    "results/env.json",
    "results/manifest.json",
    "results/audit.json",
    "results/publish",
    "results/captures",
    "results/requirements.lock",
]
VIDEO_DIRS = ["results/captures"]
VIDEO_GLOBS = ["*.mp4"]


def _cd(remote_dir: str) -> str:
    """`cd '~/x'` would not expand the tilde."""
    if remote_dir.startswith("~"):
        rest = remote_dir[1:].lstrip("/")
        return f'cd "$HOME"/{shlex.quote(rest)}' if rest else 'cd "$HOME"'
    return f"cd {shlex.quote(remote_dir)}"


def _small_cmd(remote_dir: str) -> str:
    paths = " ".join(shlex.quote(p) for p in SMALL)
    return (f"{_cd(remote_dir)} && "
            f"tar czf - --ignore-failed-read --exclude='*.mp4' {paths} "
            f"$(ls results/smi-*.csv 2>/dev/null) 2>/dev/null")


def _video_cmd(remote_dir: str, min_age_min: int) -> str:
    dirs = " ".join(shlex.quote(d) for d in VIDEO_DIRS)
    expr = " -o ".join(f"-name '{g}'" for g in VIDEO_GLOBS)
    return (f"{_cd(remote_dir)} && "
            f"find {dirs} -type f \\( {expr} \\) -mmin +{min_age_min} -print0 "
            f"2>/dev/null | tar czf - --null --files-from=- 2>/dev/null")


def _ssh_base(args) -> list[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "ServerAliveInterval=15", "-p", str(args.port)]
    if args.key:
        cmd += ["-i", str(Path(args.key).expanduser())]
    cmd.append(f"{args.user}@{args.host}")
    return cmd


def _run_remote(args, remote_cmd: str, out_path: Path) -> bool:
    with out_path.open("wb") as fh:
        p = subprocess.run(_ssh_base(args) + [remote_cmd], stdout=fh, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print(f"  ssh failed ({p.returncode}): {p.stderr.decode(errors='ignore').strip()[:200]}")
        return False
    return out_path.stat().st_size > 0


def _extract(tgz: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            for m in tf:
                if m.name.startswith("/") or ".." in m.name.split("/"):
                    continue
                tf.extract(m, dest)
                n += 1
    except (tarfile.TarError, EOFError):
        pass
    return n


def _count(dest: Path) -> int:
    f = dest / "latest" / "results" / "results.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8", errors="ignore") as fh:
        return sum(1 for line in fh if line.strip().endswith("}"))


def pull_once(args) -> int:
    dest = Path(args.dest).expanduser()
    hist = dest / "history"
    hist.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tgz = hist / f"snap-{ts}.tgz"

    if not _run_remote(args, _small_cmd(args.remote_dir), tgz):
        return 0
    n = _extract(tgz, dest / "latest")
    print(f"[{datetime.now():%H:%M:%S}] {n} files, {tgz.stat().st_size / 1024:.0f} KB, "
          f"{_count(dest)} record(s)")

    if args.with_videos:
        vtgz = hist / f"videos-{ts}.tgz"
        if _run_remote(args, _video_cmd(args.remote_dir, args.video_age_min), vtgz):
            mb = vtgz.stat().st_size / 2**20
            m = _extract(vtgz, dest / "latest")
            print(f"           + {m} media file(s), {mb:.1f} MB")
        vtgz.unlink(missing_ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", default="root")
    ap.add_argument("--key", default=None)
    ap.add_argument("--remote-dir", default="~/sparkbench")
    ap.add_argument("--dest", default="sb-results")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--with-videos", action="store_true")
    ap.add_argument("--video-age-min", type=int, default=3)
    args = ap.parse_args()

    if args.once or args.interval <= 0:
        return pull_once(args)

    print(f"pulling every {args.interval}s -> {args.dest}  (ctrl-c to stop)")
    try:
        while True:
            pull_once(args)
            import time
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
