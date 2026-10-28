"""Capture material for the podcast: terminal casts, screenshots, a shot list.

A rented box is headless, so "screen recording" means a pty cast (always works) or
X11 capture (only with a desktop session). Anything unavailable is reported as
unavailable; a missing screenshot is a fact, a fabricated one is a lie in the edit.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .harness import OUT

CAP = OUT / "captures"
CAP.mkdir(parents=True, exist_ok=True)
COLS, ROWS = 120, 36


def detect() -> dict:
    return {
        "display": os.environ.get("DISPLAY"),
        "pty": _has("pty"),
        "ffmpeg": shutil.which("ffmpeg"),
        "screenshot": next((shutil.which(t) for t in
                            ("gnome-screenshot", "scrot", "import", "grim")
                            if shutil.which(t)), None),
        "x11grab": bool(os.environ.get("DISPLAY")) and bool(shutil.which("ffmpeg")),
    }


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


class TerminalRecorder:
    """asciinema v2 cast recorded through a pty.

    A pty (not a pipe) is the point: the child sees a real terminal, so colours,
    progress bars and isatty() behave exactly as they would in a normal session.
    """

    def __init__(self, name: str, cols: int = COLS, rows: int = ROWS):
        self.path = CAP / f"{name}.cast"
        self.cols, self.rows = cols, rows

    def run(self, argv: list[str]) -> int:
        import pty
        import select

        try:
            import termios
            import tty
        except ImportError:
            termios = tty = None

        fh = self.path.open("w", encoding="utf-8")
        fh.write(json.dumps({
            "version": 2, "width": self.cols, "height": self.rows,
            "timestamp": int(time.time()),
            "env": {"SHELL": os.environ.get("SHELL", "/bin/bash"),
                    "TERM": os.environ.get("TERM", "xterm-256color")},
        }) + "\n")
        fh.flush()

        pid, fd = pty.fork()
        if pid == 0:
            os.execvp(argv[0], argv)

        saved = None
        stdin_fd = sys.stdin.fileno()
        watch_in = sys.stdin.isatty() and termios is not None
        if watch_in:
            try:
                saved = termios.tcgetattr(stdin_fd)
                tty.setraw(stdin_fd)
            except Exception:  # noqa: BLE001
                watch_in = False

        start = time.time()
        cmdline = "$ " + " ".join(argv) + "\n"
        fh.write(json.dumps([0.0, "o", cmdline]) + "\n")
        fh.flush()
        sys.stdout.write(cmdline)
        sys.stdout.flush()

        try:
            while True:
                watch = [fd] + ([stdin_fd] if watch_in else [])
                try:
                    r, _, _ = select.select(watch, [], [], 0.2)
                except OSError:
                    break
                if fd in r:
                    try:
                        data = os.read(fd, 8192)
                    except OSError:
                        break
                    if not data:
                        break
                    txt = data.decode("utf-8", errors="replace")
                    fh.write(json.dumps([round(time.time() - start, 4), "o", txt]) + "\n")
                    fh.flush()
                    sys.stdout.write(txt)
                    sys.stdout.flush()
                if watch_in and stdin_fd in r:
                    try:
                        keys = os.read(stdin_fd, 1024)
                    except OSError:
                        keys = b""
                    if keys:
                        os.write(fd, keys)
                    else:
                        watch_in = False
        finally:
            if saved is not None:
                try:
                    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, saved)
                except Exception:  # noqa: BLE001
                    pass
            fh.close()
            try:
                _, status = os.waitpid(pid, 0)
                rc = os.waitstatus_to_exitcode(status)
            except Exception:  # noqa: BLE001
                rc = 0
        return rc


def record(name: str, argv: list[str]) -> int:
    if not _has("pty"):
        print(f"[capture] pty unavailable; running without recording: {' '.join(argv)}")
        return subprocess.call(argv)
    rec = TerminalRecorder(name)
    print(f"[capture] recording -> {rec.path}")
    rc = rec.run(argv)
    print(f"\n[capture] cast saved: {rec.path}")
    return rc


def screenshot(name: str) -> str | None:
    caps = detect()
    if not caps["display"]:
        print("[capture] no DISPLAY; screenshot skipped (headless session)")
        return None
    out = CAP / f"{name}.png"
    tool = caps["screenshot"]
    if tool == "gnome-screenshot":
        subprocess.run(["gnome-screenshot", "-f", str(out)], check=False)
    elif tool == "scrot":
        subprocess.run(["scrot", str(out)], check=False)
    elif tool == "grim":
        subprocess.run(["grim", str(out)], check=False)
    elif tool == "import":
        subprocess.run(["import", "-window", "root", str(out)], check=False)
    else:
        print("[capture] no screenshot tool found")
        return None
    return str(out) if out.exists() else None


def screen_record(seconds: int, name: str = "desktop") -> str | None:
    caps = detect()
    if not caps["x11grab"]:
        print("[capture] no DISPLAY or no ffmpeg; screen recording unavailable")
        return None
    out = CAP / f"{name}.mp4"
    cmd = ["ffmpeg", "-y", "-f", "x11grab", "-framerate", "15", "-i", os.environ["DISPLAY"],
           "-t", str(seconds), "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=False, capture_output=True)
    return str(out) if out.exists() else None


def shotlist() -> str:
    caps = detect()
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 播客素材清单（sparkbench）",
        "",
        f"生成于 {ts}",
        f"- DISPLAY: `{caps['display'] or '无（纯 SSH）'}`",
        f"- 终端录制: {'可用' if caps['pty'] else '不可用'}   "
        f"截图: {caps['screenshot'] or '不可用'}   "
        f"桌面录屏: {'可用' if caps['x11grab'] else '不可用'}",
        "",
        "终端镜头：`python -m sparkbench.capture run <名字> -- <命令>`",
        "",
    ]
    shots = [
        ("01-boot", "nvidia-smi", 20, "128GB 统一内存第一次出现",
         "开场放数字。观众对 128G 没直觉，对'桌上这台机器'有。"),
        ("02-smoke", "python -m sparkbench smoke --profile gb10", 60,
         "cgroup / 统一内存那两行", "这是全片最省钱的 10 秒：一台被限到 64G 的容器。"),
        ("03-compute", "python -m sparkbench a --profile gb10", 120,
         "BF16 实测 vs 标称 125 TFLOPS", "差的那 30% 才是真话。"),
        ("04-bandwidth", "python -m sparkbench b --profile gb10", 90,
         "231/273 GB/s，以及反推的 decode 上限", "带宽决定这台机器能服务多大的模型。"),
        ("05-capacity", "python -m sparkbench c --profile gb10", 150,
         "分配到某个 GiB 突然掉速（swap 悬崖）", "统一内存不会 OOM，会悄悄变慢。"),
        ("06-video", "python -m sparkbench d --profile gb10", 300,
         "帧数翻倍、TTFT 超线性增长", "这才是'长视频'的真实成本。"),
        ("07-soak", "python -m sparkbench e --profile gb10", 1800,
         "首窗 vs 末窗的 TFLOP/s", "140W 的机器跑半小时是什么样。"),
        ("08-report", "python tools/report.py --profile gb10", 60,
         "达成率条形图和 verdict", "收尾给可比较的数字，别给形容词。"),
    ]
    for name, cmd, secs, look, tip in shots:
        lines += [f"## {name}", f"- 命令：`{cmd}`", f"- 建议时长：{secs}s",
                  f"- 看点：{look}", f"- 解说：{tip}", ""]
    lines += [
        "## 素材落点",
        "- `results/captures/<名字>.cast` → asciinema player / `agg` 转 GIF",
        "- `results/captures/*.png`（有 DISPLAY 时）",
        "- 数据：`results/publish/report.md`",
        "",
        "## 没有 GUI 怎么办",
        "纯 SSH 下截图/桌面录屏不可用，这是事实。终端 cast 加报告表格已经够叙事；",
        "需要真桌面就在控制台开 VNC 再跑 `screen_record()`。",
    ]
    return "\n".join(lines)


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sparkbench.capture")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p = sub.add_parser("shot")
    p.add_argument("name", nargs="?", default="shot")
    p = sub.add_parser("record")
    p.add_argument("seconds", type=int, nargs="?", default=30)
    p.add_argument("--name", default="desktop")
    sub.add_parser("detect")
    sub.add_parser("shots")

    a = ap.parse_args()
    if a.cmd == "run":
        argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
        return record(a.name, argv)
    if a.cmd == "shot":
        print(screenshot(a.name) or "no screenshot")
        return 0
    if a.cmd == "record":
        print(screen_record(a.seconds, a.name) or "no recording")
        return 0
    if a.cmd == "detect":
        print(json.dumps(detect(), indent=2))
        return 0
    print(shotlist())
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
