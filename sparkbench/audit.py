"""Can this results directory be used to reconstruct the run?

Same three-way verdict as the canguoai project, so results from the two repos can be
read with the same standard:

  reproducible - code, deps, weights, manifest all check out
  replay only  - what was run and what came out, but reruns may differ
  cannot replay- not even that
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .harness import OUT

TESTS = ("a_compute", "b_bandwidth", "c_capacity", "d_video", "e_sustained")


def _rows(out: Path) -> list[dict]:
    f = out / "results.jsonl"
    if not f.exists():
        return []
    rs = []
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rs.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rs


def _c(cond: bool, name: str, detail: str = "") -> dict:
    return {"status": "PASS" if cond else "WARN", "name": name, "detail": detail}


def _verify(out: Path, doc: dict) -> tuple[int, int, list[str]]:
    good = bad = 0
    broken: list[str] = []
    for f in doc.get("files", []):
        if not f.get("sha256"):
            continue
        p = out / f["path"]
        if not p.exists():
            broken.append(f["path"])
            bad += 1
        elif hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]:
            good += 1
        else:
            broken.append(f["path"])
            bad += 1
    return good, bad, broken[:5]


def collect(out: Path, verify: bool = True) -> tuple[list[dict], dict]:
    rs = _rows(out)
    env = next((r.get("env", {}) for r in rs if r.get("test") == "env"), {})
    runs = [r for r in rs if r.get("test") in TESTS]
    errs = [r for r in runs if r.get("status") == "error"]
    oks = [r for r in runs if r.get("status") != "error"]
    git = (env.get("git") or {}) if env else {}

    checks = [
        _c(bool(runs), "results.jsonl has runs", f"{len(runs)} record(s)"),
        _c(bool(env), "env snapshot", "env:start"),
        _c(bool(git.get("commit")), "code commit pinned", git.get("commit") or "no git metadata"),
        _c(git.get("dirty") is False, "working tree clean", "dirty" if git.get("dirty") else "clean/unknown"),
        _c((out / "requirements.lock").exists(), "dependency lock", "requirements.lock"),
    ]
    for key in ("device", "cc", "vram_gib", "torch", "torch_cuda"):
        checks.append(_c(bool(env.get(key)), f"env.{key}", str(env.get(key))))

    mem = env.get("memory") or {}
    checks.append(_c(mem.get("unified") is not None, "memory model detected",
                     f"unified={mem.get('unified')} cap={mem.get('cgroup_cap_gib')}"))
    if mem.get("unified") and mem.get("cgroup_cap_gib") and env.get("vram_gib"):
        ok = mem["cgroup_cap_gib"] >= env["vram_gib"] * 0.9
        checks.append(_c(ok, "container cap covers the pool",
                         f"cap {mem['cgroup_cap_gib']} vs pool {env['vram_gib']} GiB"))

    present = {r["test"] for r in oks}
    missing = [t for t in TESTS if t not in present]
    checks.append(_c(len(missing) <= 2, "test coverage", f"{len(present)}/5" +
                     (f", missing {', '.join(missing)}" if missing else "")))

    no_reason = [r["key"] for r in errs if not r.get("error")]
    no_trace = [r["key"] for r in errs
                if r.get("traceback_file") and not (out / r["traceback_file"]).exists()]
    checks.append(_c(not no_reason, "failures carry a reason", f"{len(errs)} failure(s)"))
    checks.append(_c(not no_trace, "failures carry a full traceback", f"{len(no_trace)} missing"))

    pub = out / "publish"
    for f in ("report.md", "summary.json"):
        checks.append(_c((pub / f).exists(), f"publish/{f}", ""))

    casts = list((out / "captures").glob("*.cast")) if (out / "captures").exists() else []
    checks.append(_c(bool(casts), "podcast captures", f"{len(casts)} cast(s)"))

    mdoc = {}
    mp = out / "manifest.json"
    checks.append(_c(mp.exists(), "hash manifest", "manifest.json"))
    if mp.exists():
        try:
            mdoc = json.loads(mp.read_text(encoding="utf-8"))
            if verify:
                good, bad, broken = _verify(out, mdoc)
                checks.append(_c(bad == 0, "manifest hashes verify",
                                 f"{good} ok, {bad} bad" + (f" ({', '.join(broken)})" if broken else "")))
        except Exception as exc:  # noqa: BLE001
            checks.append(_c(False, "manifest readable", str(exc)[:80]))

    return checks, {"env": env, "manifest": {k: v for k, v in mdoc.items() if k != "files"}}


def verdict(checks: list[dict]) -> tuple[str, str]:
    warn = {c["name"] for c in checks if c["status"] != "PASS"}
    if {"results.jsonl has runs", "env snapshot"} & warn:
        return "cannot replay", "缺少基础记录，连过程都还原不了"
    pins = {"code commit pinned", "working tree clean", "dependency lock", "hash manifest"}
    if pins & warn:
        return "replay only", "能还原“跑了什么、结果如何”，但不保证重跑得到同样的数"
    return "reproducible", "代码、依赖、清单都可核对，同机可复现"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sparkbench audit")
    ap.add_argument("--results", default=None, help="audit a pulled snapshot")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.results).expanduser().resolve() if args.results else OUT
    checks, meta = collect(out, verify=not args.no_verify)
    kind, why = verdict(checks)
    doc = {"results": str(out), "verdict": kind, "reason": why, "checks": checks, **meta}

    try:
        (out / "audit.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        width = max(len(c["name"]) for c in checks) + 2
        print(f"\n{out}\n" + "=" * 72)
        for c in checks:
            mark = {"PASS": " ok ", "WARN": "warn"}.get(c["status"], c["status"])
            print(f"[{mark}] {c['name'].ljust(width)} {c['detail']}")
        print("=" * 72)
        print(f"verdict: {kind} — {why}\n")
    return 0 if kind != "cannot replay" else 1


if __name__ == "__main__":
    raise SystemExit(main())
