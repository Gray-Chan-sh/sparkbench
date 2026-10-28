"""Thin vast.ai client for this workflow: find GB10 offers, create, watch, destroy.

Needs VAST_API_KEY in the environment (console.vast.ai -> Account -> API Keys).
Nothing here spends money unless you run `create` explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://cloud.vast.ai/api/v0"


def _call(method: str, path: str, body: dict | None = None, timeout: int = 30) -> dict:
    key = os.environ.get("VAST_API_KEY")
    if not key:
        raise SystemExit("VAST_API_KEY is not set")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "sparkbench/0.1")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")[:400]
        raise SystemExit(f"HTTP {exc.code} on {method} {path}: {detail}") from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"network error: {exc.reason}") from None


def offers(gpu: str, max_price: float, min_reliability: float, min_down: int) -> list[dict]:
    q = {
        "gpu_name": {"eq": gpu},
        "rentable": {"eq": True},
        "dph_total": {"lte": max_price},
        "reliability2": {"gte": min_reliability},
        "inet_down": {"gte": min_down},
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": 64,
    }
    data = _call("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)))
    return data.get("offers") or []


def _fmt(o: dict) -> str:
    return (
        f"{o['id']:>10}  {o.get('gpu_name', '?'):<10} "
        f"{round(o.get('gpu_ram', 0) / 1024):>4}GB  "
        f"${o.get('dph_total', 0):>6.2f}/h  "
        f"{o.get('inet_down', 0):>7.0f}Mb↓  "
        f"disk {o.get('disk_space', 0):>6.0f}GB  "
        f"cu{o.get('cuda_max_good', 0):>4}  "
        f"rel {o.get('reliability2', 0):.2f}  "
        f"{o.get('cpu_arch', '?'):<6} {o.get('geolocation', '?')}"
    )


def cmd_offers(args) -> None:
    rows = offers(args.gpu, args.max_price, args.min_reliability, args.min_down)
    if not rows:
        print(f"no {args.gpu} offers under ${args.max_price}/h matching the filters")
        return
    print(f"{'offer_id':>10}  {'gpu':<10} {'vram':>6}  {'price':>9}  {'net↓':>10}  "
          f"{'disk':>13}  {'cuda':<6}  {'rel':<9}  arch / geo")
    for o in rows:
        print(_fmt(o))
    print(f"\n{len(rows)} offer(s). Create with: python tools/vast.py create {rows[0]['id']} --image <img>")


def cmd_watch(args) -> None:
    seen: set[int] = set()
    print(f"watching {args.gpu} every {args.interval}s, cap ${args.max_price}/h (ctrl-c to stop)")
    while True:
        try:
            rows = offers(args.gpu, args.max_price, args.min_reliability, args.min_down)
        except SystemExit as exc:
            print(f"[warn] {exc}")
            rows = []
        for o in rows:
            if o["id"] not in seen:
                seen.add(o["id"])
                print(f"[new] {_fmt(o)}")
        if args.create and rows:
            print(f"[hit] creating from offer {rows[0]['id']}")
            cmd_create(argparse.Namespace(**{**vars(args), "offer_id": rows[0]["id"]}))
            return
        time.sleep(args.interval)


def cmd_create(args) -> None:
    onstart = args.onstart
    if onstart and os.path.exists(onstart):
        onstart = open(onstart, encoding="utf-8").read()
    body = {
        "client_id": "me",
        "image": args.image,
        "disk": args.disk,
        "runtype": "ssh",
        "label": args.label,
        "onstart": onstart or "",
    }
    if args.price is not None:
        body["price"] = args.price
    res = _call("PUT", f"/asks/{args.offer_id}/", body)
    print(json.dumps(res, indent=2))
    print("\nnext: python tools/vast.py ssh --id <new_contract_id>")


def cmd_destroy(args) -> None:
    res = _call("DELETE", f"/instances/{args.id}/")
    print(json.dumps(res, indent=2))


def cmd_ssh(args) -> None:
    res = _call("GET", f"/instances/{args.id}/")
    inst = res.get("instances", res)
    host, port = inst.get("ssh_host"), inst.get("ssh_port")
    print(f"ssh -p {port} root@{host}   # or: vastai ssh-url {args.id}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="vast")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(gpu="GB10", max_price=1.5, min_reliability=0.95, min_down=300)

    p = sub.add_parser("offers", help="list matching offers")
    p.add_argument("--gpu", default=common["gpu"])
    p.add_argument("--max-price", type=float, default=common["max_price"])
    p.add_argument("--min-reliability", type=float, default=common["min_reliability"])
    p.add_argument("--min-down", type=int, default=common["min_down"])
    p.set_defaults(fn=cmd_offers)

    p = sub.add_parser("watch", help="poll and announce new offers")
    p.add_argument("--gpu", default=common["gpu"])
    p.add_argument("--max-price", type=float, default=common["max_price"])
    p.add_argument("--min-reliability", type=float, default=common["min_reliability"])
    p.add_argument("--min-down", type=int, default=common["min_down"])
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--create", action="store_true", help="auto-create from the cheapest hit")
    p.add_argument("--image", default="")
    p.add_argument("--disk", type=int, default=200)
    p.add_argument("--label", default="sparkbench")
    p.add_argument("--onstart", default="")
    p.add_argument("--price", type=float, default=None)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("create", help="create an instance from an offer id")
    p.add_argument("offer_id", type=int)
    p.add_argument("--image", required=True)
    p.add_argument("--disk", type=int, default=200)
    p.add_argument("--label", default="sparkbench")
    p.add_argument("--onstart", default="")
    p.add_argument("--price", type=float, default=None)
    p.set_defaults(fn=cmd_create)

    p = sub.add_parser("destroy", help="destroy an instance")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_destroy)

    p = sub.add_parser("ssh", help="print the ssh command")
    p.add_argument("--id", type=int, required=True)
    p.set_defaults(fn=cmd_ssh)

    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
