"""迷路を /v1/systemone に一手ずつ解かせて、ゴールできるか測る。

  python demo/maze/play.py --mode text  --games 10 --url http://mac.local:8083 --tag text_foo
  python demo/maze/play.py --mode image --games 10 --url ... --tag image_foo     # 要 --mmproj
  python demo/maze/play.py --mode optimal --games 10                             # 参考値、サーバー不要
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze import DIRS, Maze  # noqa: E402

BENCH = Path(__file__).resolve().parents[2]

CRITERIA = {
    "up": "move one cell up, towards the top of the maze",
    "down": "move one cell down, towards the bottom of the maze",
    "left": "move one cell left",
    "right": "move one cell right",
}
INSTRUCTIONS = ("You are walking through a maze towards the goal. "
                "Which direction should you move next? Do not walk into walls.")


def ask(session, url, mz, mode, api_key=None, timeout=600, spaced=True, assistant_prefix=None,
        legal_only=False, labels=False, coords=False):
    crit = CRITERIA
    if legal_only:
        legal = mz.legal_dirs()
        crit = {d: CRITERIA[d] for d in legal} if len(legal) >= 2 else CRITERIA
    body = {
        "model": "jev-latest",
        "state": mz.image_state() if mode == "image" else mz.text_state(spaced, labels, coords),
        "questions": {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": crit}},
    }
    if mode == "image":
        body["images"] = [mz.data_url()]
    if assistant_prefix:
        body["options"] = {"assistant_prefix": assistant_prefix}
    headers = {"authorization": f"Bearer {api_key}"} if api_key else None
    r = session.post(f"{url}/v1/systemone", json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    a = r.json()["answers"]["move"]
    return a["choice"], a["probabilities"], a["confidence"]


def play(args, seed, session):
    mz = Maze(args.cells, args.cells, seed=seed, braid=args.braid)
    frames = []
    latencies = []
    limit = args.max_steps or mz.shortest * 6
    bump_streak = 0
    while not mz.done and mz.steps < limit and bump_streak < args.bump_limit:
        before = mz.remaining()
        forced = None
        if args.legal_only:
            legal = mz.legal_dirs()
            if len(legal) == 1:
                forced = legal[0]
        if forced:
            choice, probs, conf = forced, {}, 1.0
        elif args.mode in ("text", "image"):
            t0 = time.time()
            choice, probs, conf = ask(session, args.url, mz, args.mode, args.api_key,
                                      spaced=not args.dense,
                                      assistant_prefix=getattr(args, "assistant_prefix", None),
                                      legal_only=args.legal_only,
                                      labels=args.labels, coords=args.coords)
            latencies.append((time.time() - t0) * 1000)
        elif args.mode == "optimal":
            choice, probs, conf = mz.optimal_dir(), {}, 1.0
        else:  # random
            choice, probs, conf = mz.rng.choice(DIRS), {}, 0.0
        frames.append({"pos": list(mz.pos), "choice": choice,
                       "probs": {k: round(v, 4) for k, v in probs.items()},
                       "confidence": round(conf, 4), "legal": mz.legal_dirs(),
                       "forced": bool(forced), "remaining": before, "steps": mz.steps})
        moved = mz.step(choice)
        bump_streak = 0 if moved else bump_streak + 1
        frames[-1]["moved"] = moved
        frames[-1]["closer"] = moved and (mz.remaining() or 0) < (before or 0)
    cause = mz.cause or ("stuck" if bump_streak >= args.bump_limit else "step limit")
    frames.append({"pos": list(mz.pos), "choice": None, "probs": {}, "confidence": 0.0,
                   "legal": [], "remaining": mz.remaining(), "steps": mz.steps, "moved": False})
    return {"seed": seed, "solved": mz.done, "steps": mz.steps, "bumps": mz.bumps,
            "shortest": mz.shortest, "cause": cause, "frames": frames,
            "grid": ["".join(r) for r in mz.grid],
            "start": list(mz.start), "goal": list(mz.goal),
            "mean_ms": sum(latencies) / len(latencies) if latencies else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="text", choices=["text", "image", "optimal", "random"])
    ap.add_argument("--url", default="http://127.0.0.1:18160")
    ap.add_argument("--api-key")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--cells", type=int, default=5, help="セル数（盤面は 2n+1 四方）")
    ap.add_argument("--braid", type=float, default=0.0, help="行き止まりを開けてループを作る割合")
    ap.add_argument("--max-steps", type=int, default=0, help="0 なら最短手数の6倍")
    ap.add_argument("--dense", action="store_true", help="AA をスペース区切りにしない")
    ap.add_argument("--legal-only", action="store_true",
                    help="進める方向だけを選択肢にする（壁の判定はこちらで済ませる）")
    ap.add_argument("--labels", action="store_true", help="行番号・列番号を振り、軸の向きを明示する")
    ap.add_argument("--coords", action="store_true", help="現在地とゴールの座標も渡す")
    ap.add_argument("--bump-limit", type=int, default=8,
                    help="壁に連続でぶつかったら打ち切る回数")
    ap.add_argument("--assistant-prefix")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tag")
    ap.add_argument("--out-dir", default=str(BENCH / "results/maze"))
    a = ap.parse_args()

    session = requests.Session()
    games = []
    t0 = time.time()
    for i in range(a.games):
        g = play(a, a.seed0 + i, session)
        games.append(g)
        ms = f" {g['mean_ms']:.0f} ms/move" if g["mean_ms"] else ""
        print(f"game {i}: {'solved' if g['solved'] else 'failed'} steps={g['steps']}"
              f" (shortest {g['shortest']}) bumps={g['bumps']}{ms}", flush=True)

    solved = [g for g in games if g["solved"]]
    ratio = [g["steps"] / g["shortest"] for g in solved if g["shortest"]]
    lat = [g["mean_ms"] for g in games if g["mean_ms"]]
    # 手ごとの内訳
    tot = closer = legal_ok = 0
    for g in games:
        for f in g["frames"]:
            if not f["choice"] or f.get("forced"):
                continue
            tot += 1
            legal_ok += f["choice"] in f["legal"]
            closer += bool(f.get("closer"))
    causes = {}
    for g in games:
        causes[g["cause"]] = causes.get(g["cause"], 0) + 1
    summary = {"mode": a.mode, "games": a.games, "cells": a.cells, "braid": a.braid,
               "spaced": not a.dense, "assistant_prefix": a.assistant_prefix,
               "legal_only": a.legal_only, "labels": a.labels, "coords": a.coords, "causes": causes,
               "solve_rate": len(solved) / len(games),
               "mean_steps_solved": sum(g["steps"] for g in solved) / len(solved) if solved else 0,
               "mean_ratio": sum(ratio) / len(ratio) if ratio else 0,
               "mean_shortest": sum(g["shortest"] for g in games) / len(games),
               "legal_rate": legal_ok / tot if tot else 0,
               "closer_rate": closer / tot if tot else 0,
               "bumps_per_game": sum(g["bumps"] for g in games) / len(games),
               "mean_ms_per_move": sum(lat) / len(lat) if lat else 0.0,
               "total_seconds": time.time() - t0}
    print(json.dumps(summary, ensure_ascii=False))

    tag = a.tag or a.mode
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps({"summary": summary, "games": games},
                                                ensure_ascii=False), encoding="utf-8")
    print("wrote", out / f"{tag}.json")


if __name__ == "__main__":
    main()
