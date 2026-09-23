"""Snake を /v1/systemone に判断させて、どれだけスコアが取れるか測る。

同じ盤面を「AA（文字）」と「画像」で渡して比べられる。方策の参考値として random / greedy も選べる。

  python demo/snake/play.py --mode text  --games 10 --url http://127.0.0.1:18160
  python demo/snake/play.py --mode image --games 5
  python demo/snake/play.py --mode greedy --games 20          # サーバー不要
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snake import DIRS, Snake  # noqa: E402

BENCH = Path(__file__).resolve().parents[2]

CRITERIA = {
    "up": "move one cell up, towards the top wall",
    "down": "move one cell down, towards the bottom wall",
    "left": "move one cell left, towards the left wall",
    "right": "move one cell right, towards the right wall",
}
INSTRUCTIONS = ("The snake must eat the apple without hitting a wall or its own body. "
                "Which direction should the snake move next?")


def ask(session, url, game, mode, api_key=None, timeout=600, minimal=False, style="space",
        assistant_prefix=None):
    body = {
        "model": "jev-latest",
        "state": (game.image_state() if mode == "image"
                  else game.styled_state(style, minimal) if mode == "board"
                  else game.table_state(minimal) if mode == "table"
                  else game.text_state(minimal)),
        "questions": {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}},
    }
    if mode == "image":
        body["images"] = [game.data_url()]
    if assistant_prefix:
        body["options"] = {"assistant_prefix": assistant_prefix}
    headers = {"authorization": f"Bearer {api_key}"} if api_key else None
    r = session.post(f"{url}/v1/systemone", json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    a = r.json()["answers"]["move"]
    return a["choice"], a["probabilities"], a["confidence"]


def play(args, seed, session):
    g = Snake(args.w, args.h, seed=seed)
    frames = []
    latencies = []
    stall = 0  # りんごを食べずに歩いた歩数
    while g.alive and g.steps < args.max_steps and stall < args.stall_limit:
        before = g.score
        if args.mode in ("text", "table", "board", "image"):
            t0 = time.time()
            choice, probs, conf = ask(session, args.url, g, args.mode, args.api_key,
                                      minimal=getattr(args, "minimal", False),
                                      style=getattr(args, "board", "space"),
                                      assistant_prefix=getattr(args, "assistant_prefix", None))
            latencies.append((time.time() - t0) * 1000)
        elif args.mode == "greedy":
            choice, probs, conf = g.greedy_dir(), {}, 1.0
        else:  # random
            choice, probs, conf = g.rng.choice(DIRS), {}, 0.0
        frames.append({"board": g.ascii_board(), "body": [list(p) for p in g.body],
                       "apple": list(g.apple) if g.apple else None, "dir": g.dir,
                       "choice": choice, "probs": {k: round(v, 4) for k, v in probs.items()},
                       "confidence": round(conf, 4), "score": g.score, "safe": g.safe_dirs()})
        g.step(choice)
        stall = 0 if g.score > before else stall + 1
    cause = g.cause or ("step limit" if g.steps >= args.max_steps else "stalled")
    frames.append({"board": g.ascii_board(), "body": [list(p) for p in g.body],
                   "apple": list(g.apple) if g.apple else None, "dir": g.dir,
                   "choice": None, "probs": {}, "confidence": 0.0, "score": g.score, "safe": []})
    return {"seed": seed, "score": g.score, "steps": g.steps, "cause": cause,
            "reversals": g.reversals, "frames": frames,
            "mean_ms": sum(latencies) / len(latencies) if latencies else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="text",
                    choices=["text", "table", "board", "image", "greedy", "random"])
    ap.add_argument("--board", default="space",
                    choices=["ascii", "table", "table_open", "space", "comma", "tab", "json", "markdown"],
                    help="--mode board のときの盤面の書き方")
    ap.add_argument("--url", default="http://127.0.0.1:18160")
    ap.add_argument("--api-key")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--stall-limit", type=int, default=60, help="この歩数りんごを食べなければ打ち切り")
    ap.add_argument("--minimal", action="store_true", help="AA から座標などの補助を外し、画像版と同じ情報量にする")
    ap.add_argument("--assistant-prefix",
                    help="生成プロンプトの後ろに足す文字列（GLM 系なら \\n</think>\\n など）")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tag")
    ap.add_argument("--out-dir", default=str(BENCH / "results/snake"))
    a = ap.parse_args()

    session = requests.Session()
    games = []
    t0 = time.time()
    for i in range(a.games):
        g = play(a, a.seed0 + i, session)
        games.append(g)
        ms = f" {g['mean_ms']:.0f} ms/move" if g["mean_ms"] else ""
        print(f"game {i}: score={g['score']} steps={g['steps']} {g['cause']}{ms}", flush=True)
    scores = [g["score"] for g in games]
    steps = [g["steps"] for g in games]
    lat = [g["mean_ms"] for g in games if g["mean_ms"]]
    causes = {}
    for g in games:
        causes[g["cause"]] = causes.get(g["cause"], 0) + 1
    summary = {"mode": a.mode, "games": a.games, "board": f"{a.w}x{a.h}", "minimal": a.minimal,
               "style": a.board if a.mode == "board" else None,
               "assistant_prefix": a.assistant_prefix,
               "mean_score": sum(scores) / len(scores), "max_score": max(scores),
               "mean_steps": sum(steps) / len(steps), "causes": causes,
               "reversals": sum(g["reversals"] for g in games),
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
