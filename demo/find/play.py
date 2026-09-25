"""「虫のいるマスはどれ？」を /v1/systemone に聞いて、当たる割合を測る。

8×6 の 48 マスを選択肢にして、1回の推論で 48 個の確率を受け取る。虫が1匹なら
当てずっぽうは 1/48 = 2.1%、3匹なら 6.25%。

  python demo/find/play.py --mode text          --trials 20 --url http://127.0.0.1:18160
  python demo/find/play.py --mode image_labeled --trials 20 --url ...   # 要 --mmproj
  python demo/find/play.py --mode image         --trials 20 --url ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import Scene, all_cells  # noqa: E402

BENCH = Path(__file__).resolve().parents[2]
INSTRUCTIONS = "Which cell contains a bug?"
# 画像に何を書き込むか。axis は盤の外に列の文字と行の数字を振る（表計算の見出しと同じ）。
IMAGE_LABELS = {"image_labeled": "cells", "image_axis": "axis", "image": "none"}



def ask(session, url, scene, mode, api_key=None, timeout=600, assistant_prefix=None):
    labels = IMAGE_LABELS.get(mode, "none")
    body = {
        "model": "jev-latest",
        "state": scene.text_state() if mode == "text" else scene.image_state(labels),
        "questions": {"cell": {"type": "choice", "instructions": INSTRUCTIONS,
                               "criteria": {c: None for c in all_cells()}}},
    }
    if mode != "text":
        body["images"] = [scene.data_url(labels=labels)]
    if assistant_prefix:
        body["options"] = {"assistant_prefix": assistant_prefix}
    headers = {"authorization": f"Bearer {api_key}"} if api_key else None
    r = session.post(f"{url}/v1/systemone", json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    a = r.json()["answers"]["cell"]
    return a["choice"], a["probabilities"], a["confidence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="text", choices=["text", "image_labeled", "image_axis", "image", "random"])
    ap.add_argument("--url", default="http://127.0.0.1:18160")
    ap.add_argument("--api-key")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--bugs", type=int, default=3, help="虫の数（当てずっぽうは bugs/48）")
    ap.add_argument("--assistant-prefix")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tag")
    ap.add_argument("--out-dir", default=str(BENCH / "results/find"))
    a = ap.parse_args()

    session = requests.Session()
    rows, lat = [], []
    for i in range(a.trials):
        scene = Scene(seed=a.seed0 + i, bugs=a.bugs)
        if a.mode == "random":
            choice, probs, conf = scene.rng.choice(all_cells()), {}, 0.0
        else:
            t0 = time.time()
            choice, probs, conf = ask(session, a.url, scene, a.mode, a.api_key,
                                      assistant_prefix=a.assistant_prefix)
            lat.append((time.time() - t0) * 1000)
        bugs = scene.bug_cells()
        hit = choice in bugs
        rows.append({
            "seed": a.seed0 + i, "choice": choice, "hit": hit,
            "on_creature": choice in scene.occupied_cells(),
            "distance": scene.distance_to_bug(choice),
            "confidence": round(conf, 4), "bugs": bugs,
            "creatures": [[k, x, y] for k, x, y in scene.creatures],
            "top": sorted(((round(v, 4), k) for k, v in probs.items()), reverse=True)[:5],
        })
        print(f"trial {i}: {choice} {'HIT' if hit else 'miss'} "
              f"(bugs {' '.join(bugs)}, dist {rows[-1]['distance']})", flush=True)

    n = len(rows)
    summary = {
        "mode": a.mode, "trials": n, "bugs": a.bugs, "chance": a.bugs / 48,
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "on_creature_rate": sum(r["on_creature"] for r in rows) / n,
        "mean_distance": sum(r["distance"] for r in rows) / n,
        "mean_confidence": sum(r["confidence"] for r in rows) / n,
        "mean_ms": sum(lat) / len(lat) if lat else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False))

    tag = a.tag or a.mode
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps({"summary": summary, "trials": rows},
                                                ensure_ascii=False), encoding="utf-8")
    print("wrote", out / f"{tag}.json")


if __name__ == "__main__":
    main()
