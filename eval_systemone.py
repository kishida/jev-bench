"""/v1/systemone 互換サーバー（llama-server の jev ブランチや qwen3-engine）を客観問題で評価し、温度 T をフィットする。

各問題を choice の質問として送り、options.return_logits で生の logit を受け取る（温度はかけない）。
val で T をフィットし、test の正解率・ECE を校正前後で出す。汎用モデルを使うときの T はここで決める。

python scripts/eval_systemone.py --url http://127.0.0.1:18101 --tag gemma4_12b_llamaserver
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "training"))
from jev.data import UNI_DIR, read_jsonl  # noqa: E402
from jev.metrics import fit_temperature, objective_metrics  # noqa: E402

NEG = -1e4


def ask(url, item, session, prefix=None):
    body = {
        "state": item["context"] or "(none)",
        "model": "jev-latest",
        "questions": {"q": {"type": "choice", "instructions": item["question"],
                            "criteria": {c: None for c in item["choices"]}}},
        "options": {"return_logits": True, "temperature_scaling": False},
    }
    if prefix:
        body["options"]["assistant_prefix"] = prefix
    r = session.post(f"{url}/v1/systemone", json=body, timeout=600)
    r.raise_for_status()
    lg = r.json()["answers"]["q"]["logits"]
    return [lg[c] for c in item["choices"]]


def run(url, items, workers, prefix=None):
    s = requests.Session()
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda it: ask(url, it, s, prefix), items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--assistant-prefix", default=None, help="続きを書かせる書き出し（harmony 形式なら <|channel|>final<|message|> など）")
    a = ap.parse_args()

    out = {}
    data = {}
    for split in ["val", "test"]:
        items = [it for it in read_jsonl(UNI_DIR / f"objective_{split}.jsonl") if len(set(it["choices"])) == len(it["choices"])]
        t0 = time.time()
        logits = run(a.url, items, a.workers, a.assistant_prefix)
        out[f"{split}_seconds"] = time.time() - t0
        mat = np.full((len(items), 8), NEG, dtype=np.float32)
        for i, lg in enumerate(logits):
            mat[i, :len(lg)] = lg
        rows = [{"n": it["n_choices"], "gold": it["answer_idx"], "source": it["source"]} for it in items]
        data[split] = (rows, mat)
        print(split, len(items), f"{time.time() - t0:.0f}s", flush=True)

    vrows, vmat = data["val"]
    T = fit_temperature(vmat, [[1.0 if k == r["gold"] else 0.0 for k in range(8)] for r in vrows])
    trows, tmat = data["test"]
    raw, _ = objective_metrics(trows, tmat, 1.0)
    cal, _ = objective_metrics(trows, tmat, T)
    out.update({"tag": a.tag, "url": a.url, "assistant_prefix": a.assistant_prefix, "temperature": T, "raw": raw, "calibrated": cal})
    (ROOT / "results").joinpath(a.tag).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results" / a.tag / "systemone_metrics.json", "w"), indent=1)
    print(f"T={T:.3f}  acc={cal['acc']:.3f}  ECE raw={raw['ece']:.3f} -> cal={cal['ece']:.3f}  NLL raw={raw['nll']:.3f} -> cal={cal['nll']:.3f}")


if __name__ == "__main__":
    main()
