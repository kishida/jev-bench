"""Evaluate any /v1/systemone server on jev-bench.

  pip install requests numpy datasets
  python eval.py --url http://127.0.0.1:8080

Every question is sent as one `choice` question with the option texts as the criteria, so this
measures the server the way an application would use it.

Two modes, picked automatically from what the server accepts:

  logits         the server understands `options.return_logits` and `options.temperature_scaling`.
                 The raw logits come back untouched, a temperature T is fitted on the validation
                 split, and accuracy/ECE are reported before and after applying it. This is what
                 the numbers in the llama.cpp jev docs were measured with.
  probabilities  the server only returns `probabilities`. Accuracy is identical; the calibration
                 numbers describe whatever calibration the server already applies, and no T is
                 fitted. Use this for hosted services.

Pass --data to read test.jsonl / val.jsonl from a local directory instead of the Hub.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

HUB_ID = "kishida/jev-bench"
NEG = -1e4          # padding for unused labels, far below any real logit
MAX_CHOICES = 8


# --------------------------------------------------------------------------- data
def load_split(split, data_dir=None):
    if data_dir:
        path = Path(data_dir) / f"{split}.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        from datasets import load_dataset
        rows = list(load_dataset(HUB_ID, split=split))
    # a question whose options are not all distinct has no single right answer
    return [r for r in rows if len(set(r["choices"])) == len(r["choices"])]


# --------------------------------------------------------------------------- server
def build_body(item, assistant_prefix=None, want_logits=True):
    body = {
        "model": "jev-latest",
        "state": item["context"] or "(none)",
        "questions": {"q": {"type": "choice", "instructions": item["question"],
                            "criteria": {c: None for c in item["choices"]}}},
    }
    options = {}
    if want_logits:
        options["return_logits"] = True
        options["temperature_scaling"] = False
    if assistant_prefix:
        options["assistant_prefix"] = assistant_prefix
    if options:
        body["options"] = options
    return body


def ask(session, url, item, assistant_prefix, want_logits, api_key, timeout):
    headers = {"authorization": f"Bearer {api_key}"} if api_key else None
    r = session.post(f"{url}/v1/systemone", json=build_body(item, assistant_prefix, want_logits),
                     headers=headers, timeout=timeout)
    r.raise_for_status()
    a = r.json()["answers"]["q"]
    if want_logits:
        return [a["logits"][c] for c in item["choices"]]
    p = a["probabilities"]
    return [p[c] for c in item["choices"]]


def supports_logits(session, url, item, assistant_prefix, api_key, timeout):
    """One probe request. A server that rejects the option answers 4xx; anything else is a real error."""
    try:
        ask(session, url, item, assistant_prefix, True, api_key, timeout)
        return True
    except requests.HTTPError as e:
        if e.response is not None and 400 <= e.response.status_code < 500:
            return False
        raise
    except KeyError:
        return False          # accepted the option but returned no logits


def run_split(url, items, workers, assistant_prefix, want_logits, api_key, timeout):
    session = requests.Session()
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(
            lambda it: ask(session, url, it, assistant_prefix, want_logits, api_key, timeout), items))


# --------------------------------------------------------------------------- metrics
def softmax_n(logits, n, T=1.0):
    z = np.asarray(logits[:n], dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_temperature(mat, rows, lo=-3.0, hi=3.0, rounds=40):
    """T minimising the cross entropy of the gold label, searched over log T.

    A ternary search rather than LBFGS, so the script needs no torch. The loss is smooth and
    single-minimum in log T, and 40 rounds leave the answer accurate to about 1e-7.
    """
    def loss(logT):
        T = float(np.exp(logT))
        return float(np.mean([-np.log(max(softmax_n(mat[i], r["n"], T)[r["gold"]], 1e-12))
                              for i, r in enumerate(rows)]))

    for _ in range(rounds):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if loss(a) < loss(b):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2))


def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def metrics(rows, mat, T=1.0, already_probs=False):
    probs = [np.asarray(mat[i][:r["n"]], dtype=np.float64) if already_probs
             else softmax_n(mat[i], r["n"], T) for i, r in enumerate(rows)]
    conf = [float(p.max()) for p in probs]
    correct = [int(p.argmax() == r["gold"]) for r, p in zip(rows, probs)]
    return {"count": len(rows), "acc": float(np.mean(correct)), "ece": ece(conf, correct),
            "nll": float(np.mean([-np.log(max(p[r["gold"]], 1e-12)) for r, p in zip(rows, probs)])),
            "brier": float(np.mean([float(((p - np.eye(r["n"])[r["gold"]]) ** 2).sum())
                                    for r, p in zip(rows, probs)])),
            "mean_conf": float(np.mean(conf))}


def breakdown(rows, mat, key, T, already_probs):
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r[key], []).append(i)
    return {str(k): metrics([rows[i] for i in idx], [mat[i] for i in idx], T, already_probs)
            for k, idx in sorted(groups.items(), key=lambda kv: str(kv[0]))}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Evaluate a /v1/systemone server on jev-bench.")
    ap.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--api-key")
    ap.add_argument("--data", help="directory holding test.jsonl / val.jsonl (default: the Hub)")
    ap.add_argument("--workers", type=int, default=8, help="parallel requests; lower it for a hosted API")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--assistant-prefix",
                    help="text the model writes before the label, for models the server cannot detect")
    ap.add_argument("--limit", type=int, help="use only the first N questions of each split (a quick check)")
    ap.add_argument("--mode", default="auto", choices=["auto", "logits", "probabilities"])
    ap.add_argument("--out", help="write the full result as JSON here")
    a = ap.parse_args()

    splits = {s: load_split(s, a.data) for s in ("val", "test")}
    if a.limit:
        splits = {s: rows[:a.limit] for s, rows in splits.items()}

    session = requests.Session()
    if a.mode == "auto":
        use_logits = supports_logits(session, a.url, splits["val"][0], a.assistant_prefix,
                                     a.api_key, a.timeout)
        print(f"mode: {'logits' if use_logits else 'probabilities'} (detected)", flush=True)
    else:
        use_logits = a.mode == "logits"
        print(f"mode: {a.mode}", flush=True)
    if not use_logits:
        print("  the server returns probabilities only, so no temperature is fitted and the\n"
              "  calibration numbers describe the server's own calibration.", flush=True)

    data, seconds = {}, {}
    for split, items in splits.items():
        t0 = time.time()
        values = run_split(a.url, items, a.workers, a.assistant_prefix, use_logits,
                           a.api_key, a.timeout)
        seconds[split] = time.time() - t0
        mat = np.full((len(items), MAX_CHOICES), 0.0 if not use_logits else NEG, dtype=np.float64)
        for i, v in enumerate(values):
            mat[i, :len(v)] = v
        rows = [{"n": it["n_choices"], "gold": it["answer_idx"], "source": it["source"]}
                for it in items]
        data[split] = (rows, mat)
        print(f"{split}: {len(items)} questions in {seconds[split]:.0f}s", flush=True)

    trows, tmat = data["test"]
    out = {"url": a.url, "mode": "logits" if use_logits else "probabilities",
           "assistant_prefix": a.assistant_prefix, "seconds": seconds}

    if use_logits:
        vrows, vmat = data["val"]
        T = fit_temperature(vmat, vrows)
        out["temperature"] = T
        out["raw"] = metrics(trows, tmat, 1.0)
        out["calibrated"] = metrics(trows, tmat, T)
        out["by_n"] = breakdown(trows, tmat, "n", T, False)
        out["by_source"] = breakdown(trows, tmat, "source", T, False)
        print(f"\nT={T:.2f}  accuracy={out['calibrated']['acc']:.3f}  "
              f"ECE {out['raw']['ece']:.3f} -> {out['calibrated']['ece']:.3f}  "
              f"NLL {out['raw']['nll']:.3f} -> {out['calibrated']['nll']:.3f}")
    else:
        out["reported"] = metrics(trows, tmat, already_probs=True)
        out["by_n"] = breakdown(trows, tmat, "n", 1.0, True)
        out["by_source"] = breakdown(trows, tmat, "source", 1.0, True)
        rounded = sum(tmat[i][r["gold"]] == 0.0 for i, r in enumerate(trows))
        out["gold_rounded_to_zero"] = int(rounded)
        print(f"\naccuracy={out['reported']['acc']:.3f}  ECE={out['reported']['ece']:.3f}  "
              f"NLL={out['reported']['nll']:.3f}")
        if rounded:
            print(f"  NLL and Brier are inflated: the right option was rounded to 0.0 on "
                  f"{rounded} of {len(trows)} questions.\n"
                  f"  Accuracy and ECE are unaffected. Compare NLL only within this mode.")

    print(f"always guessing would give {np.mean([1.0 / r['n'] for r in trows]):.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
