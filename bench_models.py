"""docs/jev.md の比較表用: 各モデルを同じ条件で llama-server に載せ、精度・校正・平均応答時間を測る。

応答時間は「1問だけ入ったリクエスト」を直列に投げたときの平均（プロンプトキャッシュの影響を避けるため毎回別の問題）。
python temp/bench_all.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "training"))
from jev.data import UNI_DIR, read_jsonl  # noqa: E402

SERVER = r"D:\dev\llama.cpp\build\bin\Release\llama-server.exe"
GGUF = Path(os.environ.get("GGUF_DIR", "D:/dev/gguf"))  # where the off-the-shelf GGUFs live
PORT = 18160
N_LATENCY = 50
N_WARMUP = 3

MODELS = [
    # サイズの比較（Qwen3.5 系列。量子化は入手したものに準じる）
    ("Qwen3.5 4B Q4_K_M",           GGUF / "lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf", []),
    ("Qwen3.5 9B UD-Q4_K_XL",       GGUF / "unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf", []),
    ("Qwen3.6 35B A3B UD-IQ2_M",    GGUF / "unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", []),
    # 世代の比較（同じ 27B、同じ量子化）
    ("Qwen3.5 27B UD-IQ3_XXS",      GGUF / "unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-IQ3_XXS.gguf", []),
    ("Qwen3.6 27B UD-IQ3_XXS",      GGUF / "unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-IQ3_XXS.gguf", []),
    ("Qwen3.8 27B UD-IQ3_XXS",      GGUF / "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf", []),
    ("Gemma 4 12B UD-Q4_K_XL",      GGUF / "unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf", []),
    ("gpt-oss 20B MXFP4",           GGUF / "lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf", []),
    ("LLM-jp-4 8B thinking Q4_K_M", GGUF / "llm-jp/llm-jp-4-8b-thinking-gguf/llm-jp-4-8b-thinking-Q4_K_M.gguf", []),
    ("jwenv 4B poc Q8_0",           ROOT.parent / "training/models/gguf/jwenv-4b-poc-q8_0.gguf", []),
    ("jwenv 1.7B poc Q8_0",         ROOT.parent / "training/models/gguf/jwenv-1.7b-poc-q8_0.gguf", []),
    ("Qwen3.5 2B Q8_0",             GGUF / "lmstudio-community/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf", []),
    ("Qwen3 1.7B Q8_0",             GGUF / "lmstudio-community/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf", []),
    ("jwenv 0.6B poc Q8_0",         ROOT.parent / "training/models/gguf/jwenv-0.6b-poc-q8_0.gguf", []),
    ("LFM2.5 8B A1B UD-Q4_K_XL",    GGUF / "unsloth/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-UD-Q4_K_XL.gguf",
     ["--jev-assistant-prefix", "<think>\\n\\n</think>\\n\\n"]),
    ("LFM2.5 350M Q8_0",            GGUF / "LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf", []),
    ("Qwen3 0.6B Q8_0",             GGUF / "lmstudio-community/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", []),
    ("gemma-3 270m-it Q8_0",        GGUF / "lmstudio-community/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q8_0.gguf", []),
]

TAGS = {name: "bench_" + name.split()[0].lower().replace(".", "").replace("-", "") + "_" +
        Path(path).stem.lower().replace(".", "").replace("-", "_") for name, path, _ in MODELS}


def wait_ready(url, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url + "/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def latency(url, items):
    body = lambda it: {
        "state": it["context"] or "(none)", "model": "m",
        "questions": {"q": {"type": "choice", "instructions": it["question"],
                            "criteria": {c: None for c in it["choices"]}}},
    }
    s = requests.Session()
    for it in items[:N_WARMUP]:
        s.post(url + "/v1/systemone", json=body(it), timeout=600).raise_for_status()
    ts = []
    for it in items[N_WARMUP:N_WARMUP + N_LATENCY]:
        t0 = time.time()
        s.post(url + "/v1/systemone", json=body(it), timeout=600).raise_for_status()
        ts.append((time.time() - t0) * 1000)
    ts.sort()
    return {"mean_ms": sum(ts) / len(ts), "median_ms": ts[len(ts) // 2], "n": len(ts)}


def main():
    url = f"http://127.0.0.1:{PORT}"
    items = [it for it in read_jsonl(UNI_DIR / "objective_test.jsonl")
             if len(set(it["choices"])) == len(it["choices"])][:N_WARMUP + N_LATENCY]
    out_path = ROOT / "results/bench_all.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name, path, extra in MODELS:
        if name in out:
            print("skip", name, flush=True)
            continue
        if not Path(path).exists():
            print("MISSING", name, path, flush=True)
            continue
        print("===", name, flush=True)
        log = open(f"temp/bench_{TAGS[name]}.log", "w", encoding="utf-8")
        proc = subprocess.Popen([SERVER, "-m", str(path), "-ngl", "99", "--host", "127.0.0.1",
                                 "--port", str(PORT), "-c", "16384", "-np", "8", *extra],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            if not wait_ready(url):
                print("FAILED to start", name, flush=True)
                continue
            t0 = time.time()
            r = subprocess.run([sys.executable, "scripts/eval_systemone.py", "--url", url,
                                "--tag", TAGS[name], "--workers", "8"], capture_output=True, text=True)
            print(r.stdout.strip()[-200:], flush=True)
            m = json.loads((ROOT / "results" / TAGS[name] / "systemone_metrics.json").read_text())
            lat = latency(url, items)
            out[name] = {"path": str(path), "extra": extra, "temperature": m["temperature"],
                         "acc": m["calibrated"]["acc"], "ece_raw": m["raw"]["ece"],
                         "ece_cal": m["calibrated"]["ece"], "nll_raw": m["raw"]["nll"],
                         "nll_cal": m["calibrated"]["nll"], "latency": lat,
                         "eval_seconds": time.time() - t0}
            print(f"{name}: acc={out[name]['acc']:.3f} latency={lat['mean_ms']:.0f} ms", flush=True)
            out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
            time.sleep(3)

    for name in out:
        d = out[name]
        print(f"{name:30s} acc={d['acc']:.3f} ece {d['ece_raw']:.3f}->{d['ece_cal']:.3f} "
              f"T={d['temperature']:.2f} {d['latency']['mean_ms']:.0f} ms")


if __name__ == "__main__":
    main()
