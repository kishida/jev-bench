"""Snake を複数モデル・複数の渡し方で回して results/snake/ にそろえる（replay.html で見る用）。

python demo/snake/run_all.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

SERVER = r"D:\dev\llama.cpp\build\bin\Release\llama-server.exe"
GGUF = Path(os.environ.get("GGUF_DIR", "D:/dev/gguf"))  # where the off-the-shelf GGUFs live
QWEN35 = GGUF / "lmstudio-community/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf"
MMPROJ = GGUF / "lmstudio-community/Qwen3.5-2B-GGUF/mmproj-Qwen3.5-2B-BF16.gguf"
PORT = 18160
URL = f"http://127.0.0.1:{PORT}"
BENCH = Path(__file__).resolve().parents[2]
OUT = BENCH / "results/snake"

# tag -> (表示名, モデル, mmproj, 入力の渡し方, ゲーム数)
RUNS = [
    ("text_qwen35_2b",  "Qwen3.5 2B",    QWEN35, None,    "text",  10),
    ("text_jwenv17",    "jwenv 1.7B",    BENCH.parent / "training/models/gguf/jwenv-1.7b-poc-q8_0.gguf", None, "text", 10),
    ("text_qwen35_4b",  "Qwen3.5 4B",    GGUF / "lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf", None, "text", 10),
    ("text_gemma4_12b", "Gemma 4 12B",   GGUF / "unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf", None, "text", 10),
    ("image_qwen35_2b", "Qwen3.5 2B 画像", QWEN35, MMPROJ, "image", 5),
]
BASELINES = [("greedy", "りんごに近づく手", "greedy", 20), ("random", "でたらめ", "random", 20)]


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(URL + "/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def play(tag, mode, games, extra=()):
    cmd = [sys.executable, "demo/snake/play.py", "--mode", mode, "--games", str(games),
           "--url", URL, "--tag", tag, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip()[-400:] or r.stderr.strip()[-400:], flush=True)
    return r.returncode == 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []

    for tag, label, mode, games in BASELINES:
        if play(tag, mode, games):
            index.append({"tag": tag, "label": label})

    for tag, label, model, mmproj, mode, games in RUNS:
        if not Path(model).exists():
            print("MISSING", model, flush=True)
            continue
        print("===", tag, flush=True)
        log = open(f"temp/snake_{tag}.log", "w", encoding="utf-8")
        args = [SERVER, "-m", str(model), "-ngl", "99", "--host", "127.0.0.1", "--port", str(PORT),
                "-c", "16384", "-np", "4"]
        if mmproj:
            args += ["--mmproj", str(mmproj)]
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        try:
            if not wait_ready():
                print("FAILED to start", tag, flush=True)
                continue
            if play(tag, mode, games):
                index.append({"tag": tag, "label": label})
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
            time.sleep(3)

    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n--- summary")
    for e in index:
        s = json.loads((OUT / f"{e['tag']}.json").read_text(encoding="utf-8"))["summary"]
        ms = f"{s['mean_ms_per_move']:.0f} ms/move" if s["mean_ms_per_move"] else "-"
        print(f"{e['label']:18s} {s['mode']:6s} mean={s['mean_score']:.1f} max={s['max_score']} "
              f"steps={s['mean_steps']:.0f} {ms} {s['causes']}")


if __name__ == "__main__":
    main()
