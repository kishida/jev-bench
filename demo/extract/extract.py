"""文章の中から「条件に当てはまる箇所」を Jev で探すデモ。

文章を state に置き、候補の断片それぞれを noul の質問にして投げる。返ってくるのは断片ごとの確率なので、
そのまま濃淡で塗れる。

候補が多いときは **小分けにして順に投げる**（既定12個ずつ）。どのリクエストも state が同じなので、
llama-server のプロンプトキャッシュが効いて2回目以降は安い。結果は届いた順に流すので、長い文章でも
少しずつ色が付いていく。

使い方は2通り。

- 条件モード: 文章＋条件（「納期に関する記述」など）
- メッセージモード: システムプロンプト＋メッセージ。Claude Code の Web Chat と同じ形（system を state、
  message を質問側）に載せて、システムプロンプトのどのあたりが効いているかを見る

  python demo/extract/extract.py --url http://127.0.0.1:18160     # → http://127.0.0.1:8810/
  python demo/extract/extract.py --url ... --cli --text-file doc.txt --condition "納期に関する記述"
"""
import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent

# 句読点・改行で切る。助詞での分割は形態素解析が要るので、ここでは読点までを1つの塊として扱う。
SPLIT = re.compile(r"(?<=[。．！？!?\n])|(?<=、)|(?<=,)")


def split_spans(text, min_len=2, max_len=160):
    """文章を候補の断片に切る。空白と短すぎるものは落とす。"""
    out = []
    for piece in SPLIT.split(text):
        s = piece.strip()
        if len(s) < min_len:
            continue
        out.append(s[:max_len])
    return out


def message_condition(message):
    """メッセージモードの条件文。"""
    return f"次のメッセージに答えるとき、参照すべき指示が書かれている: 「{message}」"


def ask_batch(url, state, condition, spans, offset=0, api_key=None, choice=False, timeout=600):
    """spans（1バッチぶん）の noul を1リクエストで聞く。"""
    questions = {
        f"s{offset + i}": {
            "type": "noul",
            "instructions": f"次の部分は条件「{condition}」に当てはまりますか: 「{s}」",
        }
        for i, s in enumerate(spans)
    }
    if choice and 2 <= len(spans) <= 8:
        questions["best"] = {
            "type": "choice",
            "instructions": f"条件「{condition}」に最もよく当てはまる部分はどれですか",
            "criteria": {f"s{offset + i}": s for i, s in enumerate(spans)},
        }
    body = {"model": "jev-latest", "state": state, "questions": questions}
    headers = {"authorization": f"Bearer {api_key}"} if api_key else None
    t0 = time.time()
    r = requests.post(f"{url}/v1/systemone", json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    a = data["answers"]
    hits = [{"i": offset + i, "text": s, "p": a[f"s{offset + i}"]["noul"]} for i, s in enumerate(spans)]
    return {"spans": hits, "usage": data.get("usage", {}), "ms": round((time.time() - t0) * 1000),
            "model": Path(data.get("model", "")).name}


def ask_all(url, state, condition, spans, batch=12, api_key=None, on_batch=None):
    """小分けにして全部聞く。on_batch(結果) があればバッチごとに呼ぶ。"""
    out, tokens, model = [], 0, ""
    t0 = time.time()
    for k in range(0, len(spans), batch):
        res = ask_batch(url, state, condition, spans[k:k + batch], offset=k, api_key=api_key)
        out += res["spans"]
        tokens += res["usage"].get("input_tokens", 0)
        model = model or res["model"]
        if on_batch:
            on_batch(res, k)
    return {"spans": out, "usage": {"input_tokens": tokens}, "model": model,
            "ms": round((time.time() - t0) * 1000)}


# ---------------------------------------------------------------- ブラウザ版
def make_handler(args):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/model":
                name = ""
                try:
                    p = requests.get(f"{args.url}/props", timeout=10).json()
                    name = Path(p.get("model_alias") or p.get("model_path") or "").name
                except Exception:
                    pass
                self._send(json.dumps({"model": name}).encode(), "application/json")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/stream":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            state = (req.get("state") or "").strip()
            message = (req.get("message") or "").strip()
            condition = message_condition(message) if message else (req.get("condition") or "").strip()
            target = (req.get("target") or state).strip()  # 切り分ける対象。既定は state 自身
            spans = split_spans(target)
            batch = max(1, min(int(req.get("batch") or args.batch), 32))

            # NDJSON を少しずつ流す。長さが分からないので、終わったら接続を閉じて終端を伝える
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()

            def emit(obj):
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

            if not state or not condition or not spans:
                emit({"error": "文章と条件（またはメッセージ）が要ります"})
                return
            emit({"total": len(spans), "batch": batch, "condition": condition,
                  "spans_text": [s for s in spans]})
            t0 = time.time()
            tokens = 0
            try:
                for k in range(0, len(spans), batch):
                    res = ask_batch(args.url, state, condition, spans[k:k + batch], offset=k,
                                    api_key=args.api_key)
                    tokens += res["usage"].get("input_tokens", 0)
                    emit({"spans": res["spans"], "ms": res["ms"], "model": res["model"],
                          "done": k + batch >= len(spans)})
                emit({"finished": True, "ms": round((time.time() - t0) * 1000), "tokens": tokens})
            except requests.HTTPError as e:
                body = e.response.text[:300] if e.response is not None else ""
                emit({"error": f"{e} {body}"})
            except Exception as e:
                emit({"error": str(e)[:300]})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18160", help="llama-server（/v1/systemone）")
    ap.add_argument("--api-key")
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--batch", type=int, default=12, help="1リクエストで聞く候補の数")
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--text-file")
    ap.add_argument("--condition")
    ap.add_argument("--message", help="指定するとメッセージモード（--text-file がシステムプロンプト）")
    a = ap.parse_args()

    if a.cli:
        text = Path(a.text_file).read_text(encoding="utf-8") if a.text_file else sys.stdin.read()
        cond = message_condition(a.message) if a.message else a.condition
        spans = split_spans(text)
        done = [0]

        def on_batch(res, k):
            done[0] += len(res["spans"])
            print(f"  ... {done[0]}/{len(spans)} ({res['ms']} ms)", file=sys.stderr, flush=True)

        res = ask_all(a.url, text, cond, spans, batch=a.batch, api_key=a.api_key, on_batch=on_batch)
        for s in sorted(res["spans"], key=lambda x: -x["p"]):
            print(f"{s['p']:.3f} {'█' * round(s['p'] * 20):<20s} {s['text'][:70]}")
        print(f"\n{len(spans)} 候補 / {res['ms']} ms / {res['usage']['input_tokens']} tokens / {res['model']}")
        return

    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a))
    print(f"extract demo: http://127.0.0.1:{a.port}/  (model server: {a.url}, batch: {a.batch})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
