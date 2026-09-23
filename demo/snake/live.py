"""Snake をモデルに解かせ続けて、ブラウザでリアルタイムに見せるデモ。

裏で延々とゲームを回し、一手ごとに盤面とモデルの確率を SSE で配信する。落ちたら新しいゲームを始めるので、
置きっぱなしのデモに使える。

  python demo/snake/live.py --url http://127.0.0.1:18160 --mode text
  → http://127.0.0.1:8800/

サーバー側のモデル名は /props から取って表示する。--mode image はマルチモーダルのモデル（--mmproj 付き）に。
"""
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import CRITERIA, INSTRUCTIONS, ask  # noqa: E402
from snake import DIRS, Snake  # noqa: E402

HERE = Path(__file__).resolve().parent


class Hub:
    """つないでいるブラウザに配る。最後の状態は新しい接続にそのまま送る。"""

    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.last = None
        self.stats = {"games": 0, "moves": 0, "scores": [], "best": 0, "causes": {}}

    def publish(self, event):
        with self.lock:
            self.last = event
            dead = []
            for q in self.clients:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.clients.remove(q)

    def subscribe(self):
        q = queue.Queue(maxsize=64)
        with self.lock:
            self.clients.append(q)
            if self.last:
                q.put_nowait(self.last)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)


def runner(args, hub, stop):
    session = requests.Session()
    model = "?"
    try:
        p = session.get(f"{args.url}/props", timeout=10).json()
        model = Path(p.get("model_alias") or p.get("model_path") or "?").name
    except Exception:
        pass

    seed = args.seed0
    while not stop.is_set():
        g = Snake(args.w, args.h, seed=seed)
        seed += 1
        stall = 0
        t_game = time.time()
        while not stop.is_set() and g.alive and g.steps < args.max_steps and stall < args.stall_limit:
            before = g.score
            t0 = time.time()
            try:
                choice, probs, conf = ask(session, args.url, g, args.mode, args.api_key)
            except Exception as e:  # サーバーが落ちても止めない
                hub.publish({"type": "error", "message": str(e)[:200]})
                time.sleep(3)
                continue
            ms = (time.time() - t0) * 1000
            hub.publish({"type": "move", "model": model, "mode": args.mode,
                         "w": g.w, "h": g.h, "body": [list(p) for p in g.body],
                         "apple": list(g.apple) if g.apple else None, "dir": g.dir,
                         "choice": choice, "probs": {k: round(probs.get(k, 0), 4) for k in DIRS},
                         "confidence": round(conf, 4), "safe": g.safe_dirs(),
                         "score": g.score, "steps": g.steps, "ms": round(ms),
                         "game": hub.stats["games"] + 1, "stats": hub.stats})
            g.step(choice)
            stall = 0 if g.score > before else stall + 1
            if args.delay:
                time.sleep(args.delay)
        cause = g.cause or ("step limit" if g.steps >= args.max_steps else "stalled")
        s = hub.stats
        s["games"] += 1
        s["moves"] += g.steps
        s["scores"].append(g.score)
        s["best"] = max(s["best"], g.score)
        s["causes"][cause] = s["causes"].get(cause, 0) + 1
        hub.publish({"type": "over", "model": model, "mode": args.mode,
                     "w": g.w, "h": g.h, "body": [list(p) for p in g.body],
                     "apple": list(g.apple) if g.apple else None, "dir": g.dir,
                     "choice": None, "probs": {}, "confidence": 0, "safe": [],
                     "score": g.score, "steps": g.steps, "cause": cause,
                     "seconds": round(time.time() - t_game, 1),
                     "game": s["games"], "stats": s})
        time.sleep(args.pause)


def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = hub.subscribe()
                try:
                    while True:
                        try:
                            ev = q.get(timeout=15)
                            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    hub.unsubscribe(q)
                return
            path = "live.html" if self.path in ("/", "/index.html") else self.path.lstrip("/")
            f = HERE / path
            if not f.is_file() or f.suffix not in (".html", ".css", ".js"):
                self.send_error(404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18160", help="llama-server（/v1/systemone）")
    ap.add_argument("--api-key")
    ap.add_argument("--mode", default="text", choices=["text", "image"])
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--stall-limit", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--delay", type=float, default=0.0, help="一手ごとの追加の待ち（秒）")
    ap.add_argument("--pause", type=float, default=0.5, help="ゲームが終わってから次を始めるまで（秒）")
    a = ap.parse_args()

    hub = Hub()
    stop = threading.Event()
    threading.Thread(target=runner, args=(a, hub, stop), daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(hub))
    print(f"live demo: http://127.0.0.1:{a.port}/  (model server: {a.url}, mode: {a.mode})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
