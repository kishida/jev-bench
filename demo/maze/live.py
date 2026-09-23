"""迷路をモデルに解かせ続けて、ブラウザでリアルタイムに見せるデモ。

裏で延々と迷路を回し、一手ごとに盤面とモデルの確率を SSE で配信する。ゴールしても行き詰まっても
次の迷路を始めるので、置きっぱなしのデモに使える。

  python demo/maze/live.py --url http://mac.local:8086 --mode text --labels --coords --cells 5
  → http://127.0.0.1:8801/

--mode image はマルチモーダルのモデル（--mmproj 付き）に。
"""
import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze import DIRS, Maze  # noqa: E402
from play import ask  # noqa: E402

HERE = Path(__file__).resolve().parent


class Hub:
    """つないでいるブラウザに配る。最後の状態は新しい接続にそのまま送る。"""

    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.last = None
        self.stats = {"games": 0, "solved": 0, "moves": 0, "legal": 0, "closer": 0, "causes": {}}

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


def base(mz, args, model, hub):
    return {"model": model, "mode": args.mode, "labels": args.labels, "coords": args.coords,
            "grid": ["".join(r) for r in mz.grid], "goal": list(mz.goal),
            "pos": list(mz.pos), "trail": [list(p) for p in mz.visited],
            "steps": mz.steps, "bumps": mz.bumps, "shortest": mz.shortest,
            "remaining": mz.remaining(), "game": hub.stats["games"] + 1, "stats": hub.stats}


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
        mz = Maze(args.cells, args.cells, seed=seed, braid=args.braid)
        seed += 1
        limit = args.max_steps or mz.shortest * 6
        bump_streak = 0
        t_game = time.time()
        while (not stop.is_set() and not mz.done and mz.steps < limit
               and bump_streak < args.bump_limit):
            before = mz.remaining()
            t0 = time.time()
            try:
                choice, probs, conf = ask(session, args.url, mz, args.mode, args.api_key,
                                          labels=args.labels, coords=args.coords,
                                          assistant_prefix=args.assistant_prefix)
            except Exception as e:  # サーバーが落ちても止めない
                hub.publish({"type": "error", "message": str(e)[:200]})
                time.sleep(3)
                continue
            ms = (time.time() - t0) * 1000
            legal = mz.legal_dirs()
            ev = base(mz, args, model, hub)
            ev.update({"type": "move", "choice": choice, "legal": legal,
                       "probs": {k: round(probs.get(k, 0), 4) for k in DIRS},
                       "confidence": round(conf, 4), "ms": round(ms)})
            hub.publish(ev)
            moved = mz.step(choice)
            bump_streak = 0 if moved else bump_streak + 1
            s = hub.stats
            s["moves"] += 1
            s["legal"] += choice in legal
            s["closer"] += bool(moved and (mz.remaining() or 0) < (before or 0))
            if args.delay:
                time.sleep(args.delay)
        cause = mz.cause or ("stuck" if bump_streak >= args.bump_limit else "step limit")
        s = hub.stats
        s["games"] += 1
        s["solved"] += bool(mz.done)
        s["causes"][cause] = s["causes"].get(cause, 0) + 1
        ev = base(mz, args, model, hub)
        ev.update({"type": "over", "choice": None, "legal": [], "probs": {}, "confidence": 0,
                   "cause": cause, "seconds": round(time.time() - t_game, 1),
                   "game": s["games"]})
        hub.publish(ev)
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
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--cells", type=int, default=5, help="セル数（盤面は 2n+1 四方）")
    ap.add_argument("--braid", type=float, default=0.0)
    ap.add_argument("--labels", action="store_true", help="行番号・列番号を振り、軸の向きを明示する")
    ap.add_argument("--coords", action="store_true", help="現在地とゴールの座標も渡す")
    ap.add_argument("--max-steps", type=int, default=0, help="0 なら最短手数の6倍")
    ap.add_argument("--bump-limit", type=int, default=8)
    ap.add_argument("--assistant-prefix")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--delay", type=float, default=0.0, help="一手ごとの追加の待ち（秒）")
    ap.add_argument("--pause", type=float, default=2.0, help="終わってから次を始めるまで（秒）")
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
