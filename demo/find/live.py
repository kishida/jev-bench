"""鳥・うさぎ・虫がなめらかに動く野原を見せ、ときどき「虫のいるマスは？」とモデルに聞くデモ。

生き物は1ステップごとに隣のマスへ動き、ブラウザ側がその間を補間してなめらかに見せる。
数ステップに1回、その瞬間の盤面を PNG にしてモデルに投げ、返ってきた確率を重ねて表示する。

  python demo/find/live.py --url http://127.0.0.1:18160 --mode image_labeled
  → http://127.0.0.1:8802/

--mode text なら画像なしの AA で聞く（画像非対応のモデル用）。
"""
import argparse
import json
import queue
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import COLS, COUNTS, ROWS, Scene, all_cells, cell_name  # noqa: E402
from play import INSTRUCTIONS, ask  # noqa: E402

HERE = Path(__file__).resolve().parent


class Hub:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.last = None
        self.stats = {"asked": 0, "hit": 0, "on_creature": 0, "dist": 0.0}

    def publish(self, event):
        with self.lock:
            if event.get("type") != "tick" or self.last is None:
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


class Field:
    """実数座標で動く生き物たち。1ステップで隣のマスへ1つ動く。"""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        spots = [(x, y) for y in range(ROWS) for x in range(COLS)]
        self.rng.shuffle(spots)
        self.items = []
        i = 0
        for kind, n in COUNTS.items():
            for _ in range(n):
                x, y = spots[i]
                i += 1
                self.items.append([kind, float(x), float(y)])

    def step(self):
        taken = {(round(c[1]), round(c[2])) for c in self.items}
        for c in self.items:
            x, y = round(c[1]), round(c[2])
            moves = [(x + dx, y + dy) for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
                     if 0 <= x + dx < COLS and 0 <= y + dy < ROWS and (x + dx, y + dy) not in taken]
            if moves and self.rng.random() < 0.8:
                taken.discard((x, y))
                nx, ny = self.rng.choice(moves)
                taken.add((nx, ny))
                c[1], c[2] = float(nx), float(ny)

    def positions(self):
        return [(k, x, y) for k, x, y in self.items]


def runner(args, hub, stop):
    session = requests.Session()
    model = "?"
    try:
        p = session.get(f"{args.url}/props", timeout=10).json()
        model = Path(p.get("model_alias") or p.get("model_path") or "?").name
    except Exception:
        pass

    field = Field(args.seed)
    n = 0
    while not stop.is_set():
        n += 1
        field.step()
        positions = field.positions()
        hub.publish({"type": "tick", "model": model, "mode": args.mode, "step": n,
                     "cols": COLS, "rows": ROWS, "seconds": args.interval,
                     "creatures": [[k, x, y] for k, x, y in positions]})

        if n % args.every == 0:
            scene = Scene.from_positions(positions)
            t0 = time.time()
            try:
                choice, probs, conf = ask(session, args.url, scene, args.mode, args.api_key,
                                          assistant_prefix=args.assistant_prefix)
            except Exception as e:
                hub.publish({"type": "error", "message": str(e)[:200]})
                time.sleep(3)
                continue
            ms = (time.time() - t0) * 1000
            bugs = scene.bug_cells()
            hit = choice in bugs
            s = hub.stats
            s["asked"] += 1
            s["hit"] += hit
            s["on_creature"] += choice in scene.occupied_cells()
            s["dist"] += scene.distance_to_bug(choice)
            top = sorted(((v, k) for k, v in probs.items()), reverse=True)[:6]
            hub.publish({"type": "answer", "model": model, "mode": args.mode, "step": n,
                         "choice": choice, "hit": hit, "bugs": bugs,
                         "distance": scene.distance_to_bug(choice),
                         "confidence": round(conf, 4), "ms": round(ms),
                         "top": [[k, round(v, 4)] for v, k in top],
                         "heat": {k: round(v, 4) for k, v in probs.items() if v >= 0.01},
                         "stats": dict(s)})
        time.sleep(args.interval)


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
    ap.add_argument("--url", default="http://127.0.0.1:18160")
    ap.add_argument("--api-key")
    ap.add_argument("--mode", default="image_labeled",
                    choices=["image_labeled", "image", "text"])
    ap.add_argument("--port", type=int, default=8802)
    ap.add_argument("--interval", type=float, default=1.2, help="1ステップの秒数")
    ap.add_argument("--every", type=int, default=1, help="何ステップごとに聞くか")
    ap.add_argument("--assistant-prefix")
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()

    hub = Hub()
    stop = threading.Event()
    threading.Thread(target=runner, args=(a, hub, stop), daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(hub))
    print(f"live demo: http://127.0.0.1:{a.port}/  (model server: {a.url}, mode: {a.mode})",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
