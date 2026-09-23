"""迷路の生成と、モデルに渡す表現（スペース区切りの AA と PNG 画像）。

Snake と同じく依存を増やさないよう、PNG は自前で書く。

盤面は (2w+1) x (2h+1) のセル格子。壁を掘る方式（深さ優先）で作り、行き止まりが残る。
braid>0 なら行き止まりの一部を開けてループを作る（迷路が易しくなる）。
"""
import base64
import random
import struct
import zlib
from collections import deque

UP, DOWN, LEFT, RIGHT = "up", "down", "left", "right"
DIRS = [UP, DOWN, LEFT, RIGHT]
DELTA = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

WALL, OPEN = "#", "."


class Maze:
    """左上が (0,0)、x は右、y は下。grid[y][x] が '#' か '.'。"""

    def __init__(self, cells_w=5, cells_h=5, seed=0, braid=0.0):
        self.rng = random.Random(seed)
        self.w, self.h = cells_w * 2 + 1, cells_h * 2 + 1
        self.grid = [[WALL] * self.w for _ in range(self.h)]
        self._carve(cells_w, cells_h)
        if braid:
            self._braid(braid)
        self.start = (1, 1)
        self.goal = (self.w - 2, self.h - 2)
        self.pos = self.start
        self.visited = [self.start]
        self.steps = 0
        self.bumps = 0          # 壁にぶつかった回数（進めなかった手）
        self.done = False
        self.cause = None
        self.shortest = self.shortest_path_len(self.start, self.goal)

    # ------------------------------------------------------------ 生成
    def _carve(self, cw, ch):
        stack = [(0, 0)]
        seen = {(0, 0)}
        self.grid[1][1] = OPEN
        while stack:
            cx, cy = stack[-1]
            nbrs = []
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cw and 0 <= ny < ch and (nx, ny) not in seen:
                    nbrs.append((nx, ny, dx, dy))
            if not nbrs:
                stack.pop()
                continue
            nx, ny, dx, dy = self.rng.choice(nbrs)
            self.grid[cy * 2 + 1 + dy][cx * 2 + 1 + dx] = OPEN
            self.grid[ny * 2 + 1][nx * 2 + 1] = OPEN
            seen.add((nx, ny))
            stack.append((nx, ny))

    def _braid(self, rate):
        """行き止まりの一部を開けてループを作る。"""
        for y in range(1, self.h - 1, 2):
            for x in range(1, self.w - 1, 2):
                if self.grid[y][x] != OPEN:
                    continue
                walls = [(x + dx, y + dy) for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
                         if self.grid[y + dy][x + dx] == WALL
                         and 0 < x + dx < self.w - 1 and 0 < y + dy < self.h - 1]
                opens = 4 - len(walls)
                if opens <= 1 and walls and self.rng.random() < rate:
                    wx, wy = self.rng.choice(walls)
                    self.grid[wy][wx] = OPEN

    # ------------------------------------------------------------ 進行
    def open_at(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h and self.grid[y][x] == OPEN

    def legal_dirs(self):
        out = []
        for d in DIRS:
            dx, dy = DELTA[d]
            if self.open_at(self.pos[0] + dx, self.pos[1] + dy):
                out.append(d)
        return out

    def step(self, direction):
        dx, dy = DELTA[direction]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        self.steps += 1
        if not self.open_at(nx, ny):
            self.bumps += 1
            return False
        self.pos = (nx, ny)
        self.visited.append(self.pos)
        if self.pos == self.goal:
            self.done, self.cause = True, "goal"
        return True

    def shortest_path_len(self, a, b):
        """a から b までの最短手数（幅優先）。届かなければ None。"""
        q = deque([(a, 0)])
        seen = {a}
        while q:
            (x, y), d = q.popleft()
            if (x, y) == b:
                return d
            for dx, dy in DELTA.values():
                nx, ny = x + dx, y + dy
                if self.open_at(nx, ny) and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append(((nx, ny), d + 1))
        return None

    def remaining(self):
        return self.shortest_path_len(self.pos, self.goal)

    def optimal_dir(self):
        """最短経路に沿う手（参考方策）。"""
        best, bestd = None, None
        for d in self.legal_dirs():
            dx, dy = DELTA[d]
            n = self.shortest_path_len((self.pos[0] + dx, self.pos[1] + dy), self.goal)
            if n is not None and (bestd is None or n < bestd):
                best, bestd = d, n
        return best or (self.legal_dirs() or DIRS)[0]

    # ------------------------------------------------------------ 表現
    def chars(self, mark_pos=True):
        g = [row[:] for row in self.grid]
        gx, gy = self.goal
        g[gy][gx] = "G"
        if mark_pos:
            px, py = self.pos
            g[py][px] = "S"
        return g

    def ascii_board(self, spaced=True, labels=False):
        g = self.chars()
        sep = " " if spaced else ""
        if not labels:
            return "\n".join(sep.join(row) for row in g)
        wid = len(str(self.h - 1))
        head = " " * (wid + 1) + sep.join(str(x % 10) for x in range(self.w))
        rows = [f"{y:>{wid}} " + sep.join(row) for y, row in enumerate(g)]
        return "\n".join([head] + rows)

    def text_state(self, spaced=True, labels=False, coords=False):
        """labels=True で行番号・列番号を振り、軸の向きを明示する。
        coords=True で現在地とゴールの座標も渡す（Snake の座標ヒント版に相当）。"""
        out = [self.ascii_board(spaced, labels), ""]
        out.append("The grid above is a maze. S = you, G = the goal, # = wall, . = open floor.")
        if labels:
            out.append("The first line is the column number and the first column is the row number.")
            out.append("Row 0 is the top row. Moving up decreases the row number by 1, moving down "
                       "increases it by 1. Column 0 is the leftmost column. Moving left decreases "
                       "the column number by 1, moving right increases it by 1.")
        else:
            out.append("Up is towards the top row, down towards the bottom row, left and right as shown.")
        if coords:
            out.append(f"You are at row {self.pos[1]}, column {self.pos[0]}. "
                       f"The goal is at row {self.goal[1]}, column {self.goal[0]}.")
        out.append("You can only move onto open floor or the goal. Moving into a wall wastes the turn.")
        return "\n".join(out)

    def image_state(self):
        return (
            "The picture shows a maze. The blue stick figure is you, the green chequered square is "
            "the goal, the black squares are walls, and the white squares are open floor.\n"
            "Up is towards the top of the picture, down towards the bottom, left and right as seen.\n"
            "You can only move onto open floor or the goal. Moving into a wall wastes the turn."
        )

    def png(self, cell=28, border=1):
        w, h = self.w * cell, self.h * cell
        floor, wall = (248, 248, 245), (25, 25, 25)
        me, goal, trail = (40, 90, 200), (30, 150, 70), (215, 225, 240)
        px = [[floor] * w for _ in range(h)]

        def fill(x0, y0, x1, y1, c):
            for y in range(max(0, y0), min(h, y1)):
                row = px[y]
                for x in range(max(0, x0), min(w, x1)):
                    row[x] = c

        def cellfill(cx, cy, c, pad=0):
            fill(cx * cell + pad, cy * cell + pad, (cx + 1) * cell - pad, (cy + 1) * cell - pad, c)

        def checker(cx, cy, c):
            """ゴールは市松模様。色だけでなく模様でも分かるように。"""
            n = 4
            step = cell // n
            for i in range(n):
                for j in range(n):
                    if (i + j) % 2 == 0:
                        fill(cx * cell + j * step, cy * cell + i * step,
                             cx * cell + (j + 1) * step, cy * cell + (i + 1) * step, c)

        def stick(cx, cy, c):
            """現在地は棒人間。四角だと壁や床と形で区別できないため。"""
            x0, y0 = cx * cell, cy * cell
            u = max(1, cell // 14)          # 線の太さ
            mx = x0 + cell // 2
            head_r = cell // 6
            hy = y0 + cell // 4
            for dy in range(-head_r, head_r + 1):     # 頭（丸）
                half = int((head_r * head_r - dy * dy) ** 0.5)
                fill(mx - half, hy + dy, mx + half + 1, hy + dy + 1, c)
            fill(mx - u, hy + head_r, mx + u, y0 + int(cell * 0.68), c)          # 胴
            fill(mx - cell // 4, hy + head_r + u, mx + cell // 4, hy + head_r + u + 2 * u, c)  # 腕
            for sx in (-1, 1):                                                    # 脚
                for k in range(int(cell * 0.22)):
                    x = mx + sx * k // 2
                    y = y0 + int(cell * 0.68) + k
                    fill(x - u, y, x + u, y + 1, c)

        for y in range(self.h):
            for x in range(self.w):
                if self.grid[y][x] == WALL:
                    cellfill(x, y, wall)
        for x, y in self.visited[:-1]:
            cellfill(x, y, trail, pad=cell // 3)
        checker(*self.goal, goal)
        stick(*self.pos, me)

        raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in px)

        def chunk(tag, data):
            return (struct.pack(">I", len(data)) + tag + data +
                    struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6))
                + chunk(b"IEND", b""))

    def data_url(self, cell=28):
        return "data:image/png;base64," + base64.b64encode(self.png(cell)).decode()
