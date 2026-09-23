"""Snake の盤面と、モデルに渡す2つの表現（AA と PNG 画像）。

依存を増やさないよう PNG は自前で書く（無圧縮 zlib ストリーム）。
"""
import base64
import random
import struct
import zlib

UP, DOWN, LEFT, RIGHT = "up", "down", "left", "right"
DIRS = [UP, DOWN, LEFT, RIGHT]
DELTA = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class Snake:
    """左上が (0,0)、x は右、y は下。"""

    def __init__(self, w=10, h=10, seed=0):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        cx, cy = w // 2, h // 2
        self.body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]  # 先頭が頭
        self.dir = RIGHT
        self.alive = True
        self.cause = None
        self.score = 0
        self.steps = 0
        self.reversals = 0
        self.apple = self._new_apple()

    def _new_apple(self):
        free = [(x, y) for y in range(self.h) for x in range(self.w) if (x, y) not in self.body]
        return self.rng.choice(free) if free else None

    def step(self, direction):
        """direction に進む。逆走の指示は普通の Snake と同じく無視して直進する。"""
        if direction == OPPOSITE[self.dir]:
            self.reversals += 1
            direction = self.dir
        self.dir = direction
        dx, dy = DELTA[direction]
        hx, hy = self.body[0]
        nx, ny = hx + dx, hy + dy
        self.steps += 1
        if not (0 <= nx < self.w and 0 <= ny < self.h):
            self.alive, self.cause = False, "wall"
            return
        # 尾は次で空くので、尾との衝突は当たりにしない（りんごを食べたときは伸びるので当たり）
        grows = (nx, ny) == self.apple
        body = self.body if grows else self.body[:-1]
        if (nx, ny) in body:
            self.alive, self.cause = False, "self"
            return
        self.body = [(nx, ny)] + body
        if grows:
            self.score += 1
            self.apple = self._new_apple()
            if self.apple is None:
                self.alive, self.cause = False, "cleared"

    # ------------------------------------------------------------ 表現
    def ascii_board(self):
        """AA。# 壁 / H 頭 / o 胴 / A りんご / . 空き"""
        grid = [["." for _ in range(self.w)] for _ in range(self.h)]
        for x, y in self.body[1:]:
            grid[y][x] = "o"
        hx, hy = self.body[0]
        grid[hy][hx] = "H"
        if self.apple:
            ax, ay = self.apple
            grid[ay][ax] = "A"
        top = "#" * (self.w + 2)
        rows = ["#" + "".join(r) + "#" for r in grid]
        return "\n".join([top] + rows + [top])

    def text_state(self, minimal=False):
        """minimal=True では座標などの補助を外し、画像版と同じだけの情報にそろえる。"""
        if minimal:
            return (
                f"{self.ascii_board()}\n\n"
                f"Legend: H = the snake's head, o = its body, A = the apple, # = the wall, . = empty.\n"
                f"The snake is {len(self.body)} cells long and is currently moving {self.dir}.\n"
                f"Up is towards the top of the board, down towards the bottom, left and right as shown.\n"
                f"The snake dies if it hits the wall or its own body."
            )
        hx, hy = self.body[0]
        ax, ay = self.apple if self.apple else (hx, hy)
        return (
            f"{self.ascii_board()}\n\n"
            f"Legend: H = the snake's head, o = its body, A = the apple, # = the wall, . = empty.\n"
            f"The snake is {len(self.body)} cells long and is currently moving {self.dir}.\n"
            f"Moving up decreases the row, moving down increases it; left decreases the column, right increases it.\n"
            f"The head is at column {hx}, row {hy}. The apple is at column {ax}, row {ay}.\n"
            f"The snake dies if it hits the wall or its own body."
        )

    def html_board(self):
        """1マス1セルの HTML テーブル。AA だと "#....oo...#" が数トークンにまとまって
        列の対応が消えるので、td で囲んでマスごとにトークンを切る。"""
        rows = []
        for y in range(self.h):
            cells = []
            for x in range(self.w):
                if (x, y) == self.body[0]:
                    c = "H"
                elif (x, y) in self.body[1:]:
                    c = "o"
                elif self.apple and (x, y) == self.apple:
                    c = "A"
                else:
                    c = "."
                cells.append(f"<td>{c}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        return "<table>\n" + "\n".join(rows) + "\n</table>"

    def table_state(self, minimal=True):
        """HTML テーブル版。既定は座標なし（画像版と同じ情報量）。"""
        head = (
            f"{self.html_board()}\n\n"
            f"The table is the board. H = the snake's head, o = its body, A = the apple, . = empty. "
            f"Outside the table is the wall.\n"
            f"The snake is {len(self.body)} cells long and is currently moving {self.dir}.\n"
        )
        if minimal:
            return (head
                    + "Up is towards the top row, down towards the bottom row, left and right as shown.\n"
                    + "The snake dies if it hits the wall or its own body.")
        hx, hy = self.body[0]
        ax, ay = self.apple if self.apple else (hx, hy)
        return (head
                + "Moving up decreases the row, moving down increases it; "
                  "left decreases the column, right increases it.\n"
                + f"The head is at column {hx}, row {hy}. The apple is at column {ax}, row {ay}.\n"
                + "The snake dies if it hits the wall or its own body.")

    # 盤面の表し方いろいろ。AA は "....." がまとめて1トークンになり列の対応が消えるので、
    # マスごとに区切りを入れる書き方を何種類か用意して比べる。
    BOARD_STYLES = ("ascii", "table", "table_open", "space", "comma", "tab", "json", "markdown")

    def cell_char(self, x, y):
        if (x, y) == self.body[0]:
            return "H"
        if (x, y) in self.body[1:]:
            return "o"
        if self.apple and (x, y) == self.apple:
            return "A"
        return "."

    def grid_chars(self):
        return [[self.cell_char(x, y) for x in range(self.w)] for y in range(self.h)]

    def styled_board(self, style):
        g = self.grid_chars()
        if style == "ascii":
            return self.ascii_board()
        if style == "table":
            return ("<table>\n"
                    + "\n".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in g)
                    + "\n</table>")
        if style == "table_open":
            return ("<table>\n"
                    + "\n".join("<tr>" + "".join(f"<td>{c}" for c in row) for row in g)
                    + "\n</table>")
        if style == "space":
            return "\n".join(" ".join(row) for row in g)
        if style == "comma":
            return "\n".join(",".join(row) for row in g)
        if style == "tab":
            return "\n".join("\t".join(row) for row in g)
        if style == "markdown":
            head = "| " + " | ".join(str(x) for x in range(self.w)) + " |"
            sep = "|" + "---|" * self.w
            body = "\n".join("| " + " | ".join(row) + " |" for row in g)
            return "\n".join([head, sep, body])
        if style == "json":
            return "[\n" + ",\n".join("  [" + ", ".join(f'"{c}"' for c in row) + "]" for row in g) + "\n]"
        raise ValueError(style)

    def styled_state(self, style, minimal=True):
        """盤面の書き方だけ変えた state。情報量は画像版とそろえる（minimal=True）。"""
        what = {
            "ascii": "The board is drawn with characters, # is the wall.",
            "table": "The board is the HTML table below.",
            "table_open": "The board is the HTML table below (closing tags omitted).",
            "space": "The board is the grid below, one cell per character, columns separated by spaces.",
            "comma": "The board is the grid below, one cell per character, columns separated by commas.",
            "tab": "The board is the grid below, one cell per character, columns separated by tabs.",
            "json": "The board is the JSON array below, one array per row.",
            "markdown": "The board is the markdown table below; the header row is the column number.",
        }[style]
        head = (f"{self.styled_board(style)}\n\n"
                f"{what} H = the snake's head, o = its body, A = the apple, . = empty. "
                f"Outside the board is the wall.\n"
                f"The snake is {len(self.body)} cells long and is currently moving {self.dir}.\n")
        if minimal:
            return (head
                    + "Up is towards the top row, down towards the bottom row, left and right as shown.\n"
                    + "The snake dies if it hits the wall or its own body.")
        hx, hy = self.body[0]
        ax, ay = self.apple if self.apple else (hx, hy)
        return (head
                + "Moving up decreases the row, moving down increases it; "
                  "left decreases the column, right increases it.\n"
                + f"The head is at column {hx}, row {hy}. The apple is at column {ax}, row {ay}.\n"
                + "The snake dies if it hits the wall or its own body.")

    def image_state(self):
        return (
            "The picture shows a Snake board. The dark green square is the snake's head, "
            "the lighter green squares are its body, the red circle is the apple, "
            "and the black border is the wall.\n"
            f"The snake is {len(self.body)} cells long and is currently moving {self.dir}.\n"
            "Up is towards the top of the picture, down towards the bottom, left and right as seen.\n"
            "The snake dies if it hits the wall or its own body."
        )

    def png(self, cell=32, border=1):
        """盤面を PNG バイト列に。"""
        w, h = (self.w + 2) * cell, (self.h + 2) * cell
        bg, wall = (245, 245, 240), (20, 20, 20)
        head, body, apple, grid = (20, 110, 40), (110, 190, 110), (210, 50, 40), (220, 220, 214)
        px = [[bg] * w for _ in range(h)]

        def fill(x0, y0, x1, y1, c):
            for y in range(max(0, y0), min(h, y1)):
                row = px[y]
                for x in range(max(0, x0), min(w, x1)):
                    row[x] = c

        fill(0, 0, w, cell, wall)
        fill(0, h - cell, w, h, wall)
        fill(0, 0, cell, h, wall)
        fill(w - cell, 0, w, h, wall)
        for i in range(self.w + 1):  # 目盛りの薄い線
            fill(cell + i * cell - border, cell, cell + i * cell, h - cell, grid)
        for i in range(self.h + 1):
            fill(cell, cell + i * cell - border, w - cell, cell + i * cell, grid)

        def cellfill(cx, cy, c):
            x0, y0 = (cx + 1) * cell, (cy + 1) * cell
            fill(x0 + border, y0 + border, x0 + cell - border, y0 + cell - border, c)

        def celldisc(cx, cy, c):
            """マスの中に円を描く（りんご）。"""
            x0, y0 = (cx + 1) * cell, (cy + 1) * cell
            r = cell / 2 - border - 1
            ccx, ccy = x0 + cell / 2, y0 + cell / 2
            for y in range(y0, y0 + cell):
                dy = y + 0.5 - ccy
                if abs(dy) > r:
                    continue
                half = (r * r - dy * dy) ** 0.5
                fill(int(ccx - half), y, int(ccx + half) + 1, y + 1, c)

        for x, y in self.body[1:]:
            cellfill(x, y, body)
        cellfill(*self.body[0], head)
        if self.apple:
            celldisc(*self.apple, apple)

        raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in px)
        def chunk(tag, data):
            return (struct.pack(">I", len(data)) + tag + data +
                    struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6))
                + chunk(b"IEND", b""))

    def data_url(self, cell=32):
        return "data:image/png;base64," + base64.b64encode(self.png(cell)).decode()

    # ------------------------------------------------------------ 参考用の方策
    def safe_dirs(self):
        out = []
        for d in DIRS:
            if d == OPPOSITE[self.dir]:
                continue
            dx, dy = DELTA[d]
            hx, hy = self.body[0]
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.body[:-1]:
                out.append(d)
        return out

    def greedy_dir(self):
        """りんごに近づく安全な手。なければ安全な手。それも無ければ直進。"""
        safe = self.safe_dirs()
        if not safe:
            return self.dir
        hx, hy = self.body[0]
        ax, ay = self.apple
        return min(safe, key=lambda d: abs(hx + DELTA[d][0] - ax) + abs(hy + DELTA[d][1] - ay))
