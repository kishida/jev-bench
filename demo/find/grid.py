"""8×6 のマス目に鳥・うさぎ・虫を置いて、「虫のいるマスはどれ？」と聞くための盤面。

Snake や迷路と同じく、依存を増やさないよう PNG は自前で書く。

マスの名前は列が A〜H、行が 1〜6 で "H3" のように書く。同じ盤面を
  - AA（文字）
  - 画像（マス名あり）
  - 画像（マス名なし）
の3通りで渡して比べられるようにしてある。画像でマス名を消すと、モデルは見た位置を自分で
座標に直さなければならない。
"""
import base64
import random
import struct
import zlib

COLS, ROWS = 8, 6
COL_NAMES = "ABCDEFGH"
BIRD, RABBIT, BUG = "bird", "rabbit", "bug"
COUNTS = {BIRD: 4, RABBIT: 4, BUG: 3}
CHARS = {BIRD: "B", RABBIT: "R", BUG: "X"}


def cell_name(x, y):
    return f"{COL_NAMES[x]}{y + 1}"


def all_cells():
    return [cell_name(x, y) for y in range(ROWS) for x in range(COLS)]


class Scene:
    """どのマスに何がいるか。left-top が a1、x は右、y は下。"""

    def __init__(self, seed=0, rng=None):
        self.rng = rng or random.Random(seed)
        self.creatures = []          # [(kind, x, y), ...]
        self._place()

    def _place(self):
        spots = [(x, y) for y in range(ROWS) for x in range(COLS)]
        self.rng.shuffle(spots)
        i = 0
        for kind, n in COUNTS.items():
            for _ in range(n):
                x, y = spots[i]
                i += 1
                self.creatures.append((kind, x, y))

    @classmethod
    def from_positions(cls, positions):
        """実数座標 [(kind, fx, fy), ...] から作る。マスは四捨五入、端数は描画のずれに回す。
        ライブデモが「なめらかに動いている途中の一瞬」を出題するのに使う。"""
        self = cls.__new__(cls)
        self.rng = random.Random(0)
        self.creatures = []
        self.offsets = {}
        for i, (kind, fx, fy) in enumerate(positions):
            x = min(COLS - 1, max(0, round(fx)))
            y = min(ROWS - 1, max(0, round(fy)))
            self.creatures.append((kind, x, y))
            self.offsets[i] = (fx - x, fy - y)
        return self

    # ------------------------------------------------------------ 参照
    def at(self, x, y):
        for kind, cx, cy in self.creatures:
            if (cx, cy) == (x, y):
                return kind
        return None

    def bug_cells(self):
        return [cell_name(x, y) for kind, x, y in self.creatures if kind == BUG]

    def occupied_cells(self):
        return [cell_name(x, y) for kind, x, y in self.creatures]

    def distance_to_bug(self, name):
        """選んだマスから一番近い虫までのチェビシェフ距離（斜めも1）。"""
        x, y = COL_NAMES.index(name[0]), int(name[1:]) - 1
        return min(max(abs(x - bx), abs(y - by))
                   for kind, bx, by in self.creatures if kind == BUG)

    # ------------------------------------------------------------ AA
    def ascii_board(self):
        head = "   " + " ".join(COL_NAMES)
        rows = []
        for y in range(ROWS):
            cells = [CHARS.get(self.at(x, y), ".") for x in range(COLS)]
            rows.append(f"{y + 1}  " + " ".join(cells))
        return "\n".join([head] + rows)

    def text_state(self):
        return (
            f"{self.ascii_board()}\n\n"
            "The grid above is a field seen from above, 8 columns (A to H) by 6 rows (1 to 6).\n"
            "B = a bird, R = a rabbit, X = a bug, . = empty grass.\n"
            "A cell is named by its column letter followed by its row number, for example H3 is "
            "the cell in column H, row 3.\n"
            "There are 4 birds, 4 rabbits and 3 bugs."
        )

    def image_state(self, labelled):
        where = ("Each cell has its name printed in its top-left corner."
                 if labelled else
                 "The columns are A to H from left to right and the rows are 1 to 6 from top to "
                 "bottom, so the top-left cell is A1 and the bottom-right cell is H6.")
        return (
            "The picture shows a field seen from above, divided into 8 columns by 6 rows.\n"
            "The birds are blue and have a pointed beak, the rabbits are grey with two long ears, "
            "and the bugs are dark red, round, with six legs and two antennae.\n"
            f"{where}\n"
            "There are 4 birds, 4 rabbits and 3 bugs."
        )

    # ------------------------------------------------------------ 画像
    def png(self, cell=64, labelled=False, offsets=None):
        offsets = offsets if offsets is not None else getattr(self, "offsets", None)
        """offsets は {(kind, index): (dx, dy)} の実数マス単位のずれ（ライブ表示用）。"""
        w, h = COLS * cell, ROWS * cell
        grass = (238, 241, 232)
        line = (196, 201, 188)
        label = (150, 155, 143)
        px = [[grass] * w for _ in range(h)]

        def fill(x0, y0, x1, y1, c):
            for yy in range(max(0, int(y0)), min(h, int(y1))):
                row = px[yy]
                for xx in range(max(0, int(x0)), min(w, int(x1))):
                    row[xx] = c

        def disc(cx, cy, r, c):
            for dy in range(-int(r), int(r) + 1):
                half = int((r * r - dy * dy) ** 0.5)
                fill(cx - half, cy + dy, cx + half + 1, cy + dy + 1, c)

        for i in range(1, COLS):
            fill(i * cell - 1, 0, i * cell + 1, h, line)
        for i in range(1, ROWS):
            fill(0, i * cell - 1, w, i * cell + 1, line)

        if labelled:
            for y in range(ROWS):
                for x in range(COLS):
                    _text(fill, x * cell + 5, y * cell + 5, cell_name(x, y), label)

        for idx, (kind, x, y) in enumerate(self.creatures):
            dx, dy = (offsets or {}).get(idx, (0.0, 0.0))
            cx = int((x + dx + 0.5) * cell)
            cy = int((y + dy + 0.5) * cell)
            if kind == BIRD:
                _bird(fill, disc, cx, cy, cell)
            elif kind == RABBIT:
                _rabbit(fill, disc, cx, cy, cell)
            else:
                _bug(fill, disc, cx, cy, cell)

        raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in px)

        def chunk(tag, data):
            return (struct.pack(">I", len(data)) + tag + data +
                    struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6))
                + chunk(b"IEND", b""))

    def data_url(self, cell=64, labelled=False, offsets=None):
        return ("data:image/png;base64,"
                + base64.b64encode(self.png(cell, labelled, offsets)).decode())


# ---------------------------------------------------------------- 生き物の絵
# 色だけでなく形でも見分けられるようにする（Snake でりんごを丸にしたのと同じ理由）。

def _bird(fill, disc, cx, cy, cell):
    body = (48, 96, 200)
    r = cell // 5
    disc(cx, cy, r, body)                                   # 胴
    disc(cx, cy - r, r * 2 // 3, body)                      # 頭
    for k in range(r // 2 + 2):                             # くちばし（三角）
        fill(cx + r * 2 // 3 + k, cy - r - k // 2, cx + r * 2 // 3 + k + 1,
             cy - r + k // 2 + 1, (232, 150, 40))
    fill(cx - r, cy - 2, cx, cy + 2, (28, 66, 150))         # 翼


def _rabbit(fill, disc, cx, cy, cell):
    body = (140, 142, 138)
    r = cell // 5
    disc(cx, cy + r // 3, r, body)                          # 胴
    disc(cx, cy - r // 2, r * 2 // 3, body)                 # 頭
    for sx in (-1, 1):                                      # 長い耳
        ex = cx + sx * r // 3
        fill(ex - 2, cy - r * 2, ex + 2, cy - r, body)
    disc(cx + r, cy + r // 2, max(2, r // 4), (200, 200, 196))   # しっぽ


def _bug(fill, disc, cx, cy, cell):
    body = (150, 40, 40)
    r = cell // 5
    disc(cx, cy, r, body)                                   # 丸い胴
    for k in range(3):                                      # 脚6本
        yy = cy - r // 2 + k * (r // 2)
        fill(cx - r * 2, yy - 1, cx - r, yy + 1, body)
        fill(cx + r, yy - 1, cx + r * 2, yy + 1, body)
    for sx in (-1, 1):                                      # 触角
        fill(cx + sx * r // 2 - 1, cy - r * 2, cx + sx * r // 2 + 1, cy - r, body)


# ---------------------------------------------------------------- 極小フォント
# マス名だけ描ければよいので、5×7 の A〜H と 1〜6 を持つ。
_FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
}


def _text(fill, x0, y0, s, colour, scale=2):
    for ch in s:
        glyph = _FONT.get(ch)
        if glyph:
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        fill(x0 + gx * scale, y0 + gy * scale,
                             x0 + (gx + 1) * scale, y0 + (gy + 1) * scale, colour)
        x0 += 6 * scale
