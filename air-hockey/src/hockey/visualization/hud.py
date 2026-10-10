"""2-D heads-up display (scoreboard, AI decision panel, legend, captions) drawn with PIL."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
)

BLUE = (40, 110, 240)
RED = (225, 50, 45)
PANEL = (12, 16, 26, 185)
WHITE = (245, 247, 250)
GREY = (170, 178, 190)


@lru_cache(maxsize=32)
def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("Japanese font (IPAexGothic) not found; install fonts-ipaexfont-gothic")


@dataclass
class HudState:
    home_name: str = "AI"
    away_name: str = "Opponent"
    score: tuple[int, int] = (0, 0)
    time_text: str = ""
    strategy: str = ""
    mode: str = ""
    lines: list[str] = field(default_factory=list)
    caption: str = ""
    speed_label: str = "再生速度 1.0x（実時間）"
    title: str = ""
    legend: bool = True
    labels: list[tuple[float, float, str]] = field(default_factory=list)  # pixel x, y, text


LEGEND = (
    ((25, 215, 90), "予測軌道（物理モデル）"),
    ((255, 140, 15), "予測打撃点 / 防御点"),
    ((255, 215, 25), "狙いのシュート経路"),
    ((50, 115, 255), "ロボット目標位置"),
)


def _rounded(draw: ImageDraw.ImageDraw, box, fill, radius: int = 10) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def draw_hud(frame: np.ndarray, st: HudState) -> np.ndarray:
    h, w = frame.shape[:2]
    s = w / 1280.0
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # Scoreboard
    bw, bh = int(560 * s), int(58 * s)
    x0, y0 = (w - bw) // 2, int(12 * s)
    _rounded(d, (x0, y0, x0 + bw, y0 + bh), PANEL, int(12 * s))
    side = int(200 * s)
    _rounded(d, (x0 + 6, y0 + 6, x0 + side, y0 + bh - 6), (*BLUE, 235), int(8 * s))
    _rounded(d, (x0 + bw - side, y0 + 6, x0 + bw - 6, y0 + bh - 6), (*RED, 235), int(8 * s))
    f_name, f_score = font(int(20 * s)), font(int(34 * s))
    d.text(
        (x0 + 6 + (side - 6) / 2, y0 + bh / 2), st.home_name, font=f_name, fill=WHITE, anchor="mm"
    )
    d.text(
        (x0 + bw - side + (side - 6) / 2, y0 + bh / 2),
        st.away_name,
        font=f_name,
        fill=WHITE,
        anchor="mm",
    )
    d.text(
        (w / 2, y0 + bh / 2),
        f"{st.score[0]}  :  {st.score[1]}",
        font=f_score,
        fill=WHITE,
        anchor="mm",
    )
    if st.time_text:
        d.text(
            (w / 2, y0 + bh + int(14 * s)),
            st.time_text,
            font=font(int(16 * s)),
            fill=GREY,
            anchor="mm",
        )

    # Title (top-left) and playback speed (top-right)
    if st.title:
        d.text((int(18 * s), int(18 * s)), st.title, font=font(int(18 * s)), fill=WHITE)
    if st.speed_label:
        f = font(int(15 * s))
        tw = d.textlength(st.speed_label, font=f)
        _rounded(
            d, (w - tw - int(30 * s), int(14 * s), w - int(12 * s), int(42 * s)), PANEL, int(8 * s)
        )
        d.text((w - int(21 * s), int(28 * s)), st.speed_label, font=f, fill=WHITE, anchor="rm")

    # AI decision panel
    panel_lines = []
    if st.strategy:
        panel_lines.append(("AI戦略", st.strategy))
    if st.mode:
        panel_lines.append(("動作", st.mode))
    for line in st.lines:
        k, _, v = line.partition("|")
        panel_lines.append((k, v))
    if panel_lines:
        f_k, f_v = font(int(15 * s)), font(int(19 * s))
        lh = int(28 * s)
        px0, py0 = int(16 * s), int(86 * s)
        pw = int(340 * s)
        _rounded(
            d, (px0, py0, px0 + pw, py0 + lh * len(panel_lines) + int(14 * s)), PANEL, int(10 * s)
        )
        for i, (k, v) in enumerate(panel_lines):
            yy = py0 + int(9 * s) + i * lh
            d.text((px0 + int(12 * s), yy + int(3 * s)), k, font=f_k, fill=GREY)
            d.text((px0 + int(122 * s), yy), v, font=f_v, fill=WHITE)

    # Legend
    if st.legend:
        f = font(int(14 * s))
        lh = int(22 * s)
        lx0 = int(16 * s)
        ly0 = h - int(16 * s) - lh * len(LEGEND) - int(10 * s) - (int(60 * s) if st.caption else 0)
        _rounded(
            d,
            (lx0, ly0, lx0 + int(230 * s), ly0 + lh * len(LEGEND) + int(10 * s)),
            PANEL,
            int(8 * s),
        )
        for i, (col, text) in enumerate(LEGEND):
            yy = ly0 + int(6 * s) + i * lh
            d.rounded_rectangle(
                (lx0 + int(10 * s), yy + int(4 * s), lx0 + int(28 * s), yy + int(16 * s)),
                radius=3,
                fill=(*col, 255),
            )
            d.text((lx0 + int(36 * s), yy), text, font=f, fill=WHITE)

    for x, y, text in st.labels:
        f = font(int(14 * s))
        d.text((x, y), text, font=f, fill=WHITE, anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))

    # Caption
    if st.caption:
        f = font(int(24 * s))
        ch = int(50 * s)
        _rounded(
            d,
            (int(80 * s), h - ch - int(14 * s), w - int(80 * s), h - int(14 * s)),
            (0, 0, 0, 175),
            int(10 * s),
        )
        d.text((w / 2, h - int(14 * s) - ch / 2), st.caption, font=f, fill=WHITE, anchor="mm")

    return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))
