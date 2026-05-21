#!/usr/bin/env python3
"""Generate observation pipeline diagram (SVG) for thesis.

Shows how depth images and state vectors are captured, processed, and
assembled into a TensorDict that feeds the actor network.

Usage:  python gen_obs_pipeline.py
Output: obs_pipeline.svg (+ obs_pipeline.pdf if inkscape/cairosvg available)
"""

import os
import subprocess

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Layout constants ────────────────────────────────────────────────
W, H = 780, 185
BH = 40          # standard box height
RX = 5           # corner radius

# Row centres
Y1 = 44          # depth path
Y2 = 141         # state path

# Column positions (x-start of each box)
COL_SRC = 12     # source boxes
COL_PROC = 165   # processing boxes
COL_TENS = 330   # tensor boxes
COL_BUS = 492    # vertical merge bus
COL_ACT = 540    # actor box
COL_OUT = 700    # output box

# Box widths
W_SRC = 112
W_PROC = 122
W_TENS = 128
W_ACT = 120
W_OUT = 62

FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

# ── Colour palette (matches reference SVG) ──────────────────────────
GRAD_DEFS = """
    <linearGradient id="gp" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#8752f6" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#8752f6" stop-opacity="0.06"/>
    </linearGradient>
    <linearGradient id="gb" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5698f9" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#5698f9" stop-opacity="0.06"/>
    </linearGradient>
    <linearGradient id="go" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#e8a340" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#e8a340" stop-opacity="0.06"/>
    </linearGradient>"""

ARROW_MARKER = """
    <marker id="arr" viewBox="0 0 10 7" refX="10" refY="3.5"
      markerWidth="10" markerHeight="7" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#888"/>
    </marker>"""

COLORS = {
    "purple": ("#8752f6", "#4a2a8a", "gp"),
    "blue":   ("#5698f9", "#2a5a9a", "gb"),
    "orange": ("#e8a340", "#8a6420", "go"),
}


def box(x, y, w, h, color="purple"):
    stroke, _, grad = COLORS[color]
    sw = 1.6 if color == "blue" else 1.4
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{RX}" fill="url(#{grad})" stroke="{stroke}" '
            f'stroke-width="{sw}"/>')


def txt(x, y, text, size=13, weight="600", color="purple"):
    _, fill, _ = COLORS.get(color, COLORS["purple"])
    return (f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}">{text}</text>')


def sub(x, y, text):
    return (f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="9.5" fill="#888">{text}</text>')


def arr(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="1.2" marker-end="url(#arr)"/>')


def line(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="5,3"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="1.2"{d}/>')


# ── Compose SVG ─────────────────────────────────────────────────────
def cx(x, w):
    return x + w / 2

parts = []
p = parts.append

# ── Row 1: Depth path ──────────────────────────────────────────────
y1t = Y1 - BH // 2
p(box(COL_SRC, y1t, W_SRC, BH))
p(txt(cx(COL_SRC, W_SRC), Y1 - 3, "Tiefenkamera"))
p(sub(cx(COL_SRC, W_SRC), Y1 + 10, "64×64 px, 90° FoV"))

p(arr(COL_SRC + W_SRC, Y1, COL_PROC - 8, Y1))

p(box(COL_PROC, y1t, W_PROC, BH))
p(txt(cx(COL_PROC, W_PROC), Y1 - 3, "Normalisierung"))
p(sub(cx(COL_PROC, W_PROC), Y1 + 10, "clamp(d / 20 m, 0, 1)"))

p(arr(COL_PROC + W_PROC, Y1, COL_TENS - 8, Y1))

p(box(COL_TENS, y1t, W_TENS, BH))
p(txt(cx(COL_TENS, W_TENS), Y1 - 3, "Tiefenbild D", size=12.5))
p(f'<text x="{cx(COL_TENS, W_TENS) + 53}" y="{Y1}" '
  f'text-anchor="middle" font-size="9" fill="#4a2a8a" '
  f'font-weight="600">t</text>')
p(sub(cx(COL_TENS, W_TENS), Y1 + 10, "∈ [0, 1], 64×64"))

# connect to bus
p(line(COL_TENS + W_TENS, Y1, COL_BUS, Y1))

# ── Row 2: State path ──────────────────────────────────────────────
y2t = Y2 - BH // 2
p(box(COL_SRC, y2t, W_SRC, BH))
p(txt(cx(COL_SRC, W_SRC), Y2 - 3, "Zustandsgrößen", size=12))
p(sub(cx(COL_SRC, W_SRC), Y2 + 10, "p, q, v, ω, a"))

p(arr(COL_SRC + W_SRC, Y2, COL_PROC - 8, Y2))

p(box(COL_PROC, y2t, W_PROC, BH))
p(txt(cx(COL_PROC, W_PROC), Y2 - 3, "Skalierung &amp; Clip", size=12))
p(sub(cx(COL_PROC, W_PROC), Y2 + 10, "×1/15, ×0,4, ×1/π"))

p(arr(COL_PROC + W_PROC, Y2, COL_TENS - 8, Y2))

p(box(COL_TENS, y2t, W_TENS, BH))
p(txt(cx(COL_TENS, W_TENS), Y2 - 3, "Zustandsvektor o", size=12))
p(f'<text x="{cx(COL_TENS, W_TENS) + 58}" y="{Y2}" '
  f'text-anchor="middle" font-size="9" fill="#4a2a8a" '
  f'font-weight="600">t</text>')
p(sub(cx(COL_TENS, W_TENS), Y2 + 10, "∈ ℝ, 17-dimensional"))

# connect to bus
p(line(COL_TENS + W_TENS, Y2, COL_BUS, Y2))

# ── Vertical merge bus ──────────────────────────────────────────────
mid_y = (Y1 + Y2) // 2
p(line(COL_BUS, Y1, COL_BUS, Y2))
p(arr(COL_BUS, mid_y, COL_ACT - 8, mid_y))

# ── Actor box ───────────────────────────────────────────────────────
act_h = 56
act_y = mid_y - act_h // 2
p(box(COL_ACT, act_y, W_ACT, act_h, "blue"))
p(txt(cx(COL_ACT, W_ACT), mid_y - 3, "Actor-Netz", size=14, color="blue"))
p(f'<text x="{cx(COL_ACT, W_ACT)}" y="{mid_y + 13}" text-anchor="middle" '
  f'font-size="10.5" fill="#2a5a9a" font-weight="600">CNN + MLP</text>')

# ── Output ──────────────────────────────────────────────────────────
p(arr(COL_ACT + W_ACT, mid_y, COL_OUT - 8, mid_y))
out_h = 36
out_y = mid_y - out_h // 2
p(box(COL_OUT, out_y, W_OUT, out_h, "orange"))
p(txt(cx(COL_OUT, W_OUT), mid_y - 2, "a", size=14, color="orange"))
p(f'<text x="{cx(COL_OUT, W_OUT) + 8}" y="{mid_y + 2}" text-anchor="middle" '
  f'font-size="9.5" fill="#8a6420" font-weight="600">t</text>')
p(sub(cx(COL_OUT, W_OUT), mid_y + 12, "[−1,1]⁴"))

# ── Assemble ────────────────────────────────────────────────────────
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     font-family="{FONT}">
  <defs>{GRAD_DEFS}{ARROW_MARKER}
  </defs>
  <rect width="{W}" height="{H}" fill="#fafafa" rx="4"/>
  {"".join(parts)}
</svg>"""

svg_path = os.path.join(OUT_DIR, "obs_pipeline.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved {svg_path}")

# ── Try PDF conversion ──────────────────────────────────────────────
pdf_path = os.path.join(OUT_DIR, "obs_pipeline.pdf")
converted = False

try:
    import cairosvg
    cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
    converted = True
except ImportError:
    pass

if not converted:
    try:
        subprocess.run(
            ["inkscape", svg_path, "--export-type=pdf",
             f"--export-filename={pdf_path}"],
            check=True, capture_output=True,
        )
        converted = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

if converted:
    print(f"Saved {pdf_path}")
else:
    print("PDF conversion skipped (install cairosvg or inkscape)")
