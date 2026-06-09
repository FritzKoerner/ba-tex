#!/usr/bin/env python3
"""Generate architecture diagram (SVG → PDF) for thesis Abbildung 4.2.

Depth → Shared CNN → Concat with State → Actor/Critic MLP heads.
Style matches pipeline diagram (Abbildung 4.1).
"""

import os
import subprocess

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

W, H = 700, 200
FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

DEFS = """\
  <defs>
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
    </linearGradient>
    <linearGradient id="gg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#60aa64" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#60aa64" stop-opacity="0.06"/>
    </linearGradient>
    <marker id="arr" viewBox="0 0 10 7" refX="10" refY="3.5"
      markerWidth="10" markerHeight="7" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#888"/>
    </marker>
  </defs>"""

C = {
    "purple": ("#8752f6", "#4a2a8a", "gp"),
    "blue":   ("#5698f9", "#2a5a9a", "gb"),
    "orange": ("#e8a340", "#e8a340", "go"),
    "green":  ("#60aa64", "#3a6a3e", "gg"),
}


def rbox(x, y, w, h, color, sw=1.6):
    s, _, g = C[color]
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
            f'fill="url(#{g})" stroke="{s}" stroke-width="{sw}"/>')


def txt(x, y, text, color="purple", size=11, weight="600"):
    _, fill, _ = C[color]
    return (f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}">'
            f'{text}</text>')


def note(x, y, text, fill="#888", size=9):
    return (f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="{size}" fill="{fill}">{text}</text>')


def arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="1.3" marker-end="url(#arr)"/>')


def ln(x1, y1, x2, y2, color="#aaa", sw=1.2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}"/>')


parts = []
p = parts.append

# Y positions
Y_C = 52      # critic / CNN path
Y_A = 152     # actor / state path
Y_M = 102     # merge midpoint

# == INPUTS ================================================================

# Depth image thumbnail
DI_X, DI_W, DI_H = 15, 65, 48
di_y = Y_C - DI_H // 2
p(f'<rect x="{DI_X}" y="{di_y}" width="{DI_W}" height="{DI_H}" '
  f'rx="4" fill="#ddd" stroke="#bbb" stroke-width="1"/>')
bands = ["#333", "#555", "#777", "#999", "#bbb"]
bh = (DI_H - 4) // len(bands)
by = di_y + 2
for shade in bands:
    p(f'<rect x="{DI_X + 2}" y="{by}" width="{DI_W - 4}" height="{bh}" '
      f'rx="1" fill="{shade}"/>')
    by += bh
p(note(DI_X + DI_W // 2, di_y - 5, "Tiefenbild"))
p(note(DI_X + DI_W // 2, di_y + DI_H + 11, "64 × 64"))

# State vector box
ST_W, ST_H = 100, 28
st_y = Y_A - ST_H // 2
p(rbox(DI_X, st_y, ST_W, ST_H, "purple"))
p(txt(DI_X + ST_W // 2, Y_A + 4, "Zustandsvektor", "purple", size=9.5))

INP_R = DI_X + DI_W  # 80
ST_R = DI_X + ST_W   # right edge of state box

# Arrow from depth image
p(arrow(INP_R, Y_C, INP_R + 22, Y_C))

# == CONV LAYERS (tall narrow blue rectangles, decreasing height) ==========

CX0 = INP_R + 36   # 116
CW = 30
CGAP = 14

convs = [
    ("Conv 32", "8×8, stride 4", 80),
    ("Conv 64", "4×4, stride 2", 68),
    ("Conv 128", "3×3, stride 1", 56),
]

cx = CX0
for label, sub_label, h in convs:
    y_top = Y_C - h // 2
    s, _, g = C["blue"]
    p(f'<rect x="{cx}" y="{y_top}" width="{CW}" height="{h}" rx="4" '
      f'fill="url(#{g})" stroke="{s}" stroke-width="1.4"/>')
    tcx = cx + CW // 2
    p(f'<text x="{tcx - 5}" y="{Y_C}" text-anchor="middle" '
      f'font-size="9" fill="#2a5a9a" font-weight="600" '
      f'transform="rotate(-90, {tcx - 5}, {Y_C})">{label}</text>')
    p(f'<text x="{tcx + 5}" y="{Y_C}" text-anchor="middle" '
      f'font-size="7.5" fill="#2a5a9a" '
      f'transform="rotate(-90, {tcx + 5}, {Y_C})">{sub_label}</text>')
    cx += CW + CGAP

# Small connector lines between convs
for i in range(len(convs) - 1):
    lx = CX0 + (i + 1) * (CW + CGAP) - CGAP
    p(ln(lx, Y_C, lx + CGAP, Y_C, "#888"))

# Line to MaxPool (no arrowhead inside CNN bracket)
last_r = CX0 + len(convs) * (CW + CGAP) - CGAP
p(ln(last_r, Y_C, last_r + 12 + CGAP, Y_C, "#888"))

# Max Pool (tall narrow, rotated text like conv layers)
MP_X = last_r + 12 + CGAP
MP_W, MP_H = CW, 46
mp_y = Y_C - MP_H // 2
s, _, g = C["blue"]
p(f'<rect x="{MP_X}" y="{mp_y}" width="{MP_W}" height="{MP_H}" rx="4" '
  f'fill="url(#{g})" stroke="{s}" stroke-width="1.4"/>')
mp_cx = MP_X + MP_W // 2
p(f'<text x="{mp_cx}" y="{Y_C}" text-anchor="middle" '
  f'dominant-baseline="central" '
  f'font-size="9" fill="#2a5a9a" font-weight="600" '
  f'transform="rotate(-90, {mp_cx}, {Y_C})">MaxPool</text>')
MP_R = MP_X + MP_W

# Shared CNN bracket (green dashed) with label inside
br_x, br_r = CX0 - 6, MP_R + 6
br_y = Y_C - convs[0][2] // 2 - 12
br_bottom = Y_C + convs[0][2] // 2 + 10
br_h = br_bottom - br_y
p(f'<rect x="{br_x}" y="{br_y}" width="{br_r - br_x}" height="{br_h}" '
  f'rx="4" fill="none" stroke="#60aa64" stroke-width="1.1" '
  f'stroke-dasharray="5,3"/>')
p(txt((br_x + br_r) // 2, br_bottom - 4, "CNN", "green", size=9, weight="700"))

# == STATE PATH: Emp. Norm =================================================

EN_X = CX0 + 50
EN_W, EN_H = 100, 26
p(arrow(ST_R, Y_A, EN_X - 14, Y_A))
p(rbox(EN_X, Y_A - EN_H // 2, EN_W, EN_H, "purple"))
p(txt(EN_X + EN_W // 2, Y_A + 4, "Normalisierung", "purple", size=9))
EN_R = EN_X + EN_W

# == MERGE BUS =============================================================

BUS = MP_R + 28
p(ln(MP_R, Y_C, BUS, Y_C))
p(ln(EN_R, Y_A, BUS, Y_A))
p(ln(BUS, Y_C, BUS, Y_A))
# == SPLIT BUS =============================================================

SP = BUS + 18
p(ln(BUS, Y_M, SP, Y_M, "#888"))
p(ln(SP, Y_C, SP, Y_A))
p(arrow(SP, Y_C, SP + 26, Y_C))
p(arrow(SP, Y_A, SP + 26, Y_A))

# == FC CHAIN HELPERS ======================================================

FC_W, FC_H, FC_GAP = 48, 24, 16


def fc_box(x, y, label, color="blue"):
    p(rbox(x, y - FC_H // 2, FC_W, FC_H, color))
    p(txt(x + FC_W // 2, y + 4, label, color, size=9))
    return x + FC_W


# == CRITIC BRANCH (y = Y_C) ===============================================

fc_x0 = SP + 46

# Background box
critic_fc_w = 2 * FC_W + FC_GAP
p(f'<rect x="{fc_x0 - 10}" y="{Y_C - FC_H // 2 - 20}" '
  f'width="{critic_fc_w + 20}" height="{FC_H + 34}" rx="6" '
  f'fill="#5698f9" fill-opacity="0.04" stroke="#5698f9" '
  f'stroke-width="0.7" stroke-opacity="0.25"/>')
p(txt(fc_x0 + critic_fc_w // 2, Y_C - FC_H // 2 - 8, "Critic",
      "blue", size=10, weight="700"))

# FC(32) → FC(32)
x = fc_box(fc_x0, Y_C, "FC (32)")
p(ln(x, Y_C, x + FC_GAP, Y_C, "#888"))
x = fc_box(x + FC_GAP, Y_C, "FC (32)")

# → V(s) output
p(arrow(x, Y_C, x + 32, Y_C))
p(txt(x + 46, Y_C + 4, "V(s)", "orange", size=10, weight="700"))

# == ACTOR BRANCH (y = Y_A) ================================================

# Background box
actor_fc_w = 3 * FC_W + 2 * FC_GAP + FC_GAP + 56  # 3 FC + Tanh
p(f'<rect x="{fc_x0 - 10}" y="{Y_A - FC_H // 2 - 20}" '
  f'width="{actor_fc_w + 20}" height="{FC_H + 34}" rx="6" '
  f'fill="#5698f9" fill-opacity="0.04" stroke="#5698f9" '
  f'stroke-width="0.7" stroke-opacity="0.25"/>')
p(txt(fc_x0 + actor_fc_w // 2, Y_A - FC_H // 2 - 8, "Actor",
      "blue", size=10, weight="700"))

# FC(64) → FC(64) → FC(64) → Tanh
x = fc_box(fc_x0, Y_A, "FC (64)")
p(ln(x, Y_A, x + FC_GAP, Y_A, "#888"))
x = fc_box(x + FC_GAP, Y_A, "FC (64)")
p(ln(x, Y_A, x + FC_GAP, Y_A, "#888"))
x = fc_box(x + FC_GAP, Y_A, "FC (64)")
p(ln(x, Y_A, x + FC_GAP, Y_A, "#888"))

# Tanh-Gaussian box (slightly wider)
TW = 56
p(rbox(x + FC_GAP, Y_A - FC_H // 2, TW, FC_H, "blue"))
p(txt(x + FC_GAP + TW // 2, Y_A + 4, "Tanh", "blue", size=9))
x = x + FC_GAP + TW

# → a_t output
p(arrow(x, Y_A, x + 32, Y_A))
p(txt(x + 44, Y_A + 4, "a_t", "orange", size=10, weight="700"))

# == ASSEMBLE ==============================================================

svg = f"""\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     font-family="{FONT}">
{DEFS}
  <rect width="{W}" height="{H}" fill="white" rx="6"/>
  {"".join(parts)}
</svg>"""

svg_path = os.path.join(OUT_DIR, "architecture.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved {svg_path}")

pdf_path = os.path.join(OUT_DIR, "architecture.pdf")
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
