#!/usr/bin/env python3
"""Generate pipeline diagram (SVG -> PDF) for thesis Abbildung 4.1.

Slim version of the RL pipeline: stacked observations -> RL-Policy -> PID-Regler.
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
    <marker id="arr-o" viewBox="0 0 10 7" refX="10" refY="3.5"
      markerWidth="10" markerHeight="7" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#e8a340"/>
    </marker>
  </defs>"""

C = {
    "purple": ("#8752f6", "#4a2a8a", "gp"),
    "blue":   ("#5698f9", "#2a5a9a", "gb"),
    "orange": ("#e8a340", "#e8a340", "go"),
    "green":  ("#60aa64", "#3a6a3e", "gg"),
}


def box(x, y, w, h, color, sw=1.6):
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

# -- Observation column layout ---------------------------------------------
OX, OW, OH = 20, 155, 30
GAP = 6
col_cx = OX + OW // 2

# Depth image thumbnail (top)
y = 16
centers = []

DI_W, DI_H = 80, 54
di_x = OX + (OW - DI_W) // 2
di_y = y

p(f'<rect x="{di_x}" y="{di_y}" width="{DI_W}" height="{DI_H}" '
  f'rx="4" fill="#ddd" stroke="#bbb" stroke-width="1"/>')
bands = ["#333", "#555", "#777", "#999", "#bbb"]
bx, bw, bh = di_x + 2, DI_W - 4, 10
by = di_y + 2
for shade in bands:
    p(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
      f'rx="1" fill="{shade}"/>')
    by += bh

di_cy = di_y + DI_H // 2
centers.append(di_cy)
p(note(col_cx, di_y + DI_H + 12, "Tiefenbild"))
y = di_y + DI_H + 24

# Zustandsvektor (bottom)
zv_y = y + 20
p(box(OX, zv_y, OW, OH, "purple"))
zv_cy = zv_y + OH // 2
p(txt(col_cx, zv_cy + 4, "Zustandsvektor", "purple"))
centers.append(zv_cy)
orange_cy = zv_cy

# -- Merge bus -------------------------------------------------------------
BUS = OX + OW + 16
MID = (centers[0] + centers[-1]) // 2

for i, cy in enumerate(centers):
    right_edge = (di_x + DI_W) if i == 0 else (OX + OW)
    p(ln(right_edge, cy, BUS, cy))
p(ln(BUS, centers[0], BUS, centers[-1]))
p(arrow(BUS, MID, BUS + 30, MID))

# -- RL-Policy (blue, prominent) -------------------------------------------
RX = BUS + 58
RW, RH = 160, 66
rcx = RX + RW // 2

p(box(RX, MID - RH // 2, RW, RH, "blue", sw=2))
p(txt(rcx, MID + 5, "RL-Strategie", "blue", size=14, weight="700"))

# -- Arrow -> PID ----------------------------------------------------------
p(arrow(RX + RW, MID, RX + RW + 60, MID))
p(note(RX + RW + 30, MID - 8, "Sollwerte"))

# -- PID-Regler (green) ----------------------------------------------------
PX = RX + RW + 68
PW, PH = 124, 56
pcx = PX + PW // 2

p(box(PX, MID - PH // 2, PW, PH, "green"))
p(txt(pcx, MID + 5, "PID-Regler", "green", size=12))

# -- Output arrow ----------------------------------------------------------
p(arrow(PX + PW, MID, PX + PW + 38, MID))
p(note(PX + PW + 54, MID + 4, "RPM", fill="#3a6a3e", size=10))

# -- Feedback loop (dashed orange, ends right of orange box) ---------------
fb_end_x = BUS + 16
p(f'<path d="M {pcx} {MID + PH // 2} L {pcx} {orange_cy} L {fb_end_x} {orange_cy}" '
  f'fill="none" stroke="#e8a340" stroke-width="1.2" '
  f'stroke-dasharray="5,3" marker-end="url(#arr-o)"/>')
p(note((pcx + fb_end_x) // 2, orange_cy + 12, "Vorherige Aktion", fill="#e8a340", size=10))

# -- Assemble SVG ----------------------------------------------------------
svg = f"""\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     font-family="{FONT}">
{DEFS}
  <rect width="{W}" height="{H}" fill="white" rx="6"/>
  {"".join(parts)}
</svg>"""

svg_path = os.path.join(OUT_DIR, "pipeline.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved {svg_path}")

# -- PDF conversion --------------------------------------------------------
pdf_path = os.path.join(OUT_DIR, "pipeline.pdf")
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
