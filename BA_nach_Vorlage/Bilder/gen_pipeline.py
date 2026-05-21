#!/usr/bin/env python3
"""Generate combined pipeline diagram (SVG → PDF) for thesis.

Single diagram: observations → actor network → action → PID → simulator.

Usage:  python gen_pipeline.py
Output: pipeline.svg (+ pipeline.pdf if cairosvg/inkscape available)
"""

import os
import subprocess

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

W, H = 780, 128
BH = 36
RX = 5
Y1 = 34
Y2 = 94
YM = 64
FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

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
    </linearGradient>
    <linearGradient id="gg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#60aa64" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#60aa64" stop-opacity="0.06"/>
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
    "green":  ("#60aa64", "#3a6a3e", "gg"),
}


def box(x, y, w, h, color="purple"):
    stroke, _, grad = COLORS[color]
    sw = 1.6 if color in ("blue", "green") else 1.4
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{RX}" fill="url(#{grad})" stroke="{stroke}" '
            f'stroke-width="{sw}"/>')


def txt(x, y, text, size=13, weight="600", color="purple"):
    _, fill, _ = COLORS.get(color, COLORS["purple"])
    return (f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}">{text}</text>')


def sub(x, y, text, fill="#888"):
    return (f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="9.5" fill="{fill}">{text}</text>')


def arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="1.2" marker-end="url(#arr)"/>')


def ln(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="1.2"/>')


def cx(x, w):
    return x + w / 2


parts = []
p = parts.append

x = 12

# ── Inputs (two rows) ─────────────────────────────────────────────
SRC_W = 108
src_cx = cx(x, SRC_W)

p(box(x, Y1 - BH // 2, SRC_W, BH))
p(txt(src_cx, Y1 - 3, "Tiefenbild", size=12))
p(sub(src_cx, Y1 + 10, "64×64, [0, 1]"))

p(box(x, Y2 - BH // 2, SRC_W, BH))
p(txt(src_cx, Y2 - 3, "Zustandsvektor", size=12))
p(sub(src_cx, Y2 + 10, "17-dim, normalisiert"))

x += SRC_W  # 120

# ── Merge bus ──────────────────────────────────────────────────────
bus_x = x + 25
p(ln(x, Y1, bus_x, Y1))
p(ln(x, Y2, bus_x, Y2))
p(ln(bus_x, Y1, bus_x, Y2))
p(arrow(bus_x, YM, bus_x + 18, YM))
x = bus_x + 26  # 171

# ── Actor-Netz (blue) ─────────────────────────────────────────────
ACT_W, ACT_H = 108, 46
act_cx = cx(x, ACT_W)
p(box(x, YM - ACT_H // 2, ACT_W, ACT_H, "blue"))
p(txt(act_cx, YM - 4, "Actor-Netz", size=13, color="blue"))
p(sub(act_cx, YM + 10, "CNN + MLP", fill="#2a5a9a"))
x += ACT_W  # 279

p(arrow(x, YM, x + 18, YM))
x += 26  # 305

# ── a_t (orange) ──────────────────────────────────────────────────
AT_W, AT_H = 48, 34
at_cx = cx(x, AT_W)
p(box(x, YM - AT_H // 2, AT_W, AT_H, "orange"))
p(txt(at_cx, YM - 2, "a", size=14, color="orange"))
p(f'<text x="{at_cx + 8}" y="{YM + 1}" text-anchor="middle" '
  f'font-size="9" fill="#8a6420" font-weight="600">t</text>')
p(sub(at_cx, YM + 12, "[−1, 1]"))
x += AT_W  # 353

p(arrow(x, YM, x + 18, YM))
x += 26  # 379

# ── Skalierung (purple) ───────────────────────────────────────────
SK_W = 76
sk_cx = cx(x, SK_W)
p(box(x, YM - BH // 2, SK_W, BH))
p(txt(sk_cx, YM - 3, "Skalierung", size=11.5))
p(sub(sk_cx, YM + 10, "p_soll, ψ_soll"))
x += SK_W  # 455

p(arrow(x, YM, x + 18, YM))
x += 26  # 481

# ── PID-Regler (green) ────────────────────────────────────────────
PID_W = 84
pid_cx = cx(x, PID_W)
p(box(x, YM - BH // 2, PID_W, BH, "green"))
p(txt(pid_cx, YM - 3, "PID-Regler", size=12, color="green"))
p(sub(pid_cx, YM + 10, "kaskadierend", fill="#3a6a3e"))
x += PID_W  # 565

p(arrow(x, YM, x + 18, YM))
x += 26  # 591

# ── Physiksimulator (green) ───────────────────────────────────────
SIM_W = 100
sim_cx = cx(x, SIM_W)
p(box(x, YM - BH // 2, SIM_W, BH, "green"))
p(txt(sim_cx, YM - 3, "Simulator", size=12, color="green"))
p(sub(sim_cx, YM + 10, "300 × Δt = 3,0 s", fill="#3a6a3e"))
x += SIM_W  # 691

p(arrow(x, YM, x + 15, YM))
x += 22  # 713

# ── Output label (no box) ─────────────────────────────────────────
p(txt(x + 10, YM - 1, "s", size=13))
p(f'<text x="{x + 17}" y="{YM + 2}" text-anchor="middle" '
  f'font-size="9" fill="#4a2a8a" font-weight="600">t+1</text>')
p(sub(x + 10, YM + 12, ", r"))

# ── Assemble ───────────────────────────────────────────────────────
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     font-family="{FONT}">
  <defs>{GRAD_DEFS}{ARROW_MARKER}
  </defs>
  <rect width="{W}" height="{H}" fill="#fafafa" rx="4"/>
  {"".join(parts)}
</svg>"""

svg_path = os.path.join(OUT_DIR, "pipeline.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved {svg_path}")

# ── PDF conversion ─────────────────────────────────────────────────
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
