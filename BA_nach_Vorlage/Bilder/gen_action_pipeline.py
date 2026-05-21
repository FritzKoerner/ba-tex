#!/usr/bin/env python3
"""Generate action pipeline diagram (SVG) for thesis.

Shows how the actor network output is sampled, squashed, scaled to
target setpoints, and executed by the cascading PID controller through
300 physics substeps.

Usage:  python gen_action_pipeline.py
Output: action_pipeline.svg (+ action_pipeline.pdf if inkscape/cairosvg available)
"""

import os
import subprocess

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Layout constants ────────────────────────────────────────────────
W, H = 780, 210
BH = 40
RX = 5

# Two rows: neural-network decision (top) → physical execution (bottom)
Y1 = 44          # row 1: actor → sampling → action
Y2 = 155         # row 2: scaling → PID → physics

FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

# ── Colour palette ──────────────────────────────────────────────────
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


def arr(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="1.2" marker-end="url(#arr)"/>')


def line(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="5,3"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="1.2"{d}/>')


def cx(x, w):
    return x + w / 2


# ── Compose SVG ─────────────────────────────────────────────────────
parts = []
p = parts.append

# ════════════════════════════════════════════════════════════════════
# Row 1: Actor → Sampling → Action
# ════════════════════════════════════════════════════════════════════

# Actor-Netz
ax, aw, ah = 12, 125, 50
ay = Y1 - ah // 2
p(box(ax, ay, aw, ah, "blue"))
p(txt(cx(ax, aw), Y1 - 5, "Actor-Netz", size=14, color="blue"))
p(sub(cx(ax, aw), Y1 + 9, "μ, log σ", fill="#2a5a9a"))

# → Gauss + tanh
p(arr(ax + aw, Y1, 182, Y1))
sx, sw_ = 190, 155
sy = Y1 - BH // 2
p(box(sx, sy, sw_, BH))
p(txt(cx(sx, sw_), Y1 - 3, "Gauss-Stichprobe"))
p(sub(cx(sx, sw_), Y1 + 10, "N(μ, σ²) → tanh"))

# → Action
p(arr(sx + sw_, Y1, 390, Y1))
actx, actw = 398, 95
acty = Y1 - BH // 2
p(box(actx, acty, actw, BH, "orange"))
p(txt(cx(actx, actw), Y1 - 2, "a", size=14, color="orange"))
p(f'<text x="{cx(actx, actw) + 8}" y="{Y1 + 1}" text-anchor="middle" '
  f'font-size="9.5" fill="#8a6420" font-weight="600">t</text>')
p(sub(cx(actx, actw), Y1 + 11, "∈ [−1, 1]⁴"))

# Vertical connector: action → row 2
conn_x = cx(actx, actw)
p(line(conn_x, Y1 + BH // 2, conn_x, Y2 - BH // 2 - 8))
p(arr(conn_x, Y2 - BH // 2 - 8, conn_x, Y2 - BH // 2))

# ════════════════════════════════════════════════════════════════════
# Row 2: Scaling → PID → Physics → next state
# ════════════════════════════════════════════════════════════════════

# Skalierung
skx, skw = 100, 140
sky = Y2 - BH // 2
p(box(skx, sky, skw, BH))
p(txt(cx(skx, skw), Y2 - 3, "Skalierung"))
p(sub(cx(skx, skw), Y2 + 10, "×(1, 1, 2) m;  ×180°"))

# Annotation: setpoints
p(f'<text x="{cx(skx, skw)}" y="{Y2 + BH // 2 + 14}" text-anchor="middle" '
  f'font-size="9" fill="#aaa" font-style="italic">'
  f'→ p_soll, ψ_soll</text>')

# → PID
p(arr(skx + skw, Y2, 295, Y2))
px, pw = 303, 145
py = Y2 - BH // 2
p(box(px, py, pw, BH, "green"))
p(txt(cx(px, pw), Y2 - 3, "PID-Regler", color="green"))
p(sub(cx(px, pw), Y2 + 10, "kaskadierend", fill="#3a6a3e"))

# → Physics
p(arr(px + pw, Y2, 495, Y2))
phx, phw = 503, 145
phy = Y2 - BH // 2
p(box(phx, phy, phw, BH, "green"))
p(txt(cx(phx, phw), Y2 - 3, "Physiksimulator", color="green"))
p(sub(cx(phx, phw), Y2 + 10, "300 × Δt = 3,0 s", fill="#3a6a3e"))

# → Output: s_{t+1}, r_t
p(arr(phx + phw, Y2, 693, Y2))
ox, ow, oh = 701, 65, 40
oy = Y2 - oh // 2
p(box(ox, oy, ow, oh, "purple"))
p(txt(cx(ox, ow), Y2 - 2, "s", size=13))
p(f'<text x="{cx(ox, ow) + 6}" y="{Y2 + 1}" text-anchor="middle" '
  f'font-size="9" fill="#4a2a8a" font-weight="600">t+1</text>')
p(sub(cx(ox, ow), Y2 + 12, ", r"))

# ── Bracket: 300 substeps annotation ───────────────────────────────
brace_y = Y2 - BH // 2 - 15
brace_x1 = px
brace_x2 = phx + phw
brace_mid = (brace_x1 + brace_x2) / 2
p(f'<line x1="{brace_x1}" y1="{brace_y + 5}" x2="{brace_x1}" y2="{brace_y}" '
  f'stroke="#bbb" stroke-width="0.8"/>')
p(f'<line x1="{brace_x1}" y1="{brace_y}" x2="{brace_x2}" y2="{brace_y}" '
  f'stroke="#bbb" stroke-width="0.8"/>')
p(f'<line x1="{brace_x2}" y1="{brace_y}" x2="{brace_x2}" y2="{brace_y + 5}" '
  f'stroke="#bbb" stroke-width="0.8"/>')
p(f'<text x="{brace_mid}" y="{brace_y - 4}" text-anchor="middle" '
  f'font-size="9" fill="#aaa">Entscheidungsintervall (300 Substeps)</text>')

# ── Assemble ────────────────────────────────────────────────────────
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     font-family="{FONT}">
  <defs>{GRAD_DEFS}{ARROW_MARKER}
  </defs>
  <rect width="{W}" height="{H}" fill="#fafafa" rx="4"/>
  {"".join(parts)}
</svg>"""

svg_path = os.path.join(OUT_DIR, "action_pipeline.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved {svg_path}")

# ── Try PDF conversion ──────────────────────────────────────────────
pdf_path = os.path.join(OUT_DIR, "action_pipeline.pdf")
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
