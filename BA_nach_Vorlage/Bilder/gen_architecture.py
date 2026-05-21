#!/usr/bin/env python3
"""Generate network architecture diagram (SVG → PDF) for thesis.

Shows: Input → Shared CNN Encoder → Actor/Critic MLP heads → Output,
with all layer dimensions visible.

Usage:  python gen_architecture.py
Output: architecture.svg (+ architecture.pdf if cairosvg/inkscape available)
"""

import os
import subprocess

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

W, H = 1040, 265
BH = 36
CONV_BH = 42
RX = 5
FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

# Row Y-centres
Y_D = 66       # depth / CNN path
Y_S = 218      # state path
Y_M = 142      # merge / concat midpoint
Y_A = 90       # actor branch (after split)
Y_C = 194      # critic branch (after split)

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


def ln(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="5,3"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="1.2"{d}/>')


def cx(x, w):
    return x + w / 2


# ── Box widths ─────────────────────────────────────────────────
INP_W = 62
CONV_W = 78
POOL_W = 68
NORM_W = 102
CONCAT_W = 66
FC_W = 52
TANH_W = 70
OUT_W = 48
V_W = 54

parts = []
p = parts.append

# ══════════════════════════════════════════════════════════════
# DEPTH PATH (top row)
# ══════════════════════════════════════════════════════════════

# ── Input D_t ──────────────────────────────────────────────
inp_x = 12
p(box(inp_x, Y_D - BH // 2, INP_W, BH))
inp_d_cx = cx(inp_x, INP_W)
p(txt(inp_d_cx - 4, Y_D - 2, "D", size=14))
p(f'<text x="{inp_d_cx + 7}" y="{Y_D + 1}" text-anchor="middle" '
  f'font-size="9" fill="#4a2a8a" font-weight="600">t</text>')
p(sub(inp_d_cx, Y_D + 12, "64×64"))
inp_end = inp_x + INP_W  # 74

p(arr(inp_end, Y_D, inp_end + 18, Y_D))

# ── Conv 1 ─────────────────────────────────────────────────
c1x = inp_end + 26  # 100
p(box(c1x, Y_D - CONV_BH // 2, CONV_W, CONV_BH, "blue"))
p(txt(cx(c1x, CONV_W), Y_D - 5, "Conv 32", size=11, color="blue"))
p(sub(cx(c1x, CONV_W), Y_D + 9, "8×8, stride 4"))

p(arr(c1x + CONV_W, Y_D, c1x + CONV_W + 14, Y_D))

# ── Conv 2 ─────────────────────────────────────────────────
c2x = c1x + CONV_W + 22  # 200
p(box(c2x, Y_D - CONV_BH // 2, CONV_W, CONV_BH, "blue"))
p(txt(cx(c2x, CONV_W), Y_D - 5, "Conv 64", size=11, color="blue"))
p(sub(cx(c2x, CONV_W), Y_D + 9, "4×4, stride 2"))

p(arr(c2x + CONV_W, Y_D, c2x + CONV_W + 14, Y_D))

# ── Conv 3 ─────────────────────────────────────────────────
c3x = c2x + CONV_W + 22  # 300
p(box(c3x, Y_D - CONV_BH // 2, CONV_W, CONV_BH, "blue"))
p(txt(cx(c3x, CONV_W), Y_D - 5, "Conv 128", size=11, color="blue"))
p(sub(cx(c3x, CONV_W), Y_D + 9, "3×3, stride 1"))

p(arr(c3x + CONV_W, Y_D, c3x + CONV_W + 14, Y_D))

# ── Global Max Pool ────────────────────────────────────────
pool_x = c3x + CONV_W + 22  # 400
p(box(pool_x, Y_D - BH // 2, POOL_W, BH, "blue"))
p(txt(cx(pool_x, POOL_W), Y_D - 3, "Max Pool", size=10.5, color="blue"))
p(sub(cx(pool_x, POOL_W), Y_D + 10, "global → 128"))
pool_end = pool_x + POOL_W  # 468

# ── BN + ELU annotation ───────────────────────────────────
bn_y = Y_D + CONV_BH // 2 + 14
mid_conv = cx(c1x, c3x + CONV_W - c1x)
p(f'<text x="{mid_conv}" y="{bn_y}" text-anchor="middle" '
  f'font-size="9" fill="#999" font-style="italic">'
  f'+ Batch-Norm + ELU je Schicht</text>')

# ── Shared CNN bracket (dashed green) ──────────────────────
br_x = c1x - 8       # 92
br_end = pool_end + 8  # 476
br_w = br_end - br_x   # 384
br_y = Y_D - CONV_BH // 2 - 20  # 25
br_h = (bn_y + 6) - br_y  # 107 - 25 = 82
p(f'<rect x="{br_x}" y="{br_y}" width="{br_w}" height="{br_h}" '
  f'rx="4" fill="none" stroke="#60aa64" '
  f'stroke-width="1.2" stroke-dasharray="5,3"/>')
p(f'<text x="{cx(br_x, br_w)}" y="{br_y - 5}" '
  f'text-anchor="middle" font-size="9.5" fill="#3a6a3e" '
  f'font-weight="600">Geteilter CNN-Encoder</text>')

# ══════════════════════════════════════════════════════════════
# STATE PATH (bottom row)
# ══════════════════════════════════════════════════════════════

# ── Input o_t ──────────────────────────────────────────────
p(box(inp_x, Y_S - BH // 2, INP_W, BH))
inp_s_cx = cx(inp_x, INP_W)
p(txt(inp_s_cx - 3, Y_S - 2, "o", size=14))
p(f'<text x="{inp_s_cx + 6}" y="{Y_S + 1}" text-anchor="middle" '
  f'font-size="9" fill="#4a2a8a" font-weight="600">t</text>')
p(sub(inp_s_cx, Y_S + 12, "ℝ¹⁷"))

p(arr(inp_end, Y_S, inp_end + 18, Y_S))

# ── Empirical Normalization ────────────────────────────────
norm_x = inp_end + 26  # 100 — same column as Conv1
p(box(norm_x, Y_S - BH // 2, NORM_W, BH))
p(txt(cx(norm_x, NORM_W), Y_S - 3, "Emp. Norm.", size=11))
p(sub(cx(norm_x, NORM_W), Y_S + 10, "laufend μ, σ²"))
norm_end = norm_x + NORM_W  # 202

# ══════════════════════════════════════════════════════════════
# MERGE BUS
# ══════════════════════════════════════════════════════════════

merge_x = pool_end + 20  # 488
p(ln(pool_end, Y_D, merge_x, Y_D))
p(ln(norm_end, Y_S, merge_x, Y_S))
p(ln(merge_x, Y_D, merge_x, Y_S))
p(arr(merge_x, Y_M, merge_x + 18, Y_M))

# ── Concat ─────────────────────────────────────────────────
concat_x = merge_x + 26  # 514
p(box(concat_x, Y_M - BH // 2, CONCAT_W, BH))
p(txt(cx(concat_x, CONCAT_W), Y_M - 3, "Concat", size=11))
p(sub(cx(concat_x, CONCAT_W), Y_M + 10, "145-dim"))
concat_end = concat_x + CONCAT_W  # 580

# ══════════════════════════════════════════════════════════════
# SPLIT BUS → ACTOR / CRITIC
# ══════════════════════════════════════════════════════════════

split_x = concat_end + 20  # 600
p(ln(concat_end, Y_M, split_x, Y_M))
p(ln(split_x, Y_A, split_x, Y_C))
p(arr(split_x, Y_A, split_x + 18, Y_A))
p(arr(split_x, Y_C, split_x + 18, Y_C))

# ── Helper: draw 3 FC layers in a row ─────────────────────
def fc_chain(start_x, y):
    """Draw FC 128 → FC 128 → FC 128 and return x after last box."""
    x = start_x
    for i in range(3):
        p(box(x, y - BH // 2, FC_W, BH, "blue"))
        p(txt(cx(x, FC_W), y - 3, "FC 128", size=10.5, color="blue"))
        p(sub(cx(x, FC_W), y + 10, "ELU"))
        x += FC_W
        if i < 2:
            p(arr(x, y, x + 10, y))
            x += 16
    return x  # end of last FC box

# ── Actor: 3× FC 128 ──────────────────────────────────────
fc_start = split_x + 26  # 626
fc_a_end = fc_chain(fc_start, Y_A)

# "Actor" label above FC chain
fc_mid = cx(fc_start, fc_a_end - fc_start)
p(f'<text x="{fc_mid}" y="{Y_A - BH // 2 - 7}" text-anchor="middle" '
  f'font-size="9.5" fill="#2a5a9a" font-weight="600">Actor</text>')

p(arr(fc_a_end, Y_A, fc_a_end + 14, Y_A))

# ── Tanh-Gaussian distribution ─────────────────────────────
tanh_x = fc_a_end + 22
p(box(tanh_x, Y_A - BH // 2, TANH_W, BH))
p(txt(cx(tanh_x, TANH_W), Y_A - 3, "Tanh-Gauss.", size=10.5))
p(sub(cx(tanh_x, TANH_W), Y_A + 10, "μ, log σ → tanh"))
tanh_end = tanh_x + TANH_W

p(arr(tanh_end, Y_A, tanh_end + 14, Y_A))

# ── a_t output ─────────────────────────────────────────────
out_a_x = tanh_end + 22
p(box(out_a_x, Y_A - BH // 2, OUT_W, BH, "orange"))
out_a_cx = cx(out_a_x, OUT_W)
p(txt(out_a_cx - 4, Y_A - 1, "a", size=14, color="orange"))
p(f'<text x="{out_a_cx + 7}" y="{Y_A + 2}" text-anchor="middle" '
  f'font-size="9" fill="#8a6420" font-weight="600">t</text>')
p(sub(out_a_cx, Y_A + 12, "(−1, 1)⁴"))

# ── Critic: 3× FC 128 ─────────────────────────────────────
fc_c_end = fc_chain(fc_start, Y_C)

# "Critic" label above FC chain
p(f'<text x="{fc_mid}" y="{Y_C - BH // 2 - 7}" text-anchor="middle" '
  f'font-size="9.5" fill="#2a5a9a" font-weight="600">Critic</text>')

p(arr(fc_c_end, Y_C, fc_c_end + 14, Y_C))

# ── V(s) output ────────────────────────────────────────────
v_x = fc_c_end + 22
p(box(v_x, Y_C - BH // 2, V_W, BH, "orange"))
v_cx = cx(v_x, V_W)
p(txt(v_cx, Y_C - 3, "V(s)", size=13, color="orange"))
p(sub(v_cx, Y_C + 10, "skalar"))

# ══════════════════════════════════════════════════════════════
# ASSEMBLE SVG
# ══════════════════════════════════════════════════════════════

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     font-family="{FONT}">
  <defs>{GRAD_DEFS}{ARROW_MARKER}
  </defs>
  <rect width="{W}" height="{H}" fill="#fafafa" rx="4"/>
  {"".join(parts)}
</svg>"""

svg_path = os.path.join(OUT_DIR, "architecture.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved {svg_path}")

# ── PDF conversion ─────────────────────────────────────────
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
