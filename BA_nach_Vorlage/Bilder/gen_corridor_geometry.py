#!/usr/bin/env python3
"""Generate corridor geometry figure for thesis (XZ + YZ side views).

Two side views of the Corridor Navigation v2 environment showing corridor
boundaries, obstacle layers, spawn area, target, and schematic slalom path.

Usage:  python gen_corridor_geometry.py
Output: corridor_geometry.pdf (same directory)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 9,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# ── Corridor V2 geometry (metres) ───────────────────────────────────
CX = (-2.5, 2.5)
CY = (-1.0, 1.0)
CZ = (0.0, 15.0)
SX, SY, SZ = (-1.5, 1.5), (-0.5, 0.5), 12.0
TARGET = (0.0, 0.0, 1.0)

# Schematic obstacle AABBs (approximate typical GSO mesh after scaling)
OBS = [
    {"cx":  1.0, "wx": 2.5, "h": 1.8, "z": 2.0},   # lower slice, +X
    {"cx": -1.0, "wx": 2.5, "h": 1.8, "z": 7.0},   # upper slice, -X
]

# ── Colours ─────────────────────────────────────────────────────────
C = dict(
    corridor="#aaaaaa", ground="#777777", ground_f="#f0f0f0",
    spawn="#5a9bd5", target="#3d8b52", obs="#d2d2d2", obs_e="#999999",
    path="#555555", drone="#333333", dim="#aaaaaa",
)

# ── Figure ──────────────────────────────────────────────────────────
fig, (ax_xz, ax_yz) = plt.subplots(
    1, 2, figsize=(5.2, 4.2),
    gridspec_kw={"width_ratios": [3, 1.4], "wspace": 0.42},
)

for ax, axis, hr, sr in [
    (ax_xz, "x", CX, SX),
    (ax_yz, "y", CY, SY),
]:
    cw, ch = hr[1] - hr[0], CZ[1] - CZ[0]

    # Corridor boundary
    ax.add_patch(mpatches.Rectangle(
        (hr[0], CZ[0]), cw, ch,
        lw=0.8, ec=C["corridor"], fc="none", ls=(0, (4, 2.5)), zorder=2,
    ))

    # Ground
    ax.axhline(0, color=C["ground"], lw=0.8, zorder=3)
    gx = [hr[0] - 0.8, hr[1] + 0.8]
    ax.fill_between(gx, -0.8, 0, color=C["ground_f"], zorder=1)

    # Spawn region
    sw = sr[1] - sr[0]
    ax.add_patch(mpatches.Rectangle(
        (sr[0], SZ - 0.2), sw, 0.4,
        fc=C["spawn"], alpha=0.22, ec=C["spawn"], lw=0.6, zorder=4,
    ))

    # Obstacles
    for ob in OBS:
        if axis == "x":
            ox, ow = ob["cx"] - ob["wx"] / 2, ob["wx"]
        else:
            ox, ow = CY[0], CY[1] - CY[0]
        ax.add_patch(mpatches.FancyBboxPatch(
            (ox, ob["z"] - ob["h"] / 2), ow, ob["h"],
            boxstyle="round,pad=0.06",
            fc=C["obs"], ec=C["obs_e"], lw=0.6, zorder=5,
        ))

    # Target
    th = 0 if axis == "x" else 1
    ax.plot(TARGET[th], TARGET[2], "*",
            color=C["target"], ms=10, zorder=10, mew=0.4)

    # Drone at spawn
    ax.plot(0, SZ, "^", color=C["drone"], ms=7, zorder=10)

    # Axes
    ax.set_xlabel(f"${axis}$ (m)")
    ax.set_ylabel("$z$ (m)")
    ax.set_xlim(hr[0] - 0.6, hr[1] + 0.6)
    ax.set_ylim(-0.8, CZ[1] + 1.8)
    ax.set_yticks([0, 1, 2, 7, 12, 15])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Slalom path (XZ) ───────────────────────────────────────────────
z_wp = np.array([1.0, 1.8, 2.0, 4.5, 6.0, 7.0, 8.5, 10.5, 12.0])
x_wp = np.array([0.0, -0.6, -1.0, -0.3, 0.6, 1.0, 0.7, 0.2, 0.0])
cs = CubicSpline(z_wp, x_wp)
z_f = np.linspace(1.0, 12.0, 250)
x_f = cs(z_f)
ax_xz.plot(x_f, z_f, color=C["path"], lw=1.0, ls="--", alpha=0.6, zorder=8)

# Direction arrows along path (two small arrows)
for frac in (0.35, 0.70):
    i = int(frac * len(z_f))
    ax_xz.annotate(
        "", xy=(x_f[i - 4], z_f[i - 4]), xytext=(x_f[i + 4], z_f[i + 4]),
        arrowprops=dict(arrowstyle="-|>", color=C["path"], lw=0,
                        mutation_scale=9),
        zorder=9,
    )

# ── Vertical descent path (YZ) ─────────────────────────────────────
ax_yz.plot([0, 0], [SZ, TARGET[2]], color=C["path"], lw=1.0,
           ls="--", alpha=0.6, zorder=8)
ax_yz.annotate(
    "", xy=(0, 6), xytext=(0, 8),
    arrowprops=dict(arrowstyle="-|>", color=C["path"], lw=0,
                    mutation_scale=9),
    zorder=9,
)

# ── Dimension annotations ──────────────────────────────────────────
for ax, rng, label in [(ax_xz, CX, "5\\,m"), (ax_yz, CY, "2\\,m")]:
    dz = CZ[1] + 0.4
    ax.annotate(
        "", xy=(rng[1], dz), xytext=(rng[0], dz),
        arrowprops=dict(arrowstyle="<->", color=C["dim"], lw=0.6),
    )
    ax.text((rng[0] + rng[1]) / 2, dz + 0.55,
            label, ha="center", fontsize=7.5, color="#888888")

# ── Panel titles ────────────────────────────────────────────────────
ax_xz.set_title("Seitenansicht ($xz$-Ebene)", fontsize=9, pad=10)
ax_yz.set_title("Seitenansicht ($yz$-Ebene)", fontsize=9, pad=10)

# ── Legend (XZ panel, upper left) ───────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=C["corridor"], ls=(0, (4, 2.5)), lw=0.8,
           label="Korridorgrenze"),
    mpatches.Patch(fc=C["spawn"], alpha=0.22, ec=C["spawn"], lw=0.6,
                   label="Startbereich"),
    mpatches.Patch(fc=C["obs"], ec=C["obs_e"], lw=0.6,
                   label="Hindernis (AABB)"),
    Line2D([0], [0], marker="*", color=C["target"], ls="None", ms=8,
           label="Landeziel"),
    Line2D([0], [0], marker="^", color=C["drone"], ls="None", ms=5,
           label="Drohne (Start)"),
    Line2D([0], [0], color=C["path"], ls="--", lw=0.8, alpha=0.6,
           label="Flugbahn (Slalom)"),
]
ax_xz.legend(
    handles=legend_handles, loc="upper left", fontsize=6.5,
    framealpha=0.92, borderpad=0.5, handletextpad=0.4, handlelength=1.6,
)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "corridor_geometry.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
