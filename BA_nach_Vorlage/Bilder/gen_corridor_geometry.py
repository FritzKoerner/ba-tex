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

# Schematic obstacle AABBs (typical GSO mesh after scaling, clipped to 2m height)
# X offset: slalom_offset 0.6–1.2 m, alternating sides.
# Typical AABB: X~3m, Y~2m (spans full corridor Y), Z clipped to 2m.
OBS = [
    {"cx":  0.9, "wx": 3.0, "wy": 2.0, "h": 2.0, "z": 3.0},   # lower slice, +X
    {"cx": -0.9, "wx": 3.0, "wy": 2.0, "h": 2.0, "z": 7.0},   # upper slice, -X
]

# ── Colours ─────────────────────────────────────────────────────────
C = dict(
    corridor="#aaaaaa", ground="#777777", ground_f="#f0f0f0",
    spawn="#5698f9", target="#8752f6", obs="#60aa64", obs_e="#3a6a3e",
    path="#e8a340", drone="#5698f9", dim="#aaaaaa",
)

# ── Figure ──────────────────────────────────────────────────────────
fig, ax_xz = plt.subplots(1, 1, figsize=(3.8, 4.2))

for ax, axis, hr, sr in [
    (ax_xz, "x", CX, SX),
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
        (sr[0], SZ - 0.35), sw, 0.7,
        fc=C["spawn"], alpha=0.22, ec=C["spawn"], lw=0.6, zorder=4,
    ))

    # Obstacles
    for ob in OBS:
        if axis == "x":
            ox, ow = ob["cx"] - ob["wx"] / 2, ob["wx"]
        else:
            ox, ow = -ob["wy"] / 2, ob["wy"]
        ax.add_patch(mpatches.FancyBboxPatch(
            (ox, ob["z"] - ob["h"] / 2), ow, ob["h"],
            boxstyle="round,pad=0.06",
            fc=C["obs"], alpha=0.25, ec=C["obs_e"], lw=0.8, zorder=5,
        ))

    # Target
    th = 0 if axis == "x" else 1
    ax.plot(TARGET[th], TARGET[2], "o",
            color=C["target"], ms=7, zorder=10)

    # Drone at spawn
    ax.plot(0, SZ, "v", color=C["drone"], ms=7, zorder=10)

    # Axes
    ax.set_xlabel(f"${axis}$ (m)")
    ax.set_ylabel("$z$ (m)")
    ax.set_xlim(hr[0] - 0.6, hr[1] + 0.6)
    ax.set_ylim(-0.8, CZ[1] + 1.8)
    ax.set_yticks([0, 1, 2, 7, 12, 15])
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(colors="black")

# ── Slalom path (XZ) ───────────────────────────────────────────────
z_wp = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.5, 10.5, 11.6])
x_wp = np.array([0.0, -1.0, -1.3, 0.0, 1.0, 1.3, 0.5, 0.1, 0.0])
cs = CubicSpline(z_wp, x_wp)
z_f = np.linspace(1.0, 11.6, 250)
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

# ── Dimension annotations ──────────────────────────────────────────
for ax, rng, label in [(ax_xz, CX, "5\\,m")]:
    dz = CZ[1] + 0.4
    ax.annotate(
        "", xy=(rng[1], dz), xytext=(rng[0], dz),
        arrowprops=dict(arrowstyle="<->", color=C["dim"], lw=0.6),
    )
    ax.text((rng[0] + rng[1]) / 2, dz + 0.55,
            label, ha="center", fontsize=7.5, color="#888888")


# ── Legend (XZ panel, upper left) ───────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=C["corridor"], ls=(0, (4, 2.5)), lw=0.8,
           label="Korridorgrenze"),
    mpatches.Patch(fc=C["spawn"], alpha=0.22, ec=C["spawn"], lw=0.6,
                   label="Startbereich"),
    mpatches.Patch(fc=C["obs"], alpha=0.25, ec=C["obs_e"], lw=0.8,
                   label="Hindernis (AABB)"),
    Line2D([0], [0], marker="o", color=C["target"], ls="None", ms=6,
           label="Landeziel"),
    Line2D([0], [0], marker="v", color=C["drone"], ls="None", ms=5,
           label="Drohne (Start)"),
    Line2D([0], [0], color=C["path"], ls="--", lw=0.8, alpha=0.6,
           label="Flugbahn (Slalom)"),
]
fig.legend(
    handles=legend_handles, loc="lower center", ncol=3, fontsize=6.5,
    framealpha=0.92, borderpad=0.5, handletextpad=0.4, handlelength=1.6,
    bbox_to_anchor=(0.5, -0.12),
)

fig.patch.set_facecolor("#ffffff")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "corridor_geometry.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
