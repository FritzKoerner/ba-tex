#!/usr/bin/env python3
"""Generate vineyard targets heightmap figure for thesis (Abbildung 4.4).

Requires trimesh for raycasting. Run from Bilder/ directory.
Output: vineyard_targets.pdf (same directory)
"""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 10,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

OUTDIR = os.path.dirname(os.path.abspath(__file__))
SCALE = 4.0
MESH_FILE = os.path.join(
    os.path.dirname(OUTDIR), "..", "..", "genesis_v05",
    "assets", "scene", "vineyard-eltville-germany", "source",
    "vineyard_fixed_normals.obj",
)

VINEYARD_TARGETS = [
    (-4.057, -1.600, 1.191),
    (-2.881,  1.770, 1.176),
    (-2.855,  2.249, 1.182),
    (-2.171,  2.070, 1.116),
    (-2.124,  2.528, 1.125),
    (-2.048,  2.957, 1.113),
    (-1.777,  2.181, 1.080),
    (-1.390, -3.762, 0.821),
    (-1.385,  2.799, 1.068),
    (-1.260,  0.149, 1.148),
    (-1.149, -0.972, 0.968),
    (-1.057, -4.095, 0.990),
    (-1.000, -2.000, 0.789),
    ( 0.000, -5.000, 1.070),
    ( 0.392, -0.493, 0.943),
    ( 0.571, -1.177, 1.335),
    ( 0.616, -1.727, 0.766),
    ( 1.000, -3.000, 0.618),
    ( 2.000, -1.000, 0.731),
    ( 3.000, -3.000, 0.503),
]

# ── Colours (thesis palette) ──────────────────────────────────────
TARGET_COLOR = "#8752f6"
BG = "#ffffff"

CMAP_TERRAIN = mcolors.LinearSegmentedColormap.from_list(
    "green_orange", ["#60aa64", "#e8a340"]
)

# ── Load and transform mesh ───────────────────────────────────────
print("Loading mesh...")
m = trimesh.load(MESH_FILE, force="mesh")
R = Rotation.from_euler("x", 90, degrees=True).as_matrix()
verts = np.array(m.vertices @ R.T)
z_offset = float(-verts[:, 2].min())
verts[:, 2] += z_offset
verts *= SCALE
faces = m.faces

targets = np.array(VINEYARD_TARGETS) * SCALE

# ── Build heightmap ───────────────────────────────────────────────
print("Building heightmap...")
resolution = 0.05 * SCALE
x_min, y_min = verts[:, 0].min(), verts[:, 1].min()
x_max, y_max = verts[:, 0].max(), verts[:, 1].max()

xs = np.arange(x_min, x_max + resolution, resolution)
ys = np.arange(y_min, y_max + resolution, resolution)
H, W = len(ys), len(xs)

rotated_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

ray_origins = np.zeros((H * W, 3))
ray_dirs = np.zeros((H * W, 3))
idx = 0
for iy, y_val in enumerate(ys):
    for ix, x_val in enumerate(xs):
        ray_origins[idx] = [x_val, y_val, verts[:, 2].max() + 1.0]
        ray_dirs[idx] = [0, 0, -1]
        idx += 1

print(f"  Casting {H*W} rays...")
locations, index_ray, _ = rotated_mesh.ray.intersects_location(ray_origins, ray_dirs)

heightmap = np.full((H, W), np.nan, dtype=np.float32)
for loc, ray_i in zip(locations, index_ray):
    iy = ray_i // W
    ix = ray_i % W
    if np.isnan(heightmap[iy, ix]):
        heightmap[iy, ix] = loc[2]
    else:
        heightmap[iy, ix] = max(heightmap[iy, ix], loc[2])

# ── Crop to target region with padding ────────────────────────────
pad = 6.0
crop_xmin = targets[:, 0].min() - pad
crop_xmax = targets[:, 0].max() + pad
crop_ymin = targets[:, 1].min() - pad
crop_ymax = targets[:, 1].max() + pad

ix_lo = max(0, int((crop_xmin - x_min) / resolution))
ix_hi = min(W, int((crop_xmax - x_min) / resolution))
iy_lo = max(0, int((crop_ymin - y_min) / resolution))
iy_hi = min(H, int((crop_ymax - y_min) / resolution))

hmap_crop = heightmap[iy_lo:iy_hi, ix_lo:ix_hi]
extent_crop = [
    x_min + ix_lo * resolution,
    x_min + ix_hi * resolution,
    y_min + iy_lo * resolution,
    y_min + iy_hi * resolution,
]

# ── Plot ──────────────────────────────────────────────────────────
print("Generating figure...")
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG)

valid = hmap_crop[~np.isnan(hmap_crop)]
vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)

masked_hmap = np.ma.masked_invalid(hmap_crop)

im = ax.imshow(
    masked_hmap, origin="lower", extent=extent_crop,
    cmap=CMAP_TERRAIN, aspect="equal",
    vmin=vmin, vmax=vmax,
    interpolation="bilinear",
)

contour_levels = np.linspace(vmin, vmax, 12)
ax.contour(
    masked_hmap, levels=contour_levels, origin="lower",
    extent=extent_crop, colors="0.35", linewidths=0.3, alpha=0.5,
)

ax.scatter(
    targets[:, 0], targets[:, 1],
    c=TARGET_COLOR, marker="o", s=55, edgecolors="white",
    linewidths=1.0, zorder=5, label=r"Landeziel ($n=20$)",
)

ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="0.7")

cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
cbar.set_label(r"Gel\"andeh\"ohe (m)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$y$ (m)")
ax.set_facecolor("0.92")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")
ax.tick_params(colors="black")

fig.tight_layout()
out = os.path.join(OUTDIR, "vineyard_targets.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
