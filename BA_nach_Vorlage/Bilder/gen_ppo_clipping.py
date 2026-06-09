"""Generate the PPO clipping figure (Schulman et al. 2017, Fig. 1).

Two subplots: L^CLIP as a function of r for A>0 (left) and A<0 (right).
Outputs ppo_clipping.pdf in the same directory.
"""

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

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

BLUE = "#5698f9"
ORANGE = "#e8a340"
GRAY = "#aaaaaa"
BG = "#ffffff"

EPS = 0.2

r = np.linspace(0.0, 2.0, 1000)

def l_clip(r, A, eps=EPS):
    clipped = np.clip(r, 1 - eps, 1 + eps)
    return np.minimum(r * A, clipped * A)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.2, 2.2), sharey=False)
fig.patch.set_facecolor(BG)

for ax, A, ylim, title in [
    (ax1, 1.0, (-0.2, 1.4), "$\\hat{A}_t > 0$"),
    (ax2, -1.0, (-1.4, 0.2), "$\\hat{A}_t < 0$"),
]:
    y = l_clip(r, A)
    ax.set_facecolor(BG)
    ax.plot(r, y, color=BLUE, linewidth=1.8)
    ax.axvline(1 - EPS, color=GRAY, linewidth=0.6, linestyle="--")
    ax.axvline(1 + EPS, color=GRAY, linewidth=0.6, linestyle="--")
    ax.plot(1.0, l_clip(np.array([1.0]), A)[0], "o", color=ORANGE, markersize=5, zorder=5)
    ax.set_xlabel("$r$")
    ax.set_ylabel("$L^{\\mathrm{CLIP}}$")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 2)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1 - EPS, 1, 1 + EPS, 2])
    ax.set_xticklabels(["$0$", "$1{-}\\varepsilon$", "$1$", "$1{+}\\varepsilon$", "$2$"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(colors="black")

fig.tight_layout(w_pad=2.0)

import os
out = os.path.join(os.path.dirname(__file__), "ppo_clipping.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
