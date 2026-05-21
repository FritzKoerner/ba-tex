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

EPS = 0.2

r = np.linspace(0.0, 2.0, 1000)

def l_clip(r, A, eps=EPS):
    clipped = np.clip(r, 1 - eps, 1 + eps)
    return np.minimum(r * A, clipped * A)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.2, 2.2), sharey=False)

# --- A > 0 ---
A_pos = 1.0
y_pos = l_clip(r, A_pos)
ax1.plot(r, y_pos, color="C0", linewidth=1.8)
ax1.axvline(1 - EPS, color="gray", linewidth=0.5, linestyle="--")
ax1.axvline(1 + EPS, color="gray", linewidth=0.5, linestyle="--")
ax1.plot(1.0, l_clip(np.array([1.0]), A_pos)[0], "o", color="C3", markersize=5, zorder=5)
ax1.set_xlabel("$r$")
ax1.set_ylabel("$L^{\\mathrm{CLIP}}$")
ax1.set_title("$\\hat{A}_t > 0$", fontsize=10)
ax1.set_xlim(0, 2)
ax1.set_ylim(-0.2, 1.4)
ax1.set_xticks([0, 1 - EPS, 1, 1 + EPS, 2])
ax1.set_xticklabels(["$0$", "$1{-}\\varepsilon$", "$1$", "$1{+}\\varepsilon$", "$2$"])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- A < 0 ---
A_neg = -1.0
y_neg = l_clip(r, A_neg)
ax2.plot(r, y_neg, color="C0", linewidth=1.8)
ax2.axvline(1 - EPS, color="gray", linewidth=0.5, linestyle="--")
ax2.axvline(1 + EPS, color="gray", linewidth=0.5, linestyle="--")
ax2.plot(1.0, l_clip(np.array([1.0]), A_neg)[0], "o", color="C3", markersize=5, zorder=5)
ax2.set_xlabel("$r$")
ax2.set_ylabel("$L^{\\mathrm{CLIP}}$")
ax2.set_title("$\\hat{A}_t < 0$", fontsize=10)
ax2.set_xlim(0, 2)
ax2.set_ylim(-1.4, 0.2)
ax2.set_xticks([0, 1 - EPS, 1, 1 + EPS, 2])
ax2.set_xticklabels(["$0$", "$1{-}\\varepsilon$", "$1$", "$1{+}\\varepsilon$", "$2$"])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout(w_pad=2.0)

import os
out = os.path.join(os.path.dirname(__file__), "ppo_clipping.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
