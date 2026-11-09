"""Slide 4: growth rate of each 1D lattice mode for kappa near onset, and the frequency at onset (V1).

Usage: uv run python figures/slides_growth.py   (writes slides/img/growth.png)
"""
import matplotlib
import numpy as np

from mcn_project import presets
from mcn_project.linear import eigenvalues, find_onset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL, KAPPA_RANGE = presets.v1(), (0.0, 1.0)
OFFSETS = (-0.04, -0.02, 0.0, 0.02)   # kappa - kappa_c
MODES = np.arange(13)

plt.style.use("figures/style.mplstyle")
onset = find_onset(MODEL, 1, KAPPA_RANGE)
q = 2 * np.pi * MODES / MODEL.length
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained")
for p, color in zip(OFFSETS, plt.cm.viridis(np.linspace(0.1, 0.85, len(OFFSETS)))):
    nu = eigenvalues(MODEL, q, onset.kappa + p)
    bold = p == 0
    axs[0].plot(MODES, nu.real.max(axis=1), "o-", ms=3, lw=2 if bold else 1,
                color="C3" if bold else color, label=f"$p={p:+.2f}$")
lead = eigenvalues(MODEL, q, onset.kappa)
lead = lead[np.arange(MODES.size), lead.real.argmax(axis=1)]
axs[0].axhline(0, color="k", lw=0.6)
axs[0].plot(onset.k0, 0, "*", color="C3", ms=10)
axs[0].set(xlabel="mode $k$", ylabel=r"$\max\,\mathrm{Re}\,\nu$ (1/ms)")
axs[0].legend(ncol=2, fontsize=7)
axs[1].plot(MODES, 1000 * lead.imag / (2 * np.pi), "o-", ms=3, color="0.3")
axs[1].plot(onset.k0, 1000 * onset.frequency, "*", color="C3", ms=10)
for ax in axs: ax.set_xticks(MODES[::2])
axs[1].set(xlabel="mode $k$", ylabel=r"$\mathrm{Im}\,\nu/2\pi$ (Hz), rightmost", ylim=(0, None))
fig.savefig("slides/img/growth.png", dpi=200)
print(f"kappa_c={onset.kappa:.6f}, k0={onset.k0:g}, f0={1000 * onset.frequency:.3f} Hz")
