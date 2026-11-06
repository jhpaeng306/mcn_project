"""Thresholds of a static stimulus q cos(2 pi m x/L) on E, with the small-q theory (report Sec. 9).

Reads results/two_parameter_curves.json (from theory/code/two_parameter_curves.py).
Usage: uv run python figures/two_parameter_curves.py   (writes figures/two_parameter_curves.pdf)
"""
import json

import matplotlib
import numpy as np

from mcn_project import presets
from mcn_project.linear import find_onset
from mcn_project.normal_form import (
    cubic_coefficients,
    static_forcing_coefficients,
    unit_forcing_coefficients,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("figures/style.mplstyle")
model = presets.wilson_cowan()
onset = find_onset(model, 1, (4, 9))
alpha = cubic_coefficients(model, onset).alpha.real
data = json.load(open("results/two_parameter_curves.json"))
curves, near_zero = data["curves"], data["near_zero"]

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), layout="constrained")
for m, c in curves.items():
    c = np.array(c, dtype=float)
    axs[0].plot(c[:, 0], c[:, 1], label=f"$k={m}$")
axs[0].set(xlabel=r"stimulus peak $\delta$", ylabel="$a_{EE}$ at instability", ylim=(4, 10), title="(a)")
axs[0].legend(ncol=2, fontsize=6)

q = np.linspace(-0.012, 0.012, 201)
colors = {"1": "C0", "2": "C1", "3": "C2", "6": "C5"}
slope = abs(unit_forcing_coefficients(model, onset)[1].real) / alpha
for sign in (1, -1):                                   # m = 6: the two branches A = +-B
    axs[1].plot(q, sign * slope * q, color=colors["6"], lw=0.8, ls=":")
axs[1].plot(q, -slope * abs(q), color=colors["6"])
c = np.array(near_zero["6"])
axs[1].plot(c[:, 0], c[:, 1] - onset.kappa, "o", ms=2.5, color=colors["6"])
axs[1].set(xlabel=r"stimulus peak $\delta$", ylabel=r"$a_{EE}-a_{EE}^*$", ylim=(-0.3, 0.1), title="(b) $k=6=2k_0$")

for m in ("1", "2", "3"):
    chi, xi = static_forcing_coefficients(model, onset, 2 * np.pi * int(m) / model.length, model.drive / 2)
    for sign in ([1, -1] if m == "3" else [1]):        # sign 1: the first branch to destabilize
        C = -(chi.real + sign * abs(np.real(xi))) / alpha
        axs[2].plot(q, 1e3 * C * q**2, color=colors[m], ls="-" if sign == 1 else ":",
                    label=f"$k={m}$" if sign == 1 else None)
    c = np.array(near_zero[m])
    axs[2].plot(c[:, 0], 1e3 * (c[:, 1] - onset.kappa), "o", ms=2.5, color=colors[m])
axs[2].set(xlabel=r"stimulus peak $\delta$", ylabel=r"$10^3\,(a_{EE}-a_{EE}^*)$", title=r"(c) $k\leq3$")
axs[2].legend(fontsize=6)
fig.savefig("figures/two_parameter_curves.pdf")
fig.savefig("figures/two_parameter_curves.png", dpi=150)
