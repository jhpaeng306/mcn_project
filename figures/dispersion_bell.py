"""Frequency of the oscillatory branch versus wavenumber at onset (report Sec. 8.4).

Follows the complex pair through q, reports where it peaks, where it stops being
rightmost, and where it collides into two real eigenvalues.
Usage: uv run python figures/dispersion_bell.py   (writes figures/dispersion_bell.pdf)
"""
import matplotlib
import numpy as np

from mcn_project import presets
from mcn_project.linear import eigenvalues, find_onset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = {"Wilson-Cowan": (presets.wilson_cowan(), (4, 9)), "V1 (K=4)": (presets.v1(), (0, 1)),
         "V1 (K=3)": (presets.v1(3), (0, 1))}
plt.style.use("figures/style.mplstyle")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), layout="constrained")
for ax, (name, (model, rng)) in zip(axs, CASES.items()):
    onset = find_onset(model, 1, rng)
    q = np.linspace(0, 6 * onset.q0, 3001)
    ev = eigenvalues(model, q, onset.kappa)
    pair = np.where(abs(ev.imag) > 1e-12, ev.imag, -np.inf).max(axis=1)   # largest Im over branches
    pair = np.where(np.isfinite(pair), pair, 0)
    re_pair = np.array([e[np.argmax(e.imag)].real for e in ev])
    rightmost = re_pair >= ev.real.max(axis=1) - 1e-12
    i_peak = np.argmax(pair)
    lost = q[np.argmax(~rightmost & (q > 0))] if (~rightmost).any() else np.nan
    gone = q[np.argmax(pair == 0)] if (pair == 0).any() else np.nan
    print(f"{name}: omega(0)={pair[0]:.4f}, peak omega={pair[i_peak]:.4f} at q/q0={q[i_peak]/onset.q0:.2f}, "
          f"omega(q0)={onset.omega0:.4f}; pair stops being rightmost at q/q0={lost/onset.q0:.2f}; "
          f"pair becomes real at q/q0={gone/onset.q0:.2f}")
    ax.plot(q / onset.q0, np.where(rightmost, pair, np.nan), "C0", label="pair is rightmost")
    ax.plot(q / onset.q0, np.where(rightmost, np.nan, pair), "C0--", lw=0.8, label="pair exists, not rightmost")
    ax.axvline(1, color="0.6", lw=0.8)
    ax.set(title=name, xlabel="$q/q_0$", ylabel=r"$\omega$ of the complex pair")
axs[0].legend(frameon=False, fontsize=8)
fig.savefig("figures/dispersion_bell.pdf")
fig.savefig("figures/dispersion_bell.png", dpi=150)
