"""Re c_theta versus the angle between two critical waves (report Sec. 10).

Usage: uv run python figures/cross_coupling.py   (writes figures/cross_coupling.pdf)
"""
import matplotlib
import numpy as np

from mcn_project import presets
from mcn_project.linear import find_onset
from mcn_project.normal_form import cross_coefficient

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("figures/style.mplstyle")
theta = np.linspace(0.02, np.pi, 300)
fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
for name, model, rng in [("Wilson-Cowan", presets.wilson_cowan(length=58.264198846330785), (4, 9)),
                         ("V1 ($K=4$)", presets.v1(), (0, 1))]:
    onset = find_onset(model, 2, rng)
    c = np.array([cross_coefficient(model, onset, t) for t in theta])
    ax.plot(np.degrees(theta), c.real / abs(c[-1].real), label=name)
    print(name, "Re c_theta / |Re c| at 90 deg:", round(cross_coefficient(model, onset, np.pi / 2).real
                                                         / abs(c[-1].real), 3))
ax.axhline(0, color="k", lw=0.6)
ax.set(xlabel=r"angle $\theta$ between the waves (deg)", ylabel=r"$\mathrm{Re}\,c_\theta/|\mathrm{Re}\,c|$")
ax.legend()
fig.savefig("figures/cross_coupling.pdf")
fig.savefig("figures/cross_coupling.png", dpi=150)
