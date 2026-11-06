"""Standard figures shared by the notebooks."""

import matplotlib.pyplot as plt
import numpy as np

from .analysis import critical_amplitudes
from .linear import Onset, eigenvalues, growth_rate, lattice_shells, shell_q
from .model import NeuralField
from .simulate import Run


def mode_growth(model: NeuralField, kappas, modes=range(8), onset: Onset | None = None):
    """Largest real eigenvalue of each 1D lattice mode versus kappa."""
    fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    for h in modes:
        g = [growth_rate(model, 2 * np.pi * h / model.length, k) for k in kappas]
        bold = onset is not None and h == round(onset.k0)
        ax.plot(kappas, g, lw=2.5 if bold else 1, color=None if bold else "0.7",
                label=f"$k_0={h}$" if bold else None)
    ax.axhline(0, color="k", lw=0.8)
    if onset is not None:
        ax.axvline(onset.kappa, color="C3", ls="--", label=rf"$\kappa_c={onset.kappa:.4f}$")
        ax.legend(frameon=False)
    ax.set(xlabel=r"$\kappa$", ylabel=r"$\max\,\mathrm{Re}\,\nu$")
    return fig


def dispersion(model: NeuralField, kappa: float, dim: int = 1, q_max: float | None = None):
    """Real and imaginary parts of all eigenvalues of M(q); dots mark lattice shells."""
    r2 = lattice_shells(model, dim)
    q_lat = shell_q(model, r2)
    q_max = q_max or q_lat[min(len(q_lat) - 1, 12)]
    q = np.linspace(0, q_max, 400)
    nu = eigenvalues(model, q, kappa)
    keep = q_lat <= q_max
    nu_lat = eigenvalues(model, q_lat[keep], kappa)
    lead = np.argmax(nu_lat.real, axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, part, label in [(axs[0], np.real, r"$\mathrm{Re}\,\nu$"), (axs[1], np.imag, r"$\mathrm{Im}\,\nu$")]:
        ax.plot(q, part(nu), color="0.6", lw=1)
        ax.plot(q_lat[keep], part(nu_lat[np.arange(lead.size), lead]), "o", color="C0", ms=4)
        ax.set(xlabel="$q$", ylabel=label)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[1].set_ylim(bottom=0)
    return fig


def spacetime(runs: list[Run], titles: list[str], onset: Onset, population: int = 0):
    """Rows of (input, activity, |A| and |B|) for 1D runs; time runs upward."""
    fig, axs = plt.subplots(len(runs), 3, figsize=(13, 3.2 * len(runs)),
                            layout="constrained", squeeze=False)
    for row, run, title in zip(axs, runs, titles):
        extent = [run.x[0], run.x[-1], run.t[0], run.t[-1]]
        for ax, field, name in [(row[0], run.input, "input"), (row[1], run.u[:, population], "activity")]:
            lim = max(np.abs(field).max(), 1e-12)
            im = ax.imshow(field, origin="lower", aspect="auto", extent=extent,
                           cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set(title=f"{title}: {name}", xlabel="$x$", ylabel="$t$")
            fig.colorbar(im, ax=ax, shrink=0.8)
        A, B = critical_amplitudes(run, onset)
        row[2].plot(abs(A), run.t, label="$|A|$")
        row[2].plot(abs(B), run.t, label="$|B|$")
        row[2].set(title=f"{title}: critical amplitudes", xlabel="amplitude", ylabel="$t$",
                   ylim=(run.t[0], run.t[-1]))
        row[2].legend(frameon=False)
    return fig


def snapshots(run: Run, times, population: int = 0):
    """2D activity at the saved frames closest to the given times."""
    fig, axs = plt.subplots(1, len(times), figsize=(3.2 * len(times), 3), layout="constrained",
                            squeeze=False)
    lim = np.abs(run.u[:, population]).max()
    for ax, t in zip(axs[0], times):
        i = np.argmin(abs(run.t - t))
        im = ax.imshow(run.u[i, population].T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[0, run.x[-1], 0, run.x[-1]])
        ax.set(title=f"$t={run.t[i]:.0f}$", xlabel="$x$", ylabel="$y$")
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8)
    return fig
