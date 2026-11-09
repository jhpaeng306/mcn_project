"""Slides 5 and 12: 1D space-time plots below onset, no input vs uniform flicker at 2 omega0.

Layout follows the earlier talk: no input | flicker close-up (input, activity) | flicker, whole run | amplitudes.
Reads results/slides_sim1d.npz (slides/build/sim1d.py).
Usage: uv run python figures/slides_spacetime.py   (writes slides/img/spacetime_{wc,v1}.png)
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {"wc": ("Wilson–Cowan", 1.0, "$t$", "$x$"), "v1": ("V1-inspired", 1e-3, "$t$ (s)", "$x$ (0.1 mm)")}  # title, time scale, labels
data = np.load("results/slides_sim1d.npz")
plt.style.use("figures/style.mplstyle")


def field(ax, x, t, u, title, tlabel, xlabel):
    lim = np.abs(u).max()
    ax.imshow(u, origin="lower", aspect="auto", extent=[x[0], x[-1], t[0], t[-1]],
              cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set(title=title, xlabel=xlabel, ylabel=tlabel)


for key, (name, scale, tlabel, xlabel) in MODELS.items():
    d = {k[len(key) + 1:]: data[k] for k in data.files if k.startswith(key + "_")}
    fig, axs = plt.subplots(1, 5, figsize=(10.5, 2.9), layout="constrained",
                            gridspec_kw=dict(width_ratios=[1, 0.8, 0.8, 1, 1]))
    field(axs[0], d["x"], scale * d["none_t"], d["none_u"], "no input: activity", tlabel, xlabel)
    field(axs[1], d["x"], scale * d["zoom_t"], d["zoom_input"], "flicker: input (end)", tlabel, xlabel)
    field(axs[2], d["x"], scale * d["zoom_t"], d["zoom_u"], "flicker: activity (end)", "", xlabel)
    field(axs[3], d["x"], scale * d["flicker_t"], d["flicker_u"], "flicker: activity", tlabel, xlabel)
    t = scale * d["flicker_t"]
    axs[4].plot(d["flicker_A"], t, label="$|A|$, flicker")
    axs[4].plot(d["flicker_B"], t, "--", label="$|B|$, flicker")
    axs[4].plot(d["none_A"], t, color="0.6", label="$|A|$, no input")
    axs[4].axvline(d["theory"], color="k", ls=":", lw=1, label=r"theory $\epsilon\sqrt{\rho_*}$")
    axs[4].set(title="critical amplitudes", xlabel="amplitude", ylabel=tlabel, ylim=(t[0], t[-1]))
    axs[4].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axs[4].legend(fontsize=6.5, loc="lower right")
    fig.suptitle(name, fontsize=10)
    fig.savefig(f"slides/img/spacetime_{key}.png", dpi=200)

d = {k[3:]: data[k] for k in data.files if k.startswith("v1_")}
fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.9), layout="constrained")
field(axs[0], d["x"], 1e-3 * d["zoom_t"], d["zoom_input"], r"input $I(t)$, at $2\omega_0$", "$t$ (s)", "$x$ (0.1 mm)")
field(axs[1], d["x"], 1e-3 * d["zoom_t"], d["zoom_u"], "E activity", "", "$x$ (0.1 mm)")
fig.savefig("slides/img/flicker_closeup_v1.png", dpi=200)
