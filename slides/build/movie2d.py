"""Slide 14 movie: V1, input cos(K.x + 2 omega0 t) with K = (6, 0), uniform along y, on the shell |k|^2 = 10.

Parameters are those of route 1a in notebooks/v1_2d.ipynb, one seed.
Runs from a random field to T_END, then saves MOVIE_PERIODS response periods for the movie.
Usage: uv run python slides/build/movie2d.py   (writes slides/media/v1_traveling.mp4, about 1 minute)
"""
import matplotlib
import numpy as np

from mcn_project import presets
from mcn_project.analysis import critical_amplitudes
from mcn_project.linear import critical_wavenumber, find_onset
from mcn_project.normal_form import cubic_coefficients, pair_forcing_coefficient
from mcn_project.simulate import Forcing, Wave, simulate

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

SHELL, KAPPA_RANGE, K = 10, (0.0, 2.0), (6, 0)   # critical shell |k|^2 = SHELL
EPS, MU, STRENGTH, NOISE, SEED = 0.025, -1.0, 3.0, 0.1, 0
N, DT, T_END, MOVIE_PERIODS, FRAME_EVERY = 32, 1.0, 80_000.0, 2, 2

L = np.sqrt(SHELL) * 2 * np.pi / critical_wavenumber(presets.v1(), KAPPA_RANGE)
model = presets.v1(length=L)
onset = find_onset(model, 2, KAPPA_RANGE)
a = MU * cubic_coefficients(model, onset).alpha
peak = STRENGTH * abs(a) / abs(pair_forcing_coefficient(model, onset, 2 * np.pi * np.hypot(*K) / L, model.drive / 2, "2w0"))
kappa = onset.kappa + MU * EPS**2
run = simulate(model, kappa, Forcing((Wave(peak * EPS**2, K, -2 * onset.omega0),)), dim=2, n=N, dt=DT,
               t_end=T_END, save_every=1000, noise=NOISE * EPS, seed=SEED)
for mode in [(3, 1), (3, -1), (1, 3), (0, 3)]:
    print(mode, f"eps|A| at end = {abs(critical_amplitudes(run, onset, mode)[0][-1]):.5f}")
movie = simulate(model, kappa, run.forcing, dim=2, n=N, dt=DT, t_end=MOVIE_PERIODS * 2 * np.pi / onset.omega0,
                 save_every=FRAME_EVERY, u0=run.u[-1], t_start=run.t[-1])

plt.style.use("figures/style.mplstyle")
fig, axs = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
images = []
for ax, field, title in [(axs[0], movie.input, "input (uniform along $y$)"), (axs[1], movie.u[:, 0], "E activity")]:
    lim = np.abs(field).max()
    images.append(ax.imshow(field[0].T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                            extent=[0, L, 0, L], interpolation="bilinear"))
    ax.set(title=title, xlabel="$x$ (0.1 mm)", ylabel="$y$ (0.1 mm)")
stamp = fig.suptitle("")


def frame(i):
    images[0].set_data(movie.input[i].T)
    images[1].set_data(movie.u[i, 0].T)
    stamp.set_text(f"$t$ = {movie.t[i] / 1000:.3f} s")
    return images


print(f"L = {L:.3f}")
FuncAnimation(fig, frame, frames=len(movie.t)).save("slides/media/v1_traveling.mp4", writer=FFMpegWriter(fps=12), dpi=160)
