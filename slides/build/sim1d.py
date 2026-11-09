"""1D runs for the slides: no input vs uniform flicker at 2 omega0, below onset (mu = -1).

Parameters are those of notebooks/v1_1d.ipynb and notebooks/wilson_cowan_1d.ipynb.
Each forced run is continued for ZOOM_PERIODS periods, saved every step, for the close-up panels.
Usage: uv run python slides/build/sim1d.py   (writes results/slides_sim1d.npz, about 2 minutes)
"""
import numpy as np

from mcn_project import presets
from mcn_project.amplitude import standing_equilibrium
from mcn_project.analysis import critical_amplitudes
from mcn_project.linear import find_onset
from mcn_project.normal_form import cubic_coefficients, unit_forcing_coefficients
from mcn_project.simulate import Forcing, Wave, critical_seed, simulate

MU, SEED, NOISE, ZOOM_PERIODS = -1.0, 0.02, 1e-4, 4
CASES = {  # model, kappa range, eps, flicker peak (units of eps^2), n, dt, t_end, save_every
    "wc": (presets.wilson_cowan(), (4.0, 9.0), 0.1, 1.0, 60, 0.1, 3000.0, 10),
    "v1": (presets.v1(), (0.0, 1.0), 0.02, 3.0, 64, 1.0, 150_000.0, 10),
}


def run_model(model, kappa_range, eps, peak, n, dt, t_end, save_every):
    onset = find_onset(model, 1, kappa_range)
    nf = cubic_coefficients(model, onset)
    eta_t1, _ = unit_forcing_coefficients(model, onset)
    theory = eps * np.sqrt(standing_equilibrium(MU * nf.alpha, nf.b, nf.c, peak * eta_t1)["rho"])
    kappa = onset.kappa + MU * eps**2
    u0 = critical_seed(model, onset, SEED * eps, n=n) + NOISE * eps * np.random.default_rng(0).standard_normal((model.K, n))
    out = dict(x=np.arange(n) * model.length / n, theory=theory, omega0=onset.omega0, k0=onset.k0)
    for name, forcing in [("none", Forcing()), ("flicker", Forcing((Wave(peak * eps**2, (), 2 * onset.omega0),)))]:
        run = simulate(model, kappa, forcing, n=n, dt=dt, t_end=t_end, save_every=save_every, u0=u0)
        A, B = critical_amplitudes(run, onset)
        out |= {f"{name}_t": run.t, f"{name}_u": run.u[:, 0], f"{name}_A": abs(A), f"{name}_B": abs(B)}
    zoom_end = ZOOM_PERIODS * 2 * np.pi / onset.omega0
    zoom = simulate(model, kappa, forcing, n=n, dt=dt, t_end=zoom_end, save_every=1, u0=run.u[-1], t_start=run.t[-1])
    out |= dict(zoom_t=zoom.t, zoom_u=zoom.u[:, 0], zoom_input=zoom.input)
    print(f"late mean |A| = {abs(A[run.t > 0.9 * t_end]).mean():.4e}, theory eps sqrt(rho*) = {theory:.4e}")
    return out


results = {}
for key, args in CASES.items():
    print(key, end=": ")
    results |= {f"{key}_{k}": v for k, v in run_model(*args).items()}
np.savez_compressed("results/slides_sim1d.npz", **results)
