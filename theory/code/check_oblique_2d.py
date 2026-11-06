"""H2 check: an input traveling along x creates a pattern with structure along y.

Each model is put on a square torus whose critical shell is oblique, |k|^2 = 10. The input
cos(6 q1 x + 2 w0 t) has K = (6, 0) q1 = (3,1) q1 + (3,-1) q1, so it couples that pair only.
Theory (report Sec. 8): the pair obeys the Sec. 6 equations with c -> c_theta (cos theta = 0.8);
its stability inside the full shell comes from ShellSystem. The run starts from noise.
Usage: uv run python theory/code/check_oblique_2d.py [WC|V1]   (WC ~1 min, V1 ~5 min)
"""
import os
import sys

import numpy as np

from mcn_project import presets
from mcn_project.amplitude import standing_equilibrium
from mcn_project.analysis import critical_amplitudes
from mcn_project.linear import critical_wavenumber, find_onset
from mcn_project.normal_form import (
    cross_coefficient,
    cubic_coefficients,
    pair_forcing_coefficient,
)
from mcn_project.shell import ShellSystem, shell_waves
from mcn_project.simulate import Forcing, Wave, simulate

CASES = {"WC": dict(preset=presets.wilson_cowan, rng=(4, 12), eps=0.1, dt=0.1, t_end=9000.0),
         "V1": dict(preset=presets.v1, rng=(0, 2), eps=float(os.environ.get("V1_EPS", "0.05")), dt=1.0,
                    t_end=40000.0 * (0.05 / float(os.environ.get("V1_EPS", "0.05"))) ** 2)}
MU, N, STRENGTH = -1.0, 32, 3.0     # input coefficient |eta| = STRENGTH |mu alpha|


for name in sys.argv[1:] or list(CASES):
    p = CASES[name]
    L = np.sqrt(10) * 2 * np.pi / critical_wavenumber(p["preset"](), p["rng"])
    model = p["preset"](length=L)
    onset = find_onset(model, 2, p["rng"])
    nf = cubic_coefficients(model, onset)
    a, eps, H = MU * nf.alpha, p["eps"], model.drive / 2
    eta1 = pair_forcing_coefficient(model, onset, 6 * 2 * np.pi / L, H, "2w0")
    peak = STRENGTH * abs(a) / abs(eta1)
    eq = standing_equilibrium(a, nf.b, cross_coefficient(model, onset, np.arccos(0.8)), peak * eta1)
    shell = ShellSystem(model, onset, shell_waves(10), a, forcing=[((6, 0), peak * H, "2w0")])
    locked = np.zeros(len(shell.waves), complex)
    for k in [(3, 1), (3, -1)]:
        locked[shell.waves.index(k)] = eq["z"]
    top = np.sort(shell.jacobian_eigenvalues(locked).real)[::-1][:3]
    print(f"== {name}: L = {L:.3f}, |k0|^2 = {onset.k0**2:.0f}, input peak = {peak:.3f} eps^2")
    print(f"   predicted |A(3,+-1)| = {eps * np.sqrt(eq['rho']):.5f}; "
          f"largest shell eigenvalues {np.round(top, 4)} -> {'stable' if top[1] < 0 else 'unstable'}")

    forcing = Forcing((Wave(peak * eps**2, (6, 0), -2 * onset.omega0),))   # cos(6 q1 x + 2 w0 t)
    run = simulate(model, onset.kappa + MU * eps**2, forcing, dim=2, n=N, dt=p["dt"],
                   t_end=p["t_end"], save_every=20, noise=1e-3 * eps, seed=1)
    amps = {k: abs(critical_amplitudes(run, onset, k)[0]) for k in shell.waves}
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        i = np.argmin(abs(run.t - frac * run.t[-1]))
        pair = (amps[(3, 1)][i] + amps[(3, -1)][i]) / 2
        others = max(v[i] for k, v in amps.items() if k not in [(3, 1), (3, -1)])
        print(f"   t = {run.t[i]:7.0f}: mean |A(3,+-1)| = {pair:.5f}, largest other wave = {others:.2e}")
    sys.stdout.flush()
