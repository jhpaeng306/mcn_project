"""Direct 2D field runs that check the square-lattice amplitude theory (report Sec. 8).

Checks, for Wilson-Cowan (L=58.26) and V1 (L=43):
  1. c_perp: above onset, a small y-wave on an x-traveling wave grows at eps^2 Re(a + c_perp rho_T).
  2. stripe transverse eigenvalue (uses c_perp and c_q) under temporal forcing at 2 w0.
  3. square amplitude eps sqrt(rho_sq) (uses gamma_sq = b + c + 2 c_perp + c_q).
Usage: uv run python theory/code/check_square_2d.py   (about 5 minutes)
"""
import json
import os
import sys

import numpy as np

from mcn_project import presets
from mcn_project.amplitude import standing_equilibrium
from mcn_project.analysis import critical_amplitudes
from mcn_project.linear import find_onset
from mcn_project.normal_form import cubic_coefficients, square_coefficients, unit_forcing_coefficients
from mcn_project.simulate import Forcing, Wave, grid, simulate

ONLY = sys.argv[2:]
CASES = {
    "WC": dict(model=presets.wilson_cowan(length=58.264198846330785), rng=(4, 9), eps=0.1, peak=1.0,
               n=32, dt=0.1, t_end=3000.0, save_every=20),
    "V1": dict(model=presets.v1(), rng=(0, 1), eps=float(os.environ.get("V1_EPS", "0.05")), peak=3.0,
               n=32, dt=1.0, t_end=24000.0 * (0.05 / float(os.environ.get("V1_EPS", "0.05"))) ** 2,
               save_every=20),
}


def field(model, onset, n, amps):
    """eps-scaled critical field 2 Re(Phi sum_j z_j e^{i k_j . x}) for z = (z1, w1, z2, w2)."""
    _, (X, Y) = grid(model, n, 2)
    q = onset.q0
    waves = [np.exp(1j * q * X), np.exp(-1j * q * X), np.exp(1j * q * Y), np.exp(-1j * q * Y)]
    return 2 * np.real(onset.Phi[:, None, None] * sum(z * w for z, w in zip(amps, waves)))


def four_amplitudes(run, onset):
    k = round(onset.k0)
    z1, w1 = critical_amplitudes(run, onset, (k, 0))
    z2, w2 = critical_amplitudes(run, onset, (0, k))
    return np.array([z1, w1, z2, w2])


def linear_slope(t, y, ref):
    """Exponential rate of y while it is still small (below ref / 30), skipping the first third."""
    small = np.flatnonzero(y > ref / 30)
    stop = small[0] if small.size else len(t)
    keep = slice(stop // 3, stop)       # skip the initial transient
    return np.polyfit(t[keep], np.log(y[keep]), 1)[0]


results = {}
for name, p in CASES.items():
    if ONLY and name not in ONLY:
        continue
    model, eps = p["model"], p["eps"]
    onset = find_onset(model, 2, p["rng"])
    nf, sq = cubic_coefficients(model, onset), square_coefficients(model, onset)
    eta_t = p["peak"] * unit_forcing_coefficients(model, onset)[0]
    run_kw = dict(dim=2, n=p["n"], dt=p["dt"], t_end=p["t_end"], save_every=p["save_every"])
    out = {}

    # 1. c_perp, above onset (mu = +1), x-traveling wave at its amplitude plus a small y-wave
    a = nf.alpha
    rho_T = -a.real / nf.b.real
    u0 = field(model, onset, p["n"], eps * np.array([np.sqrt(rho_T), 0, 1e-3 * np.sqrt(rho_T), 0]))
    run = simulate(model, onset.kappa + eps**2, Forcing(), u0=u0, **run_kw)
    amps = four_amplitudes(run, onset)
    out["c_perp: y-wave growth"] = dict(
        measured=linear_slope(run.t, abs(amps[2]), eps * np.sqrt(rho_T)) / eps**2,
        predicted=(a + sq.c_perp * rho_T).real)

    # 2. stripe under temporal forcing (mu = -1), small transverse waves
    a = -nf.alpha
    forcing = Forcing((Wave(p["peak"] * eps**2, (), 2 * onset.omega0),))
    st = standing_equilibrium(a, nf.b, nf.c, eta_t)
    zs = st["z"]
    h, l = a + 2 * sq.c_perp * st["rho"], eta_t + sq.c_q * zs**2
    lam = (h.real + np.sqrt(complex(abs(l) ** 2 - h.imag**2))).real
    u0 = field(model, onset, p["n"], eps * np.array([zs, zs, 1e-3 * np.sqrt(st["rho"]), 0]))
    run = simulate(model, onset.kappa - eps**2, forcing, u0=u0, **run_kw)
    amps = four_amplitudes(run, onset)
    trans = np.sqrt(abs(amps[2]) ** 2 + abs(amps[3]) ** 2)
    out["stripe: transverse rate"] = dict(measured=linear_slope(run.t, trans, eps * np.sqrt(st["rho"])) / eps**2,
                                          predicted=lam)
    calm = trans < 0.03 * eps * np.sqrt(st["rho"])
    out["stripe: |z1| while 1D"] = dict(measured=float(abs(amps[0][calm][len(amps[0][calm]) // 2:]).mean()),
                                     predicted=eps * np.sqrt(st["rho"]))

    # 3. square-symmetric state under the same forcing
    gamma_sq = nf.b + nf.c + 2 * sq.c_perp + sq.c_q
    sqr = standing_equilibrium(a, gamma_sq / 2, gamma_sq / 2, eta_t)
    zq = sqr["z"]
    u0 = field(model, onset, p["n"], eps * np.array([zq] * 4))
    run = simulate(model, onset.kappa - eps**2, forcing, u0=u0, **run_kw)
    amps = four_amplitudes(run, onset)
    zbar = abs(amps).mean(axis=0)
    h, l = a + 2 * gamma_sq * sqr["rho"], -a                 # report eq. (65) with eta_x = 0
    lam_in = (h.real + np.sqrt(complex(abs(l) ** 2 - h.imag**2))).real
    out["square: in-subspace eigenvalue"] = dict(measured=np.nan, predicted=lam_in)
    out["square: |z| (first 10%)"] = dict(measured=float(zbar[: len(zbar) // 10].mean()),
                                          predicted=eps * np.sqrt(sqr["rho"]))
    out["square: max |u| start -> end"] = dict(measured=float(abs(run.u[-1]).max()),
                                               predicted=float(abs(run.u[0]).max()))
    results[name] = out
    for key, v in out.items():
        pred = "-" if v["predicted"] is None else f"{v['predicted']:+.5g}"
        print(f"{name:3s} {key:28s} measured {v['measured']:+.5g}   predicted {pred}")
    sys.stdout.flush()

json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), default=float, indent=1)
