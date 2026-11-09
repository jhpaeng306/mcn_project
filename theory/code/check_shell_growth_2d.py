"""Checks behind statements of report Secs. 10.5-10.6.

A. V1, x-traveling input K = (6, 0) on the oblique shell |k|^2 = 10 (notebooks/v1_2d.ipynb, route 1a, seeds 0-2):
   linear rates of the neighbouring shells |k|^2 = 9, 13, and their size and decay in the direct runs.
B. Wilson-Cowan, same input (theory/code/check_oblique_2d.py setup, seeds 1-3): the unstable directions of the
   shell Jacobian at the locked pair, and the growth of each other shell wave after the pair locks.
C. V1, uniform flicker, stripe on the axial shell at eps = 0.05 (theory/code/check_square_2d.py setup), run to
   saturation. A stripe does not depend on y, so the 1D field on the same length and grid gives it exactly.
Usage: uv run python theory/code/check_shell_growth_2d.py   (about 1.5 minutes)
"""
import numpy as np

from mcn_project import presets
from mcn_project.amplitude import standing_equilibrium
from mcn_project.analysis import critical_amplitudes, shell_amplitudes
from mcn_project.linear import critical_wavenumber, find_onset, growth_rate
from mcn_project.normal_form import cubic_coefficients, pair_forcing_coefficient, unit_forcing_coefficients
from mcn_project.shell import ShellSystem, locked_pair, shell_waves
from mcn_project.simulate import Forcing, Wave, grid, simulate, simulate_seeds

SHELL, PAIR, K, MU, STRENGTH, N = 10, [(3, 1), (3, -1)], (6, 0), -1.0, 3.0, 32
V1 = dict(rng=(0.0, 2.0), eps=0.025, dt=1.0, t_end=80_000.0, save_every=100, noise=0.1, seeds=[0, 1, 2])
WC = dict(rng=(4.0, 12.0), eps=0.1, dt=0.1, t_end=9000.0, save_every=20, noise=1e-3, seeds=[1, 2, 3])
WINDOWS = [(1e-5, 1e-4), (1e-4, 1e-2)]     # growth fits while the slow part of eps|A_k| is in these ranges
STRIPE = dict(eps=0.05, peak=3.0, dt=1.0, t_end=200_000.0, save_every=20)


def setup(preset, rng, eps):
    L = np.sqrt(SHELL) * 2 * np.pi / critical_wavenumber(preset(), rng)
    model = preset(length=L)
    onset = find_onset(model, 2, rng)
    a = MU * cubic_coefficients(model, onset).alpha
    eta_unit = pair_forcing_coefficient(model, onset, 2 * np.pi * np.hypot(*K) / L, model.drive / 2, "2w0")
    peak = STRENGTH * abs(a) / abs(eta_unit)
    forcing = Forcing((Wave(peak * eps**2, K, -2 * onset.omega0),))
    return model, onset, a, peak * model.drive / 2, forcing, onset.kappa + MU * eps**2


def max_fourier(run, waves, mask):
    """Largest |Fourier coefficient| over populations and the given lattice vectors, per saved time in mask."""
    coeff = np.fft.fft2(run.u[mask], axes=(2, 3)) / run.x.size**2
    return np.max([np.abs(coeff[:, :, i, j]).max(axis=1) for i, j in waves], axis=0)


def slow_part(t, A, omega0, width):
    """|s(t)| in A = s + r e^{-2i omega0 t}, fitted by least squares on sliding windows of the given width."""
    out = np.full(t.size, np.nan)
    for i, ti in enumerate(t):
        w = np.abs(t - ti) <= width / 2
        basis = np.column_stack([np.ones(w.sum()), np.exp(-2j * omega0 * t[w])])
        out[i] = abs(np.linalg.lstsq(basis, A[w], rcond=None)[0][0])
    return out


def neighbour_shells_v1():
    print("== A. V1, neighbouring shells (eps = 0.025)")
    model, onset, a, H, forcing, kappa = setup(presets.v1, V1["rng"], V1["eps"])
    q1 = 2 * np.pi / model.length
    for r2 in (9, 10, 13):
        print(f"   shell {r2:2d}: linear max Re nu = {growth_rate(model, q1 * np.sqrt(r2), kappa):+.3e} per ms")
    runs = simulate_seeds(model, kappa, forcing, V1["seeds"], dim=2, n=N, dt=V1["dt"], t_end=V1["t_end"],
                          save_every=V1["save_every"], noise=V1["noise"] * V1["eps"])
    end = lambda run: run.t == run.t[-1]
    for seed, run in zip(V1["seeds"], runs):
        late = run.t >= 10_000
        rates = {r2: np.polyfit(run.t[late], np.log(max_fourier(run, shell_waves(r2), late)), 1)[0] for r2 in (9, 13)}
        both = shell_waves(9) + shell_waves(13)
        print(f"   seed {seed}: shells 9, 13 largest |u_k| = {max_fourier(run, both, run.t >= 40_000).max():.2e} "
              f"after 40 s, {max_fourier(run, both, end(run))[0]:.2e} at 80 s; decay 10-80 s: "
              f"shell 9 {rates[9]:+.2e}, shell 13 {rates[13]:+.2e} per ms; pair |u_k| at 80 s = "
              f"{max_fourier(run, PAIR, end(run))[0]:.2e}, eps|A| = {abs(critical_amplitudes(run, onset, PAIR[0])[0][-1]):.5f}")


def growth_wc():
    print("== B. Wilson-Cowan, growth after the pair locks (eps = 0.1)")
    eps = WC["eps"]
    model, onset, a, H, forcing, kappa = setup(presets.wilson_cowan, WC["rng"], eps)
    pred = locked_pair(model, onset, a, SHELL, PAIR, K, H)
    system = ShellSystem(model, onset, shell_waves(SHELL), a, forcing=[(K, H, "2w0")])
    A = np.zeros(len(system.waves), complex)
    for k in PAIR:
        A[system.waves.index(k)] = pred["z"]
    n = len(A)
    x0 = np.concatenate([A.real, A.imag])

    def f(x):
        r = system.rhs(x[:n] + 1j * x[n:])
        return np.concatenate([r.real, r.imag])

    J = np.column_stack([(f(x0 + d) - f(x0 - d)) / 2e-7 for d in 1e-7 * np.eye(2 * n)])
    values, vectors = np.linalg.eig(J)
    for i in np.argsort(-values.real):
        if values[i].real > 1e-6:
            weight = np.abs(vectors[:n, i]) ** 2 + np.abs(vectors[n:, i]) ** 2
            print(f"   Jacobian eigenvalue {values[i].real:+.4f} per slow time on waves "
                  f"{[system.waves[j] for j in np.flatnonzero(weight > 0.05 * weight.max())]}")
    others = [k for k in system.waves if k not in PAIR]
    runs = simulate_seeds(model, kappa, forcing, WC["seeds"], dim=2, n=N, dt=WC["dt"], t_end=WC["t_end"],
                          save_every=WC["save_every"], noise=WC["noise"] * eps)
    for seed, run in zip(WC["seeds"], runs):
        amps = shell_amplitudes(run, onset, system.waves)
        slow = {k: slow_part(run.t, amps[k], onset.omega0, 2 * 2 * np.pi / onset.omega0) for k in others}
        i_lock = np.argmax(abs(amps[PAIR[0]]) > 0.9 * eps * np.sqrt(pred["rho"]))
        after = np.arange(run.t.size) >= i_lock
        print(f"   seed {seed}: pair locks at t = {run.t[i_lock]:.0f}; largest other slow part there "
              f"{max(s[i_lock] for s in slow.values()):.1e}")
        for k in others:
            fits = []
            for lo, hi in WINDOWS:
                w = after & (slow[k] > lo) & (slow[k] < hi)
                fits.append(f"{np.polyfit(run.t[w], np.log(slow[k][w]), 1)[0] / eps**2:+.2f}" if w.sum() >= 5 else "  -  ")
            print(f"      wave {str(k):8s} slow part at lock {slow[k][i_lock]:.1e}, reaches 1e-4 at t = "
                  f"{run.t[np.argmax(after & (slow[k] > 1e-4))]:.0f}; rate per slow time {fits[0]} in {WINDOWS[0]}, "
                  f"{fits[1]} in {WINDOWS[1]}")


def stripe_v1():
    print("== C. V1 stripe under flicker, eps = 0.05, run to saturation (1D field, L = 43, n = 32)")
    model, eps = presets.v1(), STRIPE["eps"]
    onset = find_onset(model, 1, (0.0, 1.0))
    nf = cubic_coefficients(model, onset)
    eq = standing_equilibrium(MU * nf.alpha, nf.b, nf.c, STRIPE["peak"] * unit_forcing_coefficients(model, onset)[0])
    x = grid(model, N, 1)[0]
    u0 = 4 * eps * np.real(eq["z"] * onset.Phi)[:, None] * np.cos(onset.q0 * x)   # A = B = z*, as in check_square_2d.py
    forcing = Forcing((Wave(STRIPE["peak"] * eps**2, (), 2 * onset.omega0),))
    run = simulate(model, onset.kappa + MU * eps**2, forcing, n=N, dt=STRIPE["dt"], t_end=STRIPE["t_end"],
                   save_every=STRIPE["save_every"], u0=u0)
    A, B = critical_amplitudes(run, onset)
    for lo, hi in [(0, 2_000), (12_000, 24_000), (45_000, 50_000), (195_000, 200_000)]:
        w = (run.t > lo) & (run.t <= hi)
        print(f"   t = {lo / 1000:5.0f}-{hi / 1000:3.0f} s: mean eps|A|, eps|B| = {abs(A[w]).mean():.5f}, {abs(B[w]).mean():.5f}")
    print(f"   theory eps sqrt(rho*) = {eps * np.sqrt(eq['rho']):.5f}")


if __name__ == "__main__":   # simulate_seeds starts worker processes
    neighbour_shells_v1()
    growth_wc()
    stripe_v1()
