"""Direct simulation against linear theory and internal consistency."""
import numpy as np
import pytest

from mcn_project import presets
from mcn_project.analysis import critical_amplitudes, dominant_mode
from mcn_project.linear import find_onset, growth_rate
from mcn_project.simulate import Forcing, Wave, critical_seed, grid, simulate


@pytest.fixture(scope="module")
def wc():
    model = presets.wilson_cowan()
    return model, find_onset(model, 1, (4, 9))


def test_small_critical_mode_grows_at_linear_rate(wc):
    model, onset = wc
    kappa = onset.kappa + 0.2
    run = simulate(model, kappa, n=60, t_end=300, u0=critical_seed(model, onset, 1e-7, n=60))
    A, _ = critical_amplitudes(run, onset)
    late = run.t > 100
    measured = np.polyfit(run.t[late], np.log(abs(A[late])), 1)[0]
    assert measured == pytest.approx(growth_rate(model, onset.q0, kappa), rel=1e-3)
    assert dominant_mode(run.u[:, 0]) == (3,)


def test_y_uniform_2d_run_equals_1d_run(wc):
    model, onset = wc
    forcing = Forcing((Wave(0.05, (), 2 * onset.omega0), Wave(0.03, (6,))))
    u0 = critical_seed(model, onset, 1e-3, n=32)
    run1 = simulate(model, onset.kappa - 0.1, forcing, n=32, t_end=20, u0=u0)
    u0_2d = np.repeat(u0[:, :, None], 32, axis=2)
    run2 = simulate(model, onset.kappa - 0.1, forcing, dim=2, n=32, t_end=20, u0=u0_2d)
    assert np.abs(run2.u[:, :, :, 5] - run1.u).max() < 1e-12


def test_critical_amplitudes_recover_known_values(wc):
    model, onset = wc
    x, _ = grid(model, 60, 1)
    t = np.linspace(0, 50, 11)
    A, B = 0.3 - 0.1j, 0.05 + 0.2j
    E_plus = np.exp(1j * (onset.omega0 * t[:, None] + onset.q0 * x))
    E_minus = np.exp(1j * (onset.omega0 * t[:, None] - onset.q0 * x))
    u = 2 * np.real(onset.Phi[None, :, None] * (A * E_plus + B * E_minus)[:, None, :])
    run = type("R", (), dict(u=u, t=t, x=x))
    A_meas, B_meas = critical_amplitudes(run, onset)
    assert np.allclose(A_meas, A) and np.allclose(B_meas, B)
