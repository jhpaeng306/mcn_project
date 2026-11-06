"""General cubic coupling, forcing coefficients, and the shell system (report Secs. 7-8)."""
import numpy as np
import pytest

from mcn_project import presets
from mcn_project.amplitude import standing_equilibrium
from mcn_project.linear import find_onset
from mcn_project.normal_form import (
    cross_coefficient,
    cubic_coefficients,
    cubic_coupling,
    pair_forcing_coefficient,
    square_coefficients,
    unit_forcing_coefficients,
)
from mcn_project.shell import ShellSystem, shell_waves


@pytest.fixture(scope="module")
def v1():
    model = presets.v1()
    return model, find_onset(model, 2, (0, 1))


def test_general_coupling_reduces_to_paper_coefficients(v1):
    nf = cubic_coefficients(*v1)
    assert cubic_coupling(*v1, [1, 0], [1, 0], [1, 0]) == pytest.approx(nf.b)
    assert cross_coefficient(*v1, np.pi) == pytest.approx(nf.c)
    assert square_coefficients(*v1).c_perp == pytest.approx(-7.489424 + 9.781553j, abs=1e-6)  # earlier draft


def test_pair_forcing_reduces_to_paper_and_draft_values(v1):
    model, onset = v1
    eta_t, eta_x = unit_forcing_coefficients(model, onset)
    H = model.drive / 2
    assert pair_forcing_coefficient(model, onset, 0, H, "2w0") == pytest.approx(eta_t)
    assert pair_forcing_coefficient(model, onset, 2 * onset.q0, H, "0") == pytest.approx(eta_x)
    eta_w = pair_forcing_coefficient(model, onset, 2 * onset.q0, H, "2w0")
    alpha = cubic_coefficients(model, onset).alpha
    assert abs(alpha) / abs(eta_w) == pytest.approx(1.10547, abs=1e-5)


@pytest.mark.parametrize("r2, pair, K", [(9, [(3, 0), (-3, 0)], (0, 0)), (10, [(3, 1), (3, -1)], (6, 0))])
def test_locked_pair_is_equilibrium_of_whole_shell(v1, r2, pair, K):
    model, onset = v1
    nf = cubic_coefficients(model, onset)
    a, H = -nf.alpha, 3 * model.drive / 2
    kX, kY = (np.array(k, float) for k in pair)
    theta = np.arccos(kX @ kY / (kX @ kX))
    eta = pair_forcing_coefficient(model, onset, 2 * np.pi * np.linalg.norm(K) / model.length, H, "2w0")
    eq = standing_equilibrium(a, nf.b, cross_coefficient(model, onset, theta), eta)
    shell = ShellSystem(model, onset, shell_waves(r2), a, forcing=[(K, H, "2w0")])
    A = np.zeros(len(shell.waves), complex)
    for k in pair:
        A[shell.waves.index(k)] = eq["z"]
    assert abs(shell.rhs(A)).max() < 1e-12


def test_continuation_equals_one_run():
    from mcn_project.simulate import Forcing, Wave, simulate
    model = presets.wilson_cowan()
    forcing = Forcing((Wave(0.01, (), 0.6), Wave(0.01, (6,))))
    full = simulate(model, 6.9, forcing, n=30, dt=0.1, t_end=20, save_every=10, seed=3)
    half = simulate(model, 6.9, forcing, n=30, dt=0.1, t_end=10, save_every=10, seed=3)
    rest = simulate(model, 6.9, forcing, n=30, dt=0.1, t_end=10, save_every=10, u0=half.u[-1], t_start=half.t[-1])
    assert np.allclose(rest.u[-1], full.u[-1], atol=1e-12) and rest.t[-1] == pytest.approx(full.t[-1])


def test_flicker_and_oblique_predictions_v1(v1):
    from mcn_project.linear import critical_wavenumber
    from mcn_project.shell import flicker_states, locked_pair
    model, onset = v1
    a = -cubic_coefficients(model, onset).alpha
    states = flicker_states(model, onset, a, 3 * model.drive / 2)
    assert states["stripe"]["rho"] == pytest.approx(0.0066795, rel=1e-4)
    assert states["stripe"]["eigenvalues"][1] == pytest.approx(-0.026074, abs=1e-5)   # transverse
    assert states["square"]["eigenvalues"][0] == pytest.approx(0.022131, abs=1e-5)    # symmetry breaking
    L = np.sqrt(10) * 2 * np.pi / critical_wavenumber(presets.v1(), (0, 2))
    m = presets.v1(length=L)
    o = find_onset(m, 2, (0, 2))
    pred = locked_pair(m, o, -cubic_coefficients(m, o).alpha, 10, [(3, 1), (3, -1)], (6, 0), 3.448 * m.drive / 2)
    assert pred["rho"] == pytest.approx(0.010359, rel=1e-3)
    assert abs(pred["eigenvalues"][0]) < 1e-8 and pred["eigenvalues"][1] < -0.1
