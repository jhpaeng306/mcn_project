import pytest

from mcn_project.amplitude import amplitude_rhs, standing_equilibrium


@pytest.mark.parametrize("eta_x", [0, 0.2 + 0.1j])
def test_standing_equilibrium_solves_amplitude_equations(eta_x):
    a, b, c, eta_t = -0.5 + 0.3j, -1 + 2j, -1.5 + 1j, 1.2 - 0.4j
    eq = standing_equilibrium(a, b, c, eta_t, eta_x)
    dA, dB = amplitude_rhs(eq["z"], eq["z"], a, b, c, eta_t, eta_x, eta_x)
    assert abs(dA) < 1e-12 and abs(dB) < 1e-12


def test_no_standing_equilibrium_below_threshold():
    assert standing_equilibrium(-1 + 0j, -1 + 0j, -1 + 0j, 0.5) is None
