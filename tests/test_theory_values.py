"""Onset and amplitude coefficients against the values printed in paper Sec. 7."""
import numpy as np
import pytest

from mcn_project import presets
from mcn_project.linear import find_onset
from mcn_project.normal_form import cubic_coefficients, unit_forcing_coefficients


@pytest.fixture(scope="module")
def v1():
    model = presets.v1()
    return model, find_onset(model, 1, (0, 1))


def test_v1_onset_matches_paper(v1):
    _, onset = v1
    assert onset.kappa == pytest.approx(0.7491489065, abs=1e-10)
    assert onset.k0 == pytest.approx(3)
    assert onset.omega0 == pytest.approx(0.0829864799, abs=1e-10)
    assert 1000 * onset.frequency == pytest.approx(13.2077085, abs=1e-6)


def test_v1_cubic_coefficients_match_paper(v1):
    nf = cubic_coefficients(*v1)
    assert nf.alpha == pytest.approx(0.0880577 + 0.0643807j, abs=1e-7)
    assert nf.b == pytest.approx(-6.706479 + 9.069454j, abs=1e-6)
    assert nf.c == pytest.approx(-6.961796 + 8.999612j, abs=1e-6)


def test_v1_forcing_coefficients_match_paper(v1):
    eta_t, eta_x = unit_forcing_coefficients(*v1)
    assert abs(eta_t) == pytest.approx(abs(0.055509 - 0.029075j), abs=1e-6)
    assert eta_x == pytest.approx(0.018902 + 0.046191j, abs=1e-6)


def test_three_population_onset_matches_legacy_readme():
    onset = find_onset(presets.v1(3), 1, (0, 1))
    assert onset.kappa == pytest.approx(0.5880341657, abs=1e-9)
    assert onset.k0 == pytest.approx(3)


def test_wilson_cowan_onset_matches_legacy_notebook():
    onset = find_onset(presets.wilson_cowan(), 1, (4, 9))
    assert onset.kappa == pytest.approx(7.024921, abs=1e-5)
    assert onset.k0 == pytest.approx(3)
    assert onset.omega0 == pytest.approx(0.318128, abs=1e-6)


def test_eigenvector_normalization(v1):
    _, onset = v1
    assert np.linalg.norm(onset.Phi) == pytest.approx(1)
    assert np.vdot(onset.Psi, onset.Phi) == pytest.approx(1)
