"""Threshold shift of a static stimulus q cos(m q1 x) on E, Wilson-Cowan (report Sec. 7).

The full-model threshold comes from the stability of the stimulus-locked steady state.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from mcn_project import presets
from mcn_project.linear import find_onset
from mcn_project.normal_form import (
    cubic_coefficients,
    static_forcing_coefficients,
    unit_forcing_coefficients,
)
from mcn_project.steady import locked_state_growth

N = 60


@pytest.fixture(scope="module")
def wc():
    model = presets.wilson_cowan()
    return model, find_onset(model, 1, (4, 9))


def full_model_growth(model, kappa, stimulus):
    return locked_state_growth(model, kappa, stimulus)[0]


@pytest.mark.parametrize("mode", [2, 3])
def test_quadratic_threshold_shift(wc, mode):
    model, onset = wc
    x = np.arange(N) * model.length / N
    q = 0.005
    stim = q * np.cos(2 * np.pi * mode * x / model.length)
    kappa_th = brentq(lambda k: full_model_growth(model, k, stim), onset.kappa - 0.05, onset.kappa + 0.05,
                      xtol=1e-10)
    chi, xi = static_forcing_coefficients(model, onset, 2 * np.pi * mode / model.length, q * model.drive / 2)
    predicted = -(chi.real + abs(np.real(xi))) / cubic_coefficients(model, onset).alpha.real
    assert kappa_th - onset.kappa == pytest.approx(predicted, rel=0.02)


def test_linear_threshold_shift_at_twice_k0(wc):
    model, onset = wc
    x = np.arange(N) * model.length / N
    q = 0.002
    stim = q * np.cos(2 * np.pi * 6 * x / model.length)
    kappa_th = brentq(lambda k: full_model_growth(model, k, stim), onset.kappa - 0.1, onset.kappa + 0.01,
                      xtol=1e-10)
    eta_x = unit_forcing_coefficients(model, onset)[1]
    predicted = -q * abs(eta_x.real) / cubic_coefficients(model, onset).alpha.real
    assert kappa_th - onset.kappa == pytest.approx(predicted, rel=0.03)
