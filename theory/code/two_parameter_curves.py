"""Thresholds of a static stimulus q cos(2 pi m x / L) on E (Wilson-Cowan, L = 60, N = 60).

Reproduces the PI's 2parcurves figure: for each m, the a_EE at which the stimulus-locked
steady state first loses stability, for q in [-1.2, 0]. Writes results/two_parameter_curves.json (curves, and fine samples near q = 0).
Usage: uv run python theory/code/two_parameter_curves.py   (about 1 minute)
"""
import json

import numpy as np
from scipy.optimize import brentq

from mcn_project import presets
from mcn_project.linear import find_onset
from mcn_project.steady import locked_state_growth

N, MODES, STEP = 60, range(1, 7), 0.05
Q = np.concatenate([np.linspace(0, -0.1, 21)[1:], np.linspace(-0.1, -1.2, 23)[1:]])
model = presets.wilson_cowan()
x = np.arange(N) * model.length / N


def first_instability(stimulus):
    """Continue the locked state in a_EE from 3 and return the first crossing (a_EE, frequency)."""
    u, prev = None, None
    for kappa in np.arange(3.0, 12.0, STEP):
        growth, _, u_new = locked_state_growth(model, kappa, stimulus, u)
        if prev is not None and prev[0] < 0 <= growth:
            try:
                kc = brentq(lambda k: locked_state_growth(model, k, stimulus, prev[1])[0],
                            kappa - STEP, kappa, xtol=1e-6)
                return kc, locked_state_growth(model, kc, stimulus, prev[1])[1]
            except ValueError:          # the locked branch ends (fold) inside this step
                return kappa - STEP / 2, np.nan
        prev, u = (growth, u_new), u_new
    return np.nan, np.nan


curves = {m: [[q, *first_instability(q * np.cos(2 * np.pi * m * x / model.length))] for q in Q]
          for m in MODES}
# Fine samples near q = 0, both signs, for the comparison with the small-q theory.
onset = find_onset(model, 1, (4, 9))
near = {}
for m in (1, 2, 3, 6):
    near[m] = []
    for q in np.linspace(-0.012, 0.012, 13):
        stim = q * np.cos(2 * np.pi * m * x / model.length)
        near[m].append([q, brentq(lambda k: locked_state_growth(model, k, stim)[0],
                                  onset.kappa - 0.4, onset.kappa + 0.1, xtol=1e-10)])
json.dump(dict(curves=curves, near_zero=near), open("results/two_parameter_curves.json", "w"), indent=0)
for m, c in curves.items():
    c = np.array(c)
    print(f"m={m}: min a_EE = {np.nanmin(c[:, 1]):.3f} at q = {c[np.nanargmin(c[:, 1]), 0]:.2f}")
