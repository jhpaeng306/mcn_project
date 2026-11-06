"""Measurements on simulated fields, in the paper's amplitude coordinates."""

import numpy as np

from .linear import Onset
from .simulate import Run


def critical_amplitudes(run: Run, onset: Onset, mode: tuple[int, ...] | None = None):
    """Project the field onto u = A Phi E_+ + B Phi E_- + c.c. (eq. u1).

    The Fourier coefficient at +k0 is A Phi e^{i w0 t} + conj(B Phi) e^{-i w0 t}.
    In 2D, mode selects the lattice vector (default (k0, 0)).
    """
    dim = run.u.ndim - 2
    mode = mode or (round(onset.k0),) + (0,) * (dim - 1)
    axes = tuple(range(2, 2 + dim))
    coeff = np.fft.fftn(run.u, axes=axes)[(slice(None), slice(None)) + mode] / run.x.size**dim
    basis = np.column_stack([onset.Phi, onset.Phi.conj()])
    c_plus, c_minus = np.linalg.lstsq(basis, coeff.T, rcond=None)[0]
    fast = np.exp(-1j * onset.omega0 * run.t)
    return c_plus * fast, np.conj(c_minus) * fast


def spatial_power(field: np.ndarray) -> np.ndarray:
    """Time-averaged power per spatial Fourier mode of field (time, *grid), mean removed."""
    axes = tuple(range(1, field.ndim))
    power = np.abs(np.fft.fftn(field, axes=axes)) ** 2
    power[(slice(None),) + (0,) * (field.ndim - 1)] = 0
    return power.mean(axis=0)


def dominant_mode(field: np.ndarray) -> tuple[int, ...]:
    """Lattice vector with the largest power, folded to nonnegative first index."""
    power = spatial_power(field)
    n = power.shape[0]
    index = np.unravel_index(np.argmax(power), power.shape)
    k = tuple(int(i if i <= n // 2 else i - n) for i in index)
    return tuple(-ki for ki in k) if k[0] < 0 else k


def dominant_frequency(t: np.ndarray, signal: np.ndarray) -> float:
    """Angular frequency of the largest nonzero peak of a uniformly sampled signal."""
    power = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    omega = 2 * np.pi * np.fft.rfftfreq(t.size, t[1] - t[0])
    return omega[1:][np.argmax(power[1:])]
