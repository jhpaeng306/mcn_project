"""Cubic amplitude-equation coefficients for the neural field (paper Secs. 4-7).

Notation follows the paper: Q_q, C_q are the Fourier quadratic and cubic maps at
output wavenumber q, R(Omega, q) = (i Omega I - M(q))^{-1}, and
    A_tau = A (a + b|A|^2 + c|B|^2) + eta_t conj(B) + eta_+ B,   a = mu alpha.
"""

from dataclasses import dataclass

import numpy as np

from .linear import Onset, linear_matrix
from .model import NeuralField


@dataclass(frozen=True)
class NormalForm:
    alpha: complex
    b: complex
    c: complex


class _Maps:
    """Q_q, C_q, R and M_1 of the field, evaluated at onset."""

    def __init__(self, model: NeuralField, onset: Onset):
        self.model, self.kappa = model, onset.kappa
        self.f1, self.f2, self.f3 = model.activation_derivatives()

    def Q(self, q, v, w):
        m = self.model
        return m.kernel_symbol(q, self.kappa) @ (self.f2 * v * w / 2) / m.tau

    def C(self, q, u, v, w):
        m = self.model
        return m.kernel_symbol(q, self.kappa) @ (self.f3 * u * v * w / 6) / m.tau

    def R(self, Omega, q, v):
        M = linear_matrix(self.model, q, self.kappa)
        return np.linalg.solve(1j * Omega * np.eye(self.model.K) - M, v)

    def M1(self, q):
        m = self.model
        return m.J1 * np.exp(-m.Sigma * q**2 / 2) * self.f1 / m.tau[:, None]


def cubic_coefficients(model: NeuralField, onset: Onset) -> NormalForm:
    """alpha, b, c of eqs. (transversality) and (abc)."""
    m = _Maps(model, onset)
    Phi, Psi, q, w = onset.Phi, onset.Psi, onset.q0, onset.omega0
    Phib = Phi.conj()
    h20p = m.R(2 * w, 2 * q, m.Q(2 * q, Phi, Phi))
    h20_0 = 2 * m.R(2 * w, 0, m.Q(0, Phi, Phi))
    h11_0 = 2 * m.R(0, 0, m.Q(0, Phi, Phib))
    h11p = 2 * m.R(0, 2 * q, m.Q(2 * q, Phi, Phib))
    Gs = 2 * m.Q(q, Phi, h11_0) + 2 * m.Q(q, Phib, h20p) + 3 * m.C(q, Phi, Phi, Phib)
    Gc = (2 * m.Q(q, Phi, h11_0) + 2 * m.Q(q, Phib, h20_0) + 2 * m.Q(q, Phi, h11p)
          + 6 * m.C(q, Phi, Phi, Phib))
    return NormalForm(alpha=np.vdot(Psi, m.M1(q) @ Phi), b=np.vdot(Psi, Gs), c=np.vdot(Psi, Gc))


def forcing_coefficients(model: NeuralField, onset: Onset, P=None, S=None):
    """(eta_t, eta_+, eta_-) for input I = eps^2 (P e^{2i w0 t} + S e^{2i q0 x} + c.c.).

    P and S are complex input harmonics over populations; a real cosine of peak d
    on drive vector e has harmonic d e / 2.
    """
    m = _Maps(model, onset)
    Phi, Psi, q, w = onset.Phi, onset.Psi, onset.q0, onset.omega0
    zero = np.zeros(model.K)
    p2w = m.R(2 * w, 0, (zero if P is None else np.asarray(P)) / model.tau)
    s2q = m.R(0, 2 * q, (zero if S is None else np.asarray(S)) / model.tau)
    eta_t = 2 * np.vdot(Psi, m.Q(q, Phi.conj(), p2w))
    eta_plus = 2 * np.vdot(Psi, m.Q(q, Phi, s2q))
    eta_minus = 2 * np.vdot(Psi, m.Q(q, Phi, s2q.conj()))
    return eta_t, eta_plus, eta_minus


def unit_forcing_coefficients(model: NeuralField, onset: Onset):
    """eta_t and eta_x for cosine peak 1 on model.drive (paper's eta^(1))."""
    harmonic = model.drive / 2
    eta_t, eta_x, _ = forcing_coefficients(model, onset, P=harmonic, S=harmonic)
    return eta_t, eta_x
