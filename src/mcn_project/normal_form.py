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


def pair_forcing_coefficient(model: NeuralField, onset: Onset, K: float, H, frequency: str) -> complex:
    """Coefficient by which an input eps^2 (H e^{i(Omega t + K.x)} + c.c.) couples two critical waves.

    frequency = "2w0": Omega = 2 w0; the wave k_Y = K - k_X gains eta conj(A_X).
    frequency = "0":   Omega = 0 (static); the wave k_Y = k_X + K gains eta A_X.
    K is |K|. The paper's eta_t, eta_+, and the traveling eta_w are special cases.
    """
    m = _Maps(model, onset)
    H = np.asarray(H) / model.tau
    if frequency == "2w0":
        return 2 * np.vdot(onset.Psi, m.Q(onset.q0, onset.Phi.conj(), m.R(2 * onset.omega0, K, H)))
    return 2 * np.vdot(onset.Psi, m.Q(onset.q0, onset.Phi, m.R(0, K, H)))


def forcing_coefficients(model: NeuralField, onset: Onset, P=None, S=None):
    """(eta_t, eta_+, eta_-) for input I = eps^2 (P e^{2i w0 t} + S e^{2i q0 x} + c.c.).

    P and S are complex input harmonics over populations; a real cosine of peak d
    on drive vector e has harmonic d e / 2.
    """
    P = np.zeros(model.K) if P is None else np.asarray(P)
    S = np.zeros(model.K) if S is None else np.asarray(S)
    q = onset.q0
    return (pair_forcing_coefficient(model, onset, 0, P, "2w0"),
            pair_forcing_coefficient(model, onset, 2 * q, S, "0"),
            pair_forcing_coefficient(model, onset, 2 * q, S.conj(), "0"))


def unit_forcing_coefficients(model: NeuralField, onset: Onset):
    """eta_t and eta_x for cosine peak 1 on model.drive (paper's eta^(1))."""
    harmonic = model.drive / 2
    eta_t, eta_x, _ = forcing_coefficients(model, onset, P=harmonic, S=harmonic)
    return eta_t, eta_x


def cubic_coupling(model: NeuralField, onset: Onset, kX, kY, kZ) -> complex:
    """Coefficient of A_X conj(A_Y) A_Z in the equation of the wave kX - kY + kZ.

    kX, kY, kZ are critical wavevectors (|k| = q0, any common unit). Each pair forms a
    second-order response at the sum or difference of its wavevectors; the coefficient
    collects them as in eqs. (Gs)-(Gc). kX = kY = kZ gives b; the 1D pair gives c.
    """
    m = _Maps(model, onset)
    Phi, Psi, q, w = onset.Phi, onset.Psi, onset.q0, onset.omega0
    kX, kY, kZ = (np.atleast_1d(np.asarray(k, dtype=float)) for k in (kX, kY, kZ))
    scale = q / np.linalg.norm(kX)

    def h20(k):   # response to two critical waves, at frequency 2 w0
        n = scale * np.linalg.norm(k)
        return m.R(2 * w, n, m.Q(n, Phi, Phi))

    def h11(k):   # response to a wave and a conjugate wave, static
        n = scale * np.linalg.norm(k)
        return 2 * m.R(0, n, m.Q(n, Phi, Phi.conj()))

    if np.allclose(kX, kZ):
        G = 2 * m.Q(q, Phi.conj(), h20(2 * kX)) + 2 * m.Q(q, Phi, h11(kX - kY)) + 3 * m.C(q, Phi, Phi, Phi.conj())
    else:
        G = (2 * m.Q(q, Phi, h11(kX - kY)) + 2 * m.Q(q, Phi.conj(), 2 * h20(kX + kZ))
             + 2 * m.Q(q, Phi, h11(kZ - kY)) + 6 * m.C(q, Phi, Phi, Phi.conj()))
    return np.vdot(Psi, G)


def cross_coefficient(model: NeuralField, onset: Onset, theta: float) -> complex:
    """c_theta: coefficient of A_X |A_Y|^2 for two critical waves at angle theta.

    theta = pi gives c of the paper; theta = pi/2 gives c_perp.
    """
    kX, kY = np.array([1.0, 0.0]), np.array([np.cos(theta), np.sin(theta)])
    return cubic_coupling(model, onset, kX, kY, kY)


@dataclass(frozen=True)
class SquareCoefficients:
    """Extra cubic coefficients on the axial square shell.

    z1' = z1 (a + b|z1|^2 + c|w1|^2 + c_perp (|z2|^2 + |w2|^2)) + c_q conj(w1) z2 w2,
    where (z1, w1) travel along -x and +x and (z2, w2) along -y and +y.
    """
    c_perp: complex
    c_q: complex


def square_coefficients(model: NeuralField, onset: Onset) -> SquareCoefficients:
    c_q = cubic_coupling(model, onset, [0, 1], [-1, 0], [0, -1])     # z2 conj(w1) w2 -> z1
    return SquareCoefficients(cross_coefficient(model, onset, np.pi / 2), c_q)


def static_forcing_coefficients(model: NeuralField, onset: Onset, km: float, S):
    """(chi, xi) for a static input eps (S e^{i km x} + c.c.), km not in {0, 2 q0}.

    The amplitude equation gains A chi (all km) and B xi (only km = q0); both are O(|S|^2).
    """
    m = _Maps(model, onset)
    Phi, Psi, q, w = onset.Phi, onset.Psi, onset.q0, onset.omega0
    sigma = m.R(0, km, np.asarray(S) / model.tau)
    chi = (2 * m.Q(q, Phi, 2 * m.R(0, 0, m.Q(0, sigma, sigma.conj())))
           + 4 * m.Q(q, sigma.conj(), m.R(w, q + km, m.Q(q + km, Phi, sigma)))
           + 4 * m.Q(q, sigma, m.R(w, abs(q - km), m.Q(abs(q - km), Phi, sigma.conj())))
           + 6 * m.C(q, Phi, sigma, sigma.conj()))
    xi = 0
    if np.isclose(km, q):
        xi = np.vdot(Psi, 2 * m.Q(q, Phi, m.R(0, 2 * q, m.Q(2 * q, sigma, sigma)))
                     + 4 * m.Q(q, sigma, m.R(w, 0, m.Q(0, Phi, sigma)))
                     + 3 * m.C(q, Phi, sigma, sigma))
    return np.vdot(Psi, chi), xi
