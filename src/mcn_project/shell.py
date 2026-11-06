"""Amplitude equations for all critical waves of a 2D lattice shell (report Sec. 8).

Each wave k of the shell |k|^2 = r2 has an amplitude A_k. The equation of A_k contains
every cubic monomial A_X conj(A_Y) A_Z with kX - kY + kZ = k, and every forcing term
allowed by the wavevector rule of Sec. 8.
"""

from itertools import combinations_with_replacement

import numpy as np

from .amplitude import standing_equilibrium
from .linear import Onset
from .model import NeuralField
from .normal_form import cubic_coupling, pair_forcing_coefficient


def shell_waves(r2: int) -> list[tuple[int, int]]:
    n = int(np.sqrt(r2)) + 1
    return [(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1) if i * i + j * j == r2]


class ShellSystem:
    """dA_k/dtau = a A_k + cubic terms + forcing terms, for k in waves.

    forcing: list of (K, H, frequency) with K an integer lattice vector, H the input harmonic
    over populations, frequency "2w0" (input e^{i(2 w0 t + K.x)}) or "0" (static, both +-K).
    """

    def __init__(self, model: NeuralField, onset: Onset, waves, a, forcing=()):
        self.waves, self.a = [tuple(k) for k in waves], a
        index = {k: i for i, k in enumerate(self.waves)}
        q1 = 2 * np.pi / model.length
        self.cubic = []          # (out, X, Y, Z, coefficient)
        for X, Z in combinations_with_replacement(range(len(waves)), 2):
            for Y in range(len(waves)):
                out = tuple(np.add(np.subtract(self.waves[X], self.waves[Y]), self.waves[Z]))
                if out in index:
                    coef = cubic_coupling(model, onset, self.waves[X], self.waves[Y], self.waves[Z])
                    self.cubic.append((index[out], X, Y, Z, coef))
        self.linear = []         # (out, source, coefficient, conjugate source?)
        for K, H, frequency in forcing:
            K = np.asarray(K)
            norm = q1 * np.linalg.norm(K)
            if frequency == "2w0":
                eta = pair_forcing_coefficient(model, onset, norm, H, "2w0")
                self.linear += [(index[tuple(K - k)], index[k], eta, True)
                                for k in self.waves if tuple(K - k) in index]
            else:
                for sign, h in ((1, np.asarray(H)), (-1, np.conj(H))):
                    eta = pair_forcing_coefficient(model, onset, norm, h, "0")
                    self.linear += [(index[tuple(np.add(k, sign * K))], index[k], eta, False)
                                    for k in self.waves if tuple(np.add(k, sign * K)) in index]

    def rhs(self, A):
        out = self.a * A
        for o, X, Y, Z, coef in self.cubic:
            out[o] += coef * A[X] * np.conj(A[Y]) * A[Z]
        for o, src, eta, conj in self.linear:
            out[o] += eta * (np.conj(A[src]) if conj else A[src])
        return out

    def jacobian_eigenvalues(self, A, h=1e-7):
        """Eigenvalues of the real Jacobian at A (central differences)."""
        n = len(A)
        x0 = np.concatenate([A.real, A.imag])

        def f(x):
            r = self.rhs(x[:n] + 1j * x[n:])
            return np.concatenate([r.real, r.imag])

        J = np.column_stack([(f(x0 + d) - f(x0 - d)) / (2 * h) for d in h * np.eye(2 * n)])
        return np.linalg.eigvals(J)


def _top_eigenvalues(system: ShellSystem, locked: dict, count: int = 3):
    A = np.zeros(len(system.waves), complex)
    for k, z in locked.items():
        A[system.waves.index(k)] = z
    return np.sort(system.jacobian_eigenvalues(A).real)[::-1][:count]


def locked_pair(model: NeuralField, onset: Onset, a, r2: int, pair, K, H) -> dict:
    """Prediction for an input H e^{i(2 w0 t + K.x)} that couples the shell pair (kX, kY).

    Returns the coupling eta, the equilibrium of report Prop. 2 (|A_kX| = |A_kY| = sqrt(rho)),
    and the largest real parts of the Jacobian in the whole shell (one zero: translation).
    """
    kX, kY = (np.asarray(k, float) for k in pair)
    theta = np.arccos(np.clip(kX @ kY / (kX @ kX), -1, 1))
    eta = pair_forcing_coefficient(model, onset, 2 * np.pi * np.linalg.norm(K) / model.length, H, "2w0")
    b = cubic_coupling(model, onset, kX, kX, kX)
    eq = standing_equilibrium(a, b, cubic_coupling(model, onset, kX, kY, kY) if theta > 0 else 0, eta)
    system = ShellSystem(model, onset, shell_waves(r2), a, forcing=[(K, H, "2w0")])
    return dict(eta=eta, theta=theta, rho=eq["rho"], z=eq["z"],
                eigenvalues=_top_eigenvalues(system, {tuple(pair[0]): eq["z"], tuple(pair[1]): eq["z"]}))


def flicker_states(model: NeuralField, onset: Onset, a, H) -> dict:
    """Stripe and square under uniform input H e^{2i w0 t} on the axial shell (report Sec. 10.4)."""
    k = round(onset.k0)
    system = ShellSystem(model, onset, shell_waves(k * k), a, forcing=[((0, 0), H, "2w0")])
    eta = pair_forcing_coefficient(model, onset, 0, H, "2w0")
    b = cubic_coupling(model, onset, [1, 0], [1, 0], [1, 0])
    c = cubic_coupling(model, onset, [1, 0], [-1, 0], [-1, 0])
    c_perp = cubic_coupling(model, onset, [1, 0], [0, 1], [0, 1])
    c_x = cubic_coupling(model, onset, [0, 1], [-1, 0], [0, -1])
    out = {}
    for name, gamma, waves in [("stripe", b + c, [(k, 0), (-k, 0)]),
                               ("square", b + c + 2 * c_perp + c_x, [(k, 0), (-k, 0), (0, k), (0, -k)])]:
        eq = standing_equilibrium(a, gamma / 2, gamma / 2, eta)
        out[name] = dict(rho=eq["rho"], eigenvalues=_top_eigenvalues(system, {w: eq["z"] for w in waves}))
    return out
