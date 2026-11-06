"""Amplitude equations for all critical waves of a 2D lattice shell (report Sec. 8).

Each wave k of the shell |k|^2 = r2 has an amplitude A_k. The equation of A_k contains
every cubic monomial A_X conj(A_Y) A_Z with kX - kY + kZ = k, and every forcing term
allowed by the wavevector rule of Sec. 8.
"""

from itertools import combinations_with_replacement

import numpy as np

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
