"""K-population neural field with Gaussian kernels (paper Sec. 8).

    D_tau u_t = -u + W(kappa) * F(u) + I(x, t),
    W_ij-hat(q) = J_ij(kappa) exp(-Sigma_ij q^2 / 2),   J(kappa) = J0 + kappa J1,
    F_j(u) = l(g_j (u - theta_j)) - l(-g_j theta_j),     l = logistic.

Rows index target populations and columns index source populations.
The paper's bifurcation parameter is p = kappa - kappa_c.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class NeuralField:
    J0: np.ndarray
    J1: np.ndarray
    Sigma: np.ndarray
    tau: np.ndarray
    gain: np.ndarray
    theta: np.ndarray
    length: float
    names: tuple[str, ...]
    drive: np.ndarray

    @property
    def K(self) -> int:
        return len(self.names)

    def J(self, kappa: float) -> np.ndarray:
        return self.J0 + kappa * self.J1

    def kernel_symbol(self, q, kappa: float) -> np.ndarray:
        """W-hat(q) with shape q.shape + (K, K)."""
        q = np.asarray(q, dtype=float)[..., None, None]
        return self.J(kappa) * np.exp(-self.Sigma * q**2 / 2)

    def activation(self, u: np.ndarray) -> np.ndarray:
        """F(u) for u with population axis first."""
        shape = (self.K,) + (1,) * (u.ndim - 1)
        g, th = self.gain.reshape(shape), self.theta.reshape(shape)
        return expit(g * (u - th)) - expit(-g * th)

    def activation_derivatives(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """F'(0), F''(0), F'''(0) (paper eq. v1-activation-derivatives)."""
        r, g = expit(-self.gain * self.theta), self.gain
        s = r * (1 - r)
        return g * s, g**2 * s * (1 - 2 * r), g**3 * s * (1 - 6 * r + 6 * r**2)
