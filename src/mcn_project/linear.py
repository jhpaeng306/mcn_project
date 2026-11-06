"""Linear analysis at u = 0: dispersion matrix, spectra, and the Hopf onset (paper Secs. 1, 8)."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .model import NeuralField


def linear_matrix(model: NeuralField, q, kappa: float) -> np.ndarray:
    """M(q) = D_tau^{-1} [-I + W-hat(q) D_F], shape q.shape + (K, K)."""
    f1, _, _ = model.activation_derivatives()
    return (-np.eye(model.K) + model.kernel_symbol(q, kappa) * f1) / model.tau[:, None]


def eigenvalues(model: NeuralField, q, kappa: float) -> np.ndarray:
    return np.linalg.eigvals(linear_matrix(model, q, kappa))


def growth_rate(model: NeuralField, q, kappa: float) -> np.ndarray:
    """Largest real part of the spectrum at each q."""
    return eigenvalues(model, q, kappa).real.max(axis=-1)


def lattice_shells(model: NeuralField, dim: int, tol: float = 1e-10) -> np.ndarray:
    """Distinct |k|^2 on the periodic lattice (k in Z^dim), up to negligible coupling.

    The wavenumber of shell r2 is q = 2 pi sqrt(r2) / L.
    """
    q_max = np.sqrt(2 * np.log(1 / tol) / model.Sigma[model.J1 + model.J0 != 0].min())
    n_max = int(np.ceil(q_max * model.length / (2 * np.pi)))
    n = np.arange(n_max + 1)
    squares = n**2 if dim == 1 else np.add.outer(n**2, n**2).ravel()
    return np.unique(squares)


def shell_q(model: NeuralField, r2) -> np.ndarray:
    return 2 * np.pi * np.sqrt(r2) / model.length


@dataclass(frozen=True)
class Onset:
    """First loss of stability of u = 0 as kappa increases."""
    kappa: float
    k0: float          # lattice wavenumber index, q0 = 2 pi k0 / L
    q0: float
    omega0: float
    Phi: np.ndarray    # M Phi = i omega0 Phi, |Phi| = 1
    Psi: np.ndarray    # M^* Psi = -i omega0 Psi, Psi^* Phi = 1

    @property
    def frequency(self) -> float:
        """Cycles per model time unit (Hz x 1e-3 when time is in ms)."""
        return self.omega0 / (2 * np.pi)


def find_onset(model: NeuralField, dim: int = 1, kappa_range=(0.0, 20.0),
               scan_points: int = 401) -> Onset:
    q = shell_q(model, lattice_shells(model, dim))

    def leading(kappa):
        return growth_rate(model, q, kappa).max()

    grid = np.linspace(*kappa_range, scan_points)
    values = np.array([leading(k) for k in grid])
    first = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))[0]
    kappa_c = brentq(leading, grid[first], grid[first + 1], xtol=1e-13)
    q0 = q[np.argmax(growth_rate(model, q, kappa_c))]
    Phi, Psi, nu = critical_eigenvectors(linear_matrix(model, q0, kappa_c))
    return Onset(kappa_c, q0 * model.length / (2 * np.pi), q0, abs(nu.imag), Phi, Psi)


def critical_eigenvectors(M: np.ndarray):
    """Right/left eigenvectors of the rightmost eigenvalue with positive imaginary part."""
    values, vectors = np.linalg.eig(M)
    top = np.flatnonzero(np.isclose(values.real, values.real.max(), atol=1e-8, rtol=0))
    j = top[np.argmax(values[top].imag)]
    nu, Phi = values[j], vectors[:, j]
    left_values, left_vectors = np.linalg.eig(M.conj().T)
    Psi = left_vectors[:, np.argmin(abs(left_values - np.conj(nu)))]
    return Phi, Psi / np.conj(np.vdot(Psi, Phi)), nu


def critical_wavenumber(model: NeuralField, kappa_range=(0.0, 20.0)) -> float:
    """Wavenumber at the minimum of the continuum neutral curve kappa_H(q).

    A torus of length L = 2 pi sqrt(r2) / q_c puts this wavenumber on the lattice shell r2.
    """
    q0 = find_onset(model, 1, kappa_range).q0

    def neutral(q):
        return brentq(lambda k: growth_rate(model, q, k), *kappa_range)

    return minimize_scalar(neutral, bounds=(0.7 * q0, 1.3 * q0), method="bounded").x
