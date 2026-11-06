"""Fourier, bifurcation, resolvent, and normal-form calculations.

The coefficient names follow equations (34), (43), (51), and (53) of
``final_report.pdf``.  Unlike the abstract report, the multilinear maps here
are derived from the concrete thresholded-logistic field in ``simulation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .simulation import Array, ModelParameters, activation_derivatives


@dataclass(frozen=True)
class BifurcationPoint:
    p0: float
    mode: int
    q: float
    eigenvalue: complex
    phi: Array
    psi: Array

    @property
    def omega0(self) -> float:
        return abs(self.eigenvalue.imag)

    @property
    def frequency_cycles_per_time(self) -> float:
        return self.omega0 / (2.0 * np.pi)

    @property
    def frequency_hz(self) -> float:
        """Frequency in Hz when the model's time unit is milliseconds."""
        return 1000.0 * self.frequency_cycles_per_time


@dataclass(frozen=True)
class NormalFormCoefficients:
    critical: BifurcationPoint
    alpha: complex
    b: complex
    c: complex
    f1: Array
    f2: Array
    f3: Array
    M0: Array
    M1: Array
    q20: Array
    q11: Array
    h20_plus: Array
    h20_zero: Array
    h11_zero: Array
    h11_plus: Array
    Gs: Array
    Gc: Array

    @property
    def supercritical(self) -> bool:
        return self.b.real < 0.0 and (self.b + self.c).real < 0.0

    @property
    def preferred_pattern(self) -> str:
        if not self.supercritical:
            return "not saturated by the cubic truncation"
        return "traveling wave" if self.c.real < self.b.real else "standing wave"


@dataclass(frozen=True)
class ForcingCoefficients:
    p2omega: Array
    s2k: Array
    eta_t: complex
    eta_x: complex


def gaussian_symbol(model: ModelParameters, q: float, p: float = 0.0) -> Array:
    """Return ``(A+p*A_p) exp(-Sigma*q^2/2)``."""
    if q < 0.0:
        q = abs(q)
    return model.connectivity(p) * np.exp(-0.5 * model.Sigma * q**2)


def linear_symbol(model: ModelParameters, q: float, p: float = 0.0) -> Array:
    """Linear Fourier matrix at physical angular wavenumber ``q``."""
    f1, _, _ = activation_derivatives(model)
    return (-np.eye(model.K) + gaussian_symbol(model, q, p) * f1[None, :]) / model.tau[
        :, None
    ]


def modal_spectrum(
    model: ModelParameters, p_values: Array, *, max_mode: int = 16
) -> tuple[Array, Array]:
    """Return modes and all K eigenvalues for every bifurcation parameter."""
    p_values = np.asarray(p_values, dtype=float)
    modes = np.arange(max_mode + 1)
    eigenvalues = np.empty((p_values.size, modes.size, model.K), dtype=complex)
    for i, p in enumerate(p_values):
        for j, mode in enumerate(modes):
            q = 2.0 * np.pi * mode / model.length
            values = np.linalg.eigvals(linear_symbol(model, q, float(p)))
            eigenvalues[i, j] = values[np.argsort(values.real)[::-1]]
    return modes, eigenvalues


def _leading_eigenvalue(model: ModelParameters, mode: int, p: float) -> complex:
    q = 2.0 * np.pi * mode / model.length
    values = np.linalg.eigvals(linear_symbol(model, q, p))
    return complex(values[np.argmax(values.real)])


def compute_bifurcation(
    model: ModelParameters,
    *,
    p_bounds: tuple[float, float] = (0.0, 3.0),
    max_mode: int = 16,
    scan_points: int = 401,
    tolerance: float = 1e-11,
) -> BifurcationPoint:
    """Find the first loss of stability as ``p`` increases.

    Every modal crossing is bracketed separately, which avoids assuming that
    the identity of the leading mode stays fixed over the whole interval.
    """
    p_min, p_max = map(float, p_bounds)
    if not p_min < p_max or scan_points < 3:
        raise ValueError("p_bounds must be increasing and scan_points >= 3")
    if (
        max(
            _leading_eigenvalue(model, mode, p_min).real for mode in range(max_mode + 1)
        )
        >= 0.0
    ):
        raise ValueError("the lower p bound is not linearly stable")

    grid = np.linspace(p_min, p_max, scan_points)
    crossings: list[tuple[float, int]] = []
    for mode in range(max_mode + 1):
        growth = np.array([_leading_eigenvalue(model, mode, p).real for p in grid])
        indices = np.flatnonzero((growth[:-1] < 0.0) & (growth[1:] >= 0.0))
        if indices.size:
            index = int(indices[0])
            root = brentq(
                lambda value, selected_mode=mode: (
                    _leading_eigenvalue(model, selected_mode, value).real
                ),
                grid[index],
                grid[index + 1],
                xtol=tolerance,
            )
            crossings.append((float(root), mode))
    if not crossings:
        raise ValueError("no modal instability was found in p_bounds")

    p0, mode = min(crossings)
    q = 2.0 * np.pi * mode / model.length
    matrix = linear_symbol(model, q, p0)
    values, vectors = np.linalg.eig(matrix)
    max_real = float(np.max(values.real))
    candidates = np.flatnonzero(np.isclose(values.real, max_real, atol=1e-8, rtol=0.0))
    index = int(candidates[np.argmax(values[candidates].imag)])
    eigenvalue = complex(values[index])
    phi = vectors[:, index]

    left_values, left_vectors = np.linalg.eig(matrix.conj().T)
    left_index = int(np.argmin(np.abs(left_values - np.conj(eigenvalue))))
    psi = left_vectors[:, left_index]
    pairing = np.vdot(psi, phi)
    if abs(pairing) < 1e-12:
        raise np.linalg.LinAlgError("critical left/right eigenvectors are orthogonal")
    psi = psi / np.conj(pairing)
    return BifurcationPoint(p0, mode, q, eigenvalue, phi, psi)


def compute_normal_form(
    model: ModelParameters, critical: BifurcationPoint
) -> NormalFormCoefficients:
    """Compute the report's ``alpha``, ``b``, and ``c`` and all intermediates."""
    if critical.omega0 <= 1e-10:
        raise ValueError("the critical eigenvalue is not a Hopf pair")
    K = model.K
    q = critical.q
    omega = critical.omega0
    phi, psi, p0 = critical.phi, critical.psi, critical.p0
    f1, f2, f3 = activation_derivatives(model)

    def matrix(k: float) -> Array:
        return linear_symbol(model, abs(k), p0)

    def quadratic(v: Array, w: Array, output_k: float) -> Array:
        product = 0.5 * f2 * v * w
        return (gaussian_symbol(model, abs(output_k), p0) @ product) / model.tau

    def cubic(u: Array, v: Array, w: Array, output_k: float) -> Array:
        product = (f3 / 6.0) * u * v * w
        return (gaussian_symbol(model, abs(output_k), p0) @ product) / model.tau

    def resolve(frequency: float, k: float, rhs: Array) -> Array:
        return np.linalg.solve(1j * frequency * np.eye(K) - matrix(k), rhs)

    q20 = quadratic(phi, phi, 2.0 * q)
    q11 = quadratic(phi, np.conj(phi), 0.0)
    h20_plus = resolve(2.0 * omega, 2.0 * q, q20)
    h20_zero = 2.0 * resolve(2.0 * omega, 0.0, quadratic(phi, phi, 0.0))
    h11_zero = 2.0 * resolve(0.0, 0.0, q11)
    h11_plus = 2.0 * resolve(0.0, 2.0 * q, quadratic(phi, np.conj(phi), 2.0 * q))

    Gs = (
        2.0 * quadratic(phi, h11_zero, q)
        + 2.0 * quadratic(np.conj(phi), h20_plus, q)
        + 3.0 * cubic(phi, phi, np.conj(phi), q)
    )
    Gc = (
        2.0 * quadratic(phi, h11_zero, q)
        + 2.0 * quadratic(np.conj(phi), h20_zero, q)
        + 2.0 * quadratic(phi, h11_plus, q)
        + 6.0 * cubic(phi, phi, np.conj(phi), q)
    )
    perturbation_symbol = model.A_p * np.exp(-0.5 * model.Sigma * q**2)
    M1 = (perturbation_symbol * f1[None, :]) / model.tau[:, None]
    M0 = matrix(q)
    alpha = complex(np.vdot(psi, M1 @ phi))
    return NormalFormCoefficients(
        critical=critical,
        alpha=alpha,
        b=complex(np.vdot(psi, Gs)),
        c=complex(np.vdot(psi, Gc)),
        f1=f1,
        f2=f2,
        f3=f3,
        M0=M0,
        M1=M1,
        q20=q20,
        q11=q11,
        h20_plus=h20_plus,
        h20_zero=h20_zero,
        h11_zero=h11_zero,
        h11_plus=h11_plus,
        Gs=Gs,
        Gc=Gc,
    )


def compute_forcing_coefficients(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    *,
    temporal_harmonic: Array | None = None,
    spatial_harmonic: Array | None = None,
) -> ForcingCoefficients:
    """Project the report's resonant 2:1 temporal and spatial forcing.

    Inputs are the *complex Fourier coefficients*.  Therefore a real forcing
    ``d*cos(2*omega*t)`` is passed here as ``d/2``.  The resulting amplitude
    equations contain ``eta_t*conj(B)`` and ``eta_x*B``.
    """
    critical = normal.critical
    phi, psi = critical.phi, critical.psi
    q, omega, p0 = critical.q, critical.omega0, critical.p0
    temporal = (
        np.zeros(model.K)
        if temporal_harmonic is None
        else np.asarray(temporal_harmonic, dtype=complex)
    )
    spatial = (
        np.zeros(model.K)
        if spatial_harmonic is None
        else np.asarray(spatial_harmonic, dtype=complex)
    )
    if temporal.shape != (model.K,) or spatial.shape != (model.K,):
        raise ValueError("forcing harmonics must have length K")

    def resolve(frequency: float, k: float, rhs: Array) -> Array:
        matrix = linear_symbol(model, abs(k), p0)
        return np.linalg.solve(1j * frequency * np.eye(model.K) - matrix, rhs)

    def quadratic(v: Array, w: Array, output_k: float) -> Array:
        product = 0.5 * normal.f2 * v * w
        return (gaussian_symbol(model, abs(output_k), p0) @ product) / model.tau

    p2omega = resolve(2.0 * omega, 0.0, temporal / model.tau)
    s2k = resolve(0.0, 2.0 * q, spatial / model.tau)
    eta_t = complex(2.0 * np.vdot(psi, quadratic(np.conj(phi), p2omega, q)))
    eta_x = complex(2.0 * np.vdot(psi, quadratic(phi, s2k, q)))
    return ForcingCoefficients(p2omega, s2k, eta_t, eta_x)
