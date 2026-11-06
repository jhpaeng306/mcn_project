"""Two-wave amplitude dynamics and projection of numerical field data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fft import BifurcationPoint, ForcingCoefficients, NormalFormCoefficients
from .simulation import Array, SimulationResult


@dataclass(frozen=True)
class AmplitudeTrajectory:
    t: Array
    A: Array
    B: Array
    a: complex
    eta_t: complex
    eta_x: complex


def project_critical_amplitudes(
    result: SimulationResult, critical: BifurcationPoint
) -> tuple[Array, Array]:
    """Project a direct simulation onto the counter-propagating Hopf pair."""
    coefficient = np.fft.fft(result.u, axis=2)[:, :, critical.mode] / result.u.shape[2]
    basis = np.column_stack([critical.phi, np.conj(critical.phi)])
    modal = np.linalg.lstsq(basis, coefficient.T, rcond=None)[0].T
    A = modal[:, 0] * np.exp(-1j * critical.omega0 * result.t)
    B = np.conj(modal[:, 1] * np.exp(1j * critical.omega0 * result.t))
    return A, B


def amplitude_rhs(
    A: complex,
    B: complex,
    *,
    a: complex,
    b: complex,
    c: complex,
    eta_t: complex = 0.0,
    eta_x: complex = 0.0,
) -> tuple[complex, complex]:
    """Equation (53), retaining the conjugation hidden by PDF extraction."""
    dA = A * (a + b * abs(A) ** 2 + c * abs(B) ** 2) + eta_t * np.conj(B) + eta_x * B
    dB = B * (a + b * abs(B) ** 2 + c * abs(A) ** 2) + eta_t * np.conj(A) + eta_x * A
    return dA, dB


def zero_state_eigenvalues(
    normal: NormalFormCoefficients,
    *,
    p: float,
    forcing: ForcingCoefficients | None = None,
) -> Array:
    """Return the four critical-mode eigenvalues at ``A=B=0``.

    This is the exact doubled complex linearization in report equation (89).
    It provides a quantitative distinction between a resonant input and a
    resonant input that is still too weak to destabilize the critical mode.
    Off-resonant forcing has no autonomous leading-order projection, so pass
    ``forcing=None`` for that case.
    """
    a = normal.alpha * (float(p) - normal.critical.p0)
    eta_t = 0.0j if forcing is None else forcing.eta_t
    eta_x = 0.0j if forcing is None else forcing.eta_x
    M = np.array([[a, eta_x], [eta_x, a]], dtype=complex)
    N = np.array([[0.0, eta_t], [eta_t, 0.0]], dtype=complex)
    jacobian = np.block([[M, N], [np.conj(N), np.conj(M)]])
    return np.linalg.eigvals(jacobian)


def zero_state_growth_rate(
    normal: NormalFormCoefficients,
    *,
    p: float,
    forcing: ForcingCoefficients | None = None,
) -> float:
    """Largest real critical-mode growth rate at the zero-amplitude state."""
    return float(np.max(zero_state_eigenvalues(normal, p=p, forcing=forcing).real))


def integrate_amplitudes(
    t: Array,
    normal: NormalFormCoefficients,
    *,
    p: float,
    initial: tuple[complex, complex] = (1e-3 + 0j, 8e-4 + 2e-4j),
    forcing: ForcingCoefficients | None = None,
    max_amplitude: float = 20.0,
) -> AmplitudeTrajectory:
    """Integrate the normal form on the supplied (typically simulation) axis."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be a strictly increasing one-dimensional array")
    eta_t = 0.0j if forcing is None else forcing.eta_t
    eta_x = 0.0j if forcing is None else forcing.eta_x
    a = normal.alpha * (float(p) - normal.critical.p0)
    A_history = np.full(t.size, np.nan + 1j * np.nan)
    B_history = np.full(t.size, np.nan + 1j * np.nan)
    A_history[0], B_history[0] = map(complex, initial)

    def rhs(A: complex, B: complex) -> tuple[complex, complex]:
        return amplitude_rhs(
            A, B, a=a, b=normal.b, c=normal.c, eta_t=eta_t, eta_x=eta_x
        )

    for index, step in enumerate(np.diff(t), start=1):
        A, B = A_history[index - 1], B_history[index - 1]
        k1a, k1b = rhs(A, B)
        k2a, k2b = rhs(A + 0.5 * step * k1a, B + 0.5 * step * k1b)
        k3a, k3b = rhs(A + 0.5 * step * k2a, B + 0.5 * step * k2b)
        k4a, k4b = rhs(A + step * k3a, B + step * k3b)
        next_A = A + step * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0
        next_B = B + step * (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0
        if (
            not np.isfinite(next_A)
            or not np.isfinite(next_B)
            or max(abs(next_A), abs(next_B)) > max_amplitude
        ):
            break
        A_history[index], B_history[index] = next_A, next_B
    return AmplitudeTrajectory(t, A_history, B_history, a, eta_t, eta_x)


def unforced_branch_amplitudes(
    normal: NormalFormCoefficients, p_values: Array
) -> dict[str, Array]:
    """Analytical cubic branch magnitudes versus the absolute parameter p."""
    p_values = np.asarray(p_values, dtype=float)
    growth = normal.alpha.real * (p_values - normal.critical.p0)
    traveling_squared = -growth / normal.b.real
    standing_squared = -growth / (normal.b + normal.c).real
    traveling = np.sqrt(np.maximum(traveling_squared, 0.0))
    standing = np.sqrt(np.maximum(standing_squared, 0.0))
    traveling[traveling_squared <= 0.0] = np.nan
    standing[standing_squared <= 0.0] = np.nan
    return {
        "homogeneous": np.zeros_like(p_values),
        "traveling": traveling,
        "standing": standing,
    }


def classify_wave(
    A: Array,
    B: Array,
    *,
    tail_fraction: float = 0.2,
    min_amplitude: float = 5e-4,
) -> str:
    """Classify late projected amplitudes as homogeneous/traveling/standing/mixed."""
    A = np.asarray(A)
    B = np.asarray(B)
    start = max(0, int((1.0 - tail_fraction) * A.size))
    a = float(np.nanmean(np.abs(A[start:])))
    b = float(np.nanmean(np.abs(B[start:])))
    high, low = max(a, b), min(a, b)
    if min_amplitude < 0.0:
        raise ValueError("min_amplitude must be non-negative")
    if not np.isfinite(high) or high < min_amplitude:
        return "homogeneous"
    ratio = low / high
    if ratio < 0.35:
        return "traveling wave"
    if ratio > 0.72:
        return "standing-wave-like"
    return "mixed counter-propagating wave"
