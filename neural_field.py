"""Compact multi-population neural-field and amplitude-equation tools.

The field equation is

    tau_i u_i,t = -u_i + F_i(sum_j W_ij G_ij * u_j + bias_i + input_i),

where W[row, col] means source ``col`` acts on target ``row`` and every
G_ij is a unit-mass Gaussian.  Positive columns are excitatory and negative
columns are inhibitory (Dale's-law convention).

Time is in ms.  Length is in the same spatial unit as ``widths``.  The V1
preset uses 0.1 mm as one spatial unit.  This file intentionally keeps the
model, coefficient calculation, simulation, and plotting in one place so the
assumptions are easy to revise.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter

Array = np.ndarray


def _vector(value: float | Iterable[float] | Array | None, size: int) -> Array:
    if value is None:
        return np.zeros(size)
    a = np.asarray(value, dtype=float)
    return np.full(size, float(a)) if a.ndim == 0 else a


@dataclass
class PopulationModel:
    """Parameters for an arbitrary number of cell populations."""

    names: tuple[str, ...]
    tau: Array
    weights: Array
    widths: Array
    rest: Array
    gain: Array
    length: float
    length_unit: str = "model units"
    provenance: str = "user supplied"

    def __post_init__(self) -> None:
        self.tau = np.asarray(self.tau, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.widths = np.asarray(self.widths, dtype=float)
        self.rest = np.asarray(self.rest, dtype=float)
        self.gain = np.asarray(self.gain, dtype=float)
        p = len(self.names)
        if self.tau.shape != (p,) or self.rest.shape != (p,) or self.gain.shape != (p,):
            raise ValueError("tau, rest, and gain must have one entry per population")
        if self.weights.shape != (p, p) or self.widths.shape != (p, p):
            raise ValueError(
                "weights and widths must be population-by-population matrices"
            )
        if np.any(self.tau <= 0) or np.any(self.widths <= 0):
            raise ValueError("time constants and Gaussian widths must be positive")
        if np.any((self.rest <= 0) | (self.rest >= 1)):
            raise ValueError("rest must lie strictly between zero and one")

    @property
    def populations(self) -> int:
        return len(self.names)

    def bias(self, coupling_scale: float = 1.0) -> Array:
        """Bias that keeps ``rest`` fixed as coupling is varied."""
        inverse = np.log(self.rest / (1.0 - self.rest)) / self.gain
        return inverse - coupling_scale * self.weights @ self.rest


def v1_three_population() -> PopulationModel:
    """Data-informed E/PV/SST research preset near a finite-wave Hopf point.

    Gaussian distance ratios use the E, Pvalb, and Sst values tabulated by
    Billeh et al. (Neuron 2020).  Their exp(-r^2/s^2) widths are divided by
    sqrt(2) to obtain standard deviations for exp(-r^2/(2 sigma^2)).  The
    signed rate-model weights are calibrated, not direct synaptic measurements.
    They place mode 3 near a 14 Hz finite-wavenumber Hopf bifurcation on a
    4.76 mm patch while the homogeneous mode remains damped.
    """
    widths_um = np.array(
        [
            [114.0, 95.0, 85.0],
            [92.0, 120.0, 120.0],
            [103.0, 120.0, 120.0],
        ]
    ) / np.sqrt(2.0)
    # Calibrated at rest=0.2 from a constrained search using the width matrix.
    weights = np.array(
        [
            [22.10349069, -14.35151870, -10.34719739],
            [18.11602148, -8.94069411, -12.82056905],
            [9.49422121, -2.59781804, -7.30344651],
        ]
    )
    return PopulationModel(
        names=("E", "PV", "SST"),
        tau=np.array([10.0, 12.0, 35.0]),
        weights=weights,
        widths=widths_um / 100.0,
        rest=np.full(3, 0.2),
        gain=np.ones(3),
        length=4.760231719455392 / 0.1,
        length_unit="0.1 mm",
        provenance=(
            "Billeh et al. 2020 Gaussian width ratios; dimensionless Wilson-Cowan "
            "weights calibrated near finite-wave Hopf onset"
        ),
    )


def v1_functional_five_population() -> PopulationModel:
    """E/PV/SST-GER/SST-local/VIP hypothesis constrained by V1 data.

    The E/PV/SST block inherits the Billeh et al. Gaussian width ratios and
    the calibrated three-population rate weights.  SST output is divided in
    the 66.1/33.9 proportion reported for deep-L2/3 GER/non-GER SST cells by
    Drinnenberg et al. (bioRxiv 2025).  The broader E-to-SST-GER kernel,
    distinct time constants, VIP loop, and all five-population rate gains are
    explicit modeling hypotheses; they are not an experimentally measured
    conductance matrix.  This preset asks whether the newly observed
    functional split is sufficient to retain a finite-wave Hopf instability.
    """
    ger_fraction = 0.661
    local_fraction = 1.0 - ger_fraction
    weights = np.array(
        [
            [
                22.10349069,
                -14.35151870,
                -10.34719739 * ger_fraction,
                -10.34719739 * local_fraction,
                -1.2,
            ],
            [
                18.11602148,
                -8.94069411,
                -12.82056905 * ger_fraction,
                -12.82056905 * local_fraction,
                -1.5,
            ],
            [
                12.5,
                -2.59781804,
                -7.30344651 * ger_fraction,
                -7.30344651 * local_fraction,
                -6.0,
            ],
            [
                7.5,
                -2.59781804,
                -7.30344651 * ger_fraction,
                -7.30344651 * local_fraction,
                -6.0,
            ],
            [
                8.0,
                -2.0,
                -5.0 * ger_fraction,
                -5.0 * local_fraction,
                -1.0,
            ],
        ]
    )
    # Standard deviations in 0.1 mm.  The 1.75 entry represents the broad
    # E-to-SST-GER integration range explored in the 2025 probabilistic model.
    widths = np.array(
        [
            [0.806, 0.672, 0.601, 0.601, 0.849],
            [0.651, 0.849, 0.849, 0.849, 0.849],
            [1.750, 0.849, 0.849, 0.849, 0.849],
            [0.728, 0.849, 0.849, 0.849, 0.849],
            [0.728, 0.849, 0.849, 0.849, 0.849],
        ]
    )
    return PopulationModel(
        names=("E", "PV", "SST-GER", "SST-local", "VIP"),
        tau=np.array([10.0, 12.0, 45.0, 25.0, 20.0]),
        weights=weights,
        widths=widths,
        rest=np.full(5, 0.2),
        gain=np.ones(5),
        length=4.760231719455392 / 0.1,
        length_unit="0.1 mm",
        provenance=(
            "Billeh et al. 2020 width ratios; Drinnenberg et al. 2025 "
            "SST-GER functional split; Wilson-Cowan gains calibrated/hypothesized"
        ),
    )


def rule_2011() -> PopulationModel:
    """Two-population benchmark from Rule, Stoffregen & Ermentrout (2011).

    The paper used exp(-(x/s)^2), hence the division by sqrt(2).  The absolute
    spatial scale was nondimensional.  This is a historical pattern-forming
    benchmark, not a claim that the weights or widths are measured V1 values.
    """
    return PopulationModel(
        names=("E", "I"),
        tau=np.array([10.0, 20.0]),
        weights=np.array([[10.0, -8.5], [12.0, -3.0]]),
        widths=np.array([[1.0, 2.5], [1.0, 2.5]]) / np.sqrt(2.0),
        rest=np.array([0.19386251, 0.16043885]),
        gain=np.ones(2),
        length=20.0,
        provenance="Rule, Stoffregen & Ermentrout 2011 benchmark",
    )


def sigmoid(x: Array, gain: Array) -> Array:
    shape = (gain.size,) + (1,) * (x.ndim - 1)
    z = np.clip(gain.reshape(shape) * x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def firing_derivatives(model: PopulationModel) -> tuple[Array, Array, Array]:
    """First three derivatives of each sigmoid at the chosen rest state."""
    r, g = model.rest, model.gain
    f1 = g * r * (1.0 - r)
    f2 = g**2 * r * (1.0 - r) * (1.0 - 2.0 * r)
    f3 = g**3 * r * (1.0 - r) * (1.0 - 6.0 * r + 6.0 * r**2)
    return f1, f2, f3


def gaussian_symbol(mu: Array, sigma: Array, q_squared: float) -> Array:
    """Fourier symbol ``mu * exp(-sigma**2*q_squared/2)`` for K cell types."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mu.ndim != 2 or mu.shape[0] != mu.shape[1] or sigma.shape != mu.shape:
        raise ValueError("mu and sigma must be matching square matrices")
    if np.any(sigma <= 0.0) or q_squared < 0.0:
        raise ValueError("Gaussian widths must be positive and q_squared nonnegative")
    return mu * np.exp(-0.5 * sigma**2 * q_squared)


def mixed_gaussian_symbol(
    mu: Array,
    sigma: Array,
    *,
    orientation_mode: int,
    planar_wavenumber: float,
    orientation_scale: float = 1.0,
    planar_sigma: Array | None = None,
) -> Array:
    """Gaussian symbol on S1 x R2, with at most one periodic coordinate.

    With one width matrix, distance is measured using
    q_squared=(n/orientation_scale)**2+|k|**2.  Supplying ``planar_sigma``
    allows independent orientation and planar widths.
    """
    if orientation_scale <= 0.0 or planar_wavenumber < 0.0:
        raise ValueError("orientation scale must be positive and |k| nonnegative")
    n_scaled = orientation_mode / orientation_scale
    if planar_sigma is None:
        return gaussian_symbol(
            mu, sigma, n_scaled**2 + planar_wavenumber**2
        )
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    planar_sigma = np.asarray(planar_sigma, dtype=float)
    if (
        mu.ndim != 2
        or mu.shape[0] != mu.shape[1]
        or sigma.shape != mu.shape
        or planar_sigma.shape != mu.shape
    ):
        raise ValueError("all connectivity parameters must be matching square matrices")
    if np.any(sigma <= 0.0) or np.any(planar_sigma <= 0.0):
        raise ValueError("Gaussian widths must be positive")
    decay = (sigma * n_scaled) ** 2 + (planar_sigma * planar_wavenumber) ** 2
    return mu * np.exp(-0.5 * decay)


def fourier_weights(
    model: PopulationModel, q: float, coupling_scale: float = 1.0
) -> Array:
    """Gaussian connectivity matrix at physical angular wavenumber ``q``."""
    return gaussian_symbol(
        coupling_scale * model.weights, model.widths, q_squared=q**2
    )


def linear_symbol(
    model: PopulationModel, q: float, coupling_scale: float = 1.0
) -> Array:
    f1, _, _ = firing_derivatives(model)
    return (
        -np.eye(model.populations)
        + f1[:, None] * fourier_weights(model, q, coupling_scale)
    ) / model.tau[:, None]


@dataclass(frozen=True)
class CriticalMode:
    scale: float
    mode: int
    q: float
    eigenvalue: complex
    phi: Array
    psi: Array

    @property
    def omega(self) -> float:
        return abs(self.eigenvalue.imag)

    @property
    def frequency_hz(self) -> float:
        return 1000.0 * self.omega / (2.0 * np.pi)


def leading_mode(
    model: PopulationModel, coupling_scale: float, max_mode: int = 16
) -> tuple[float, int, complex]:
    candidates: list[tuple[float, int, complex]] = []
    for mode in range(max_mode + 1):
        q = 2.0 * np.pi * mode / model.length
        eig = np.linalg.eigvals(linear_symbol(model, q, coupling_scale))
        value = eig[np.argmax(eig.real)]
        candidates.append((float(value.real), mode, complex(value)))
    return max(candidates, key=lambda item: item[0])


def find_critical_mode(
    model: PopulationModel,
    scale_bracket: tuple[float, float] = (0.85, 1.15),
    max_mode: int = 16,
    tolerance: float = 1e-11,
) -> CriticalMode:
    """Find the first instability while all weights are scaled together."""
    lo, hi = scale_bracket
    if (
        leading_mode(model, lo, max_mode)[0] >= 0
        or leading_mode(model, hi, max_mode)[0] <= 0
    ):
        raise ValueError("scale_bracket does not bracket the first instability")
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        if leading_mode(model, mid, max_mode)[0] < 0:
            lo = mid
        else:
            hi = mid
    _, mode, _ = leading_mode(model, hi, max_mode)
    q = 2.0 * np.pi * mode / model.length
    matrix = linear_symbol(model, q, hi)
    values, vectors = np.linalg.eig(matrix)
    j = np.argmax(values.imag)
    eigenvalue, phi = values[j], vectors[:, j]
    left_values, left_vectors = np.linalg.eig(matrix.conj().T)
    psi = left_vectors[:, np.argmin(np.abs(left_values + 1j * eigenvalue.imag))]
    psi = psi / np.conj(np.vdot(psi, phi))
    return CriticalMode(hi, mode, q, complex(eigenvalue), phi, psi)


@dataclass(frozen=True)
class NormalForm:
    alpha: complex
    b: complex
    c: complex
    critical: CriticalMode

    def pattern(self) -> str:
        if self.b.real >= 0:
            return "cubic truncation does not saturate supercritically"
        return "traveling wave" if self.c.real < self.b.real else "standing wave"


@dataclass(frozen=True)
class WaveEquilibrium:
    """One support equilibrium of an arbitrary cubic wave-amplitude system."""

    support: tuple[int, ...]
    inactive: tuple[int, ...]
    intensity: Array
    active_eigenvalues: Array
    invasion_rates: Array
    phase_velocity: Array
    stable_modulo_phase: bool


def wave_support_equilibria(
    linear: Array, cubic: Array, *, tolerance: float = 1e-10
) -> list[WaveEquilibrium]:
    """Enumerate all nonnegative equilibria of dA=a*A+A*(g@|A|^2).

    ``linear`` has one complex detuning per wave and ``cubic`` is the complex
    wave-by-wave interaction matrix.  Stability is radial stability plus
    resistance to invasion by every inactive wave; unforced phase directions
    remain neutral.
    """
    linear = np.asarray(linear, dtype=complex)
    cubic = np.asarray(cubic, dtype=complex)
    if linear.ndim != 1 or cubic.shape != (linear.size, linear.size):
        raise ValueError("linear must have length m and cubic shape (m, m)")
    m = linear.size
    growth, interaction = linear.real, cubic.real
    states: list[WaveEquilibrium] = []
    for mask in range(1 << m):
        support = tuple(index for index in range(m) if mask & (1 << index))
        inactive = tuple(index for index in range(m) if index not in support)
        intensity = np.zeros(m)
        if support:
            active = np.ix_(support, support)
            try:
                intensity[list(support)] = np.linalg.solve(
                    interaction[active], -growth[list(support)]
                )
            except np.linalg.LinAlgError:
                continue
            if np.any(intensity[list(support)] <= tolerance):
                continue
            radial = 2.0 * np.diag(intensity[list(support)]) @ interaction[active]
            active_eigenvalues = np.linalg.eigvals(radial)
        else:
            active_eigenvalues = np.empty(0, dtype=complex)
        invasion_rates = (
            2.0
            * (
                growth[list(inactive)]
                + interaction[np.ix_(inactive, range(m))] @ intensity
            )
            if inactive
            else np.empty(0)
        )
        phase_velocity = linear.imag + cubic.imag @ intensity
        stable = bool(
            np.all(active_eigenvalues.real < -tolerance)
            and np.all(invasion_rates < -tolerance)
        )
        states.append(
            WaveEquilibrium(
                support,
                inactive,
                intensity,
                active_eigenvalues,
                invasion_rates,
                phase_velocity,
                stable,
            )
        )
    return states


def normal_form_coefficients(
    model: PopulationModel, critical: CriticalMode
) -> NormalForm:
    """Compute alpha, self-coupling b, and counter-wave coupling c.

    The unfolding scales all connections and co-varies the bias so the stated
    homogeneous rest state remains fixed.  This makes M_1 unambiguous.
    """
    p, q, omega = model.populations, critical.q, critical.omega
    phi, psi, scale = critical.phi, critical.psi, critical.scale
    f1, f2, f3 = firing_derivatives(model)

    def w_hat(k: float) -> Array:
        return fourier_weights(model, abs(k), scale)

    def matrix(k: float) -> Array:
        return (-np.eye(p) + f1[:, None] * w_hat(k)) / model.tau[:, None]

    def ell(v: Array, k: float) -> Array:
        return w_hat(k) @ v

    def quadratic(v: Array, w: Array, kv: float, kw: float) -> Array:
        return f2 * ell(v, kv) * ell(w, kw) / (2.0 * model.tau)

    def cubic(u: Array, v: Array, w: Array, ku: float, kv: float, kw: float) -> Array:
        return f3 * ell(u, ku) * ell(v, kv) * ell(w, kw) / (6.0 * model.tau)

    def resolve(frequency: float, k: float, rhs: Array) -> Array:
        return np.linalg.solve(1j * frequency * np.eye(p) - matrix(k), rhs)

    q20 = quadratic(phi, phi, q, q)
    q11 = quadratic(phi, np.conj(phi), q, -q)
    h20_plus = resolve(2.0 * omega, 2.0 * q, q20)
    h20_zero = 2.0 * resolve(2.0 * omega, 0.0, quadratic(phi, phi, q, -q))
    h11_zero = 2.0 * resolve(0.0, 0.0, q11)
    h11_plus = 2.0 * resolve(0.0, 2.0 * q, quadratic(phi, np.conj(phi), q, q))

    gs = (
        2.0 * quadratic(phi, h11_zero, q, 0.0)
        + 2.0 * quadratic(np.conj(phi), h20_plus, -q, 2.0 * q)
        + 3.0 * cubic(phi, phi, np.conj(phi), q, q, -q)
    )
    gc = (
        2.0 * quadratic(phi, h11_zero, q, 0.0)
        + 2.0 * quadratic(np.conj(phi), h20_zero, q, 0.0)
        + 2.0 * quadratic(phi, h11_plus, -q, 2.0 * q)
        + 6.0 * cubic(phi, phi, np.conj(phi), q, -q, q)
    )
    # Derivative with respect to relative coupling p: W -> (1+p) W.
    m1 = f1[:, None] * w_hat(q) / model.tau[:, None]
    return NormalForm(
        alpha=complex(np.vdot(psi, m1 @ phi)),
        b=complex(np.vdot(psi, gs)),
        c=complex(np.vdot(psi, gc)),
        critical=critical,
    )


def angular_cross_coefficient(
    model: PopulationModel, normal_form: NormalForm, angle_degrees: float
) -> complex:
    """Two-dimensional cross-coupling for critical wavevectors at an angle."""
    critical = normal_form.critical
    p, q, omega = model.populations, critical.q, critical.omega
    phi, psi, scale = critical.phi, critical.psi, critical.scale
    _, f2, f3 = firing_derivatives(model)

    def w_hat(k: float) -> Array:
        return fourier_weights(model, abs(k), scale)

    def matrix(k: float) -> Array:
        f1, _, _ = firing_derivatives(model)
        return (-np.eye(p) + f1[:, None] * w_hat(k)) / model.tau[:, None]

    def quadratic(v: Array, w: Array, kv: float, kw: float) -> Array:
        return f2 * (w_hat(kv) @ v) * (w_hat(kw) @ w) / (2.0 * model.tau)

    def cubic(u: Array, v: Array, w: Array) -> Array:
        fields = (w_hat(q) @ u) * (w_hat(q) @ v) * (w_hat(q) @ w)
        return f3 * fields / (6.0 * model.tau)

    def resolve(frequency: float, k: float, rhs: Array) -> Array:
        return np.linalg.solve(1j * frequency * np.eye(p) - matrix(k), rhs)

    theta = np.deg2rad(angle_degrees)
    q_sum = 2.0 * q * np.cos(theta / 2.0)
    q_difference = 2.0 * q * np.sin(theta / 2.0)
    q20 = quadratic(phi, phi, q, q)
    q11 = quadratic(phi, np.conj(phi), q, q)
    h11_zero = 2.0 * resolve(0.0, 0.0, q11)
    h_sum = 2.0 * resolve(2.0 * omega, q_sum, q20)
    h_difference = 2.0 * resolve(0.0, q_difference, q11)
    cross = (
        2.0 * quadratic(phi, h11_zero, q, 0.0)
        + 2.0 * quadratic(np.conj(phi), h_sum, q, q_sum)
        + 2.0 * quadratic(phi, h_difference, q, q_difference)
        + 6.0 * cubic(phi, phi, np.conj(phi))
    )
    return complex(np.vdot(psi, cross))


@dataclass
class WeakForcing:
    """Separable weak forcing components added at the sigmoid input."""

    temporal: Array | float | None = None
    spatial: Array | float | None = None
    traveling: Array | float | None = None
    omega: float = 0.0  # radians / ms
    spatial_mode: tuple[int, ...] = (0,)
    phase: float = 0.0
    start: float = 0.0
    stop: float = np.inf


@dataclass
class Simulation:
    t: Array
    u: Array
    input: Array
    coordinates: tuple[Array, ...]
    model: PopulationModel
    boundary: str
    coupling_scale: float
    forcing: WeakForcing
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FloquetSpectrum:
    """Growth rates of spatial modes about a periodically forced uniform orbit."""

    frequency_hz: float
    modes: Array
    growth_per_ms: Array

    @property
    def pattern_mode(self) -> int:
        if self.growth_per_ms.size < 2:
            return 0
        return int(self.modes[1 + np.argmax(self.growth_per_ms[1:])])

    @property
    def pattern_growth_per_ms(self) -> float:
        if self.growth_per_ms.size < 2:
            return float("nan")
        return float(np.max(self.growth_per_ms[1:]))


def _forcing_value(
    forcing: WeakForcing, model: PopulationModel, t: float, grids: tuple[Array, ...]
) -> Array:
    shape = (model.populations,) + grids[0].shape
    result = np.zeros(shape)
    if not (forcing.start <= t < forcing.stop):
        return result
    modes = tuple(forcing.spatial_mode) + (0,) * (
        len(grids) - len(forcing.spatial_mode)
    )
    spatial_phase = sum(
        2.0 * np.pi * mode * grid / model.length for mode, grid in zip(modes, grids)
    )
    p = model.populations
    result += _vector(forcing.temporal, p).reshape((p,) + (1,) * len(grids)) * np.cos(
        forcing.omega * t + forcing.phase
    )
    result += _vector(forcing.spatial, p).reshape((p,) + (1,) * len(grids)) * np.cos(
        spatial_phase + forcing.phase
    )
    result += _vector(forcing.traveling, p).reshape((p,) + (1,) * len(grids)) * np.cos(
        spatial_phase - forcing.omega * t + forcing.phase
    )
    return result


def simulate(
    model: PopulationModel,
    *,
    dimension: int = 1,
    grid_points: int = 128,
    boundary: str = "periodic",
    coupling_scale: float = 1.01,
    forcing: WeakForcing | None = None,
    total_ms: float = 1200.0,
    dt_ms: float = 0.5,
    save_every: int = 4,
    initial_noise: float = 1e-3,
    seed: int = 1,
) -> Simulation:
    """Euler simulation in 1D or 2D with periodic, reflecting, or zero edges."""
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    if boundary not in ("periodic", "reflect", "zero"):
        raise ValueError("boundary must be periodic, reflect, or zero")
    forcing = WeakForcing() if forcing is None else forcing
    x = np.linspace(0.0, model.length, grid_points, endpoint=False)
    grids = tuple(np.meshgrid(*([x] * dimension), indexing="ij"))
    shape = (model.populations,) + (grid_points,) * dimension
    rng = np.random.default_rng(seed)
    u = model.rest.reshape(
        (model.populations,) + (1,) * dimension
    ) + initial_noise * rng.standard_normal(shape)
    bias = model.bias(coupling_scale).reshape((model.populations,) + (1,) * dimension)
    dx = model.length / grid_points

    if boundary == "periodic":
        frequencies = 2.0 * np.pi * np.fft.fftfreq(grid_points, d=dx)
        wave_grids = np.meshgrid(*([frequencies] * dimension), indexing="ij")
        q2 = sum(grid**2 for grid in wave_grids)
        matrix_shape = (model.populations, model.populations) + (1,) * dimension
        kernels = np.exp(
            -0.5
            * model.widths.reshape(matrix_shape) ** 2
            * q2.reshape((1, 1) + q2.shape)
        )
        weighted_kernels = model.weights.reshape(matrix_shape) * kernels

        def recurrent(state: Array) -> Array:
            state_hat = np.fft.fftn(state, axes=tuple(range(1, dimension + 1)))
            field_hat = coupling_scale * np.einsum(
                "ij...,j...->i...", weighted_kernels, state_hat
            )
            return np.fft.ifftn(field_hat, axes=tuple(range(1, dimension + 1))).real

    else:
        mode = "reflect" if boundary == "reflect" else "constant"

        def recurrent(state: Array) -> Array:
            field_value = np.zeros_like(state)
            for i in range(model.populations):
                for j in range(model.populations):
                    blurred = gaussian_filter(
                        state[j], model.widths[i, j] / dx, mode=mode, cval=0.0
                    )
                    field_value[i] += coupling_scale * model.weights[i, j] * blurred
            return field_value

    steps = round(total_ms / dt_ms)
    saved_t: list[float] = []
    saved_u: list[Array] = []
    saved_input: list[Array] = []
    for step in range(steps + 1):
        t = step * dt_ms
        external = _forcing_value(forcing, model, t, grids)
        if step % save_every == 0:
            saved_t.append(t)
            saved_u.append(u.copy())
            saved_input.append(external.copy())
        if step == steps:
            break
        rate = sigmoid(recurrent(u) + bias + external, model.gain)
        u += (
            dt_ms
            * (-u + rate)
            / model.tau.reshape((model.populations,) + (1,) * dimension)
        )
    return Simulation(
        t=np.asarray(saved_t),
        u=np.asarray(saved_u),
        input=np.asarray(saved_input),
        coordinates=grids,
        model=model,
        boundary=boundary,
        coupling_scale=coupling_scale,
        forcing=forcing,
        metadata={"dt_ms": dt_ms, "dimension": dimension, "grid_points": grid_points},
    )


def forcing_suite(
    model: PopulationModel, critical: CriticalMode, amplitude: float = 0.025
) -> dict[str, WeakForcing]:
    drive = np.zeros(model.populations)
    drive[0] = amplitude
    k = critical.mode
    return {
        "unforced": WeakForcing(),
        "direct_1to1": WeakForcing(
            traveling=drive, omega=critical.omega, spatial_mode=(k,)
        ),
        "temporal_2to1": WeakForcing(temporal=drive, omega=2.0 * critical.omega),
        "spatial_2to1": WeakForcing(spatial=drive, spatial_mode=(2 * k,)),
        "combined_2to1": WeakForcing(
            temporal=0.5 * drive,
            spatial=0.5 * drive,
            omega=2.0 * critical.omega,
            spatial_mode=(2 * k,),
        ),
    }


def floquet_spectrum(
    model: PopulationModel,
    *,
    coupling_scale: float,
    drive_frequency_hz: float,
    drive_amplitude: float = 0.025,
    drive_population: int = 0,
    max_mode: int = 12,
    settle_cycles: int = 24,
    samples_per_cycle: int = 120,
) -> FloquetSpectrum:
    """Compute the forced finite-wave instability without fitting amplitudes.

    A spatially uniform sinusoid is applied at the sigmoid input.  The uniform
    periodic orbit is first computed, then the variational equation for every
    Fourier mode is integrated for one period.  A positive finite-mode Floquet
    exponent means that the *model* predicts growth of a spatial pattern.  It
    is not a human hallucination threshold or a safety boundary.
    """
    if drive_frequency_hz <= 0.0:
        raise ValueError("drive_frequency_hz must be positive")
    if not 0 <= drive_population < model.populations:
        raise ValueError("drive_population is outside the population range")
    p = model.populations
    period = 1000.0 / drive_frequency_hz
    omega = 2.0 * np.pi * drive_frequency_hz / 1000.0
    drive = np.zeros(p)
    drive[drive_population] = drive_amplitude
    bias = model.bias(coupling_scale)

    def uniform_rhs(t: float, state: Array) -> Array:
        total = coupling_scale * model.weights @ state + bias
        total += drive * np.cos(omega * t)
        return (-state + sigmoid(total, model.gain)) / model.tau

    settled = solve_ivp(
        uniform_rhs,
        (0.0, settle_cycles * period),
        model.rest,
        rtol=2e-7,
        atol=1e-9,
        max_step=period / samples_per_cycle,
    )
    modes = np.arange(max_mode + 1)
    symbols = [
        fourier_weights(
            model, 2.0 * np.pi * mode / model.length, coupling_scale
        )
        for mode in modes
    ]
    identity = np.eye(p)
    initial = np.concatenate(
        [settled.y[:, -1]] + [identity.ravel() for _ in modes]
    )

    def joint_rhs(t: float, state: Array) -> Array:
        u = state[:p]
        total = coupling_scale * model.weights @ u + bias
        total += drive * np.cos(omega * t)
        rate = sigmoid(total, model.gain)
        derivative = model.gain * rate * (1.0 - rate)
        parts = [(-u + rate) / model.tau]
        offset = p
        for symbol in symbols:
            fundamental = state[offset : offset + p * p].reshape(p, p)
            jacobian = (
                -identity + derivative[:, None] * symbol
            ) / model.tau[:, None]
            parts.append((jacobian @ fundamental).ravel())
            offset += p * p
        return np.concatenate(parts)

    orbit = solve_ivp(
        joint_rhs,
        (0.0, period),
        initial,
        rtol=2e-7,
        atol=1e-9,
        max_step=period / samples_per_cycle,
    )
    final = orbit.y[:, -1]
    growth = np.empty(modes.size)
    offset = p
    for index in range(modes.size):
        monodromy = final[offset : offset + p * p].reshape(p, p)
        multiplier = np.max(np.abs(np.linalg.eigvals(monodromy)))
        growth[index] = np.log(max(multiplier, np.finfo(float).tiny)) / period
        offset += p * p
    return FloquetSpectrum(float(drive_frequency_hz), modes, growth)


def temporal_model_input(
    frequency_hz: float,
    *,
    duration_ms: float = 500.0,
    dt_ms: float = 0.5,
    amplitude: float = 0.025,
    phase: float = 0.0,
) -> tuple[Array, Array]:
    """Return a numerical sinusoidal drive for simulation, never a display."""
    if frequency_hz <= 0.0 or duration_ms <= 0.0 or dt_ms <= 0.0:
        raise ValueError("frequency, duration, and time step must be positive")
    t = np.arange(0.0, duration_ms + 0.5 * dt_ms, dt_ms)
    omega = 2.0 * np.pi * frequency_hz / 1000.0
    return t, amplitude * np.cos(omega * t + phase)


def save_model_input_bundle(
    output_directory: str | Path,
    critical: CriticalMode,
    *,
    pixels: int = 600,
    duration_ms: float = 500.0,
    dt_ms: float = 0.5,
    amplitude: float = 0.025,
) -> dict[str, Path]:
    """Save static planforms and numerical drives, with no flashing media."""
    import matplotlib.pyplot as plt

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = retinotopic_pattern("target", pixels=pixels)
    pinwheel = retinotopic_pattern("pinwheel", pixels=pixels)
    target_array = np.nan_to_num(target, nan=0.0)
    pinwheel_array = np.nan_to_num(pinwheel, nan=0.0)
    direct_t, direct = temporal_model_input(
        critical.frequency_hz,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        amplitude=amplitude,
    )
    resonant_t, resonant = temporal_model_input(
        2.0 * critical.frequency_hz,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        amplitude=amplitude,
    )
    array_path = directory / "model_inputs.npz"
    np.savez_compressed(
        array_path,
        target=target_array,
        pinwheel=pinwheel_array,
        t_ms=direct_t,
        direct_temporal=direct,
        two_to_one_temporal=resonant,
    )
    paths: dict[str, Path] = {"arrays": array_path}
    for name, values in (("static_target", target), ("static_pinwheel", pinwheel)):
        path = directory / f"{name}.png"
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.imshow(values, origin="lower", cmap="magma", vmin=-1.0, vmax=1.0)
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"{name.replace('_', ' ')} - static model input")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[name] = path
    waveform_path = directory / "temporal_model_drives.png"
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, constrained_layout=True)
    axes[0].plot(direct_t, direct)
    axes[0].set_title(f"direct model drive: {critical.frequency_hz:.2f} Hz")
    axes[1].plot(resonant_t, resonant)
    axes[1].set_title(f"2:1 model drive: {2 * critical.frequency_hz:.2f} Hz")
    for ax in axes:
        ax.set_ylabel("input")
    axes[1].set_xlabel("time (ms)")
    fig.savefig(waveform_path, dpi=180)
    plt.close(fig)
    paths["waveforms"] = waveform_path
    metadata_path = directory / "README.json"
    metadata = {
        "status": "model-only inputs; not validated human hallucination stimuli",
        "autoplay_or_flashing_media": False,
        "direct_frequency_hz": critical.frequency_hz,
        "two_to_one_frequency_hz": 2.0 * critical.frequency_hz,
        "amplitude_at_sigmoid_input": amplitude,
        "static_array_aperture": "values outside the unit disc are zero",
        "warning": (
            "Do not convert the temporal arrays into a human-viewable flashing "
            "display without an approved photosensitivity safety protocol. "
            "Static high-contrast patterns can also affect pattern-sensitive "
            "viewers; they are model inputs, not validated exposure stimuli."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    paths["metadata"] = metadata_path
    return paths


def critical_amplitudes(
    simulation: Simulation, critical: CriticalMode
) -> tuple[Array, Array]:
    """Project a periodic 1D simulation onto the two critical Hopf waves."""
    if simulation.metadata["dimension"] != 1 or simulation.boundary != "periodic":
        raise ValueError(
            "critical amplitude projection requires a 1D periodic simulation"
        )
    coefficient = (
        np.fft.fft(simulation.u, axis=2)[:, :, critical.mode] / simulation.u.shape[2]
    )
    basis = np.column_stack([critical.phi, np.conj(critical.phi)])
    modal = np.linalg.lstsq(basis, coefficient.T, rcond=None)[0].T
    a = modal[:, 0] * np.exp(-1j * critical.omega * simulation.t)
    b = np.conj(modal[:, 1] * np.exp(1j * critical.omega * simulation.t))
    return a, b


def amplitude_rhs(
    z: Array,
    *,
    a: complex,
    b: complex,
    c: complex,
    eta_t: complex = 0.0,
    eta_x: complex = 0.0,
    additive: tuple[complex, complex] = (0.0, 0.0),
) -> Array:
    """Autonomous forced normal form in four real coordinates."""
    A, B = z[0] + 1j * z[1], z[2] + 1j * z[3]
    dA = (
        A * (a + b * abs(A) ** 2 + c * abs(B) ** 2)
        + eta_t * np.conj(B)
        + eta_x * B
        + additive[0]
    )
    dB = (
        B * (a + b * abs(B) ** 2 + c * abs(A) ** 2)
        + eta_t * np.conj(A)
        + eta_x * A
        + additive[1]
    )
    return np.array([dA.real, dA.imag, dB.real, dB.imag])


def integrate_amplitudes(
    *,
    a: complex,
    b: complex,
    c: complex,
    eta_t: complex = 0.0,
    eta_x: complex = 0.0,
    additive: tuple[complex, complex] = (0.0, 0.0),
    initial: Array | None = None,
    total: float = 200.0,
    dt: float = 0.02,
) -> tuple[Array, Array]:
    """RK4 integration of the two complex amplitude equations."""
    z = (
        np.array([0.05, 0.01, 0.03, -0.02])
        if initial is None
        else np.asarray(initial, float).copy()
    )
    t = np.arange(0.0, total + dt, dt)
    history = np.empty((t.size, 4))
    kwargs = {
        "a": a,
        "b": b,
        "c": c,
        "eta_t": eta_t,
        "eta_x": eta_x,
        "additive": additive,
    }
    for n in range(t.size):
        history[n] = z
        if n == t.size - 1:
            break
        k1 = amplitude_rhs(z, **kwargs)
        k2 = amplitude_rhs(z + 0.5 * dt * k1, **kwargs)
        k3 = amplitude_rhs(z + 0.5 * dt * k2, **kwargs)
        k4 = amplitude_rhs(z + dt * k3, **kwargs)
        z += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return t, history


def real_forced_equilibria(
    a: float,
    b: float,
    c: float,
    eta_t: float = 0.0,
    eta_x: float = 0.0,
    additive: float = 0.0,
    *,
    tolerance: float = 1e-9,
) -> list[dict[str, object]]:
    """All isolated real equilibria and their exact four stability eigenvalues."""

    q, p = eta_t + eta_x, eta_x - eta_t
    s, d = b + c, b - c
    candidates: list[tuple[float, float, str]] = []

    def real_roots(coefficients: list[float]) -> list[float]:
        coefficients = list(coefficients)
        while coefficients and abs(coefficients[0]) <= tolerance:
            coefficients.pop(0)
        if len(coefficients) <= 1:
            return []
        return [
            float(root.real)
            for root in np.roots(coefficients)
            if abs(root.imag) <= tolerance
        ]

    for r in real_roots([s, 0.0, a + q, additive]):
        candidates.append((r, r, "symmetric"))

    if abs(s) > tolerance:
        for total in real_roots([b * d, 0.0, d * a - 2.0 * b * q, -s * additive]):
            product = (a - q + b * total**2) / s
            discriminant = total**2 - 4.0 * product
            if discriminant <= tolerance:
                continue
            difference = np.sqrt(discriminant)
            candidates.extend(
                [
                    (
                        0.5 * (total + difference),
                        0.5 * (total - difference),
                        "exchange-broken",
                    ),
                    (
                        0.5 * (total - difference),
                        0.5 * (total + difference),
                        "exchange-broken",
                    ),
                ]
            )
    elif abs(b) > tolerance:
        total_squared = (q - a) / b
        if total_squared > tolerance:
            for total in (np.sqrt(total_squared), -np.sqrt(total_squared)):
                product = (q + additive / total) / (2.0 * b)
                discriminant = total**2 - 4.0 * product
                if discriminant <= tolerance:
                    continue
                difference = np.sqrt(discriminant)
                candidates.extend(
                    [
                        (
                            0.5 * (total + difference),
                            0.5 * (total - difference),
                            "exchange-broken",
                        ),
                        (
                            0.5 * (total - difference),
                            0.5 * (total + difference),
                            "exchange-broken",
                        ),
                    ]
                )

    states: list[dict[str, object]] = []
    unique: set[tuple[int, int]] = set()
    scale = max(tolerance, 1e-14)
    for A, B, branch in candidates:
        key = (round(A / scale), round(B / scale))
        if key in unique:
            continue
        unique.add(key)
        real_block = np.array(
            [
                [a + 3.0 * b * A**2 + c * B**2, q + 2.0 * c * A * B],
                [q + 2.0 * c * A * B, a + 3.0 * b * B**2 + c * A**2],
            ]
        )
        imaginary_block = np.array(
            [
                [a + b * A**2 + c * B**2, p],
                [p, a + b * B**2 + c * A**2],
            ]
        )
        eigenvalues = np.concatenate(
            [np.linalg.eigvalsh(real_block), np.linalg.eigvalsh(imaginary_block)]
        )
        states.append(
            {
                "A": float(A),
                "B": float(B),
                "branch": branch,
                "eigenvalues": eigenvalues,
                "stable": bool(np.all(eigenvalues < -tolerance)),
            }
        )
    return sorted(states, key=lambda state: (str(state["branch"]), state["A"], state["B"]))


def symmetric_forced_equilibria(
    a: float,
    b: float,
    c: float,
    eta_t: float = 0.0,
    eta_x: float = 0.0,
    additive: float = 0.0,
) -> list[dict[str, object]]:
    """Real, exchange-symmetric phase-locked roots and their exact eigenvalues."""
    q = eta_t + eta_x
    roots = np.roots([b + c, 0.0, a + q, additive])
    states: list[dict[str, object]] = []
    for root in roots:
        if abs(root.imag) > 1e-9:
            continue
        r = float(root.real)
        eigenvalues = np.array(
            [
                a + q + 3.0 * (b + c) * r**2,
                a - q + (3.0 * b - c) * r**2,
                a + (b + c) * r**2 - eta_t + eta_x,
                a + (b + c) * r**2 + eta_t - eta_x,
            ]
        )
        states.append(
            {
                "A": complex(r),
                "B": complex(r),
                "eigenvalues": eigenvalues,
                "stable": bool(np.all(eigenvalues < -1e-9)),
            }
        )
    return states


def retinotopic_pattern(
    kind: str, pixels: int = 600, cycles: float = 5.0, inner_radius: float = 0.035
) -> Array:
    """Map simple cortical planforms through x=log(r/r0), y=theta."""
    axis = np.linspace(-1.0, 1.0, pixels)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    log_radius = np.log(np.maximum(radius, inner_radius) / inner_radius)
    k = 2.0 * np.pi * cycles / np.log(1.0 / inner_radius)
    if kind == "target":
        image = np.cos(k * log_radius)
    elif kind == "pinwheel":
        image = np.cos(round(cycles) * theta)
    elif kind == "spiral":
        image = np.cos(k * log_radius + round(cycles) * theta)
    elif kind in ("lattice", "honeycomb"):
        x_cortical, y_cortical = log_radius, theta
        image = (
            sum(
                np.cos(k * (np.cos(angle) * x_cortical + np.sin(angle) * y_cortical))
                for angle in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
            )
            / 3.0
        )
    else:
        raise ValueError("kind must be target, pinwheel, spiral, lattice, or honeycomb")
    image[radius > 1.0] = np.nan
    return image


def save_simulation_figure(simulation: Simulation, output: str | Path) -> Path:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dim = int(simulation.metadata["dimension"])
    if dim == 1:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
        extent = [0, simulation.model.length, simulation.t[0], simulation.t[-1]]
        axes[0].imshow(
            simulation.input[:, 0],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
        )
        axes[1].imshow(
            simulation.u[:, 0],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
        )
        axes[2].plot(
            np.linspace(
                0, simulation.model.length, simulation.u.shape[-1], endpoint=False
            ),
            simulation.u[-1, 0],
        )
        axes[0].set_title("E input")
        axes[1].set_title("E activity")
        axes[2].set_title("final E profile")
        for ax in axes[:2]:
            ax.set(xlabel="cortical position", ylabel="time (ms)")
        axes[2].set(xlabel="cortical position", ylabel="activity")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for i, ax in enumerate(axes):
            index = min(i, simulation.model.populations - 1)
            image = ax.imshow(simulation.u[-1, index], origin="lower", cmap="viridis")
            ax.set_title(f"{simulation.model.names[index]} final activity")
            ax.set(xticks=[], yticks=[])
            fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def save_retinotopic_gallery(output: str | Path) -> Path:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kinds = ("target", "pinwheel", "spiral", "honeycomb")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    for ax, kind in zip(axes, kinds):
        ax.imshow(
            retinotopic_pattern(kind), origin="lower", cmap="magma", vmin=-1, vmax=1
        )
        ax.set_title(kind)
        ax.set(xticks=[], yticks=[])
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--boundary", choices=("periodic", "reflect", "zero"), default="periodic"
    )
    parser.add_argument(
        "--forcing",
        choices=(
            "unforced",
            "direct_1to1",
            "temporal_2to1",
            "spatial_2to1",
            "combined_2to1",
        ),
        default="combined_2to1",
    )
    parser.add_argument("--preset", choices=("three", "five"), default="five")
    parser.add_argument(
        "--export-model-inputs",
        action="store_true",
        help="save static images and numerical temporal arrays; never animation",
    )
    parser.add_argument("--output", default="output/simulations/demo.png")
    args = parser.parse_args()
    model = (
        v1_three_population()
        if args.preset == "three"
        else v1_functional_five_population()
    )
    bracket = (0.85, 1.15) if args.preset == "three" else (0.3, 2.0)
    critical = find_critical_mode(model, scale_bracket=bracket)
    forcing = forcing_suite(model, critical)[args.forcing]
    result = simulate(
        model,
        dimension=args.dimension,
        grid_points=128 if args.dimension == 1 else 64,
        boundary=args.boundary,
        coupling_scale=critical.scale * 1.015,
        forcing=forcing,
    )
    save_simulation_figure(result, args.output)
    if args.export_model_inputs:
        save_model_input_bundle(Path(args.output).parent / "model_inputs", critical)
    print(
        f"saved {args.output}; critical mode={critical.mode}, "
        f"frequency={critical.frequency_hz:.3f} Hz"
    )
