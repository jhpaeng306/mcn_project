"""Pure numerical simulation of a periodic K-population neural field.

Rows of ``A`` are target populations and columns are source populations.  The
``(i, j)`` kernel is a unit-mass Gaussian with variance ``Sigma[i, j]`` and
Fourier multiplier ``exp(-Sigma[i, j] * q**2 / 2)``.  Thus ``A[i, j]`` is the
signed integrated coupling gain, not the peak height of an unnormalised
Gaussian.  The perturbed coupling is exactly ``A + p * A_p``.

The dynamics use the convention in PI's Wilson-Cowan code:

    tau_i u_i,t = -u_i + sum_j W_ij * f_j(u_j) + forcing_i,

where the thresholded logistic ``f_j`` is applied before convolution and is
shifted so that ``f_j(0) = 0``.  Consequently ``u = 0`` is a homogeneous
equilibrium in the absence of forcing for every value of ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.special import expit

Array = np.ndarray


def _array_or_default(value: Array | list[float] | None, default: Array) -> Array:
    return np.array(default if value is None else value, dtype=float, copy=True)


@dataclass
class ModelParameters:
    """Inputs for a K-population model.

    Only ``K`` is required.  The K=2 defaults reproduce the parameter layout
    of PI's E-I example; presets in :mod:`simulations.pipeline` provide the
    complete published/calibrated cases used by the notebook.
    """

    K: int = 2
    A: Array | None = None
    Sigma: Array | None = None
    tau: Array | None = None
    A_p: Array | None = None
    theta: Array | None = None
    gain: Array | None = None
    length: float = 60.0
    names: tuple[str, ...] | None = None
    time_unit: str = "model time"
    space_unit: str = "model space"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.K, int) or self.K < 1:
            raise ValueError("K must be a positive integer")
        if self.K == 2:
            default_a = np.array([[5.4, -10.0], [10.0, -8.0]])
            default_sigma = np.array([[8.0, 32.0], [8.0, 32.0]])
            default_tau = np.array([3.0, 6.6])
            default_ap = np.array([[1.0, 0.0], [0.0, 0.0]])
            default_theta = np.array([0.5, 0.3])
            default_gain = np.full(2, 4.0)
            default_names = ("E", "I")
        else:
            default_a = np.zeros((self.K, self.K))
            default_sigma = np.ones((self.K, self.K))
            default_tau = np.ones(self.K)
            default_ap = np.zeros((self.K, self.K))
            default_theta = np.full(self.K, 0.5)
            default_gain = np.full(self.K, 4.0)
            default_names = tuple(f"population_{index}" for index in range(self.K))

        self.A = _array_or_default(self.A, default_a)
        self.Sigma = _array_or_default(self.Sigma, default_sigma)
        self.tau = _array_or_default(self.tau, default_tau)
        self.A_p = _array_or_default(self.A_p, default_ap)
        self.theta = _array_or_default(self.theta, default_theta)
        self.gain = _array_or_default(self.gain, default_gain)
        self.names = default_names if self.names is None else tuple(self.names)

        square = (self.K, self.K)
        if (
            self.A.shape != square
            or self.Sigma.shape != square
            or self.A_p.shape != square
        ):
            raise ValueError("A, Sigma, and A_p must all have shape (K, K)")
        if self.tau.shape != (self.K,) or self.theta.shape != (self.K,):
            raise ValueError("tau and theta must have length K")
        if self.gain.shape != (self.K,) or len(self.names) != self.K:
            raise ValueError("gain and names must have length K")
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.A_p)):
            raise ValueError("A and A_p must contain finite values")
        if np.any(~np.isfinite(self.Sigma)) or np.any(self.Sigma < 0.0):
            raise ValueError(
                "Sigma stores variances and must be finite and non-negative"
            )
        if np.any(~np.isfinite(self.tau)) or np.any(self.tau <= 0.0):
            raise ValueError("all time constants must be finite and positive")
        if np.any(~np.isfinite(self.gain)) or np.any(self.gain <= 0.0):
            raise ValueError("all logistic gains must be finite and positive")
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive")

    def connectivity(self, p: float) -> Array:
        """Return the signed connectivity matrix ``A + p A_p``."""
        return self.A + float(p) * self.A_p


@dataclass
class Forcing:
    """Additive forcing in the same units as the recurrent field.

    A temporal cosine is spatially uniform.  A spatial cosine is static.
    Their sum gives the report's separated temporal-plus-spatial forcing.
    ``traveling_amplitude`` is optional and is not used in the four requested
    forcing comparisons.
    """

    temporal_amplitude: Array | float | None = None
    spatial_amplitude: Array | float | None = None
    traveling_amplitude: Array | float | None = None
    temporal_omega: float = 0.0
    spatial_mode: int = 0
    phase: float = 0.0
    start: float = -np.inf
    stop: float = np.inf

    def vector(self, value: Array | float | None, K: int) -> Array:
        if value is None:
            return np.zeros(K)
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return np.full(K, float(array))
        if array.shape != (K,):
            raise ValueError("each forcing amplitude must be scalar or have length K")
        return array

    def evaluate(self, model: ModelParameters, t: float, x: Array) -> Array:
        result = np.zeros((model.K, x.size))
        if not (self.start <= t < self.stop):
            return result
        temporal = self.vector(self.temporal_amplitude, model.K)
        spatial = self.vector(self.spatial_amplitude, model.K)
        traveling = self.vector(self.traveling_amplitude, model.K)
        temporal_phase = self.temporal_omega * t + self.phase
        spatial_phase = 2.0 * np.pi * self.spatial_mode * x / model.length + self.phase
        result += temporal[:, None] * np.cos(temporal_phase)
        result += spatial[:, None] * np.cos(spatial_phase)[None, :]
        result += (
            traveling[:, None]
            * np.cos(spatial_phase - self.temporal_omega * t)[None, :]
        )
        return result


@dataclass(frozen=True)
class SimulationResult:
    t: Array
    x: Array
    u: Array
    forcing: Array
    p: float
    seed: int
    model: ModelParameters
    forcing_spec: Forcing
    dt: float


def activation(u: Array, model: ModelParameters) -> Array:
    """Thresholded logistic used by PI's code, with ``f(0)=0``."""
    extra = (1,) * (u.ndim - 1)
    gain = model.gain.reshape((model.K,) + extra)
    theta = model.theta.reshape((model.K,) + extra)
    return expit(gain * (u - theta)) - expit(-gain * theta)


def activation_derivatives(model: ModelParameters) -> tuple[Array, Array, Array]:
    """Return ``f'(0)``, ``f''(0)``, and ``f'''(0)`` population-wise."""
    s = expit(-model.gain * model.theta)
    f1 = model.gain * s * (1.0 - s)
    f2 = model.gain**2 * s * (1.0 - s) * (1.0 - 2.0 * s)
    f3 = model.gain**3 * s * (1.0 - s) * (1.0 - 6.0 * s + 6.0 * s**2)
    return f1, f2, f3


def simulate(
    model: ModelParameters,
    *,
    p: float = 0.0,
    forcing: Forcing | None = None,
    seed: int = 1,
    grid_points: int = 96,
    total_time: float = 500.0,
    dt: float = 0.1,
    save_every: int = 5,
    initial_scale: float = 0.01,
    initial_symmetry: Literal["none", "reflection"] = "none",
) -> SimulationResult:
    """Integrate the field directly with fourth-order Runge--Kutta.

    The spatial convolution is evaluated exactly for the periodic Gaussian
    model by multiplying Fourier coefficients.  No amplitude equation enters
    this numerical simulation.  RK4 is used because second-order explicit
    stepping can spuriously amplify a nearly neutral Hopf pair even when its
    continuous-time real part is slightly negative.
    """
    if grid_points < 4 or total_time <= 0.0 or dt <= 0.0 or save_every < 1:
        raise ValueError("grid_points, total_time, dt, and save_every must be positive")
    forcing = Forcing() if forcing is None else forcing
    x = np.linspace(0.0, model.length, grid_points, endpoint=False)
    q = 2.0 * np.pi * np.fft.fftfreq(grid_points, d=model.length / grid_points)
    kernel_symbol = np.exp(-0.5 * model.Sigma[:, :, None] * q[None, None, :] ** 2)
    weighted_symbol = model.connectivity(p)[:, :, None] * kernel_symbol
    tau = model.tau[:, None]

    rng = np.random.default_rng(seed)
    state = initial_scale * rng.standard_normal((model.K, grid_points))
    if initial_symmetry == "reflection":
        reflected = state[:, (-np.arange(grid_points)) % grid_points]
        state = 0.5 * (state + reflected)
    elif initial_symmetry != "none":
        raise ValueError("initial_symmetry must be 'none' or 'reflection'")

    def rhs(value: Array, time: float) -> tuple[Array, Array]:
        rates_hat = np.fft.fft(activation(value, model), axis=1)
        recurrent_hat = np.einsum("ijk,jk->ik", weighted_symbol, rates_hat)
        recurrent = np.fft.ifft(recurrent_hat, axis=1).real
        external = forcing.evaluate(model, time, x)
        return (-value + recurrent + external) / tau, external

    steps = round(total_time / dt)
    saved_t: list[float] = []
    saved_u: list[Array] = []
    saved_forcing: list[Array] = []
    for step in range(steps + 1):
        time = step * dt
        if step % save_every == 0:
            _, external = rhs(state, time)
            saved_t.append(time)
            saved_u.append(state.copy())
            saved_forcing.append(external)
        if step == steps:
            break
        k1, _ = rhs(state, time)
        k2, _ = rhs(state + 0.5 * dt * k1, time + 0.5 * dt)
        k3, _ = rhs(state + 0.5 * dt * k2, time + 0.5 * dt)
        k4, _ = rhs(state + dt * k3, time + dt)
        state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("the numerical field became non-finite")

    return SimulationResult(
        t=np.asarray(saved_t),
        x=x,
        u=np.asarray(saved_u),
        forcing=np.asarray(saved_forcing),
        p=float(p),
        seed=int(seed),
        model=model,
        forcing_spec=forcing,
        dt=float(dt),
    )
