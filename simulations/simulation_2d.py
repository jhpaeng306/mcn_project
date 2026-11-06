"""Direct two-dimensional simulations of the shifted-logistic neural field.

This module is the two-dimensional counterpart of :mod:`simulations.simulation`.
It integrates exactly the same K-population Wilson--Cowan equation and uses the
same ``ModelParameters`` and activation function.  Only the spatial domain and
convolution operator change.

For a square domain ``[0, L) x [0, L)`` the equation is

    tau_i u_i,t = -u_i + sum_j W_ij^(p) * F_j(u_j) + I_i,

where ``W_ij^(p)`` is an isotropic, unit-mass 2D Gaussian with covariance
``Sigma[i, j] I`` multiplied by ``A[i, j] + p A_p[i, j]``.  Periodic
convolutions use the exact Fourier symbol.  A bounded reflecting domain uses
an even extension at every edge; a zero-exterior option is also available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter

from .simulation import Array, Forcing, ModelParameters, activation

Boundary2D = Literal["periodic", "reflect", "zero"]
Integrator2D = Literal["rk2", "rk4"]
SpatialPattern2D = Literal["plane", "cosine_product"]


@dataclass
class Forcing2D:
    """Additive temporal, spatial, or traveling forcing on a square domain.

    ``spatial_mode=(m_x, m_y)`` means the phase
    ``2*pi*(m_x*x + m_y*y)/L``.  The standard 2:1 experiment uses
    ``(2*k0, 0)`` so that the stimulus is doubled along the x direction.
    """

    temporal_amplitude: Array | float | None = None
    spatial_amplitude: Array | float | None = None
    traveling_amplitude: Array | float | None = None
    temporal_omega: float = 0.0
    spatial_mode: tuple[float, float] = (0.0, 0.0)
    spatial_pattern: SpatialPattern2D = "plane"
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

    def evaluate(
        self, model: ModelParameters, t: float, x_grid: Array, y_grid: Array
    ) -> Array:
        result = np.zeros((model.K,) + x_grid.shape)
        if not (self.start <= t < self.stop):
            return result
        temporal = self.vector(self.temporal_amplitude, model.K)
        spatial = self.vector(self.spatial_amplitude, model.K)
        traveling = self.vector(self.traveling_amplitude, model.K)
        temporal_phase = self.temporal_omega * t + self.phase
        mx, my = self.spatial_mode
        if self.spatial_pattern == "plane":
            spatial_wave = np.cos(
                2.0 * np.pi * (mx * x_grid + my * y_grid) / model.length
                + self.phase
            )
            traveling_wave = np.cos(
                2.0 * np.pi * (mx * x_grid + my * y_grid) / model.length
                + self.phase
                - self.temporal_omega * t
            )
        elif self.spatial_pattern == "cosine_product":
            spatial_wave = np.cos(
                np.pi * mx * x_grid / model.length + self.phase
            ) * np.cos(np.pi * my * y_grid / model.length + self.phase)
            traveling_wave = spatial_wave * np.cos(self.temporal_omega * t)
        else:
            raise ValueError("spatial_pattern must be plane or cosine_product")
        result += temporal[:, None, None] * np.cos(temporal_phase)
        result += spatial[:, None, None] * spatial_wave[None, :, :]
        result += traveling[:, None, None] * traveling_wave[None, :, :]
        return result

    @classmethod
    def from_1d(cls, forcing: Forcing, *, spatial_axis: int = 0) -> "Forcing2D":
        """Lift a 1D forcing to x- or y-directed forcing on the 2D square."""
        if spatial_axis not in (0, 1):
            raise ValueError("spatial_axis must be 0 (x) or 1 (y)")
        mode = (
            (forcing.spatial_mode, 0)
            if spatial_axis == 0
            else (0, forcing.spatial_mode)
        )
        return cls(
            temporal_amplitude=forcing.temporal_amplitude,
            spatial_amplitude=forcing.spatial_amplitude,
            traveling_amplitude=forcing.traveling_amplitude,
            temporal_omega=forcing.temporal_omega,
            spatial_mode=mode,
            spatial_pattern="plane",
            phase=forcing.phase,
            start=forcing.start,
            stop=forcing.stop,
        )


@dataclass(frozen=True)
class SimulationResult2D:
    """Saved 2D fields plus inexpensive full-rate diagnostics."""

    t: Array
    x: Array
    y: Array
    u: Array
    diagnostic_t: Array
    probe: Array
    spatial_rms: Array
    mode_signal: Array
    diagnostic_wavevector: tuple[float, float]
    p: float
    seed: int
    model: ModelParameters
    forcing_spec: Forcing2D
    boundary: Boundary2D
    dt: float
    integrator: Integrator2D


def _zero_exterior_recurrent_operator(
    model: ModelParameters,
    p: float,
    dx: float,
):
    coupling = model.connectivity(p)
    active = [
        (i, j, float(coupling[i, j]), float(np.sqrt(model.Sigma[i, j]) / dx))
        for i in range(model.K)
        for j in range(model.K)
        if coupling[i, j] != 0.0
    ]

    def recurrent(rates: Array) -> Array:
        field = np.zeros_like(rates)
        for target, source, weight, sigma_pixels in active:
            if sigma_pixels <= 1e-14:
                blurred = rates[source]
            else:
                blurred = gaussian_filter(
                    rates[source],
                    sigma=sigma_pixels,
                    mode="constant",
                    cval=0.0,
                    truncate=4.0,
                )
            field[target] += weight * blurred
        return field

    return recurrent


def _reflect_recurrent_operator(
    model: ModelParameters, p: float, grid_points: int
):
    """Exact Gaussian convolution for an even extension of a cell grid."""
    q = np.pi * np.arange(grid_points) / model.length
    qx, qy = np.meshgrid(q, q, indexing="ij")
    q_squared = qx**2 + qy**2
    symbol = model.connectivity(p)[:, :, None, None] * np.exp(
        -0.5 * model.Sigma[:, :, None, None] * q_squared[None, None, :, :]
    )

    def recurrent(rates: Array) -> Array:
        rates_hat = dctn(rates, type=2, axes=(-2, -1), norm="ortho")
        field_hat = np.einsum("ijxy,jxy->ixy", symbol, rates_hat)
        return idctn(field_hat, type=2, axes=(-2, -1), norm="ortho")

    return recurrent


def _periodic_recurrent_operator(
    model: ModelParameters, p: float, grid_points: int, dx: float
):
    q = 2.0 * np.pi * np.fft.fftfreq(grid_points, d=dx)
    qx, qy = np.meshgrid(q, q, indexing="ij")
    q_squared = qx**2 + qy**2
    symbol = model.connectivity(p)[:, :, None, None] * np.exp(
        -0.5 * model.Sigma[:, :, None, None] * q_squared[None, None, :, :]
    )

    def recurrent(rates: Array) -> Array:
        rates_hat = np.fft.fftn(rates, axes=(-2, -1))
        field_hat = np.einsum("ijxy,jxy->ixy", symbol, rates_hat)
        return np.fft.ifftn(field_hat, axes=(-2, -1)).real

    return recurrent


def _mode_coefficient(
    state: Array,
    wavevector: tuple[float, float],
    boundary: Boundary2D,
) -> Array:
    """Coefficient at a wavevector measured in periodic-domain mode units."""
    mx, my = wavevector
    n = state.shape[-1]
    if boundary == "periodic":
        transformed = np.fft.fftn(state, axes=(-2, -1)) / (n * n)
        ix, iy = int(round(mx)) % n, int(round(my)) % n
        return transformed[:, ix, iy]
    # Reflecting eigenfunctions are cos(n*pi*x/L), so the same physical
    # wavenumber as periodic integer m has DCT index 2m.
    transformed = dctn(state, type=2, axes=(-2, -1), norm="ortho")
    ix, iy = int(round(2 * abs(mx))), int(round(2 * abs(my)))
    if ix >= n or iy >= n:
        return np.full(state.shape[0], np.nan + 0.0j)
    return transformed[:, ix, iy].astype(complex)


def simulate_2d(
    model: ModelParameters,
    *,
    p: float = 0.0,
    forcing: Forcing2D | None = None,
    seed: int = 1,
    grid_points: int = 64,
    total_time: float = 12000.0,
    dt: float = 0.5,
    save_every: int = 200,
    diagnostic_every: int = 1,
    initial_scale: float = 3e-4,
    boundary: Boundary2D = "periodic",
    integrator: Integrator2D = "rk2",
    diagnostic_wavevector: tuple[float, float] = (0.0, 0.0),
) -> SimulationResult2D:
    """Integrate the full shifted-logistic neural field on a 2D square.

    ``reflect`` is the notebook's bounded setting: the activity is extended
    evenly across each edge before convolution.  ``zero`` instead assumes no
    activity outside the square.  RK2 is the practical default for the larger
    2D state; RK4 is available for convergence checks.
    """
    if grid_points < 8:
        raise ValueError("grid_points must be at least 8")
    if total_time <= 0.0 or dt <= 0.0 or save_every < 1 or diagnostic_every < 1:
        raise ValueError("time and save/diagnostic intervals must be positive")
    if boundary not in ("periodic", "reflect", "zero"):
        raise ValueError("boundary must be periodic, reflect, or zero")
    if integrator not in ("rk2", "rk4"):
        raise ValueError("integrator must be rk2 or rk4")
    forcing = Forcing2D() if forcing is None else forcing

    dx = model.length / grid_points
    coordinates = (np.arange(grid_points) + 0.5) * dx
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="ij")
    if boundary == "periodic":
        recurrent = _periodic_recurrent_operator(model, p, grid_points, dx)
    elif boundary == "reflect":
        recurrent = _reflect_recurrent_operator(model, p, grid_points)
    else:
        recurrent = _zero_exterior_recurrent_operator(model, p, dx)
    tau = model.tau[:, None, None]
    rng = np.random.default_rng(seed)
    state = initial_scale * rng.standard_normal((model.K, grid_points, grid_points))

    def rhs(value: Array, time: float) -> Array:
        rates = activation(value, model)
        external = forcing.evaluate(model, time, x_grid, y_grid)
        return (-value + recurrent(rates) + external) / tau

    steps = round(total_time / dt)
    saved_t: list[float] = []
    saved_u: list[Array] = []
    diagnostic_t: list[float] = []
    probe: list[Array] = []
    spatial_rms: list[Array] = []
    mode_signal: list[Array] = []
    center = grid_points // 2

    for step in range(steps + 1):
        time = step * dt
        if step % save_every == 0:
            saved_t.append(time)
            saved_u.append(state.astype(np.float32, copy=True))
        if step % diagnostic_every == 0:
            centered = state - state.mean(axis=(-2, -1), keepdims=True)
            diagnostic_t.append(time)
            probe.append(state[:, center, center].copy())
            spatial_rms.append(np.sqrt(np.mean(centered**2, axis=(-2, -1))))
            mode_signal.append(
                _mode_coefficient(state, diagnostic_wavevector, boundary)
            )
        if step == steps:
            break
        if integrator == "rk2":
            k1 = rhs(state, time)
            k2 = rhs(state + dt * k1, time + dt)
            state += 0.5 * dt * (k1 + k2)
        else:
            k1 = rhs(state, time)
            k2 = rhs(state + 0.5 * dt * k1, time + 0.5 * dt)
            k3 = rhs(state + 0.5 * dt * k2, time + 0.5 * dt)
            k4 = rhs(state + dt * k3, time + dt)
            state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("the 2D numerical field became non-finite")

    return SimulationResult2D(
        t=np.asarray(saved_t),
        x=coordinates,
        y=coordinates,
        u=np.asarray(saved_u),
        diagnostic_t=np.asarray(diagnostic_t),
        probe=np.asarray(probe),
        spatial_rms=np.asarray(spatial_rms),
        mode_signal=np.asarray(mode_signal),
        diagnostic_wavevector=tuple(map(float, diagnostic_wavevector)),
        p=float(p),
        seed=int(seed),
        model=model,
        forcing_spec=forcing,
        boundary=boundary,
        dt=float(dt),
        integrator=integrator,
    )


def dominant_spatial_mode(
    result: SimulationResult2D,
    *,
    population: int = 0,
    tail_fraction: float = 0.2,
) -> dict[str, float]:
    """Return the dominant final-tail wavevector in periodic mode units."""
    if not 0 <= population < result.model.K:
        raise ValueError("population index is out of range")
    start = max(0, int((1.0 - tail_fraction) * result.t.size))
    snapshots = result.u[start:, population].astype(float)
    snapshots -= snapshots.mean(axis=(-2, -1), keepdims=True)
    n = snapshots.shape[-1]
    if result.boundary == "periodic":
        transformed = np.fft.fftn(snapshots, axes=(-2, -1)) / (n * n)
        power = np.mean(np.abs(transformed) ** 2, axis=0)
        indices = np.rint(np.fft.fftfreq(n) * n).astype(int)
        ix, iy = np.unravel_index(np.argmax(power), power.shape)
        mx, my = float(indices[ix]), float(indices[iy])
    else:
        transformed = dctn(snapshots, type=2, axes=(-2, -1), norm="ortho")
        power = np.mean(np.abs(transformed) ** 2, axis=0)
        power[0, 0] = 0.0
        ix, iy = np.unravel_index(np.argmax(power), power.shape)
        mx, my = 0.5 * float(ix), 0.5 * float(iy)
    return {
        "mode_x": mx,
        "mode_y": my,
        "radial_mode": float(np.hypot(mx, my)),
        "wavelength_mm": float(
            0.1 * result.model.length / max(np.hypot(mx, my), 1e-15)
        ),
    }


def dominant_temporal_frequency_hz(
    result: SimulationResult2D,
    *,
    population: int = 0,
    tail_fraction: float = 0.5,
    use_mode_signal: bool = False,
) -> float:
    """Dominant nonzero temporal frequency of a probe or selected mode."""
    start = max(0, int((1.0 - tail_fraction) * result.diagnostic_t.size))
    if use_mode_signal:
        values = result.mode_signal[start:, population].real
    else:
        values = result.probe[start:, population]
    values = np.asarray(values, dtype=float)
    values -= values.mean()
    if values.size < 4 or np.max(np.abs(values)) < 1e-14:
        return float("nan")
    spacing_ms = float(np.mean(np.diff(result.diagnostic_t[start:])))
    frequencies = 1000.0 * np.fft.rfftfreq(values.size, d=spacing_ms)
    power = np.abs(np.fft.rfft(values)) ** 2
    power[0] = 0.0
    return float(frequencies[np.argmax(power)])
