"""Direct simulation of the field on a periodic 1D or 2D grid (FFT convolution, RK4)."""

from dataclasses import dataclass

import numpy as np

from .model import NeuralField


@dataclass(frozen=True)
class Wave:
    """One input component: peak * cos(2 pi k.x / L - omega t + phase).

    k = () or all zeros is spatially uniform (temporal forcing); omega = 0 is static
    (spatial forcing); both nonzero is a traveling wave along k.
    """
    peak: float
    k: tuple[int, ...] = ()
    omega: float = 0.0
    phase: float = 0.0


@dataclass(frozen=True)
class Forcing:
    waves: tuple[Wave, ...] = ()
    on: float = -np.inf
    off: float = np.inf
    drive: np.ndarray | None = None  # population weights; default model.drive


@dataclass(frozen=True)
class Run:
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray        # shape (time, K, *grid)
    input: np.ndarray    # scalar input profile, shape (time, *grid)
    kappa: float
    forcing: Forcing


def grid(model: NeuralField, n: int, dim: int):
    x = np.arange(n) * model.length / n
    return x, np.meshgrid(*([x] * dim), indexing="ij")


class _Input:
    """Precomputed cos/sin profiles so each step costs one multiply-add per wave."""

    def __init__(self, model, forcing, mesh):
        self.forcing = forcing
        self.waves = []
        for w in forcing.waves:
            k = w.k + (0,) * (len(mesh) - len(w.k))
            theta = sum(2 * np.pi * ki * xi / model.length for ki, xi in zip(k, mesh)) + w.phase
            self.waves.append((w.peak * np.cos(theta), w.peak * np.sin(theta), w.omega))
        self.zero = np.zeros(mesh[0].shape)

    def __call__(self, t):
        if not (self.forcing.on <= t < self.forcing.off):
            return self.zero
        return sum((c * np.cos(o * t) + s * np.sin(o * t) for c, s, o in self.waves), self.zero)


def simulate(model: NeuralField, kappa: float, forcing: Forcing = Forcing(), *,
             dim: int = 1, n: int = 64, dt: float = 0.1, t_end: float = 500.0,
             save_every: int = 10, u0: np.ndarray | None = None,
             noise: float = 1e-3, seed: int = 0) -> Run:
    x, mesh = grid(model, n, dim)
    axes = tuple(range(1, dim + 1))
    freqs = [2 * np.pi * np.fft.fftfreq(n, model.length / n)] * (dim - 1)
    freqs.append(2 * np.pi * np.fft.rfftfreq(n, model.length / n))
    q = np.sqrt(sum(f**2 for f in np.meshgrid(*freqs, indexing="ij")))
    W = np.moveaxis(model.kernel_symbol(q, kappa), (-2, -1), (0, 1))  # (K, K, *rfft grid)
    tau = model.tau.reshape((model.K,) + (1,) * dim)
    drive = (model.drive if forcing.drive is None else forcing.drive).reshape(tau.shape)
    I = _Input(model, forcing, mesh)

    def rhs(u, t):
        rates = np.fft.rfftn(model.activation(u), axes=axes)
        recurrent = np.fft.irfftn(np.einsum("ij...,j...->i...", W, rates), s=(n,) * dim, axes=axes)
        return (-u + recurrent + drive * I(t)) / tau

    rng = np.random.default_rng(seed)
    u = (noise * rng.standard_normal((model.K,) + (n,) * dim) if u0 is None
         else np.array(u0, dtype=float))
    steps = round(t_end / dt)
    t_out, u_out, i_out = [], [], []
    for step in range(steps + 1):
        t = step * dt
        if step % save_every == 0:
            t_out.append(t), u_out.append(u.copy()), i_out.append(I(t))
        if step == steps:
            break
        k1 = rhs(u, t)
        k2 = rhs(u + dt / 2 * k1, t + dt / 2)
        k3 = rhs(u + dt / 2 * k2, t + dt / 2)
        k4 = rhs(u + dt * k3, t + dt)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Run(np.array(t_out), x, np.array(u_out), np.array(i_out), kappa, forcing)


def critical_seed(model: NeuralField, onset, size: float, *, dim: int = 1, n: int = 64,
                  mode: tuple[int, ...] | None = None) -> np.ndarray:
    """Standing critical wave size * Re(Phi exp(2 pi i k.x / L)), default k = (k0, 0, ...)."""
    _, mesh = grid(model, n, dim)
    mode = mode or (round(onset.k0),) + (0,) * (dim - 1)
    phase = sum(2 * np.pi * k * xi / model.length for k, xi in zip(mode, mesh))
    return size * np.real(onset.Phi.reshape((-1,) + (1,) * dim) * np.exp(1j * phase))
