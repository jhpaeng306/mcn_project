"""Stationary states of the 1D field under a static input, and their stability (report Sec. 7)."""

import numpy as np
from scipy.special import expit

from .model import NeuralField


def convolution_matrix(model: NeuralField, kappa: float, n: int) -> np.ndarray:
    """Dense (K n) x (K n) matrix of u -> W * u on the periodic grid."""
    q = 2 * np.pi * np.fft.fftfreq(n, model.length / n)
    symbol = model.kernel_symbol(q, kappa)                  # (n, K, K)
    eye_hat = np.fft.fft(np.eye(n), axis=0)
    return np.block([[np.real(np.fft.ifft(eye_hat * symbol[:, i, j, None], axis=0))
                      for j in range(model.K)] for i in range(model.K)])


def locked_state_growth(model: NeuralField, kappa: float, stimulus: np.ndarray, u=None):
    """Largest real eigenvalue at the steady state of 0 = -u + W*F(u) + e stimulus(x).

    Newton from u (default 0). Returns (growth, its frequency, steady state).
    """
    n = stimulus.size
    W = convolution_matrix(model, kappa, n)
    g, th = np.repeat(model.gain, n), np.repeat(model.theta, n)
    I = np.kron(model.drive, stimulus)
    u = np.zeros(model.K * n) if u is None else u.copy()
    for _ in range(50):
        s = expit(g * (u - th))
        J = -np.eye(model.K * n) + W * (g * s * (1 - s))
        step = np.linalg.solve(J, -(-u + W @ (s - expit(-g * th)) + I))
        u += step
        if abs(step).max() < 1e-13:
            break
    ev = np.linalg.eigvals(J / np.repeat(model.tau, n)[:, None])
    top = np.argmax(ev.real)
    return ev[top].real, abs(ev[top].imag), u
