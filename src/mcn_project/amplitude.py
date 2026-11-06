"""Amplitude equations of paper Sec. 6 and their standing-wave equilibrium."""

import numpy as np
from scipy.integrate import solve_ivp


def amplitude_rhs(A, B, a, b, c, eta_t=0, eta_plus=0, eta_minus=0):
    dA = A * (a + b * abs(A) ** 2 + c * abs(B) ** 2) + eta_t * np.conj(B) + eta_plus * B
    dB = B * (a + b * abs(B) ** 2 + c * abs(A) ** 2) + eta_t * np.conj(A) + eta_minus * A
    return dA, dB


def integrate_amplitudes(tau, A0, B0, a, b, c, eta_t=0, eta_plus=0, eta_minus=0):
    """Solve eq. (amplitude) on slow times tau. Returns A(tau), B(tau)."""
    def rhs(_, y):
        return amplitude_rhs(y[0], y[1], a, b, c, eta_t, eta_plus, eta_minus)

    sol = solve_ivp(rhs, (tau[0], tau[-1]), np.array([A0, B0], dtype=complex),
                    t_eval=tau, rtol=1e-9, atol=1e-12)
    return sol.y[0], sol.y[1]


def standing_equilibrium(a, b, c, eta_t, eta_x=0):
    """A = B = z* of the Proposition in Sec. 6, with its stability conditions.

    Returns None when |eta_t| <= |a + eta_x| (no forced standing pattern).
    """
    a_plus, gamma = a + eta_x, b + c
    roots = np.roots([abs(gamma) ** 2, 2 * (a_plus * np.conj(gamma)).real,
                      abs(a_plus) ** 2 - abs(eta_t) ** 2])
    positive = roots[(abs(roots.imag) < 1e-12) & (roots.real > 0)].real
    if abs(eta_t) <= abs(a_plus) or positive.size == 0:
        return None
    rho = positive.max()
    phase = -(a_plus + gamma * rho) / eta_t           # = exp(-2 i phi*)
    phi = -np.angle(phase) / 2
    conditions = [(a_plus + 2 * gamma * rho).real, (a - eta_x + 2 * b * rho).real]
    if eta_x != 0:  # with eta_x = 0 the third direction is the neutral translation
        conditions.append(((a + 2 * b * rho) * np.conj(eta_x)).real)
    conditions = np.array(conditions)
    return dict(rho=rho, phi=phi, z=np.sqrt(rho) * np.exp(1j * phi),
                stability_conditions=conditions, stable=bool((conditions < 0).all()))


def square_rhs(z, a, b, c, c_perp, c_q, eta_t=0, eta_x=0, eta_w=0):
    """Four-wave amplitude equations on the square lattice, z = (z1, w1, z2, w2).

    z1 ~ e^{i(w0 t + q0 x)} and w1 ~ e^{i(w0 t - q0 x)} (w1 moves toward +x);
    (z2, w2) likewise along y. Forcing: uniform temporal (eta_t), static stripes
    cos(2 q0 x) (eta_x), traveling cos(2 q0 x - 2 w0 t) (eta_w, acts on w1).
    """
    z1, w1, z2, w2 = z
    n1, n2 = abs(z1) ** 2 + abs(w1) ** 2, abs(z2) ** 2 + abs(w2) ** 2
    return np.array([
        z1 * (a + b * abs(z1) ** 2 + c * abs(w1) ** 2 + c_perp * n2) + c_q * np.conj(w1) * z2 * w2
        + eta_t * np.conj(w1) + eta_x * w1,
        w1 * (a + b * abs(w1) ** 2 + c * abs(z1) ** 2 + c_perp * n2) + c_q * np.conj(z1) * z2 * w2
        + eta_t * np.conj(z1) + eta_x * z1 + eta_w * np.conj(w1),
        z2 * (a + b * abs(z2) ** 2 + c * abs(w2) ** 2 + c_perp * n1) + c_q * np.conj(w2) * z1 * w1
        + eta_t * np.conj(w2),
        w2 * (a + b * abs(w2) ** 2 + c * abs(z2) ** 2 + c_perp * n1) + c_q * np.conj(z2) * z1 * w1
        + eta_t * np.conj(z2),
    ])
