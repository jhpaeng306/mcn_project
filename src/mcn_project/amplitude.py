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
