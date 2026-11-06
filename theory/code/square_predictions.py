"""Square-lattice predictions under uniform temporal forcing at 2 w0 (report Sec. 8).

Prints the coefficients, the stripe and square equilibria, the stripe transverse
eigenvalue from the closed formula, and the eigenvalues of the full 8x8 amplitude
Jacobian at both states. Usage: uv run python theory/code/square_predictions.py
"""
import numpy as np

from mcn_project import presets
from mcn_project.amplitude import square_rhs, standing_equilibrium
from mcn_project.linear import find_onset
from mcn_project.normal_form import cubic_coefficients, square_coefficients, unit_forcing_coefficients

CASES = {"Wilson-Cowan, L=58.26": (presets.wilson_cowan(length=58.264198846330785), (4, 9), 1.0),
         "V1 (K=4), L=43": (presets.v1(), (0, 1), 3.0)}
MU = -1.0


def jacobian_eigenvalues(state, args):
    """Eigenvalues of the real 8x8 Jacobian of square_rhs (central differences)."""
    x0 = np.concatenate([state.real, state.imag])
    def f(x):
        out = square_rhs(x[:4] + 1j * x[4:], *args)
        return np.concatenate([out.real, out.imag])
    J = np.column_stack([(f(x0 + h) - f(x0 - h)) / 2e-7 for h in 1e-7 * np.eye(8)])
    return np.sort_complex(np.linalg.eigvals(J))


for name, (model, rng, peak) in CASES.items():
    onset = find_onset(model, 2, rng)
    nf, sq = cubic_coefficients(model, onset), square_coefficients(model, onset)
    eta_t = peak * unit_forcing_coefficients(model, onset)[0]
    a = MU * nf.alpha
    print(f"== {name}: |k0|^2 = {onset.k0**2:.0f}, temporal peak {peak} eps^2")
    print(f"   b = {nf.b:.4f}, c = {nf.c:.4f}, c_perp = {sq.c_perp:.4f}, c_q = {sq.c_q:.4f}")
    args = (a, nf.b, nf.c, sq.c_perp, sq.c_q, eta_t)
    stripe = standing_equilibrium(a, nf.b, nf.c, eta_t)
    z = stripe["z"]
    h, l = a + 2 * sq.c_perp * stripe["rho"], eta_t + sq.c_q * z**2
    lam = h.real + np.sqrt(complex(abs(l) ** 2 - h.imag**2))
    print(f"   stripe: rho = {stripe['rho']:.5f}, 1D-stable = {stripe['stable']}, "
          f"transverse lambda (formula) = {lam.real:+.5f}")
    print("     8x8 Jacobian real parts:", np.round(jacobian_eigenvalues(np.array([z, z, 0, 0]), args).real, 5))
    gamma_sq = nf.b + nf.c + 2 * sq.c_perp + sq.c_q
    square = standing_equilibrium(a, gamma_sq / 2, gamma_sq / 2, eta_t)   # same scalar equation, gamma = gamma_sq
    zs = square["z"]
    print(f"   square: gamma = {gamma_sq:.4f}, rho = {square['rho']:.5f}")
    print("     8x8 Jacobian real parts:", np.round(jacobian_eigenvalues(np.array([zs] * 4), args).real, 5))
