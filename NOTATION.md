# Notation

One symbol, one meaning, across code, paper, and slides. Add a row before using a new symbol. Macros live in `paper/def.tex` (and the preamble of `paper/main.tex`).
Taken from the notation table of `paper/main.tex` (Sec. 6) and Secs. 7–8.

## Symbols

| Symbol | Meaning | Shape / type | Macro | Name in code |
|---|---|---|---|---|
| $u$ | field (activity of each population) | $\RR^K$ per point | — | `u` (time, K, *grid) |
| $K$ ($d$ in Secs. 1–6) | number of populations | integer | — | `model.K` |
| $\epsilon$ | small amplitude scale | scalar | `\epsilon` | `EPS` |
| $\kappa$, $\kappa_c$ | gain scale $J=J_0+\kappa J_1$; its onset value | scalar | — | `kappa`, `onset.kappa` |
| $p$ | bifurcation parameter $p=\kappa-\kappa_c=\epsilon^2\mu$ | scalar | — | `kappa - onset.kappa` |
| $\mu$ | scaled distance from onset | scalar | — | `MU` |
| $J$, $G$ | integrated coupling gains (rows = targets); V1: $J=\kappa G$ | $K\times K$ | — | `model.J(kappa)`, `presets.V1_G` |
| $\Sigma$ | kernel variances | $K\times K$ | — | `model.Sigma` |
| $D_\tau$ | diagonal of time constants $\tau_i$ | $K\times K$ | — | `model.tau` |
| $g_j,\theta_j$ | sigmoid gain and threshold | $K$ | — | `model.gain`, `model.theta` |
| $F_j$, $D_F$ | activation, $D_F=\diag F_j'(0)$ | — | — | `model.activation`, `activation_derivatives()` |
| $e$ | input population weights | $K$ | — | `model.drive` |
| $L$ | domain length; $q_h=2\pi h/L$ | scalar | — | `model.length` |
| $M(q,p)$ | dispersion matrix $D_\tau^{-1}[-I+\widehat W(q;p)D_F]$ | $K\times K$ | — | `linear_matrix` |
| $\nu$ | eigenvalue of $M$ | complex | — | `eigenvalues` |
| $k_0,q_0,\omega_0$ | critical lattice index, wavenumber, frequency | scalars | — | `onset.k0`, `onset.q0`, `onset.omega0` |
| $\Phi,\Psi$ | right / adjoint critical eigenvectors, $\lVert\Phi\rVert_2=1$, $\Psi^*\Phi=1$ | $\CC^K$ | — | `onset.Phi`, `onset.Psi` |
| $\alpha$ | $\Psi^*M_1\Phi$ | complex | — | `nf.alpha` |
| $a,b,c$ | $a=\mu\alpha$; self and cross cubic coefficients | complex | — | `MU*nf.alpha`, `nf.b`, `nf.c` |
| $A,B$ | amplitudes of $\Phi E_\pm$, $E_\pm=\ee^{\ii(\omega_0 s\pm k_0x)}$ | complex | — | `critical_amplitudes` (returns $\epsilon A,\epsilon B$) |
| $P,S$ | complex temporal and spatial input harmonics | $\CC^K$ | — | `forcing_coefficients(P=, S=)` |
| $\eta_t,\eta_\pm,\eta_x$ | forcing coefficients in the amplitude equations | complex | — | `forcing_coefficients`; $\eta^{(1)}$ = `unit_forcing_coefficients` |
| $\rho_*,\phi_*,z_*$ | standing equilibrium $A=B=z_*=\sqrt{\rho_*}\ee^{\ii\phi_*}$ | — | — | `standing_equilibrium` |
| $s,\tau$ | fast time $s=t$ and slow time $\tau=\epsilon^2 t$ | scalar | — | `t` (fast only) |

## Terms

| Term | Our one meaning | Do not use for it |
|---|---|---|
| forcing | the weak external input $I(x,t)$ | drive, stimulus (in prose) |
| drive vector | $e$, which populations receive the forcing | input weights |
| temporal forcing | spatially uniform input $\cos(\omega t)$ | flicker (except when citing Rule et al.) |
| spatial forcing | static input $\cos(kx)$ | grating |
| traveling forcing | input $\cos(kx-\omega t)$ | moving stimulus |
| onset | the first $\kappa$ where $u=0$ loses stability | bifurcation point, threshold |
| threshold | the forcing strength at which the zero amplitude loses stability | onset |
| pattern | a non-uniform oscillating state at $k_0$ | structure |

## Known clashes to fix (PAP-1)
- $\tau$ is both the slow time (Sec. 1) and the time constants $\tau_i$, $D_\tau$ (Secs. 7–8).
- $d$ (Secs. 1–6) and $K$ (Secs. 7–8) both count populations.
