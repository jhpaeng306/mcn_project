# Journal

Append-only. Newest entry at the bottom. Only the main agent writes here.
When this file passes ~300 lines, condense older entries into the Summary section and delete them.

Entry format:

    ## YYYY-MM-DD — <task ID>: <short title>
    **Did:** what was run or written, with paths, configs, seeds
    **Found:** the result, in one or two sentences
    **Failed / surprised:** what broke or looked wrong, and what was done about it
    **Confidence:** low / medium / high, and why
    **Next:** the single most useful next step

## Summary of older entries
Before this journal: the paper (`paper/main.tex`) was written through Sec. 8; a first package (`archive/legacy_code/simulations/`) produced the 1D and 2D V1 results; the 2D runs did not give the intended patterns.

## Entries

## 2026-11-06 — SET-1: Agent files, clean-up, and a reusable package
**Did:**
- Applied the `Research_convention` templates (AGENTS, VISION draft, PLAN, NOTATION, JOURNAL, `.claude/` agents and /fig).
- Moved the paper to `paper/`, earlier notes to `theory/notes/`, reference PDFs to `papers/`, and superseded code, notebooks, and tests to `archive/` (see `archive/README.md`).
- Wrote `src/mcn_project/`: one field model for both Wilson–Cowan ($K=2$) and V1 ($K=3,4$), plus onset, coefficients, amplitude equations, a 1D/2D RK4 solver, and plots. Wrote 12 tests and two notebooks (`notebooks/wilson_cowan_1d.ipynb`, `notebooks/v1_1d.ipynb`).

**Found:**
- The new code reproduces the paper's Sec. 7 values: $\kappa_c=0.7491489065$, $\omega_0=0.0829864799$, $\alpha$, $b$, $c$, $\eta_x^{(1)}$, and $|\eta_t^{(1)}|$. The phase of $\eta_t$ depends on the phase of $\Phi$, so the tests compare only $|\eta_t^{(1)}|$.
- The new solver equals the legacy `simulation.py` to $2\times10^{-16}$ on a forced Wilson–Cowan run.
- Theory vs simulation, 1D, temporal forcing at $2\omega_0$, below onset ($\mu=-1$), standing equilibrium of Sec. 6:
  - Wilson–Cowan, $\epsilon=0.1$, peak $1\cdot\epsilon^2$: measured $|A|=2.331\times10^{-2}$, predicted $\epsilon\sqrt{\rho_*}=2.345\times10^{-2}$.
  - V1, $\epsilon=0.02$, peak $3\epsilon^2$: measured $1.594\times10^{-3}$, predicted $1.635\times10^{-3}$. Combined $(3,2)$: $1.49\times10^{-3}$ and $1.50\times10^{-3}$ for $|A|,|B|$, predicted $1.538\times10^{-3}$.
  - The relative gaps (0.6% and 2.5%) are consistent with the expected $O(\epsilon)$ correction.
- 2D, possibly why the 2D runs failed: on the $60\times60$ Wilson–Cowan torus, the first instability is the oblique shell $|k|^2=10$ (modes $(3,1)$, $(1,3)$) at $a_{EE}=7.02297$. That is below the axial onset $7.02492$, so the 1D reduction is not the right one in 2D. For V1 with $L=43$ the axial shell $|k|^2=9$ is first.

**Failed / surprised:** A noise-only initial state did not saturate within 150 s for V1: the forced growth rate is about $3.6\times10^{-5}$ per ms. As in the legacy code, the notebooks seed the critical mode at size $0.02\epsilon$.
**Confidence:** high for the 1D coefficients and solver (exact agreement with the paper and the legacy code); medium for the 2D onset remark (linear only).
**Next:** THY-1. Derive the square-torus amplitude equations for the critical shell, including oblique shells.

Notes for PAP-1 (paper Secs. 6–8):
- Sec. 6 says "$\eta_x=0$ when $S=0$" before it defines $\eta_x$.
- $\tau$ is both the slow time and the time constants, and $d$ and $K$ both count populations (see NOTATION.md).
- With $\eta_t=0$ (spatial forcing only), the Proposition does not apply. On $A=B$ the reduced equation gives $\rho=-\RePart a_+/\RePart\gamma$ (unchecked); the notebooks show this case without a prediction.
