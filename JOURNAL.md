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

## 2026-11-06 — ORG-2, THY-1, THY-2, THY-3: Report Secs. 8.4–10 (frequency shape, forcing strength, two dimensions)
**Did:**
- Moved the report to `theory/report/` (from `paper/`) and every earlier tex file to `theory/legacy/<name>/`. The source of `jinheepaeng.pdf` is kept unchanged in `theory/legacy/report_jinheepaeng/`.
- In `theory/report/main.tex`, Secs. 1–6 are unchanged. Edits elsewhere: removed the commented-out drafts after Sec. 6 (kept in legacy); added `graphicx`; added the maps $Q_q,C_q$ (fixes the undefined reference) and a Wilson–Cowan paragraph to Sec. 8.1; new Sec. 8.4 (shape of $\omega(q)$), Sec. 9 (forcing strength and wavenumber), Sec. 10 (two dimensions), and a notation table.
- Code: `normal_form.py` gained `pair_forcing_coefficient`, `cubic_coupling`, `cross_coefficient`, `square_coefficients`, `static_forcing_coefficients`; new `shell.py` (all resonant cubic and forcing terms of a lattice shell) and `steady.py` (stimulus-locked states); `amplitude.square_rhs`. Removed `traveling_forcing_coefficient` (now a case of `pair_forcing_coefficient`).
- Tests: `tests/test_two_dimensions.py`, `tests/test_static_forcing.py` (19 tests in total pass).
- Checks: `theory/code/two_parameter_curves.py`, `square_predictions.py`, `check_square_2d.py`, `check_oblique_2d.py`. Figures: `figures/two_parameter_curves.py`, `dispersion_bell.py`, `cross_coupling.py`.
- Installed LaTeX in the container (apt); the report builds with no errors or undefined references.

**Found:**
- The PI's `2parcurves.fig` is the threshold of a static stimulus $q\cos(2\pi mx/60)$ on E (Wilson–Cowan, $N=60$). The recomputed curves match it (for example, $m=6$ minimum 4.75 at $q=-0.6$). Near $q=0$: $m=2k_0$ gives a V (slope $\pm22.26$; the loop), and $m\neq2k_0$ gives parabolas. The derived $\chi,\xi$ match the full model at $q=0.005$ to 3–4 digits for $m\le3$.
- Counting rule: an input couples the critical wave first with one factor if $k\in\{0,2k_0\}$ (amplitude $\propto\epsilon^2$), otherwise with two factors (amplitude $\propto\epsilon$). Sec. 2.3's exclusion of $m=1$ is too strict for non-resonant input; noted in Sec. 9, Sec. 2.3 untouched.
- $\omega(q)\to0$ as $q\to\infty$, so $\omega$ is decreasing or bell shaped. Wilson–Cowan: decreasing from $q=0$. V1: maximum at $2.61q_0$, and the pair stops being rightmost at $2.54q_0$, so it looks monotone on the relevant modes.
- 2D: an input $(\Omega=2\omega_0,\boldsymbol K)$ couples waves with $\boldsymbol k_X+\boldsymbol k_Y=\boldsymbol K$. For $0<|\boldsymbol K|<2q_0$ (oblique shell, $\boldsymbol K=(6,0)$) an input traveling along $x$ locks a pattern traveling along $x$ and standing along $y$ (H2). The pair obeys Proposition 1 with $\gamma=b+c_\theta$. Direct runs from noise select the pair in both models.
- Stability in the shell is set by $c_\theta$. V1: $\mathrm{Re}\,c_\theta<0$ at all angles; the H2 pair is stable ($0.00397$/$0.00242$ measured vs $0.00509$/$0.00254$ predicted at $\epsilon=0.05$/$0.025$). Wilson–Cowan: $\mathrm{Re}\,c_\theta>0$ for $71^\circ$–$146^\circ$; the pair reaches the predicted amplitude, then the $(\pm1,\pm3)$ waves grow at the predicted rate and the field leaves the weakly nonlinear range. This is a likely reason the earlier 2D Wilson–Cowan runs "did not work".
- Under flicker on the axial shell: V1 stripes are stable, squares unstable; Wilson–Cowan stripes and squares are both unstable. All V1 gaps shrink when $\epsilon$ is halved.

**Failed / surprised:** First measurement windows for the Wilson–Cowan 2D checks were wrong (the transverse waves had already saturated); fixed by fitting only while the perturbation is small. The spike of $\mathrm{Re}\,c_\theta$ near $141^\circ$ is a near 2:1 resonance ($\omega(0.67q_0)\approx2\omega_0$), not a bug.
**Confidence:** high for the 1D forcing-strength results and the V1 2D results (independent checks, convergence in $\epsilon$); medium for the Wilson–Cowan 2D end state (beyond cubic order, only qualitative).
**Next:** SIM-1/SIM-2 (2D notebooks), then THY-4 (why V1 has negative $c_\theta$ at all angles).

**Referee (THY-1/THY-2/THY-3), same day.** Verdict: error, with all seven numbered formula groups correct and matching the code. The referee also built an independent grid-based check of eq. (general-cubic), which agrees with `cubic_coupling` to $5\times10^{-13}$ for $b$, $c$, $c_\perp$, $c_\times$ and the oblique couplings.
- Error: the statement that the axial shell allows only $|\boldsymbol K|\in\{0,2q_0\}$ was false; $\boldsymbol K=(k_0,k_0)$ couples $(k_0,0)$ with $(0,k_0)$. Fixed: the restriction now applies only to inputs along an axis, and the diagonal case is stated.
- Gaps, all fixed:
  - Sec. 8.4 now claims only that $\omega$ cannot increase for all $q$; the single maximum is marked as observed. The three-population model is defined.
  - The static row of the wavevector table now includes $-\boldsymbol K$ ($\eta_s^\pm$).
  - The repeated-wave case $A_X^2\bar A_Y$ is written out; it, not the boxed formula, gives $b$.
  - The Proposition names Assumptions 1–3 (new: isotropy, one critical shell, 2D quadratic non-resonance), requires $b+c_\theta\neq0$, and drops the unused hypothesis "the input couples no other pair".
  - "Only $k_0,2k_0$ lock the pattern" now says "at this order".
  - The threshold shift applies for $k\notin\{0,2k_0\}$ and cites $\RePart\alpha>0$.
  - $\lambda_\perp$ takes the real part.
  - The temporal-forcing counting is marked unchecked.
  - The first row of the test table is labelled as a growth rate.
- Notation: the static response is $s_k$ (as in Sec. 4) instead of $\sigma$; the input peak is $\delta$; the stimulus wavenumber is $k$; the quartet coefficient is $c_\times$.
- Not changed: the referee's note that NOTATION.md lacked the new symbols (they had been added in the same session).
