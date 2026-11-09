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

## 2026-11-06 — SIM-2: V1 in 2D, route 1 (x-traveling input → y-structure) and route 2 (uniform flicker)
**Did:** `notebooks/v1_2d.ipynb` (about 13 min headless). Random initial fields of size $0.1\epsilon$, seeds run in parallel (`simulate_seeds`). Route 1: $\epsilon=0.025$, input $\cos(6q_1x+2\omega_0t)$ with $|\eta|=3|\mu\alpha|$, on shells $|\boldsymbol k|^2=10$ ($L=45.01$) and $25$ ($L=71.17$), seeds 0–2, $8\times10^4$ ms. Route 2: uniform flicker, peak $3\epsilon^2$, $\epsilon=0.025$, $L=42.70$, seeds 0–5, $4\times10^5$ ms. New helpers: `linear.critical_wavenumber`, `simulate.simulate_seeds`, `simulate(t_start=...)` for continuations, `analysis.shell_amplitudes`, `shell.locked_pair`, `shell.flicker_states`, `plotting.shell_traces`, `plotting.xt_cut`; 2 new tests (21 in total pass).
**Found:**
- Route 1, shell 10: all three seeds select $(3,\pm1)$ with $\epsilon|A|=0.00242$ (theory $0.00254$, 5%); other waves $<2\times10^{-6}$. The pattern travels along $x$ with the input and is $\cos(q_1y)$ along $y$, at a seed-dependent $y$-position.
- Route 1, shell 25: all seeds select $(3,\pm4)$ ($y$-mode 4) with $0.00271$ (theory $0.00289$, 6%).
- Route 2: six seeds give six stripes, three along $x$ and three along $y$, no squares; amplitude $0.00277$ (theory $0.00288$, 4%).
**Failed / surprised:**
- At $\epsilon=0.05$ the route-2 stripe was 17% below theory. A 1D run gives the same value, so the gap is the $O(\epsilon)$ correction of the stripe itself; it falls to 4% at $\epsilon=0.025$.
- Correction to the THY-1 entry: the "$0.00378$ while 1D" stripe value there was measured before the stripe had saturated.
- Skeptic: reproduced route 1 with seed 7 and with a $48\times48$ grid, both to three digits, using an independent FFT projection. Its cautions: only two $\epsilon$ values were tested, and the neighbouring shells $9$ and $13$ decay almost as slowly as the critical one at these lengths. It also noted that `critical_amplitudes` uses a least-squares fit on $(\Phi,\bar\Phi)$, not the $\Psi$ projection. This changes non-critical waves at the $10^{-6}$ level only.
**Confidence:** high (3–6 seeds, identical outcomes, independent reproduction, $O(\epsilon)$ gaps).
**Next:** THY-4 (why V1 has $\mathrm{Re}\,c_\theta<0$ at all angles).

## 2026-11-06 — SIM-1: Wilson–Cowan in 2D, same two routes
**Did:** `notebooks/wilson_cowan_2d.ipynb` (about 3.5 min). Same structure; $\epsilon=0.1$; shells 10 ($L=61.42$) and 25 ($L=97.11$); flicker on $L=58.26$; random initial fields.
**Found:** In every seed the locked pair first grows to near the predicted amplitude ($\approx0.025$ vs $0.027$ on shell 10). Then the other shell waves grow and the field settles in a large-amplitude state ($|A|\sim0.05$–$0.36$). Under flicker, all four seeds end in large-amplitude squares ($\approx0.5$). This matches the theory (positive shell eigenvalues $0.84$, $0.27$; stripe and square both unstable). The end states are beyond cubic order.
**Confidence:** high that the weakly nonlinear patterns are unstable; the large-amplitude end states are only described, not explained.
**Next:** none required for this task.

## 2026-11-09 — SIM-1/SIM-2 follow-up: corrections to report Secs. 10.5–10.6
**Did:**
- New check `theory/code/check_shell_growth_2d.py` (about 1.5 min). Output: `results/check_shell_growth_2d.txt`.
- Part A: V1 route 1a (notebook setup, $\epsilon=0.025$, seeds 0–2). It gives the linear rates of the shells $|\boldsymbol k|^2=9,13$, and their size and decay in the runs.
- Part B: Wilson–Cowan, `check_oblique_2d.py` setup, seeds 1–3. It gives the Jacobian eigenvectors at the locked pair and the growth of the other waves after the lock. The fits use the slow part of $A_{\boldsymbol k}$. A sliding least-squares fit on $\{1,\ee^{-2\ii\omega_0t}\}$ removes the leak of the pair into `critical_amplitudes`.
- Part C: the V1 stripe under flicker at $\epsilon=0.05$, from $A=B=z_*$, as a 1D field to 200 s.
- Reran `theory/code/check_square_2d.py` for V1 at $\epsilon=0.05$ and $0.025$. Output: `results/check_square_2d_V1_eps{0.05,0.025}.json`.
- Edited report Secs. 10.5–10.6 and rebuilt `report.pdf`. Updated slide 14.

**Found:**
- The "22% / 5%" gaps are the V1 $x$-traveling pair of the Sec. 10.5 table. The gap falls by 4.7 when $\epsilon$ is halved. The report no longer calls the gaps $O(\epsilon)$ corrections.
- Two V1 table entries were stale. The current `check_square_2d.py` gives the stripe $0.00341$ (table: $0.00378$) and the square $0.00273$ (table: $0.00256$) at $\epsilon=0.05$. The 1D run of part C also gives $0.00341$, from 12 s to 200 s. All other V1 entries reproduce. With the new entries, every V1 gap shrinks by a factor of 2.8 (square) to 4.7 (pair) when $\epsilon$ is halved.
- Neighbouring shells, linear rates at $\epsilon=0.025$: $-7.3\times10^{-5}$ (shell 9), $-5.5\times10^{-5}$ (shell 10), $-2.0\times10^{-4}$ (shell 13) per ms.
- In the runs, shells 9 and 13 decay steadily from the initial noise. Shell 9 decays at $-1.6$ to $-1.7\times10^{-4}$ per ms. Their largest population Fourier coefficient is below $3\times10^{-6}$ after 40 s and at most $3.4\times10^{-9}$ at 80 s. The pair has $1.6\times10^{-3}$. The earlier "below $10^{-10}$" had no source.
- Wilson–Cowan Jacobian at the locked pair: $+0.8385$ on $(-1,\pm3)$, $+0.5992$ and $+0.5$ on $(-3,\pm1)$, $+0.4399$ on $(1,\pm3)$.
- After the lock, $(-1,\pm3)$ grow at $0.70$–$0.71$ per slow time in all three seeds (slow part between $10^{-5}$ and $10^{-4}$). This is 16% below $0.84$; the cause is not tested. They reach $10^{-4}$ first in seeds 1 and 2, and together with $(-3,-1)$ in seed 3. Other waves give scattered, window-dependent rates.
- At the lock, the slow part of every other wave is at most $2.5\times10^{-5}$. So the Sec. 10.5 bound "below $10^{-5}$" now applies to V1 only, with the Sec. 10.6 value $2\times10^{-6}$ as its source.

**Failed / surprised:**
- Correction to the THY-1/THY-2/THY-3 entry and to report Sec. 10.5. "The $(\pm1,\pm3)$ waves grow at the predicted rate $+0.59$" was wrong. $0.5992$ is the eigenvalue on $(-3,\pm1)$. The waves with the largest eigenvalue, and the first to reach $10^{-4}$, are $(-1,\pm3)$.
- Correction to the SIM-2 entry: the stripe $0.00378$ is not reproduced by the current `check_square_2d.py`; it came from an earlier version of the script.
- A discarded first version of the check fitted the raw $|A_{\boldsymbol k}|$ and was not saved. Two checker agents showed its rate depended on the window, and that "grow first" failed in one seed.

**Confidence:** high for the corrected numbers. Three seeds were used where the initial state matters. A skeptic agent reproduced them with other seeds and measures. The V1 table entries were rerun with the current scripts. Medium for the $\epsilon$-scaling remark, which rests on two $\epsilon$ values.
**Next:** test why the measured Wilson–Cowan growth is 16% below the Jacobian eigenvalue (Jacobian at the measured pair, or a smaller $\epsilon$).
