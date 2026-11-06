# Vision — Pattern emergence under weak, unrelated forcing

Owned by me. Agents read it but never edit it. Change it only when the direction changes.
<!-- Draft written by Claude from papers/mcn2026_project_prompt.pdf and the paper. Edit freely; then delete this line. -->

## Question
A neural field near a finite-wavenumber Hopf point forms spatio-temporal patterns. How does a weak input whose structure differs from the pattern create or select it? An example is a spatially uniform flicker that produces a spatial pattern. We want the analytic answer: which forcing orders $\epsilon^n,\epsilon^m$ and which frequencies $\omega$ and wavenumbers $k$, relative to $\omega_0,k_0$, produce a pattern, and which pattern.

## Why it matters
Flicker-induced hallucinations (Rule et al. 2011) and pattern-sensitive epilepsy are cases where the input does not carry the perceived pattern. If the cortex supplies the pattern and the input only selects it, then the amplitude equations predict the stimulus regimes. This gives testable rules for which stimuli are uncomfortable or evoke patterns.

## Current hypotheses
1. H1: Below onset, input at $2\omega_0$ (temporal) or $2k_0$ (spatial) at order $\epsilon^2$ creates a stable standing pattern at $k_0$ once its strength passes the threshold $|\eta_t|>|\mu\alpha|$. Detuned input does not.
2. H2: In 2D, the same mechanism holds, and the amplitude equations for the critical shell predict which planform appears (stripes, squares, or traveling stripes).
3. H3: A biologically constrained multi-population V1 model has the same structure, and its coefficients set the thresholds.

## What would count as an answer
- Evidence for H1: simulated late amplitudes match $\epsilon\sqrt{\rho_*}$ within $O(\epsilon)$, and the threshold matches, for Wilson–Cowan and V1 in 1D.
- Evidence for H2: 2D simulations show the planform that the 2D amplitude equations predict as stable.
- Evidence against: patterns where the theory predicts none, or a different planform or wavenumber.

## Out of scope for now
- Color opponency; reflecting boundaries; fits to measured synaptic strengths.

## Milestones
- <date>: <draft / talk / submission>
