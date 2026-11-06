# Review of MCN_Project-5 and proposed research storyline

Reviewed 28 September 2026. Source: `/Users/jhpaeng306/Downloads/MCN_Project-5.pdf`, all eight pages. Equation numbers below refer to that PDF. This is an audit and work plan; the supplied report and existing model sources have not been edited.

Your clarified 2D goal is **stripes parallel to y, with activity varying along x**, driven by an input traveling along x. Your frequency question concerns **increasing spatial wavenumber**.

## 1. Overall assessment

The central formal multiple-scales calculation is sound for real, pointwise symmetric multilinear nonlinearities and an isolated finite-wavenumber Hopf pair. In particular, the quadratic response, the factors 3 and 6 in the cubic projection, and the self- and cross-coupling coefficients are correct. The temporal parametric coupling correctly involves the conjugate of the opposite traveling-wave amplitude.

The report needs a correction to spatial forcing for general complex S, more precise spectral hypotheses, and several qualifications about solvability and pattern stability. Section 6 derives conditions for fixed amplitudes, not a complete classification of oscillatory patterns. Section 7 is currently empty. There are no numerical parameter values or simulation results in the supplied PDF to validate; the additional numerical checks below concern explicitly identified examples already present in the project.

## 2. Equation audit

| Location | Finding | Revision |
|---|---|---|
| Before and in (2) | The Fourier space incorrectly excludes h=0. Constants cannot be represented with only nonzero Fourier modes. | Use h in Z throughout. The mean mode is essential in (9), (33), and the temporal forcing response. |
| Section 1.1 before (4) | “All eigenvalues are nonzero, except for one” describes neither Hopf nor the real conjugate pair. | At p=0, M0(±k0) each has one simple pair ±iω0; all other eigenvalues of all modes have strictly negative real part. For a rigorous infinite-dimensional reduction, specify a suitable spectral gap and operator/function-space assumptions. |
| (3)–(9) | Reality/reflection relations, adjoint sign, and eigenvalue derivative are correct. | Keep the conjugations. Re α>0 is the transversality condition; α≠0 alone does not ensure a crossing. Resolvents exist only at invertible multipliers. |
| (10)–(12), (24) | The slow-time scaling is correct, but the order equations are not yet complete at the point where they are introduced. | Before p1=0, retain p1 L1 u2 at cubic order. If L=L0+pL1+p²L2+…, also retain p1²L2u1. Both disappear after (29). Present the simplified cubic equation after establishing (29). |
| (15), (18) | The real span of four complex waves is not the real-valued kernel as written. | Write ker_R D={AΦE+ + BΦE− + c.c.: A,B in C}, or use real/imaginary parts as four real basis vectors. Make the analogous correction for D*. |
| (16)–(22) | The adjoint and pairings are correct on the stated periodic Fourier space. | (19) requires temporal frequencies in the same Fourier lattice, Ω in ω0 Z. An arbitrary off-resonant pump is not T0-periodic. Use a common period when commensurate, long-time/multiphase averaging otherwise, and a detuning expansion near resonance. State the functional setting before using the full Fredholm equality. |
| Section 2.3, (23)–(24) | n=m=2 is the correct scale for the proposed quadratic parametric feedback. | State that exponents are integers, or handle intermediate fractional orders. If an exponent equals 3, the corresponding direct forcing belongs in the full u3 equation, even though the separate pure temporal/static spatial inputs in (1) have zero critical projection for k0,ω0>0. Retained terms in (53) assume the corresponding exponent is 2. |
| (25)–(29) | Correct for a nonzero leading pattern. | Describe p1=0 as the consistency condition for this distinguished near-onset scaling. The trivial A=B=0 solution does not itself force p1=0. |
| (30)–(45) | Correct, including all quadratic harmonic factors and cubic permutation counts. | Clarify that Q and C are real pointwise maps. If they are nonlinear convolution operators, use their Fourier multilinear symbols instead; see Section 7 below. |
| (46)–(48) | Correct, including the bars in temporal resonance. | Preserve the distinction between ηt conjugate(B) and spatial coupling proportional to B. |
| (49), XB, (53b) onward | The reverse spatial interaction uses the negative spatial harmonic, so it requires conjugate(s_2k0). | Either restrict to a real even spatial input, or use separate ξ+ and ξ− below. Reflection symmetry of the unforced operator does not make a general complex S real. |
| (56)–(61) | Correct for the symmetric spatial-forcing equations (53). | Require x>0, ζ≠0; reconstruct the phase and cover zero/one-amplitude branches separately. “Closed-form solution” is premature: the PDF stops before supplying it. |
| Section 6 generally | Fixed complex amplitudes describe frequency-locked solutions. | Add relative equilibria with phase drift. Unforced traveling and standing waves generally have constant magnitudes but nonconstant complex amplitudes. |

The expansion should also assume that Q and C preserve the reality and translation/reflection symmetries used in the phase bookkeeping. If Q=0, the particular order-three forcing mechanism derived here vanishes; a different order/scaling may then be required.

### Corrected spatial forcing

Let s=R(0,2k0)S and define

\[
\eta_t=2\Psi^*Q(\bar\Phi,R(2\omega_0,0)P),\qquad
\xi_+=2\Psi^*Q(\Phi,s),\qquad
\xi_-=2\Psi^*Q(\Phi,\bar s).
\]

For n=m=2, the general resonant system is

\[
\begin{aligned}
A_\tau&=A(a+b|A|^2+c|B|^2)+\eta_t\bar B+\xi_+B,\\
B_\tau&=B(a+b|B|^2+c|A|^2)+\eta_t\bar A+\xi_-A.
\end{aligned}
\]

In general ξ− is neither ξ+ nor its complex conjugate: the projection contains complex Φ and Ψ. If S is real, the static resolvent is real, s is real, and ξ+=ξ−=ηx, recovering (53). A common spatial phase S=e^{iθ}S0 with real S0 can also be removed by shifting the spatial origin. Population-specific phases cannot generally be removed with one shift.

The corresponding general versions of (58), (59), and (61) are

\[
\begin{aligned}
0&=(1-|\zeta|^2)[a+bx(1+|\zeta|^2)]+\xi_+\zeta-\xi_-\bar\zeta,\\
|\xi_-+\zeta[a+(b|\zeta|^2+c)x]|&=|\eta_t|,\\
0&=D(a+bT)+(\xi_+-\xi_-)h-i(\xi_++\xi_-)k,
\end{aligned}
\]

where A conjugate(B)=h+ik. Here k in the last identity is an invariant coordinate, not a spatial wavenumber; rename it to avoid confusion.

For the symmetric case and ηt≠0, reconstruct

\[
u=-\frac{\eta_x+\zeta[a+(b|\zeta|^2+c)x]}{\eta_t},
\quad |u|=1,\quad A=\sqrt{x}e^{i\phi},\quad e^{-2i\phi}=u,\quad B=\zeta A.
\]

If ηt=0, (59) sets its numerator to zero; together with (58) this still gives both equilibrium equations, but the common phase is free. Do not divide by ηt in that case. For example, B=0 and A≠0 additionally require a+b|A|²=0 and ηt conjugate(A)+ξ−A=0; pure traveling branches are not generically preserved by forcing.

### Independent checks performed

An independent Fourier-harmonic expansion using random real symmetric quadratic and cubic tensors was compared with (33), (40), and (43) for 30 complex amplitude pairs. The largest cubic residual was 1.9×10⁻¹⁴. Corrected forcing residuals were below 1.8×10⁻¹⁵; the uncorrected reverse spatial term failed for general complex S. The algebraic identities behind (58) and (61) agreed to below 4.6×10⁻¹³. These numerical identities support, but do not replace, the phase-counting derivation.

Six existing focused tests passed, covering calibrated onset and 2D Fourier shells, dispersion derivatives, and invariance of y-independent solutions. This was not a rerun of all long nonlinear simulations.

## 3. Stability analysis to add

### Unforced patterns first

Write g=Re a, β=Re b, χ=Re c and x=|A|², y=|B|². Then

\[
x_\tau=2x(g+\beta x+\chi y),\qquad
y_\tau=2y(g+\beta y+\chi x).
\]

For a supercritical branch above onset, g>0:

| Pattern | Squared amplitudes | Stability within the two-wave center dynamics |
|---|---|---|
| Zero | x=y=0 | Stable for g<0; unstable for g>0 |
| Traveling wave | x=−g/β, y=0, or reversed | Exists for β<0; stable modulo phase if χ<β |
| Standing wave | x=y=−g/(β+χ) | Exists for β+χ<0; stable modulo the symmetry phases if β<χ |

For the traveling wave the radial eigenvalue is −2g and the opposite-wave growth has real part g+χ(−g/β). For the standing wave the two magnitude eigenvalues are −2g and 2r²(β−χ). At β=χ cubic theory is degenerate and cannot select the pattern.

The phases obey

\[
\theta_{A,\tau}=\Im a+\Im b\,x+\Im c\,y,
\quad
\theta_{B,\tau}=\Im a+\Im b\,y+\Im c\,x.
\]

Consequently the physical angular frequencies to the retained order are

\[
\omega_{TW}=\omega_0+\varepsilon^2\Im(a+br^2),\qquad
\omega_{SW}=\omega_0+\varepsilon^2\Im(a+(b+c)r^2).
\]

Their neutral phases reflect continuous symmetries. Claim orbital stability, not asymptotic convergence to one fixed phase. Stability in this reduced space is not by itself stability to every Fourier mode in a large domain or to the extra critical directions in 2D.

### Forced equilibria

For z=(A,B), form the doubled complex representation of the real four-dimensional Jacobian,

\[
J=\begin{pmatrix}H&K\\\bar K&\bar H\end{pmatrix},
\]

\[
H=\begin{pmatrix}
a+2b|A|^2+c|B|^2&cA\bar B+\xi_+\\
cB\bar A+\xi_-&a+2b|B|^2+c|A|^2
\end{pmatrix},\qquad
K=\begin{pmatrix}
bA^2&cAB+\eta_t\\cAB+\eta_t&bB^2
\end{pmatrix}.
\]

All eigenvalues strictly in the left half-plane imply linear asymptotic stability of an isolated amplitude equilibrium. Treat symmetry-required zeros separately. For periodic amplitude solutions, use Floquet multipliers.

At zero, with symmetric spatial forcing ξ+=ξ−=ηx, the four exponents are

\[
\lambda_{s,\pm}=\Re(a+s\eta_x)
\pm\sqrt{|\eta_t|^2-[\Im(a+s\eta_x)]^2},\qquad s=\pm1.
\]

The square root may be imaginary. This gives a forcing threshold, not just a resonance condition. With ηx=0 and Re a<0, temporal forcing destabilizes zero when |ηt|²>|a|². Near-resonant input ωf=2ω0+ε²δ replaces a by a−iδ/2 in the forcing-locked frame. Zero amplitude here means the small forced background ε²u2F, not the exactly zero field.

## 4. The 2D experiment you actually want

Use the torus T² and a traveling input

\[
F_w=\varepsilon^2[H e^{2i(\omega_0t-k_0x)}+\bar H e^{-2i(\omega_0t-k_0x)}].
\]

Its phase fronts move in +x. A resonant response proportional to e^{i(ω0t−k0x)} varies along x, remains constant along y, and therefore has stripes parallel to y. This is consistent with your clarified goal.

This input is a **single spatiotemporal harmonic**. It is not the sum of the spatial and temporal forcings in (1), and its amplitude equation must be derived accordingly. Put

\[
h_w=R(2\omega_0,-2k_0\mathbf e_x)H,\qquad
\eta_w=2\Psi^*Q(\bar\Phi,h_w).
\]

For the positive-temporal-frequency waves e^{i(ω0t±k0x)}, the resonant term is ηw conjugate(B) in the B equation. There is no corresponding leading forcing term in A from this one traveling pump. Reversing the pump reverses the selected propagation direction. If H denotes physical input before division by the time constants, replace H in the resolvent by Dτ⁻¹H.

The phase-selection rule for a pump (Ωf,kf) acting through a conjugate critical amplitude is

\[
\Omega_f=2\omega_0,\qquad \mathbf k_\nu+\mathbf k_\mu=\mathbf k_f.
\]

For |kf|=2k0 and critical wavevectors of length k0, only the corresponding co-directed pair satisfies this rule. A pump at (ω0,−k0 ex) instead directly resonates with a critical wave; its scaling and reduction differ from this 2:1 mechanism.

There are two useful levels of the 2D claim:

1. **Existence / demonstration:** y-independent initial conditions and y-independent input remain y-independent, by translation symmetry and uniqueness. A 1D solution lifts exactly to a vertical-stripe solution on the torus.
2. **Stability / selection:** add small y-dependent perturbations and establish whether they decay. The invariant subspace alone does not establish stability.

For radial kernels on a square torus, the axis modes (±k0,0),(0,±k0) are simultaneously critical. Other integer pairs with m²+n²=k0² may also be critical; inspect the whole shell. For the existing k0=3 example there are exactly four such modes, requiring four complex amplitudes. Cubic square-symmetry interactions include cross-axis intensity terms and resonant terms of the form A_y+ A_y− conjugate(A_x−); simply copying two uncoupled 1D systems misses these interactions.

For a periodically driven stripe U(x,t), the full transverse variational equation of the source-nonlinearity field is

\[
D_\tau v_t=-v+W*[f'(U)v].
\]

Decompose v into transverse indices n. Within each n, the patterned base can couple different x modes. Compute the largest Floquet exponent γn=(1/T)log ρ(Un(T,0)); require γn<0 for every nonzero transverse mode, with truncation convergence checks. This is the direct test for stable vertical stripes. If the base is not periodic, use the corresponding time-dependent variational analysis instead.

## 5. Dispersion and when frequency increases

For the source-nonlinearity convention already used in the project,

\[
D_\tau u_t=-u+W*f(u)+I,\qquad D_\tau=\operatorname{diag}(\tau_i),
\]

linearize about a specified homogeneous equilibrium u*. With target rows and source columns,

\[
\widehat W_{ij}(q;p)=J_{ij}(p)e^{-\Sigma_{ij}q^2/2},\qquad
M(q,p)=D_\tau^{-1}[-I+\widehat W(q;p)\operatorname{diag}(f'_j(u_j^*))].
\]

This assumes isotropic periodized Gaussian kernels. For an anisotropic covariance replace Σij q² with qᵀΣij q. On a length-L torus, allowed wavenumbers are q=2π|k|/L. A continuous q curve is useful as an interpolating dispersion calculation; mark actual torus modes on it.

If instead the nonlinearity acts on each target's summed input, u_t=Dτ⁻¹[−u+f(W*u+I)], the slope matrix is on the **left** of W. Pick one convention and derive everything consistently.

Let λ(q)=σ(q)+iω(q) be a simple eigenvalue, with Mv=λv, w* M=λw*, w*v=1. Holding the equilibrium and p fixed,

\[
\frac{d\omega}{dq}=\Im(w^*M_qv),\qquad
M_q=-qD_\tau^{-1}[(\Sigma\odot\widehat W)\operatorname{diag}(f')].
\]

Thus ω increases precisely where this imaginary part is positive. Decreasing individual Gaussian coupling magnitudes does not determine its sign: signed feedback loops and phase relationships in v,w matter.

Two instructive checks:

- If every time constant equals τ and every Gaussian variance equals Σ, then λν(q)=[−1+e^{−Σq²/2}ν]/τ for ν in the spectrum of J diag(f'). Every positive-frequency branch satisfies ω′=−Σqω<0 for q>0. This special case cannot explain increasing frequency.
- With strictly positive variances, M(q) tends to −Dτ⁻¹ as q→∞. All eigenvalue imaginary parts tend to zero. A positive frequency cannot increase monotonically over all q≥0 in this instantaneous Gaussian model.

Distinguish continuation of one oscillatory eigenvalue from the frequency of the eigenvalue with largest real part. If a real root becomes dominant, the latter can drop abruptly to zero even while the complex branch remains oscillatory. Also distinguish linear eigenfrequency from nonlinear saturated-pattern frequency and imposed stimulus frequency.

### Rechecked existing examples

The stored calibration in `simulations/output/v1_extension/calibration.json` defines two design examples at p0=0.25, L=47.6023 model units, k0=3, q0≈0.395980, ω0=0.08 rad/ms≈12.7324 Hz. One model length unit is 0.1 mm.

| Quantity | E/PV/SST | E/PV/SST/VIP |
|---|---:|---:|
| Re b | −15.181255 | −15.113961 |
| Re c | −15.172141 | −15.504617 |
| Re(c−b) | +0.009114 | −0.390656 |
| Cubic 1D selection above onset | Standing wave | Traveling wave |
| dω/dq at q0, q in model units | +0.0331673 | +0.0457008 |
| dω/dqphys, qphys in rad/mm | +0.00331673 | +0.00457008 |

I independently rebuilt M from the saved matrices and activation parameters. Eigenvector derivatives agreed with centered finite differences to 4.4×10⁻¹¹ or better. Recomputing α, b and c with the existing normal-form implementation reproduced all saved coefficients exactly at the reported floating-point precision. The onset, shell and derivative tests also passed.

The three-population sign difference is only about 0.06% of |Re b|. Higher-order terms or parameter uncertainty can readily matter; present standing-wave selection as a local cubic prediction and test its robustness. The separately recalibrated models do not constitute an isolated VIP deletion experiment.

The existing branch data show local increases and later decreases or dominance switches. Reuse these as conditional model results, not as general biological laws. Linear growth rate and observed early-time envelope slope need not match during multimode transients; use an eigenmode-aligned perturbation and an appropriate fitting window for the dispersion validation.

## 6. Biologically motivated coefficient selection

The supplied eight-page PDF has no bibliography. I used the references already identified in your project's V1 notes and checked their primary-source records.

| Parameter family | Evidence and defensible use | What must remain an explicit assumption or fit |
|---|---|---|
| Population identities and signs | E/PV/SST/VIP is a useful mouse-V1 reduction. Pfeffer supports PV→PV, SST inhibition of other retained classes, and preferential VIP→SST. | Exact zero edges are a reduction, not universal absence. |
| Connection range | Billeh's distance rule P(r)=C exp(−r²/s²) supplies pair-dependent scales. | VIP-specific substitutions and applying one layer's ranges elsewhere require qualification. |
| Effective gain Jij | Combine connection probability, source density, and effective connection strength with stated units. | Probability alone is not an effective neural-field gain. |
| Transfer functions | Fit baseline activity and response curves; retain f′, f″, f‴ at the operating point. | Arbitrary sigmoid decimals are not measurements. |
| Effective time constants | Fit population response dynamics and test frequency sensitivity. | A rate-model time constant is not automatically a membrane constant. |
| Input direction | Billeh's model provides a rationale for E/PV feedforward drive. | Equal E and PV input is an extra modeling choice. |

Sources: [Pfeffer et al. (2013)](https://www.nature.com/articles/nn.3446), [Billeh et al. (2020), Methods](https://jiaxx.github.io/assets/img/2020%20Systematic%20Integration%20of%20Structural%20and%20Functional%20Data%20into%20Multi-Scale%20Models%20of%20Mouse%20Primary%20Visual%20Cortex.pdf). [Campagnola et al. (2022)](https://www.science.org/doi/10.1126/science.abj5861) supplies cell-class-dependent connectivity and synaptic-dynamics evidence; it does not directly identify this reduced model's gain matrix.

For the Billeh convention, s is not the standard deviation of exp(−r²/(2σ²)): σ=s/√2 and Σ=s²/2. Examples in micrometers are sE→E=114, sE→PV=92, sE→SST=103, sPV→E=95, sSST→E=85. These are source-compiled scales with mixed provenance, not all direct mouse-V1 measurements. Htr3a and VIP are not interchangeable population labels.

As a transparent planar coarse-graining assumption, if source density is ρj and effective per-connection strength is wij, then

\[
J_{ij}=\rho_j w_{ij}\int_{\mathbb R^2}P_{ij}(|x|)\,dx
=\rho_j w_{ij}\pi s_{ij}^2C_{ij},
\]

up to the chosen activity and input normalization. If only a disk-averaged probability Pbar within radius R is known under uniform planar sampling,

\[
\bar P=C\frac{s^2}{R^2}(1-e^{-R^2/s^2}).
\]

This derives the conversion; it does not supply missing densities or effective strengths. If preserving a 2D integrated gain in a 1D field, use the Gaussian marginal consistently.

The existing project gains, sigmoid parameters, and time constants are calibrated hypotheses. The optimization explicitly targets a finite-wave onset and a chosen frequency; it demonstrates existence under those constraints, not independent empirical confirmation. Keep a provenance table labeling each value as measured, borrowed, reduced, or fitted. Use perturbation/sensitivity studies to show how much of the phenomenon survives away from the calibration.

## 7. Bridge from biological field to the abstract report

For the source-nonlinearity model around u*, with f2=f″(u*) and f3=f‴(u*), the quadratic and cubic Fourier maps are

\[
Q_{k_1,k_2}(v,w)=\tfrac12D_\tau^{-1}\widehat W(k_1+k_2)[f_2\odot v\odot w],
\]

\[
C_{k_1,k_2,k_3}(u,v,w)=\tfrac16D_\tau^{-1}\widehat W(k_1+k_2+k_3)[f_3\odot u\odot v\odot w].
\]

Consequently Q(Φ,Φ) at output 0 and at output 2k0 are different vectors. Keep the phase-counting structure of the report, but insert the correct output-wavevector multiplier for each term. The current `simulations/fft.py` already implements this distinction. Derive it explicitly before presenting numerical b and c as an application of the abstract formula. Also divide physical input vectors by Dτ before using the report's forcing resolvents. For a real cosine of peak d, the complex harmonic coefficient is d/2.

## 8. Proposed storyline

**Central question:** How do a biologically motivated recurrent circuit and a weak resonant moving stimulus select the onset, frequency, and stability of stripe patterns?

1. **Model and biological assumptions.** Define populations, units, operating point, nonlinearity placement, kernel normalization, and input. Provide coefficient provenance.
2. **Linear mode selection.** Derive M(q,p), verify a finite-wavenumber Hopf crossing, and show growth and frequency dispersion. Explain the local criterion for increasing ω.
3. **Weakly nonlinear selection in 1D.** Present the two-amplitude reduction and numerical b,c. Put lengthy harmonic tables in an appendix. Distinguish existence from stability and amplitudes from phases.
4. **Resonant forcing.** Compare uniform temporal, static spatial, and traveling spatiotemporal pumps. Derive their different couplings, thresholds, and detuning behavior.
5. **Vertical stripes on the 2D torus.** Lift the x-dependent traveling solution, display stripes parallel to y, then test competing axis modes and transverse stability.
6. **Full-model numerical validation.** Compare measured growth, amplitudes, frequencies, and perturbation decay with theory; include negative controls and convergence.
7. **Interpretation and limits.** Separate measured constraints, calibrated design choices, and predictions. State where cubic approximation and biological robustness remain uncertain.

This ordering turns the current calculation into an explanation of a phenomenon. Do not lead with the full generic closed-form equilibrium classification; it is less informative than the stable branches and the simulations that distinguish them.

## 9. Work plan and decisive figures

| Priority | Action | Completion criterion / figure |
|---|---|---|
| 1 | Repair assumptions and spatial forcing; explicitly choose S real for the first report version if that matches the intended cosine input. | All forcing coefficients, conjugations, and hypotheses match the numerical implementation. |
| 2 | Choose one biological model and freeze one parameter record. | One table of populations, Jij, Σij, τi, activation parameters, operating point, units, and provenance. |
| 3 | Complete dispersion computation. | Plot all eigenvalue real parts and the continued positive-frequency branch against q; mark discrete modes and dominance switches; verify q-derivatives. |
| 4 | Add 1D branch and forced-state stability. | A branch diagram labeled stable/unstable; a forcing-amplitude versus detuning threshold plot. |
| 5 | Validate with 1D full-field simulations. | Space-time heatmaps plus projected A,B and fitted linear growth/frequency; compare with the normal form in the same time and amplitude units. |
| 6 | Add the moving-input 2D experiment. | Show input and field at several times, an x–t slice, and power in (±k0,0) versus (0,±k0). |
| 7 | Establish transverse stability. | Seed y-dependent modes and compare their decay/growth with variational/Floquet exponents; cover all relevant transverse modes. |
| 8 | Test robustness and convergence. | Halve time step, refine spatial grid, vary seed and uncertain biological parameters, and show the conclusions persist over a documented range. |

Use matched below-onset controls: no forcing; resonant forcing below threshold; resonant forcing above threshold; temporal/spatial detuning; reversed traveling direction. Track the critical k0 mode separately from the directly imposed 2k0 component. Otherwise an image of the imposed pattern can be mistaken for an instability. Compare full field u≈εu1+ε²u2 using physical time t and slow time τ=ε²t consistently.

For dispersion validation, initialize small perturbations along a selected eigenvector at one spatial Fourier mode. Fit the complex modal coefficient's log magnitude and unwrapped phase during the linear regime to obtain σ and ω. Compare several q values on the increasing interval and beyond its endpoint. High-q oscillations may decay rapidly, so use a short linear-regime fit rather than only a long-time spectrum.

Measure branch stability with perturbations that leave the exact symmetry subspace. A standing wave held by reflection-symmetric initial conditions, or a 2D stripe held by y-invariant initial data, is not evidence of attraction from general nearby states. Run long enough to resolve the weakest predicted decay/growth exponent, especially the nearly degenerate three-population example.

Existing starting points: `simulations/fft.py` for dispersion/normal-form coefficients, `simulations/amplitude.py` for two-wave dynamics, `simulations/v1_extension.py` for calibrated torus experiments including a traveling pump, `simulations/v1_dispersion_detail.py` for branch and dominance distinctions, and the two V1 LaTeX notes for parameter provenance and 2D variational equations. Their existing results should be reconciled with the corrected assumptions before being incorporated into the new report.
