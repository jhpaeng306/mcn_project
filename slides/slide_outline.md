# Slide outline: Pattern formation under weak forcing

15 slides total, including the title. This is the slide content outline, not an implementation task plan.

Sources:
- Previous talk: `16_Jinhee.pdf` (kept outside the repo).
- Current paper: `theory/report/main.tex`.
- Recorded numerical comparisons: `JOURNAL.md`.

The previous talk's pages 10–20 become three proof slides, slides 6–8 below.
Keep its boxed equations and progressive explanation.
Use the current paper's notation and model convention.

## 1. Pattern formation under weak forcing

**Subtitle:** How temporal flicker creates spatial patterns below onset

**Slide text**
- Spatially uniform flicker can evoke geometric visual patterns.
- A proposed mechanism is pattern formation in cortical activity.
- We ask when weak input creates a pattern absent from the input itself.

**Visual:** Uniform flicker beside a spatial pattern.

**Claim:** Biological motivation and a modeling question. V1 is a proposed mechanism, not an established explanation of all hallucinations.

**Reference:** Rule, Stoffregen, and Ermentrout (2011), https://doi.org/10.1371/journal.pcbi.1002158.

## 2. Wilson–Cowan neural fields

**Slide text**
- The model describes interacting excitatory and inhibitory populations.
- Activity changes through decay, spatial coupling, and external input.
- Coupling strengths and spatial ranges determine which modes grow.

$$
D_\tau\partial_tu=-u+W*F(u)+I(x,t).
$$

Here, $u=(u_E,u_I)^\mathsf T$ contains population activities.
The matrix $D_\tau$ contains time constants, $W$ contains coupling kernels, and $F$ is the activation function.
The input is $I$, and $*$ denotes spatial convolution.

**Visual:** Reuse the previous talk's E–I diagram, with excitatory and inhibitory kernels.

**Source correction:** The current paper uses $W*F(u)$; the previous slides use $F(W*u)$. These give different nonlinear coefficients.

## 3. A general model near the homogeneous state

Expand around the homogeneous state, represented by $u=0$:

$$
\partial_tu=\mathcal L(p)u+\mathcal Q(u,u)+\mathcal C(u,u,u)+I(x,t).
$$

**Slide text**
- $\mathcal L(p)$ describes linear growth and decay.
- $\mathcal Q$ and $\mathcal C$ describe quadratic and cubic interactions.
- The parameter $p=\kappa-\kappa_c$ measures distance from unforced onset.
- This formulation covers two-population and multiple-population neural fields.

Here, $\kappa$ is the gain parameter and $\kappa_c$ is its onset value.

**Message:** The reduction depends on critical modes and nonlinear interactions, rather than a particular population count.

## 4. Onset selects a spatial and temporal scale

For each spatial Fourier mode, the linear operator becomes a matrix $M$.

**Slide text**
- The real part of an eigenvalue gives growth or decay.
- The imaginary part gives oscillation frequency.
- At onset, the critical modes have spatial indices $\pm k_0$ and frequency $\omega_0$.

$$
\nu(\pm k_0,0)=i\omega_0.
$$

Here, $\nu$ is the critical eigenvalue branch.
On a domain of length $L$, the physical wavenumber is $q_0=2\pi k_0/L$.
Use the normalized domain of length $2\pi$ for the proof slides.

**Assumptions**
1. The critical eigenvalue is simple within each spatial Fourier block, with reflection-related indices $\pm k_0$.
2. All other modes are strictly stable, and the critical pair crosses transversely.
3. The required quadratic resolvents exist.

**Visual:** Growth rate against wavenumber, showing the critical peak reaching zero.

## 5. Weak flicker below onset

Consider spatially uniform input:

$$
I(t)=\epsilon^2\left(Pe^{2i\omega_0t}+\overline P e^{-2i\omega_0t}\right),
\qquad p=\epsilon^2\mu,\quad \mu<0.
$$

Here, $\epsilon$ is small, $P$ gives the input's population weights and phase, and $\mu$ is the scaled distance below onset.

**Slide text**
- Without input, small perturbations decay.
- The input has no spatial pattern.
- The predicted response has spatial index $k_0$ and amplitude of order $\epsilon$.
- The response oscillates at half the input frequency.

**Visual:** Input and activity space–time plots, placed side by side.

**Claim:** Prediction under resonance and the amplitude conditions introduced on slide 9.

## 6. Proof I: Critical waves and Fredholm solvability

Following pages 10–12 of the previous talk, introduce the expansion and two times:

$$
u=\epsilon u_1+\epsilon^2u_2+\epsilon^3u_3+\cdots,
\qquad s=t,\qquad \tau=\epsilon^2t.
$$

Define $\mathcal D=\partial_s-\mathcal L(0)$. At leading order,

$$
\mathcal D u_1=0,
\qquad
u_1=A(\tau)\Phi e^{i(\omega_0s+k_0x)}
+B(\tau)\Phi e^{i(\omega_0s-k_0x)}
+\text{complex conjugate}.
$$

Here, $\Phi$ is the critical population eigenvector, normalized by $\|\Phi\|_2=1$.
The complex amplitudes $A$ and $B$ describe oppositely traveling waves.

**Boxed principle**

$$
\mathcal Dw=f
\quad\Longleftrightarrow\quad
\langle V,f\rangle=0
\quad\text{for every }V\in\ker\mathcal D^*.
$$

The brackets denote the space–time inner product, and $\mathcal D^*$ is the adjoint operator.
This equivalence uses the Fredholm assumptions, including closed range and invertibility on the noncritical complement.

**Source correction:** Replace “no waves of natural frequencies in the range” with “orthogonality to the adjoint critical modes.”

## 7. Proof II: The scale and resonance conditions

Compress pages 13–19 of the previous talk.
Initially allow

$$
p=\epsilon p_1+\epsilon^2\mu+\cdots,
$$

where $p_1$ is the possible first-order parameter correction.
At second order, the quadratic terms and input have zero projection onto the adjoint critical modes. Thus,

$$
p_1\alpha A=p_1\alpha B=0
\quad\Longrightarrow\quad
\boxed{p_1=0}
$$

for a nonzero leading pattern. Here, $\alpha=\partial_p\nu(k_0,0)$ and $\operatorname{Re}\alpha>0$.

At third order, the input response interacts with a critical wave.
Frequencies and spatial indices must add to a critical wave:

| Input | Resonant interaction |
|---|---|
| Temporal flicker | $(2\omega_0,0)+(-\omega_0,k_0)=(\omega_0,k_0)$ |
| Static spatial input | $(0,2k_0)+(\omega_0,-k_0)=(\omega_0,k_0)$ |

**Boxed result:** Temporal input at $2\omega_0$ and static spatial input at $2k_0$ contribute resonant coupling at this order.

**Scope:** This statement concerns order-$\epsilon^2$ input and the cubic reduction. It does not exclude near-resonant detuning or other forcing scalings.

## 8. Proof III: Projection gives the amplitude equations

Apply the same Fredholm condition at third order, following page 20 of the previous talk:

$$
\begin{aligned}
A_\tau&=A(a+b|A|^2+c|B|^2)+\eta_t\overline B+\eta_xB,\\
B_\tau&=B(a+b|B|^2+c|A|^2)+\eta_t\overline A+\eta_xA.
\end{aligned}
$$

**Slide text**
- $a=\mu\alpha$ describes the distance from onset.
- $b$ and $c$ describe self-interaction and interaction between the two waves.
- $\eta_t$ describes temporal forcing.
- $\eta_x$ describes a reflection-symmetric static input.

**Message:** Solvability determines how the critical amplitudes evolve.

**Visual:** Keep the previous talk's boxed amplitude equations. Label each group of terms.

## 9. A standing pattern exists below onset

For pure temporal forcing, set $\eta_x=0$.
The subspace $A=B=z$ satisfies

$$
z_\tau=z\left[a+(b+c)|z|^2\right]+\eta_t\overline z.
$$

Write $z=\sqrt{\rho}\,e^{i\phi}$, where $\rho$ is squared amplitude and $\phi$ is phase.
The equilibrium condition gives

$$
|a+(b+c)\rho|^2=|\eta_t|^2.
$$

**Proof idea:** If $b+c\neq0$ and $|\eta_t|>|a|$, the corresponding quadratic is negative at zero and positive for large $\rho$.
Its leading coefficient is positive and its constant term is negative, so it has exactly one positive root $\rho_*$.

**Proven within the amplitude system**

$$
\boxed{
|\eta_t|>|\mu\alpha|,
\qquad \operatorname{Re}b<0,
\qquad \operatorname{Re}(b+c)<0
}
$$

These conditions give a standing pattern stable modulo spatial translations.
Each critical mode has leading amplitude $\epsilon\sqrt{\rho_*}$.

**Scope:** The paper proves this stability statement for the amplitude system, not a full-field persistence theorem.

## 10. A V1-inspired population model

**Slide text**
- Replace one inhibitory population with PV, SST, and VIP populations.
- Use population-dependent time constants and connection widths.
- Apply input to E and PV populations.
- Use literature-informed connectivity structure, with effective gains chosen as model parameters.

For the paper's four-population example:

$$
\kappa_c\approx0.74915,
\qquad k_0=3,
\qquad \frac{\omega_0}{2\pi}\approx13.21\ \mathrm{Hz}.
$$

**Visual:** The E/PV/SST/VIP connectivity diagram and kernel-variance matrix.

**Qualification:** This is a V1-inspired model, not a fit to measured synaptic strengths.
Use the current paper's parameters, rather than the previous talk's numerical values.

## 11. Broader excitation: the precise comparison

**Two-population restriction**

For a real two-dimensional dispersion matrix, an oscillatory pair has

$$
\operatorname{Re}\nu(q)=\tfrac12\operatorname{tr}M(q).
$$

Assume Gaussian kernels, positive activation slopes and time constants, $J_{EE}>0$, and $J_{II}\leq0$.
If the homogeneous mode is strictly stable, the following condition prevents finite-wavenumber Hopf at the same parameter:

$$
\Sigma_{EE}\geq\Sigma_{II}.
$$

Here, $J$ contains integrated coupling gains and $\Sigma$ contains kernel variances; rows index targets.
Under source-dependent widths, broader excitation therefore prevents this onset mechanism while the homogeneous mode remains stable.

**V1 example**

Excitation onto E cells is broader than inhibition onto E cells:

$$
\Sigma_{EE}=0.6498>\Sigma_{E,PV}=0.45125,
\qquad \Sigma_{EE}>\Sigma_{E,SST}=0.36125.
$$

With more populations, one oscillatory pair can cross while other eigenvalues keep the total trace negative.

**Qualifications**
- The V1 example does not have excitation broader than every inhibitory connection.
- The two-population restriction does not exclude stationary instabilities or forced patterns.
- Target-specific width comparisons differ from source-dependent width ordering.

## 12. One-dimensional simulations agree with the predicted amplitude

Show unforced and temporally forced activity below onset.

| Model | Measured modal amplitude | Predicted $\epsilon\sqrt{\rho_*}$ |
|---|---:|---:|
| Wilson–Cowan | $0.02331$ | $0.02345$ |
| V1-inspired | $0.001594$ | $0.001635$ |

**Slide text**
- The input is spatially uniform.
- The response contains the critical spatial mode.
- The reported amplitude differences are approximately $0.6\%$ and $2.5\%$.

**Status:** Numerical agreement for the reported configurations.

**Visual:** Space–time plots above a measured-versus-predicted amplitude comparison.

**Source:** SET-1 entry in `JOURNAL.md`. Values are physical modal amplitudes $\epsilon|A|$, not pointwise field maxima.

## 13. Two-dimensional flicker: stability requires competing directions

**Slide text**
- In two dimensions, several wavevectors can share the critical wavelength.
- Uniform flicker couples each wave to its opposite.
- A standing stripe must also resist perturbations in other directions.

**Reported results**
- V1-inspired model: standing stripes resist transverse perturbations.
- V1-inspired model: the symmetric square pattern is unstable.
- Wilson–Cowan example: both stripe and square branches are unstable.

**Visual:** Stripe and square examples beside $\operatorname{Re}c_\theta$.
Here, $c_\theta$ is the cross-interaction coefficient for two critical waves separated by angle $\theta$.
Use `figures/cross_coupling.pdf`.

**Message:** A standing-wave solution need not be stable within the full critical shell.

**Scope:** These results concern the reported parameter sets. Stability requires the full linearization, not only the sign of $\operatorname{Re}c_\theta$.

## 14. Traveling input can create a pattern across its direction

Let $\boldsymbol K$ be the input wavevector.
Temporal resonance couples critical wavevectors $\boldsymbol k_X$ and $\boldsymbol k_Y$ satisfying

$$
\boxed{\boldsymbol k_X+\boldsymbol k_Y=\boldsymbol K}.
$$

On the oblique lattice shell, an example is

$$
(3,1)+(3,-1)=(6,0),
$$

with all vectors expressed in lattice-index units.

**Slide text**
- The input travels along $x$ and is uniform along $y$.
- The response travels along $x$ and forms a standing modulation along $y$.
- The V1 pair persists in the reported run.
- The Wilson–Cowan pair later loses stability to competing modes.

**Visual:** Input and response movies, plus the three wavevectors.

**Status:** Parameter-specific numerical examples. Robust selection across three seeds remains to be checked.

## 15. Dispersion curves show opposite frequency trends

Show the dispersion matrices and their eigenvalue curves.
Define the tracked branch's frequency by

$$
\omega(q)=\operatorname{Im}\nu(q).
$$

**Slide text**
- Wilson–Cowan example: frequency decreases near the critical wavenumber.
- V1-inspired example: frequency increases near the critical wavenumber.
- At large wavenumber, Gaussian coupling vanishes and oscillation frequencies approach zero.

For input frequency $\Omega$, the linear resonance condition is

$$
2\omega(q)=\Omega.
$$

Locally, faster flicker corresponds to coarser candidate patterns in Wilson–Cowan and finer candidate patterns in V1.

**Visual:** Use `figures/dispersion_bell.pdf`.
Distinguish the tracked branch from the eigenvalue with largest real part.

**Qualification:** Resonance identifies candidates; growth rates and nonlinear stability determine which patterns persist.

## Review notes

- The previous PDF was read in full as extracted text. Pages 14, 19, and 20 were also rendered and inspected visually.
- The proof outline preserves its sequence: expansion, solvability, resonance selection, amplitude equations.
- Added explicit adjoint orthogonality, stability conditions, and qualifications on the broader-excitation comparison.
- Removed repeated intermediate proof slides and the claim that the operator range contains no waves at critical frequencies.
- The paper and journal supplied the numerical results. No new simulations were run for this outline.
- A referee checked the conditional two-population trace argument. A skeptic checked the summary against existing sources.
