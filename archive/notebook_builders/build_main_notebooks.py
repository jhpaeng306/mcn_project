"""Create and execute the two main notebooks, retaining their figure families."""
from pathlib import Path
import json
import os
import uuid
ROOT=Path(__file__).resolve().parents[1]


def notebook(dim):
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=text.splitlines(keepends=True)))
    def code(text):cells.append(dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},source=text.splitlines(keepends=True),execution_count=None,outputs=[]))
    md(r'''# Current V1-inspired analysis'''+(' — square domains' if dim==2 else ' — one-dimensional torus')+r'''

This is a main notebook for the current report. It uses `v1_literature.py`:
the previous cell-pair widths, connectivity matrix, relaxation times and activation
parameters, with a simple domain length. No fitting toward a reference gain
matrix is performed. Missing or parameter-mismatched runs are recomputed;
`RECOMPUTE=True` forces a fresh run.

**Units.** Time is ms; one spatial unit is 0.1 mm. The domain side is
$L=43$ units = 4.3 mm. Activity and additive input are dimensionless.
$J=\kappa G$ is an integrated gain; $p=\kappa-\kappa_c$ measures distance from onset.
The solver uses absolute $\kappa$.

**Biological constraints versus model parameters.** The retained inhibitory motif
follows [Pfeffer et al. (2013)](https://doi.org/10.1038/nn.3446).
[Billeh et al. (2020)](https://doi.org/10.1016/j.neuron.2020.01.040) supplies the
cell-pair distance scales. Its rule $\exp(-r^2/s_{ij}^2)$ becomes
$\Sigma_{ij}=(s_{ij}/100)^2/2$; $s$ itself is not the Gaussian standard deviation
in our $\exp[-r^2/(2\sigma^2)]$ convention. Htr3a-to-VIP reduction and unmeasured
pair substitutions are assumptions. The exact effective gains, $\tau,g,\theta$
are retained model parameters, not values identified uniquely by these papers.
Connection probabilities alone cannot determine integrated rate-model gains.

Billeh's STAR Methods, p. e11, lists $s_{E\to E}=114$,
$s_{E\to PV}=92$, $s_{E\to SST}=103$, $s_{PV\to E}=95$, and
$s_{SST\to E}=85$ μm from Levy and Reyes (2012), and
$s_{PV\to PV}\simeq120$ μm from internal Allen Institute data.
Other inhibitory pairs use 120 μm; E–Htr3a pairs use the corresponding
E–SST scales. Here Htr3a is reduced to VIP. A dash in the displayed
$s$ and $\Sigma$ matrices denotes an absent connection, not zero width.

**Input populations.** Direct visual/thalamic drive to E **and PV** is supported
by [Ji et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/25979090/) and used in Billeh's
model. Equal normalized E/PV strengths are our simplifying assumption; the papers
do not establish that ratio. E-only drive represents a selective-input assumption.
The 1D notebook includes an E-only comparison at the same E-input peak.

**Why $L=43$?** It is a nearby integer choice relative to the earlier 47.6023.
With the restored matrix, the nearest integers 44–51 give an oblique first mode
on the square lattice; 43 gives the axial critical shell $(\pm3,0),(0,\pm3)$.
This supports the axis-aligned stripe construction while preserving increasing
frequency near onset. The domain is a modeling choice, not a measured cortical
period. No gain is changed to obtain this geometry.

The main model has four populations. Simply deleting VIP from the unchanged
matrix gives a **homogeneous** ($k=0$) first Hopf instability. '''+('''Section 7 of this
notebook therefore uses the E/PV/SST submatrix with E→E strengthened by 8.04%
(6.22009 → 6.72), which moves its first crossing to $k_0=3$; its onset
$\kappa_c$ is recomputed by the same sweep.''' if dim==1 else '''The 1D notebook
(`analysis.ipynb`, Section 7) analyzes a three-population model with E→E
strengthened by 8.04%; on the square torus that model's first shell is the
oblique $|k|^2=8$, so it is not included in this 2D notebook.''')+r'''
''')
    code('''from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from io import BytesIO
def showfig(fig):
    buffer=BytesIO();fig.savefig(buffer,format="png",dpi=130)
    display(Image(data=buffer.getvalue()));plt.close(fig)
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'simulations/v1_literature.py').exists())
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from simulations.v1_literature import make_model, coefficients, audit, S_UM
from simulations.main_analysis import run_1d, run_2d, EPS, model_audit
K=4
RECOMPUTE=False
model=make_model(K)
critical,normal,extra=coefficients(model)
print('Populations:',model.names)
print('G (fixed dimensionless effective gains):\\n',model.A_p)
def print_widths(title,values):
    print(title)
    for i,row in enumerate(values):
        print(' '.join(f'{value:9.5g}' if model.A_p[i,j]!=0 else '        -' for j,value in enumerate(row)))
print_widths('Billeh distance scale s (micrometers):',S_UM)
print_widths('Sigma (variance in model coordinates; one length unit = 0.1 mm):',model.Sigma)
print_widths('Gaussian standard deviations (micrometers):',100*np.sqrt(model.Sigma))
print('tau (ms):',model.tau,'; sigmoid gains:',model.gain,'; thresholds:',model.theta)
print(f'L = {model.length} model units = {model.length*.1:g} mm')
print(f'Periodic onset: kappa_c={critical.p0:.12f}, k0={critical.mode}, f0={critical.frequency_hz:.6f} Hz')
print('alpha, b, c (1/ms):',normal.alpha,normal.b,normal.c)
print('c_perp (1/ms):',extra['c_perp'])
print('Square-torus spectral/stability checks:',audit(model,critical,normal,extra))
''')
    if dim==1:
        md(r'''## 1. Runs and connectivity
All below-onset cases use $\epsilon=0.02$, $p=-\epsilon^2=-0.0004$ and equal
E/PV input. The cosine peaks are $\epsilon^2$ times the following numbers:

| case | temporal | spatial | temporal frequency | spatial mode |
|---|---:|---:|---:|---:|
| none | 0 | 0 | — | — |
| temporal | 3 | 0 | $2\omega_0$ | 0 |
| spatial | 0 | 8 | 0 | $2k_0$ |
| spatial + temporal | 3 | 2 | $2\omega_0$ | $2k_0$ |

The leading temporal threshold is $|\alpha|/|\eta_t^{\rm unit}|\approx1.741$.
The selected peak 3 exceeds it; the control value 1 is below it. Spatial peak 8
exceeds the leading spatial threshold $\Re\alpha/\Re\eta_x^{\rm unit}\approx4.659$.
The combined peaks (3,2) satisfy the printed cubic stability inequalities.
These are explicit threshold-based choices; no gain-matrix fit is performed.

The direct nonlinear field uses FFT convolution and RK4, with $N=64$,
$\Delta t=1$ ms and duration 150 s ($\epsilon^2t=60$). Diagnostics are stored
every 5 ms. These are field simulations, not integrations of the amplitude equations.
The code checks cached parameters before reusing a run.
''')
        code('''model,critical,normal,extra,result,summary,OUT=run_1d(K,recompute=RECOMPUTE)
def show(name): display(Image(filename=str(OUT/name)))
print('Output directory:',OUT)
show('connectivity.png')
''')
        md(r'''## 2. Bifurcation and dispersion
The gray curves show all eigenvalues. The highlighted eigenvalue has maximal real
part; the positive imaginary representative is used for a conjugate pair.
Dotted vertical lines mark changes in the selected branch. The torus only admits
$q=2\pi k/L$; the continuous curve is its dispersion interpolation.
''')
        code("show('bifurcation.png')\nshow('dispersion.png')\nshow('frequency_near_onset.png')\nprint('Dispersion diagnostics:',summary['dispersion'])\n")
        md(r'''### Increasing frequency restored
At onset, the restored model gives $d\omega/dq=+0.0471593$ rad/ms per
rad/model-length, or $+0.00471593$ mm/ms when differentiating with respect to
physical wavenumber in rad/mm. The rightmost-mode frequencies at $k=0,1,2,3,4$
are approximately $11.087,11.376,12.159,13.208,14.283$ Hz.
The increase is local and over this mode range, not a claim of global monotonicity.
All branches and changes of the rightmost branch are displayed above.

For a simple branch, with $w^*v=1$,
$$\frac{d\omega}{dq}=\Im(w^*M_qv),\qquad
M_q=-qD_\tau^{-1}\big[(\Sigma\odot\widehat W(q))D_{F'(0)}\big].$$
Changing $L$ changes the discrete wave numbers sampled, not the sign of this
derivative at a fixed $q$ and $\kappa$. Restoring the cell-pair widths and other
previous model coefficients recovers the increasing branch. At a branch switch,
use one-sided branches; a real rightmost eigenvalue has zero oscillation frequency.
''')
        md(r'''## 3. Four forcing cases
The long-time panels show 100-ms RMS input/activity so that thousands of fast
cycles do not alias into apparent spatial patterns. The late zoom shows the
signed field over the final 0.5 s. The critical mode is $k_0=3$; the quadratic
harmonic is $2k_0=6$. All activity panels within each figure share a color scale.
''')
        code("show('forcing_comparison.png')\nshow('forcing_comparison_late_zoom.png')\nshow('frequency_doubling_verification.png')\n")
        md(r'''## 4. Detuning and weaker-forcing controls
The weak temporal case uses peak $\epsilon^2$. Detuned temporal input uses
$1.5\omega_0$ instead of $2\omega_0$; detuned spatial input uses $3k_0$ instead
of $2k_0$. The last control detunes both components. All share the same small
initial field as the resonant cases. A small directly imposed response can
remain even when the critical-mode perturbation decays.
''')
        code("show('off_resonance_and_subthreshold.png')\nfor label in ['none','temporal','spatial','spatial + temporal','weak temporal','detuned temporal','detuned spatial','detuned combined']:\n    print(label,summary['cases'][label])\n")
        md(r"""### E-only versus E/PV drive
These two runs use the same temporal cosine peak $3\epsilon^2$ on E and the same
initial field. The E-only case sets the PV input to zero. Its leading temporal
threshold is approximately 0.724, compared with 1.741 for equal E/PV drive.
A difference in response amplitude reflects forcing projection onto the critical
mode, not evidence that one input pattern is biologically exclusive.
""")
        code("show('input_population_comparison.png')\nprint(summary['cases']['temporal E only'])\n")
        md(r'''## 5. Unforced homogeneous, standing and traveling patterns
The branch diagram is the cubic prediction. With the current coefficients the
unforced traveling branch is stable at cubic order; the standing branch is
unstable to an imbalance of its two counterpropagating amplitudes.
The ideal-pattern panels are analytic illustrations.

The direct runs use $p=-\epsilon^2$ for the below-onset case and $p=+\epsilon^2$
for both above-onset cases. An exactly reflection-symmetric standing seed
preserves the standing-wave subspace; it does **not** establish stability to
general perturbations. A traveling seed tests the traveling branch. These
unforced statements are distinct from the forced standing-wave stability result.
''')
        code("show('normal_form_branches.png')\nshow('ideal_wave_patterns.png')\nshow('unforced_parameter_sweep.png')\nshow('unforced_parameter_sweep_late_zoom.png')\nfor label in ['below onset','standing seed above onset','traveling seed above onset']:\n    print(label,summary['cases'][label])\n")
        md('## 6. All-population late-time fields\nSelect a case below. Activities remain dimensionless; coordinates are shown in mm and seconds.')
        code('''SELECT_CASE='spatial + temporal'
j=list(result['labels']).index(SELECT_CASE)
keep=result['snapshot_t']>=result['snapshot_t'][-1]-500
u=result['snapshots'][keep,j]
fig,axs=plt.subplots(1,K,figsize=(3.5*K,3.5),layout='constrained',squeeze=False)
for pop,ax in enumerate(axs[0]):
    image=ax.imshow(u[:,pop].T,origin='lower',aspect='auto',extent=[result['snapshot_t'][keep][0]/1000,result['snapshot_t'][-1]/1000,0,model.length*.1],cmap='RdBu_r')
    ax.set(title=model.names[pop],xlabel='time (s)',ylabel='x (mm)')
    fig.colorbar(image,ax=ax,shrink=.7)
showfig(fig)
''')
        md(r'''## 7. Three-population model (E/PV/SST)
**Connectivity.** Widths, relaxation times, activation parameters, signs and zeros are the
E/PV/SST block of the four-population model. With the unchanged gains the first
instability is homogeneous ($k=0$, $\kappa_c\approx0.65373$), so the finite-wave amplitude
equations do not apply. A single-entry scan (±10% of each nonzero gain, same sign) shows that
only strengthening E→E moves the first crossing to a nonzero wave number. We therefore use
$$G^{(3)}=\begin{pmatrix}6.72&-1.45455&-6.48944\\5.09798&-0.90615&-8.04066\\2.67174&0&0\end{pmatrix},
\qquad G^{(3)}_{EE}=6.72=1.0804\times6.22009 .$$
All other entries equal the four-population values. `make_model(3, submatrix=True)` returns the
unmodified block.

**Onset.** $\kappa_c$, $k_0$, $\omega_0$ and the normal-form coefficients are recomputed exactly as
for four populations (first modal crossing of $J=\kappa G^{(3)}$). On the 1D torus with $L=43$ the
first crossing is the Hopf mode $k_0=3$. On the square torus the oblique shell $|k|^2=8$ crosses
slightly earlier, so this model is analyzed in 1D only; the audit below checks the 1D mode lattice.
''')
        code('''K3=3
model3=make_model(K3)
critical3,normal3,extra3=coefficients(model3)
checks3=model_audit(K3,model3,critical3,normal3,extra3)
print('Populations:',model3.names)
print('G (E->E strengthened; all other entries from the 4-population matrix):\\n',model3.A_p)
block=make_model(3,submatrix=True).A_p
print('Changed entries (target, source, submatrix value, used value, relative change):')
for i,j in zip(*np.nonzero(model3.A_p!=block)):
    print(' ',model3.names[i],model3.names[j],block[i,j],model3.A_p[i,j],f'{model3.A_p[i,j]/block[i,j]-1:+.2%}')
print(f'Periodic onset: kappa_c={critical3.p0:.12f}, k0={critical3.mode}, f0={critical3.frequency_hz:.6f} Hz')
print('alpha, b, c (1/ms):',normal3.alpha,normal3.b,normal3.c)
print('c_perp (1/ms):',extra3['c_perp'])
print('eta_t, eta_x per unit peak (1/ms):',extra3['eta_t_unit'],extra3['eta_x_unit'])
print(f"d omega/dq at q0: {extra3['nu_q'].imag:.7f} (model units) = {.1*extra3['nu_q'].imag:.8f} mm/ms")
print('1D spectral/stability checks:',checks3)
''')
        md(r'''### Forcing peaks for three populations
The cases use **the same peaks as for four populations**, $(0,0),(3,0),(0,8),(3,2)$ in units of
$\epsilon^2$ (temporal, spatial), with $\epsilon=0.02$, $p=-\epsilon^2$ and equal E/PV input, so the
two models see identical input. The thresholds are this model's own:

- Temporal: $|\alpha|/|\eta_t^{\rm unit}|\approx1.51$. Peak 3 exceeds it and the weak control 1 is below it.
- Spatial: $\Re\eta_x^{\rm unit}<0$, so real spatial input destabilizes the standing branch $A=-B$
  rather than $A=B$. On $A=\sigma B$, $z_\tau=z(a+\sigma\eta_x+\gamma|z|^2)+\sigma\eta_t\bar z$ with
  $\sigma=\operatorname{sign}\Re\eta_x$. Its threshold is $\Re\alpha/|\Re\eta_x^{\rm unit}|\approx26.6$,
  so **spatial peak 8 is below threshold** and the spatial-only case is expected to decay.
- Combined (3,2) satisfies the standing-wave stability inequalities with $\eta_x\to\sigma\eta_x$.
  Its zero-state growth is only $\approx1.5\times10^{-5}$ ms$^{-1}$. The three-population runs therefore last
  **300 s** ($\epsilon^2t=120$) instead of 150 s, so that this case saturates.

The standing seed is shifted by a quarter wavelength ($\sin k_0x$), which is the $A=-B$ pattern.
All other numerical settings equal the four-population runs.

**Why not a spatial peak above 26.6?** $\Re\eta_x^{\rm unit}=-0.0033$ is small compared with
$\Im\eta_x^{\rm unit}=0.055$. A peak of order $30\epsilon^2$ makes the neglected $O((\delta\epsilon^2)^2)$
terms larger than the $O(\delta\epsilon^2\Re\eta_x)$ term that sets the threshold. The next cell
computes the forced steady state of the full 1D field for a time-independent input
$\delta\epsilon^2(1,1,0)^T\cos2q_0x$ and the leading eigenvalue of its Jacobian. Without input
the two agree. At peak 8 both are negative, with the field somewhat more stable. Beyond the
predicted threshold the amplitude equation predicts growth, but the field becomes *more* stable,
so spatial-only forcing does not create the pattern at $\epsilon=0.02$ in this model.
''')
        code('''from scipy.optimize import fsolve
from scipy.special import expit
def forced_state_growth(m,cr,delta,n=64):
    K=m.K;kappa=cr.p0-EPS**2;x=(np.arange(n)+.5)*m.length/n
    q=2*np.pi*np.fft.fftfreq(n,d=m.length/n)
    sym=m.connectivity(kappa)[:,:,None]*np.exp(-m.Sigma[:,:,None]*q**2/2)
    g,th,tau=m.gain[:,None],m.theta[:,None],m.tau[:,None];off=expit(-g*th)
    I=delta*EPS**2*np.array(m.metadata['drive_weights'])[:,None]*np.cos(2*cr.q*x)[None]
    def rhs(u):return (-u+np.fft.ifft(np.einsum('ijz,jz->iz',sym,np.fft.fft(expit(g*(u-th))-off,axis=1)),axis=1).real+I)/tau
    u=fsolve(lambda v:rhs(v.reshape(K,n)).ravel(),np.zeros(K*n),xtol=1e-14).reshape(K,n)
    h=1e-7;f0=rhs(u).ravel();J=np.empty((K*n,K*n))
    for i in range(K*n):
        v=u.ravel().copy();v[i]+=h;J[:,i]=(rhs(v.reshape(K,n)).ravel()-f0)/h
    return np.linalg.eigvals(J).real.max()
sig=checks3['spatial_branch_sign']
print(f"{'peak':>5s} {'field Re lambda':>16s} {'amplitude eq. (A=sigma B)':>26s}")
for delta in [0,8,27,45]:
    pred=EPS**2*(-normal3.alpha+sig*delta*extra3['eta_x_unit']).real
    print(f'{delta:5d} {forced_state_growth(model3,critical3,delta):16.3e} {pred:26.3e}')
''')
        code('''model3,critical3,normal3,extra3,result3,summary3,OUT3=run_1d(K3,recompute=RECOMPUTE)
def show3(name): display(Image(filename=str(OUT3/name)))
print('Output directory:',OUT3)
show3('connectivity.png')
show3('bifurcation.png')
show3('dispersion.png')
show3('frequency_near_onset.png')
print('Dispersion diagnostics:',summary3['dispersion'])
''')
        md(r'''### Four forcing cases, controls and E-only input
Same layout as Sections 3–4, over 300 s. The critical mode is $k_0=3$ and the quadratic harmonic is $2k_0=6$.
''')
        code('''show3('forcing_comparison.png')
show3('forcing_comparison_late_zoom.png')
show3('frequency_doubling_verification.png')
show3('off_resonance_and_subthreshold.png')
show3('input_population_comparison.png')
print('E-only temporal threshold:',summary3['e_only_temporal_threshold'])
for label in ['none','temporal','spatial','spatial + temporal','weak temporal','detuned temporal','detuned spatial','detuned combined','temporal E only']:
    print(label,summary3['cases'][label])
''')
        md(r'''### Unforced branches
Cubic-order stability is read from the signs of $\Re(c-b)$ (traveling) and $\Re(b-c)$, $\Re(b+c)$
(standing); the figure labels are computed from them. For three populations $\Re(b-c)\approx-0.007$,
so the **standing** branch is stable and the traveling branch unstable. This is the opposite of four
populations, but only marginally: $\Re b$ and $\Re c$ nearly coincide. The selection rate
$\epsilon^2\Re\alpha\,(\Re c/\Re b-1)\approx7\times10^{-8}$ ms$^{-1}$ is far too slow to act within
300 s, so both seeded runs above onset keep their initial shape and reach nearly the same amplitude.
''')
        code('''print('Re(c-b) =',(normal3.c-normal3.b).real,'; Re(b+c) =',(normal3.b+normal3.c).real)
show3('normal_form_branches.png')
show3('unforced_parameter_sweep.png')
show3('unforced_parameter_sweep_late_zoom.png')
for label in ['below onset','standing seed above onset','traveling seed above onset']:
    print(label,summary3['cases'][label])
''')
        md(r'''### Three versus four populations
Late critical-mode amplitude (mean over the last 10% of each run) and the dominant frequency of the
critical E mode. For decayed cases the frequency is that of residual noise. The cubic predictions are
$\epsilon\sqrt{2\rho_*}$ for the temporal and combined cases. The three-population combined case is
still approaching its equilibrium at 300 s (about 94% of the cubic prediction), because its zero-state
growth rate is small. The three-population spatial-only case decays because peak 8 is below its threshold.
''')
        code('''for KK,cc,ss in [(3,checks3,summary3),(4,audit(model,critical,normal,extra),summary)]:
    print(f"K={KK}: predicted temporal amplitude {EPS*np.sqrt(2*cc['temporal_rho']):.3e}, combined {EPS*np.sqrt(2*cc['combined_rho']):.3e}")
print(f"{'case':24s} {'A_late (K=3)':>13s} {'A_late (K=4)':>13s} {'f (K=3) Hz':>11s} {'f (K=4) Hz':>11s}")
for label in ['none','temporal','spatial','spatial + temporal','weak temporal','detuned temporal','detuned spatial','detuned combined','temporal E only']:
    a,b=summary3['cases'][label],summary['cases'][label]
    print(f"{label:24s} {a['critical_late']:13.3e} {b['critical_late']:13.3e} {a['mode_frequency_hz']:11.3f} {b['mode_frequency_hz']:11.3f}")
''')
    else:
        md(r'''## 1. Periodic torus and reflecting square
The periodic model uses two-dimensional Fourier multipliers
$$\widehat W_{ij}(q_x,q_y)=\kappa G_{ij}
\exp[-\Sigma_{ij}(q_x^2+q_y^2)/2],\quad q_\ell=2\pi n_\ell/L.$$
The reflecting comparison uses the same multiplier with $q_\ell=\pi n_\ell/L$
in a DCT-II cosine basis. It is an even-reflected kernel model; it is not a torus.

Each boundary's first onset is recomputed on its own mode lattice. For four
populations, the periodic representative is $(3,0)$, while the reflecting
representative is $(6,1)$ in the cosine basis. Their onset gains differ slightly.
Each run uses $\kappa=\kappa_c^{\rm boundary}-\epsilon^2$ and the corresponding
$2\omega_0^{\rm boundary}$ input frequency. Thus the comparison fixes distance
below onset, not the absolute gain. Symmetry-related critical modes are included
in the initial perturbation.

Four forcing cases retain peaks $(0,0),(3,0),(0,8),(3,2)$ in units of
$\epsilon^2$, in the order (temporal, spatial). On the torus the spatial input is
$\cos(2q_0x)$. For the reflecting square it is
$\cos(2n_x\pi x/L)\cos(2n_y\pi y/L)$, matched to its selected cosine mode.
The periodic square has an additional traveling-input case.

Direct field runs use $32\times32$ cells, RK4 with $\Delta t=1$ ms, 150 s duration,
and diagnostics every 5 ms. The reflecting comparison is a numerical extension;
the torus amplitude equations are not a proof of its stability.
''')
        code('''model,geometries,results,summary,OUT=run_2d(K,recompute=RECOMPUTE)
def show(name): display(Image(filename=str(OUT/name)))
print('Output directory:',OUT)
for boundary,geometry in geometries.items():
    print(boundary,'kappa_c:',geometry.point.p0,'wavevector:',geometry.wavevector,'frequency (Hz):',geometry.point.frequency_hz)
''')
        md(r'''## 2. Four-case comparison and evolution
Final panels subtract the spatial mean and use a common color scale. Contrast
includes all nonconstant modes. Selected-mode amplitudes are normalized as real
Fourier/cosine reconstruction amplitudes and displayed as 100-ms RMS values.
Snapshot magnitudes also depend on the oscillation phase at the displayed time.
''')
        code("show('final_state_comparison.png')\nshow('pattern_contrast.png')\nshow('critical_mode_amplitude.png')\nshow('combined_evolution.png')\n")
        md(r'''## 3. Traveling input along x: stripes parallel to y
On the square torus,
$$I(x,y,t)=2\epsilon^2(1,1,0,0)^T\cos(2q_0x-2\omega_0t)$$
The input is uniform in $y$. The desired
stripes are parallel to the $y$ axis: activity varies along $x$.
The initial condition includes an orthogonal critical mode and generic noise.
The mode and transverse-RMS histories test whether these perturbations decay.
This is a finite-duration field test, alongside the cubic stability inequalities;
it is not a general nonlinear PDE stability proof.
''')
        code("show('traveling_stripes.png')\nprint('Traveling case:',summary['periodic']['cases']['traveling'])\n")
        md('## 4. Quantitative diagnostics\n`critical_late` uses the Hopf-eigenvector projection for periodic fields and the population-vector norm of the cosine coefficient for reflecting fields; these quantities should not be directly compared. `E_mode_late_rms` is the raw coefficient RMS. Frequency peaks from decayed modes have negligible power and are not evidence of an oscillatory pattern.')
        code("for boundary,record in summary.items():\n    print('\\n',boundary)\n    for label,metrics in record['cases'].items(): print(label,metrics)\n")
        md('## 5. Inspect every population\nChange the boundary and case selectors. These panels have separate color scales; use the earlier common-scale figure for comparisons across cases.')
        code('''SELECT_BOUNDARY='periodic'
SELECT_CASE='traveling'
selected=results[SELECT_BOUNDARY]
j=list(selected['labels']).index(SELECT_CASE)
fig,axs=plt.subplots(1,K,figsize=(3.5*K,3.5),layout='constrained',squeeze=False)
for pop,ax in enumerate(axs[0]):
    u=selected['final'][j,pop];u=u-u.mean()
    image=ax.imshow(u.T,origin='lower',extent=[0,model.length*.1,0,model.length*.1],cmap='RdBu_r')
    ax.set(title=model.names[pop],xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(image,ax=ax,shrink=.7)
showfig(fig)
''')
    md(r'''## Reproducibility and numerical limits
The parameter manifest, direct field arrays and figure files are saved under the
printed output directory. The cache fingerprint includes all model coefficients,
boundary, forcing, grid, duration, time step and seed. Setting `RECOMPUTE=True`
reruns the full simulations. The four-population matrix is fixed. The unmodified
three-population block, `make_model(3, submatrix=True)`, has a homogeneous first onset;
`make_model(3)` is the E→E-strengthened variant used in Section 7 of the 1D notebook.

`tests/test_main_analysis.py` verifies restoration of the previous coefficients,
the increasing eigenvalue derivative, cubic stability conditions, input targeting,
FFT agreement, time-step refinement and cache sensitivity. `v1_three_four_population.ipynb`
compares the three- and four-population dispersion curves with these same coefficients;
`output/v1_simple/` results use the superseded integer-gain example.
Finite-grid simulations support the mechanism; they do not replace a full
convergence or field-stability proof.
''')
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (project .venv)',language='python',name='python3'),language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)


def execute(nb,path):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell=InteractiveShell.instance();shell.reset(new_session=False);count=0
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        count+=1
        with capture_output() as captured:result=shell.run_cell(''.join(cell['source']),store_history=False)
        outputs=[]
        for name in ['stdout','stderr']:
            if getattr(captured,name):outputs.append(dict(output_type='stream',name=name,text=getattr(captured,name)))
        for item in captured.outputs:outputs.append(dict(output_type='display_data',data=item.data,metadata=item.metadata))
        cell.update(execution_count=count,outputs=outputs)
        if result.error_before_exec or result.error_in_exec:raise RuntimeError(captured.stdout+captured.stderr)
        print(path.name,'executed cell',count,flush=True)
    embedded=sum('image/png' in out.get('data',{}) for cell in nb['cells'] for out in cell.get('outputs',[]))
    if embedded<6: raise RuntimeError(f'Expected embedded figures, found {embedded}')
    path.write_text(json.dumps(nb,indent=1)+'\n')

if __name__=='__main__':
    os.chdir(ROOT)
    import sys
    dims=[int(a) for a in sys.argv[1:]] or [1,2]
    for dim,name in [(1,'analysis.ipynb'),(2,'analysis_2d.ipynb')]:
        if dim in dims:execute(notebook(dim),ROOT/'simulations'/name)
