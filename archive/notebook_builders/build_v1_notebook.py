"""Build and execute the current notebook from the fresh simple-model results."""
from pathlib import Path
import json
import os
import uuid
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'simulations/v1_three_four_population.ipynb'
cells=[]
def md(s):cells.append(dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=s.splitlines(keepends=True)))
def code(s):cells.append(dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},source=s.splitlines(keepends=True),execution_count=None,outputs=[]))
md(r'''# Simple V1-inspired examples: weak forcing below onset

This notebook accompanies Sections 7–9 of `final_report/main.tex`. All results
use the new integer gains and round parameters below. No earlier parameter file,
fit, or simulation output is loaded.

- Domain: **2 mm** (20 spatial units, each 0.1 mm).
- Common sigmoid: gain **4**, threshold **1/4**, shifted to vanish at zero.
  The nonzero second derivative supplies quadratic forcing interaction.
- Time constants: **(10, 15, 30, 20) ms** for E/PV/SST/VIP.
  These are simple effective fast/slow relaxation assumptions.
- Gaussian standard deviations: **75 μm** for E sources, **100 μm** for
  inhibitory sources. Broader inhibition is an explicit modeling assumption.
- Integer gains are tested for a first finite-wavenumber Hopf instability and
  stable cubic amplitude dynamics. Adding VIP uses unit-strength links.
- Literature motivates the reduced circuit motif and Gaussian distance rule,
  not these exact numerical gains, widths, or time constants.

The overall gain $\kappa$ is increased from zero to find its first instability.
$p=\kappa-\kappa_c$ measures distance from this computed onset. The frequency
and critical mode are outputs, not imposed targets. Activity and input are
dimensionless; the model does not specify voltage/current/rate conversion.
''')
code('''from pathlib import Path
import sys, json, subprocess
import numpy as np
from IPython.display import display, Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'simulations/v1_simple.py').exists())
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from simulations.v1_simple import make_model, coefficients, audit, OUT
RECOMPUTE=False
if RECOMPUTE:
    subprocess.run([sys.executable,'-m','simulations.v1_simple'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','simulations.v1_simple_validation'],cwd=ROOT,check=True)
records=json.loads((OUT/'parameters.json').read_text())
validation=json.loads((OUT/'validation.json').read_text())
print('Results:',OUT)
''')
md('## 1. Parameters, onset, and coefficient calculation\nThe following recomputes the eigenvectors and normal-form coefficients directly from the explicit model, then checks the newly saved results.')
code('''models={}; analyses={}
for K in (3,4):
    model=make_model(K); critical,normal,extra=coefficients(model)
    checks=audit(model,critical,normal,extra)
    models[K]=model; analyses[K]=(critical,normal,extra)
    record=records[str(K)]
    np.testing.assert_array_equal(model.A_p,record['G'])
    for name,value in [('alpha',normal.alpha),('b',normal.b),('c',normal.c)]:
        target=complex(record[name]['real'],record[name]['imag'])
        assert abs(value-target)<1e-9
    print(f'\\n{K} populations\\nG =\\n{model.A_p}')
    print('kappa_c, mode, frequency (Hz):',critical.p0,critical.mode,1000*critical.omega0/(2*np.pi))
    print('alpha, b, c (1/ms):',normal.alpha,normal.b,normal.c)
    print('c_perp (1/ms):',extra['c_perp'])
    print('Other square-torus modes: spectral gap (1/ms)',checks['square_spectral_gap'])
''')
md(r'''## 2. Dispersion
The continuous curve interpolates the discrete torus modes (dots). Both examples
have a **negative local frequency slope** at onset. Selecting a rightmost
branch can introduce switches; the general derivative conditions are in the report.
''')
code("display(Image(filename=str(OUT/'dispersion.png')))\n")
md(r'''## 3. Direct nonlinear field simulations
$\epsilon=0.02$, $p=-\epsilon^2$. Actual cosine input peaks equal
$\epsilon^2\delta$: temporal $\delta_t=8$, spatial $\delta_x=8$, and combined
$(\delta_t,\delta_x)=(8,4)$. These simple values satisfy the relevant threshold
conditions. Input acts equally on E and PV. The direct field uses FFT convolution
and RK4; it does not integrate the amplitude equations.

The common color scale compares the late fields. The critical spatial mode is
$k=2$, while the spatial input and quadratic harmonics are at $k=4$. Lower panels
show 100-ms RMS Fourier magnitudes to remove fast oscillation clutter.
''')
code('''for K in (3,4):
    display(Image(filename=str(OUT/f'forcing_1d_{K}pop.png')))
    print(K,records[str(K)]['metrics_1d'])
''')
md(r'''## 4. Traveling input on the square torus
The input $8\epsilon^2 e\cos(2\omega_0t-2q_0x)$ travels along $x$.
The resulting stripes run parallel to $y$. These runs contain a transverse
critical-mode seed and generic spatial noise; the transverse components decay.
''')
code('''for K in (3,4):
    display(Image(filename=str(OUT/f'torus_2d_{K}pop.png')))
    print(K,records[str(K)]['metrics_2d'])
''')
md(r'''## 5. Scaling, weaker-forcing control, and numerical accuracy
The new runs at $\epsilon=0.01,0.02,0.03$ all use $p=-\epsilon^2$ and temporal
peak $8\epsilon^2$. The predicted combined critical amplitude is
$\sqrt{2\rho_*}\epsilon$. Additional peak-$4\epsilon^2$ runs remain below the
instability threshold and their critical perturbations decay.

Refinement checks use the same initial field: a 5-s time-step comparison and a
0.5-s 2D grid/time refinement. They are finite-duration checks, not a proof of
full-field stability. Core property tests are in `tests/test_v1_simple.py`.
''')
code('''display(Image(filename=str(OUT/'epsilon_scaling.png')))
for K in (3,4):
    print(K,json.dumps(validation[str(K)],indent=2))
print('All notebook coefficient and spectral checks passed.')
''')
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (project .venv)',language='python',name='python3'),language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)
if __name__=='__main__':
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(ROOT);shell=InteractiveShell.instance();count=0
    for cell in cells:
        if cell['cell_type']!='code':continue
        count+=1
        with capture_output() as captured:result=shell.run_cell(''.join(cell['source']),store_history=False)
        outputs=[]
        for name in ['stdout','stderr']:
            if getattr(captured,name):outputs.append(dict(output_type='stream',name=name,text=getattr(captured,name)))
        for item in captured.outputs:outputs.append(dict(output_type='display_data',data=item.data,metadata=item.metadata))
        cell.update(execution_count=count,outputs=outputs)
        if result.error_before_exec or result.error_in_exec:raise RuntimeError(captured.stdout+captured.stderr)
        print('Executed cell',count,flush=True)
    PATH.write_text(json.dumps(nb,indent=1)+'\n');print(PATH)
