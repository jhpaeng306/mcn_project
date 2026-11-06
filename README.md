# Pattern emergence under weak, unrelated forcing

A neural field near a finite-wavenumber Hopf point receives a weak input whose structure differs from the pattern, for example a spatially uniform flicker. We derive the amplitude equations that say when and which pattern appears, and test them by direct simulation of a Wilson–Cowan E–I field and a four-population V1 model (E, PV, SST, VIP).

## Start here
- `theory/report/main.tex` the report (PDF: `theory/report/report.pdf`). `VISION.md` the question, `PLAN.md` the tasks, `JOURNAL.md` the log.
- `notebooks/wilson_cowan_1d.ipynb` and `notebooks/v1_1d.ipynb` reproduce the 1D results.
- `src/mcn_project/` the reusable code. All parameter values are in `src/mcn_project/presets.py`.

## Setup and checks
```bash
uv sync
```
```bash
uv run pytest -q
```
```bash
uv run jupyter lab
```

## Minimal example
```python
from mcn_project import presets
from mcn_project.linear import find_onset
from mcn_project.normal_form import cubic_coefficients
from mcn_project.simulate import Forcing, Wave, simulate

model = presets.v1()                       # or presets.wilson_cowan()
onset = find_onset(model, dim=1, kappa_range=(0, 1))
nf = cubic_coefficients(model, onset)      # alpha, b, c of the paper
eps = 0.02
run = simulate(model, onset.kappa - eps**2,
               Forcing((Wave(3 * eps**2, (), 2 * onset.omega0),)),   # uniform input at 2 omega0
               n=64, dt=1.0, t_end=150_000)
```

## Folders
| Folder | Content |
|---|---|
| `src/mcn_project/` | model, presets, linear analysis, amplitude coefficients, simulation, plots |
| `notebooks/` | thin notebooks; parameters in the first code cell |
| `tests/` | checks against the paper's numbers and theory-vs-simulation checks |
| `theory/report/` | the report, its macros (`def.tex`), and the built PDF |
| `theory/legacy/` | earlier tex documents, one folder each |
| `theory/code/` | numerical checks of the derivations |
| `figures/` | one script per report figure |
| `results/` | small computed tables used by figures |
| `papers/` | reference PDFs and `notes.md` |
| `data/` | Billeh et al. (2020) cell classes and connection probabilities |
| `archive/` | superseded code, notebooks, and drafts (see `archive/README.md`) |
