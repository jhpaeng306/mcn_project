# Current report and notebooks

The current report is [`final_report/main.tex`](final_report/main.tex).
Use [`simulations/analysis.ipynb`](simulations/analysis.ipynb) and
[`simulations/analysis_2d.ipynb`](simulations/analysis_2d.ipynb) as the main notebooks.
Both use the restored literature-based cell-pair widths and previous effective
gains on a 4.3-mm domain, without fitting to a reference matrix.
See [`simulations/README.md`](simulations/README.md) for parameters and run commands.

The following describes the earlier implementation and its presets, which are
retained for reference and are not the current report's parameter source.

# Multi-population visual-cortex neural field

This project derives and simulates the two-amplitude normal form associated
with a finite-wavenumber Hopf instability.  The implementation supports any
number of cell populations, pairwise Gaussian connectivity, weak temporal and
spatial forcing, 1D or 2D domains, and periodic, reflecting, or zero-exterior
boundaries.  It also provides the Gaussian Fourier symbol on the mixed domain
`S1 x R^d` and closed support equilibria for an arbitrary number of interacting
critical waves.

The default five-population preset uses `E/PV/SST-GER/SST-local/VIP`.
Gaussian width ratios come from mouse V1 data, while the SST-GER split is
motivated by Drinnenberg et al. (bioRxiv 2025).  Its dimensionless
Wilson-Cowan weights are calibrated hypotheses near a finite-wave Hopf point;
they are not measured synaptic conductances.  The original `E/PV/SST` preset
remains available.

## Reproduce the analysis

```bash
uv run python run_analysis.py
uv run pytest -q
```

The first command rebuilds the coefficient tables, phase portraits, static
retino-cortical pattern gallery, and 1D/2D simulations under
`output/simulations/`.

Run one case directly with:

```bash
uv run python neural_field.py \
  --preset five \
  --dimension 2 \
  --boundary reflect \
  --forcing combined_2to1 \
  --output output/simulations/custom_2d.png
```

Add `--export-model-inputs` to save static target/pinwheel images and
numerical temporal arrays beside the requested figure.  This never creates an
animation or an autoplay flicker.

Available boundaries are `periodic`, `reflect`, and `zero`.  Available forcing
cases are `unforced`, `direct_1to1`, `temporal_2to1`, `spatial_2to1`, and
`combined_2to1`.

## Revise the biological assumptions

Edit or replace `v1_three_population()` in `neural_field.py`, or construct a
`PopulationModel` directly.  For every parameter matrix, row `i` is the target
population and column `j` is the source population.  `weights[i, j]` is the
signed Wilson-Cowan gain and `widths[i, j]` is the standard deviation of its
unit-mass Gaussian.  Time is in milliseconds; spatial lengths use the same
unit as `widths`.  `gaussian_symbol()`, `mixed_gaussian_symbol()`,
`real_forced_equilibria()`, and `wave_support_equilibria()` expose the general
K-population and forced-amplitude mathematics directly.

The derivation and source audit are in
`output/pdf/hallucination_model_report.pdf`.  The report's proposed visual
inputs are model inputs, not human-viewable flicker.  Do not turn them into a
flashing display without an approved photosensitivity safety protocol.

The full published 17-population Billeh inventory and its 17-by-17 connection
probability table are under `data/`.  Blank matrix entries mean
"undetermined," not zero.  `output/simulations/floquet_resonance_scan.csv`
contains the model-derived finite-mode growth rate versus forcing frequency
and amplitude.
