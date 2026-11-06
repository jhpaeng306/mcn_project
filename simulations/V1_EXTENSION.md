# Current V1 analysis

The main notebooks are **analysis.ipynb** and **analysis_2d.ipynb**.
The authoritative report is `../final_report/main.tex`, Sections 7–9.
See [README.md](README.md) for current parameters, evidence, and run commands.

The current source `v1_literature.py` restores the previous effective gain
matrix and literature-based cell-pair widths. It uses a nearby simple length
`L=43` (4.3 mm), with increasing local frequency and an axial critical shell.
It does not fit a gain matrix toward an earlier matrix. Default visual input
acts on E/PV; the 1D notebook also compares E-only input.

Current field data and figures are in `output/main_literature/`.
`v1_three_four_population.ipynb` uses these same coefficients to compare the
three- and four-population dispersion curves. The integer-gain examples in
`output/v1_simple/` and the reference-matrix fit in `v1_extension.py` are
superseded experiments. Their results are not validation of the current model.
