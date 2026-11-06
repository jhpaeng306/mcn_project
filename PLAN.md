# Plan — Pattern emergence under weak, unrelated forcing

Owned by me. Agents change only the Status column, unless I ask them to add tasks.
One row = one task that one session can finish. Group tasks by workstream.
The method for a task, if written, is in `plans/<ID>.md` (see STYLE.md, Task plans).
Examples of good rows: `Research_convention/examples/PLAN-examples.md`.

Status values: todo · in progress · blocked · done

| ID | Workstream | Task | Done when | Status |
|---|---|---|---|---|
| SET-1 | Setup | Agent files, layout, clean-up, reusable package, 1D notebooks | `uv run pytest -q` passes; `notebooks/wilson_cowan_1d.ipynb` and `notebooks/v1_1d.ipynb` run headless with no errors | done |
| ORG-2 | Setup | Theory folder: `theory/legacy/<name>/` for every earlier tex file, `theory/report/` for the report | Folders exist; `theory/report/main.tex` builds | done |
| THY-1 | Theory | Derive the 2D amplitude equations on the square torus: wavevector rule, general cubic coefficient, locked pair, stability in the shell | Report Sec. 10; `referee` verdict in JOURNAL.md; checks in `tests/test_two_dimensions.py` and `theory/code/check_*_2d.py` | done |
| THY-2 | Theory | Write the PI's forcing-strength scalings ($m=k_0$ vs $m=2k_0$) and the two-parameter diagram | Report Sec. 9; `tests/test_static_forcing.py` passes; `figures/two_parameter_curves.pdf` | done |
| THY-3 | Theory | Explain the bell-shaped frequency curve | Report Sec. 8.4; `figures/dispersion_bell.pdf` | done |
| SIM-1 | Simulation | 2D Wilson–Cowan notebook: flicker, stripes, and the oblique $x$-traveling input | `notebooks/wilson_cowan_2d.ipynb` runs; amplitudes compared with report Sec. 10 | done |
| SIM-2 | Simulation | 2D V1 notebook: the same comparison for the multi-population model | `notebooks/v1_2d.ipynb` runs; comparison table with 3 seeds | done |
| THY-4 | Theory | H3: why V1 allows broader excitation and negative $c_\theta$ at all angles while the E–I model does not | A report subsection states a checkable condition on $(J,\Sigma,\tau)$ and tests it on both models | todo |
| PAP-1 | Paper | Revise Sections 6–8 in my style, using SET-1 numbers | `theory/report/main.tex` Secs. 6–8 revised; every number comes from the code; list of changes in the journal | todo |
| PAP-2 | Paper | Move the bibliography to `references.bib` | Each entry checked against its source (DOI); `theory/report/main.tex` uses `\bibliography{../../references}` | todo |
