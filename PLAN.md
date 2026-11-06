# Plan — Pattern emergence under weak, unrelated forcing

Owned by me. Agents change only the Status column, unless I ask them to add tasks.
One row = one task that one session can finish. Group tasks by workstream.
The method for a task, if written, is in `plans/<ID>.md` (see STYLE.md, Task plans).
Examples of good rows: `Research_convention/examples/PLAN-examples.md`.

Status values: todo · in progress · blocked · done

| ID | Workstream | Task | Done when | Status |
|---|---|---|---|---|
| SET-1 | Setup | Agent files, layout, clean-up, reusable package, 1D notebooks | `uv run pytest -q` passes; `notebooks/wilson_cowan_1d.ipynb` and `notebooks/v1_1d.ipynb` run headless with no errors | done |
| THY-1 | Theory | Derive the amplitude equations on the square torus for the critical shell (axial and oblique modes), with temporal, spatial, and traveling forcing | `theory/square-torus/main.tex` states the equations and their coefficients; `referee` verdict reported | todo |
| CODE-1 | Code | Add the 2D coefficients of THY-1 to `normal_form.py` | Depends on THY-1. A test checks each new coefficient against a 2D simulation growth rate or amplitude | blocked |
| SIM-1 | Simulation | 2D Wilson–Cowan notebook: forcing below onset on the square torus | Depends on CODE-1. `notebooks/wilson_cowan_2d.ipynb` runs; the late planform and amplitude are compared with THY-1 | blocked |
| SIM-2 | Simulation | 2D V1 notebook: the same comparison for the multi-population model | Depends on CODE-1. `notebooks/v1_2d.ipynb` runs; comparison table with 3 seeds | blocked |
| PAP-1 | Paper | Revise Sections 6–8 in my style, using SET-1 numbers | `paper/main.tex` Secs. 6–8 revised; every number comes from the code; list of changes in the journal | todo |
| PAP-2 | Paper | Move the bibliography to `references.bib` | Each entry checked against its source (DOI); `paper/main.tex` uses `\bibliography{../references}` | todo |
