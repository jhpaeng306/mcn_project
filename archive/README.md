# Archive

Superseded material, kept for reference. Nothing here is imported by `src/` or `tests/`.

| Path | What it is | Replaced by |
|---|---|---|
| `legacy_code/simulations/` | previous package (`simulation.py`, `fft.py`, `pipeline*.py`, `main_analysis.py`, `v1_*.py`) | `src/mcn_project/` |
| `legacy_code/neural_field.py`, `run_analysis.py` | earlier five-population implementation | `src/mcn_project/` |
| `legacy_code/wc2dn.py` | the PI's 2D Wilson–Cowan code (was `wc2dn.txt`); source of the Wilson–Cowan parameters | `presets.wilson_cowan()` |
| `legacy_notebooks/` | previous notebooks with their saved outputs | `notebooks/` |
| `legacy_tests/` | tests of the previous package | `tests/` |
| `notebook_builders/` | scripts that generated the previous notebooks | — |
| `audits/v1_final_report/` | numerical audit of the earlier report | `tests/test_theory_values.py` |
| `main_before_sec8_rewrite.tex` | paper draft before the Sec. 8 rewrite | `paper/main.tex` |

To run the legacy code: `cd archive/legacy_code && uv run python -m simulations.main_analysis --dim 1`.
The new solver reproduces `legacy_code/simulations/simulation.py` to $10^{-16}$ on a forced Wilson–Cowan run.
