# V1 simulation versus final report

Audited 2026-09-28 against the current `final_report/main.tex` source.

**Verdict: the three/four-population V1 notebook explicitly named by the final report implements the report's concrete model and forcing protocols consistently. The older V1 notebooks and root analysis command are different models or calibrations. The current experiments support the reported patterns, but do not complete every stability or asymptotic validation described in the report.**

No existing simulation, report, notebook, or saved result was changed. This directory contains the audit, a runnable check script, and its numerical results.

## Which simulation matches?

| Entry point | Relation to the current report |
|---|---|
| `simulations/v1_three_four_population.ipynb` → `simulations/v1_extension.py` | Correct reference for the report's separately calibrated K=3 and K=4 examples. |
| `simulations/analysis.ipynb` → `simulations/pipeline.py` | Same source-nonlinearity equation family, but legacy K=4 gains and different forcing calibration. Does not reproduce the current report's V1 parameter set. |
| `simulations/analysis_2d.ipynb` → `simulations/pipeline_2d.py` | Uses the legacy preset and includes reflecting boundaries, whereas the report's stripe calculation is on a periodic square torus. |
| Root README command `run_analysis.py` → `neural_field.py` | Different formulation: applies the sigmoid after recurrent convolution and stimulus input; the report convolves the source sigmoid and then adds input. Also defaults to a five-population preset. |

The report explicitly identifies the matching notebook at [main.tex:1230](/Users/jhpaeng306/GitHub/mcn_project/final_report/main.tex:1230). The legacy gains are in [pipeline.py:125](/Users/jhpaeng306/GitHub/mcn_project/simulations/pipeline.py:125). Its largest entrywise difference from the report's K=4 G is 0.258391588. Its onset offset is 0.001 and forcing growth target is 0.0002/ms; the report's are 0.0005 and 0.0001/ms. The root solver's different activation placement is at [neural_field.py:703](/Users/jhpaeng306/GitHub/mcn_project/neural_field.py:703).

## Model and equation checks

| Report requirement | Implementation and result |
|---|---|
| `D_tau u_t = -u + W*f(u) + I`, shifted logistic with `f(0)=0` | Matches the direct RHS, including division of external forcing by target tau. |
| Target rows, source columns; periodized unit-mass Gaussian | Matches FFT multiplier `J_ij exp(-Sigma_ij |q|²/2)`. No missing grid-spacing or normalization factor. |
| `Sigma=(s_um/100)²/2`, stated tau/gain/thresholds and L | Matches the full-precision saved calibration and all four saved simulation files. |
| Onset-centered `J=(0.75+p_report)G` | Exactly equivalent to code `J=(0.5+p_notebook)G`, with `p_report=p_notebook-0.25`. Saved run point 0.2495 therefore means report p=-0.0005. |
| Q and C use the **output** wavenumber and factors 1/2 and 1/6 | Matches `compute_normal_form`, including resolvent factors and cubic permutation counts. |
| Temporal input at `2 omega0`; static spatial input at `2 k0` | Matches the direct solver. E and PV receive equal input; SST/VIP receive none directly. Cosine peak d is correctly passed to the coefficient calculation as d/2. |
| Traveling input `cos(2 omega0 t - 2 q0 x)` | Matches `cos(2 q0 x - 2 omega0 t)` by cosine symmetry. It drives the report's A-minus direction. |
| Square-torus radial wavenumbers | Matches `q_x²+q_y²`; exact y-independent data stay y-independent. |

Sources: [model constants and construction](/Users/jhpaeng306/GitHub/mcn_project/simulations/v1_extension.py:29), [direct solver](/Users/jhpaeng306/GitHub/mcn_project/simulations/v1_extension.py:153), [normal form](/Users/jhpaeng306/GitHub/mcn_project/simulations/fft.py:175), [forcing projection](/Users/jhpaeng306/GitHub/mcn_project/simulations/fft.py:248).

The direct solver uses the full shifted sigmoid, rather than truncating it to Q+C. This is consistent with the concrete field in the report: the cubic system is its near-onset approximation. Likewise, the amplitude helper uses physical time and physical modal amplitudes. With U=epsilon A, slow time=epsilon²t, and p_report=epsilon²mu, its coefficients `alpha*p_report` and physical forcing harmonics are the rescaled report equations; epsilon has not been omitted.

## Recomputed numerical values

| Quantity | K=3 | K=4 |
|---|---:|---:|
| Notebook onset p | 0.25 | 0.25 |
| Critical integer mode | 3 | 3 |
| omega0 (rad/ms) | 0.08 | 0.08 |
| Frequency (Hz) | 12.732395 | 12.732395 |
| Homogeneous growth at onset (/ms) | -0.001 | -0.001 |
| Frequency derivative at q0 | 0.0331672606 | 0.0457007971 |
| Re(c_perp) | -15.34932555 | -16.42225054 |
| Traveling scaled cosine threshold, mu=-1 | 1.05854045 | 1.02588348 |
| Temporal/traveling physical cosine peak | 0.00221810132 | 0.00212728756 |

These reproduce the rounded values in [the dispersion section](/Users/jhpaeng306/GitHub/mcn_project/final_report/main.tex:1061), [the traveling stability section](/Users/jhpaeng306/GitHub/mcn_project/final_report/main.tex:1190), and [the strength calibration](/Users/jhpaeng306/GitHub/mcn_project/final_report/main.tex:1300).

For both models, the temporal existence inequality holds and the three traveling-wave critical stability inequalities are negative at the actual run point. The traveling scaled peaks are 4.43620 and 4.25458, above their respective thresholds.

Saved 2D axis-only results have exactly zero transverse RMS. With a transverse seed, traveling-pump RMS falls from 1.22e-4 to 1.33e-6 (K=3), and 1.06e-4 to 3.89e-8 (K=4). Uniform temporal pumping instead grows transverse activity, as the notebook describes. The traveling response is dominated by A-minus: late magnitudes 0.003109 and 0.002332, versus A-plus approximately 5.08e-6 and 3.50e-6. The cubic predictions for A-minus are 0.003237 and 0.002780, respectively. Thus the wave direction agrees; the cubic approximation is not an exact finite-amplitude prediction.

## Remaining limitations and mismatches

1. **The default instructions do not select the report's calibration.** Use the explicitly referenced three/four-population notebook when reproducing this final report. Update entry-point documentation if the intended default is now this report.

2. **Full-field Floquet stability is not implemented in the V1 extension.** The report defines transverse period maps and Gamma_n for every retained transverse index at [main.tex:1202](/Users/jhpaeng306/GitHub/mcn_project/final_report/main.tex:1202). The notebook measures evolution of a particular seeded perturbation instead. Its decay supports stability for that experiment; it does not check every transverse perturbation, compute Floquet multipliers, or bound omitted modes. This is a validation gap, not a mismatch in the direct field equation. The root module's separate Floquet utility belongs to a different model and does not supply this stripe calculation.

3. **Cached runs do not fully identify or validate their forcing protocol.** `RECOMPUTE=False` is the notebook default. Its loader checks G, tau, Sigma, gain, and theta; the 1D caller also checks p. It does not check the forcing amplitudes/frequencies, seed, domain length, solver version, or requested duration/grid against a complete saved specification; the 2D caller does not check p. The NPZ files omit several of those inputs entirely. A future forcing or solver edit can therefore reuse stale results. Current files passed the independent replay checks below. See [the notebook builder's loader](/Users/jhpaeng306/GitHub/mcn_project/notes/build_v1_notebook.py:235).

4. **The amplitude helper covers the real spatial-forcing special case.** The report's general equations have distinct eta-plus and eta-minus for complex S; [amplitude.py:35](/Users/jhpaeng306/GitHub/mcn_project/simulations/amplitude.py:35) uses one eta-x for both. This agrees with the present V1 real, zero-phase cosine inputs. Arbitrary spatial phases or complex S would require extending that helper or explicitly restricting its API.

5. **Asymptotic and long-time convergence claims remain limited.** The V1 notebook uses one onset distance. It does not demonstrate an epsilon-refinement sequence or the report's asymptotic remainder bounds. Existing timestep tests run to 1,000 ms; the saved spatial refinement record is at 2,000 ms. These are shorter than the 60,000/80,000 ms experiments. The present audit does not establish full-duration convergence or rigorous field stability.

## Verification performed

- Entire existing test suite: **40 passed in 44.94 s** using the project `.venv`.
- Independently evaluated the report's perpendicular coupling and traveling threshold formulas; verified the displayed values to their rounding precision.
- Recomputed onset, normal-form coefficients, forcing calibration, and diagnostics from the saved arrays.
- Checked all saved model matrices, sigmoid parameters, and run p against the current calibration.
- Freshly replayed the first **1,000 ms** for K=3 and K=4, in both 1D and 2D, including every case in each protocol. Maximum modal difference from each saved trajectory: **0.0**.
- Inspected full-duration saved diagnostics. The full 60/80-second experiments were **not** rerun in this audit.

Reproduce the additional checks with:

```sh
MPLCONFIGDIR=/private/tmp/mcn-audit-mpl .venv/bin/python output/audits/v1_final_report/check.py
```

Machine-readable results: [checks.json](/Users/jhpaeng306/GitHub/mcn_project/output/audits/v1_final_report/checks.json). Runnable audit: [check.py](/Users/jhpaeng306/GitHub/mcn_project/output/audits/v1_final_report/check.py).
