# Task plan — EXAMPLE: Estimate grid spacing and orientation from rate maps

This file is an example of the task plan format. Agents ignore files named `EXAMPLE*.md`.
For a real task, copy this file to `plans/<task ID>.md` and replace the content.

Status: approved
<!-- proposed = a draft by me or an agent, may have gaps; the agent fills them and asks for approval.
     approved = the agent follows it exactly. -->
Written by: me

## Input and output
- Input: rate maps $r_i(\bm x)$ for each cell $i$, in `data/ratemaps/` (one `.npy` per cell, 2 cm bins).
- Output: `results/grid_params.csv` with columns `cell, spacing_cm, orientation_deg, gridness`.

## Steps
1. Estimate the spatial autocorrelation $C_i(\bm\Delta)$ of each rate map, with the overlap-normalized Pearson correlation.
   Check: $C_i(\bm 0) = 1$ for every cell.
2. Find the six local maxima of $C_i$ closest to the origin, excluding the central peak.
   Check: exactly six maxima; if not, mark the cell `irregular` and skip steps 3–4 for it.
3. Compute the spacing as the median distance of the six maxima from the origin, in cm.
4. Compute the orientation as the angle of the maximum nearest to $0^\circ$, modulo $60^\circ$.
5. Compute the gridness score from the correlations of $C_i$ rotated by $60^\circ, 120^\circ$ versus $30^\circ, 90^\circ, 150^\circ$.
6. Save one row per cell. Save the plot of $C_i$ with the six maxima marked for 5 random cells in `figures/check_autocorr.pdf`.

## Rules for this task plan
- Follow the steps in order. Do not change the method.
- If a step fails or seems wrong, stop and report the step and the reason.
- Print or save the result of each step, so each step can be checked alone.
