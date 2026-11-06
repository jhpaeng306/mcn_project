---
name: fig
description: Make or revise one figure through quick iterations. Use when I describe a plot or a change to a plot.
---
1. One figure = one script in figures/, named after the figure (figures/loss_landscape.py → figures/loss_landscape.pdf and .png).
2. Use figures/style.mplstyle. Do not set fonts, sizes, or colors inside the script unless I ask.
3. Load data from results/ or experiments/; never recompute expensive results inside a figure script.
4. After each change: run the script, then look at the PNG yourself (Read the image) and check labels, units, legend, and that nothing overlaps.
5. Reply with: the change in one line, and anything you noticed that I might want changed. Keep iterating on my next spoken or typed instruction.
6. Axis labels and titles use LaTeX notation from NOTATION.md.
