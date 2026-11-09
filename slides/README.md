# Slides

`meeting_2026-11-09.html` is the deck for the 2026-11-09 meeting (15 slides, content from `slide_outline.md`).
It uses the shell of the `dissociation` and `LLM_Polarization` decks: arrows move, `E` edits a slide, Cmd+S saves.

## Build
Run from the repo root.

    uv run python slides/build/sim1d.py          # 1D runs -> results/slides_sim1d.npz (~30 s, not tracked)
    uv run python slides/build/movie2d.py        # 2D V1 run -> slides/media/v1_traveling.mp4 (~40 s)
    uv run python figures/slides_growth.py       # slides/img/growth.png
    uv run python figures/slides_spacetime.py    # slides/img/spacetime_{wc,v1}.png, flicker_closeup_v1.png
    uv run python slides/build/assemble.py       # sections/*.html -> meeting_2026-11-09.html
    bash slides/build/shot.sh slides/meeting_2026-11-09.html OUTDIR 1 2 3   # PNG checks (CHROME= overrides the binary)

## Images without a generator in this repo
- `hallucination_patterns.png`, `ei_fields.png`, `ei_kernels.png`, `v1_network.png`: cropped from the previous talk (`16_Jinhee.pdf`, pages 2 and 23).
- `v1_2d_flicker_seeds.png`, `wc_2d_flicker_seeds.png`, `v1_2d_traveling_traces.png`: outputs of `notebooks/v1_2d.ipynb` and `notebooks/wilson_cowan_2d.ipynb`.
- `cross_coupling.png`, `dispersion_bell.png`: copies of `figures/cross_coupling.png` and `figures/dispersion_bell.png`.
