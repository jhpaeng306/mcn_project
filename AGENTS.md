# AGENTS.md — Pattern emergence under weak, unrelated forcing

Read by Claude Code (via CLAUDE.md), Codex, and Antigravity. Personal style is in the global STYLE.md; this file adds project rules.
STYLE.md lives in `jhpaeng306/Research_convention`. On my machine `link-global-style.sh` links it. In a cloud session it is not linked: attach that repo and read its `STYLE.md` first.

## Every session
1. Read VISION.md, PLAN.md, NOTATION.md, and the last 5 entries of JOURNAL.md.
2. Work only on the PLAN.md task or tasks named in the first prompt. When they are done, stop and wait for me. Do not pick another task yourself.
   - Take the next `todo` task only if the prompt says so (for example, "then continue" or "work through PLAN.md").
   - Set a task's Status to `in progress` when you start it. Never take a task that is already `in progress`; another session owns it.
   - Several tasks that touch different files (for example, one theory task and one coding task) may run in parallel: one subagent or one session per task, each in its own git worktree.
3. Stop when each task's "Done when" condition holds, or when you are blocked.
4. Before you stop: set each task's Status in PLAN.md, and append one JOURNAL.md entry per task, whose title starts with the task ID.

## Environment
- Python: `uv run python ...`. Set up with `uv sync`. The package `mcn_project` (in `src/`) is installed in editable mode, so notebooks can `import mcn_project`.
- Quick checks: `uv run pytest -q` (about 2 seconds).
- Notebooks: `uv run jupyter lab`. Run one headless with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`.
- LaTeX: `latexmk -pdf main.tex` inside `paper/`. The cloud container has no LaTeX; say so instead of claiming a build.
- Simulations are small and run on this machine. Ask before any run longer than ~30 minutes.

## Layout
- `src/mcn_project/` all model logic: `model.py` (the field), `presets.py` (all parameter values), `linear.py` (dispersion, onset), `normal_form.py` (amplitude coefficients), `amplitude.py` (amplitude equations), `simulate.py` (direct field runs), `analysis.py` (measurements), `plotting.py` (standard figures).
- `notebooks/` thin notebooks: parameters in the first code cell, then calls into `mcn_project`. No model logic in notebooks.
- `tests/` checks of the code against the paper's numbers and of theory against simulation.
- `paper/` the theory paper: `paper/main.tex`, with macros in `paper/def.tex` (the one macro file). `paper/jinheepaeng.pdf` is the last compiled version.
- `theory/notes/` earlier derivation notes and reports (reference only). New derivations go in `theory/<topic>/main.tex` and load `../../paper/def.tex`. `theory/code/` SymPy and numerical checks.
- `figures/` one script per paper figure, named like the figure. `results/` tables and summaries.
- `references.bib` the one bibliography. `papers/` reference PDFs + `papers/notes.md`. `data/` V1 connectivity tables.
- `archive/` superseded code, notebooks, tests, and drafts. Read-only; do not import from it.

## Paper rules
- Sections 1–5 of `paper/main.tex` are mine and nearly fixed. Edit them only to fix an error, and list each edit for me.
- Section 6 and later are open for revision. Match my writing in sections 1–5: high level first, short motivating sentences before each equation, boxed key results, tables of wave types.
- Code and paper share conventions: rows of $J$ index targets, $\Sigma$ is a kernel variance, $p=\kappa-\kappa_c$, $\lVert\Phi\rVert_2=1$, $\Psi^*\Phi=1$.

## Checks before claiming a result
- Compare every simulated pattern with the theory: predicted wavenumber, threshold, and amplitude ($\epsilon\sqrt{\rho_*}$).
- The 1D cases are the baseline: `uv run pytest -q` must pass before new 2D numbers are trusted.
- Run with at least 3 seeds when a result depends on the initial state; report mean and spread.
- A surprising result means: look for a bug first, and say so in the journal.

## Agents
- At most 3 agents at once. Main agent = worker; call `skeptic` after any claimed result, `referee` after any new proof or derivation, and `scout` for literature.
- Only the main agent writes JOURNAL.md and PLAN.md. Subagents report back.

## Outputs
- Paper and notes: LaTeX, compact.
- Figures: `figures/style.mplstyle`; PDF for the paper, PNG for quick looks; use /fig.
- Notebooks are committed with outputs at 80 dpi, so results are readable on GitHub.

## Boundaries
- Do not edit VISION.md, or any PLAN.md column except Status, unless I ask.
- Do not delete data, results, or reference papers; move them to `archive/`.
- Ask before pushing to `main`.

## Done means
- The task's "Done when" condition holds, checks pass, and the journal entry states the conclusion and its confidence.
