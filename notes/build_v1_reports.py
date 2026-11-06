"""Compile the authoritative LaTeX V1 reports (requires latexmk / TeX Live)."""
from pathlib import Path
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'tmp/pdfs/v1_latex'
OUT = ROOT / 'output/pdf'
BUILD.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env['PATH'] = '/Library/TeX/texbin:' + env.get('PATH', '')
for name in ('v1_coefficient_report', 'v1_dispersion_and_2d_report'):
    with (BUILD / (name + '.build.log')).open('w') as log:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode',
                        '-halt-on-error', f'-outdir={BUILD}', name + '.tex'],
                       cwd=ROOT / 'notes', env=env, check=True, stdout=log)
    shutil.copy2(BUILD / (name + '.pdf'), OUT / (name + '.pdf'))
    print(OUT / (name + '.pdf'))
