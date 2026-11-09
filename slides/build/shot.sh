#!/bin/bash
# Screenshot slides of a deck: bash slides/build/shot.sh DECK.html OUTDIR N1 N2 ...  (1280x720 PNGs, OUTDIR/sN.png)
deck=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); out=$2; shift 2; mkdir -p "$out"
for n in "$@"; do
  "${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}" --headless=new --disable-gpu --hide-scrollbars \
    --window-size=1280,720 --virtual-time-budget=8000 --screenshot="$out/s$n.png" "file://$deck#$n" >/dev/null 2>&1
done
