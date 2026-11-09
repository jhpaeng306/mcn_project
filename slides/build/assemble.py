"""Assemble the deck from the section files: uv run python slides/build/assemble.py [--only NAME ...] [--out PATH]

The order of the sections is ORDER; each section file holds <section class="slide ..."> blocks only."""
import argparse
from pathlib import Path

HERE = Path(__file__).parent
ORDER = ("00_intro", "10_proof", "20_v1", "30_simulations")
DECK = HERE.parent / "meeting_2026-11-09.html"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", type=Path, default=DECK)
    args = parser.parse_args()
    names = [n for n in ORDER if not args.only or n in args.only]
    body = "\n".join((HERE / "sections" / f"{n}.html").read_text() for n in names)
    args.out.write_text((HERE / "shell_head.html").read_text() + body + "\n" + (HERE / "shell_tail.html").read_text())
    print(args.out, body.count('<section class="slide'), "slides")


if __name__ == "__main__":
    main()
