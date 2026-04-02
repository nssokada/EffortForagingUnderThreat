#!/usr/bin/env python3
"""
Full Analysis Pipeline — run_all.py
Master script that executes Parts 1-4 sequentially.
Each part is a separate module that can also be run independently.

Usage:
  python scripts/full_pipeline/run_all.py          # run everything
  python scripts/full_pipeline/part1_choice.py      # run Part 1 only
  python scripts/full_pipeline/part2_vigor.py       # run Part 2 only
  etc.
"""

import sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("FULL ANALYSIS PIPELINE")
    print("=" * 70)

    from part1_choice import run_part1
    from part2_vigor import run_part2
    from part3_imminence import run_part3
    from part4_policy_clinical import run_part4

    p1 = run_part1()
    if p1.get('STOP'):
        print(f"\n*** STOPPED AT PART 1: {p1['STOP']} ***")
        sys.exit(1)

    p2 = run_part2(p1)
    if p2.get('STOP'):
        print(f"\n*** STOPPED AT PART 2: {p2['STOP']} ***")
        sys.exit(1)

    p3 = run_part3(p1, p2)
    if p3.get('STOP'):
        print(f"\n*** STOPPED AT PART 3: {p3['STOP']} ***")
        sys.exit(1)

    p4 = run_part4(p1, p2, p3)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE — {elapsed:.1f} min")
    print(f"{'=' * 70}")
