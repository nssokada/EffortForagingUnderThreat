"""
Prepare model input files from Stage 5+6 outputs.

Generates the clean CSVs that the MCMC model comparison reads:
  - data/model_input/choice_trials.csv    (subj_idx, threat, distance_H, choice)
  - data/model_input/vigor_cell_means.csv (subj_idx, T_round, actual_dist, ..., mean_rate, n_trials)
  - data/model_input/subject_mapping.csv  (subj, subj_idx)

Usage:
  python scripts/preprocessing/prepare_model_input.py
  python scripts/preprocessing/prepare_model_input.py --stage5_dir <path>
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def find_latest_dir(base, prefix):
    candidates = sorted(base.glob(f"{prefix}*"))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Prepare model input files")
    parser.add_argument("--stage5_dir", type=str, default=None)
    parser.add_argument("--vigor_dir", type=str, default="results/stats/vigor_analysis")
    parser.add_argument("--output_dir", type=str, default="data/model_input")
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    args = parser.parse_args()

    # Find stage5 dir
    if args.stage5_dir:
        stage5_dir = Path(args.stage5_dir)
    else:
        for sample in ["exploratory_350", "confirmatory_350"]:
            base = Path(f"data/{sample}/processed")
            d = find_latest_dir(base, "stage5_")
            if d:
                stage5_dir = d
                break
        else:
            print("ERROR: No stage5 output found.")
            return

    vigor_dir = Path(args.vigor_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage 5: {stage5_dir}")
    print(f"Vigor:   {vigor_dir}")
    print(f"Output:  {out_dir}")

    # Load behavior (choice trials only)
    beh = pd.read_csv(stage5_dir / "behavior_rich.csv", low_memory=False)
    if args.exclude:
        beh = beh[~beh['subj'].isin(args.exclude)]
    choice = beh[beh['type'] == 1].copy()

    # Load cell means
    cell_means = pd.read_csv(vigor_dir / "cell_means.csv")
    if args.exclude:
        cell_means = cell_means[~cell_means['subj'].isin(args.exclude)]

    # Build subject mapping (subj -> 0-indexed subj_idx)
    all_subjs = sorted(set(choice['subj'].unique()) | set(cell_means['subj'].unique()))
    subj_map = {s: i for i, s in enumerate(all_subjs)}
    mapping_df = pd.DataFrame({'subj': all_subjs, 'subj_idx': range(len(all_subjs))})

    # Choice trials
    choice_out = pd.DataFrame({
        'subj_idx': choice['subj'].map(subj_map).values,
        'threat': choice['threat'].values,
        'distance_H': choice['distance_H'].values,
        'choice': choice['choice'].values,
    })

    # Vigor cell means
    vigor_out = cell_means.copy()
    vigor_out['subj_idx'] = vigor_out['subj'].map(subj_map)
    vigor_cols = ['subj_idx', 'T_round', 'actual_dist', 'actual_R', 'actual_req',
                  'is_heavy', 'mean_rate', 'n_trials']
    # Keep only columns that exist
    vigor_cols = [c for c in vigor_cols if c in vigor_out.columns]
    vigor_out = vigor_out[vigor_cols]

    # Save
    choice_out.to_csv(out_dir / "choice_trials.csv", index=False)
    vigor_out.to_csv(out_dir / "vigor_cell_means.csv", index=False)
    mapping_df.to_csv(out_dir / "subject_mapping.csv", index=False)

    print(f"\nSaved:")
    print(f"  choice_trials.csv:    {len(choice_out)} trials, {choice_out['subj_idx'].nunique()} subjects")
    print(f"  vigor_cell_means.csv: {len(vigor_out)} cells, {vigor_out['subj_idx'].nunique()} subjects")
    print(f"  subject_mapping.csv:  {len(mapping_df)} subjects")


if __name__ == "__main__":
    main()
