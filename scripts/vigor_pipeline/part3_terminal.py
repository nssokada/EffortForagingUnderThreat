#!/usr/bin/env python3
"""
Part 3: Terminal persistence

Analysis 3.1: Terminal vigor by threat
Analysis 3.2: Terminal collapse
Analysis 3.3: Persistence metric
Analysis 3.4: Acceleration vs deceleration
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, pearsonr, ttest_1samp
from pathlib import Path

OUT_DIR = Path("/workspace/results/stats/vigor_analysis")

METRICS = ['median_ipi', 'norm_rate', 'relative_vigor', 'frac_full',
           'press_sd', 'pause_freq', 'n_presses']


def run_part3(metrics_df=None):
    print(f"\n{'='*70}")
    print("PART 3: TERMINAL PERSISTENCE")
    print(f"{'='*70}")

    if metrics_df is None:
        metrics_df = pd.read_csv(OUT_DIR / "vigor_metrics.csv")

    # Load params
    params = pd.read_csv("results/stats/full_analysis/part1_params_full.csv")

    # ═══════════════════════════════════════════════════════════════
    # 3.1: Terminal vigor by threat
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 3.1: Terminal vigor by threat (attack trials only) ---")

    term = metrics_df[metrics_df['epoch'] == 'terminal'].copy()
    antic = metrics_df[metrics_df['epoch'] == 'anticipatory'].copy()

    print(f"  Terminal trials: {len(term)}")

    print(f"\n  {'Metric':<20} {'T=0.1':>10} {'T=0.5':>10} {'T=0.9':>10} {'t(0.9v0.1)':>12} {'p':>10}")
    print(f"  {'-'*75}")

    for metric in ['relative_vigor', 'frac_full', 'pause_freq', 'norm_rate']:
        vals = {}
        for T in [0.1, 0.5, 0.9]:
            sub = term[term['T_round'] == T].groupby('subj')[metric].mean()
            vals[T] = sub

        means = {T: v.mean() for T, v in vals.items()}
        shared = sorted(set(vals[0.1].index) & set(vals[0.9].index))
        if len(shared) > 20:
            t, p = ttest_rel(vals[0.9].loc[shared], vals[0.1].loc[shared])
            sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
            print(f"  {metric:<20} {means[0.1]:>10.4f} {means[0.5]:>10.4f} {means[0.9]:>10.4f} {t:>12.3f} {p:>10.4f} {sig}")
        else:
            print(f"  {metric:<20} {means.get(0.1, np.nan):>10.4f} {means.get(0.5, np.nan):>10.4f} {means.get(0.9, np.nan):>10.4f} {'N/A':>12}")

    # ═══════════════════════════════════════════════════════════════
    # 3.3: Persistence metric
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 3.3: Persistence (terminal / anticipatory) ---")

    # Merge terminal and anticipatory for same trials
    term_m = term[['subj', 'trial', 'T_round', 'distance', 'cookie',
                    'relative_vigor', 'frac_full', 'norm_rate', 'pause_freq']].copy()
    antic_m = antic[['subj', 'trial', 'relative_vigor', 'frac_full', 'norm_rate', 'pause_freq']].copy()
    merged = term_m.merge(antic_m, on=['subj', 'trial'], suffixes=('_term', '_antic'))

    # Persistence = terminal / anticipatory
    for metric in ['relative_vigor', 'norm_rate']:
        col_t = f'{metric}_term'
        col_a = f'{metric}_antic'
        merged[f'persist_{metric}'] = merged[col_t] / merged[col_a].replace(0, np.nan)

    valid = merged.dropna(subset=['persist_relative_vigor'])
    valid = valid[np.isfinite(valid['persist_relative_vigor'])]
    valid = valid[valid['persist_relative_vigor'] < 10]  # remove extreme outliers

    print(f"  Trials with valid persistence: {len(valid)}")
    print(f"  Mean persistence (relative vigor): {valid['persist_relative_vigor'].mean():.3f}")
    print(f"  Median: {valid['persist_relative_vigor'].median():.3f}")
    print(f"  SD: {valid['persist_relative_vigor'].std():.3f}")

    # By threat
    print(f"\n  Persistence by threat:")
    for T in [0.1, 0.5, 0.9]:
        sub = valid[valid['T_round'] == T]
        print(f"    T={T}: M={sub['persist_relative_vigor'].mean():.3f}, "
              f"SD={sub['persist_relative_vigor'].std():.3f}, N={len(sub)}")

    # By distance
    print(f"\n  Persistence by distance:")
    for D in [1, 2, 3]:
        sub = valid[valid['distance'] == D]
        print(f"    D={D}: M={sub['persist_relative_vigor'].mean():.3f}, N={len(sub)}")

    # cd → persistence
    subj_persist = valid.groupby('subj')['persist_relative_vigor'].mean()
    merged_cd = pd.DataFrame({'subj': subj_persist.index, 'persistence': subj_persist.values})
    merged_cd = merged_cd.merge(params, on='subj')
    if 'log_cd_z' in merged_cd.columns:
        r_cd, p_cd = pearsonr(merged_cd['log_cd_z'], merged_cd['persistence'])
        print(f"\n  cd → persistence: r={r_cd:+.3f}, p={p_cd:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 3.2: Terminal collapse
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 3.2: Terminal collapse ---")

    # Collapse = terminal rate < 50% of anticipatory rate
    valid['collapsed'] = (valid['persist_relative_vigor'] < 0.5).astype(int)

    collapse_rate = valid['collapsed'].mean()
    print(f"  Collapse rate (terminal < 50% of anticipatory): {collapse_rate*100:.1f}%")

    # By threat
    print(f"\n  Collapse rate by threat:")
    for T in [0.1, 0.5, 0.9]:
        sub = valid[valid['T_round'] == T]
        print(f"    T={T}: {sub['collapsed'].mean()*100:.1f}% (N={len(sub)})")

    # By distance
    print(f"\n  Collapse rate by distance:")
    for D in [1, 2, 3]:
        sub = valid[valid['distance'] == D]
        print(f"    D={D}: {sub['collapsed'].mean()*100:.1f}% (N={len(sub)})")

    # By cookie
    print(f"\n  Collapse rate by cookie:")
    for ck, label in [(1, 'Heavy'), (0, 'Light')]:
        sub = valid[valid['cookie'] == ck]
        print(f"    {label}: {sub['collapsed'].mean()*100:.1f}% (N={len(sub)})")

    # Who collapses? cd/ce differences
    subj_collapse = valid.groupby('subj')['collapsed'].mean()
    merged_collapse = pd.DataFrame({'subj': subj_collapse.index, 'collapse_rate': subj_collapse.values})
    merged_collapse = merged_collapse.merge(params, on='subj')
    for param, label in [('log_k_z', 'k'), ('log_beta_z', 'β'), ('log_cd_z', 'cd')]:
        if param in merged_collapse.columns:
            r, p = pearsonr(merged_collapse[param], merged_collapse['collapse_rate'])
            print(f"\n  {label} → collapse rate: r={r:+.3f}, p={p:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 3.4: Acceleration vs deceleration
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 3.4: Terminal acceleration vs deceleration ---")

    valid['accelerated'] = (valid['persist_relative_vigor'] > 1.0).astype(int)
    valid['decelerated'] = (valid['persist_relative_vigor'] < 1.0).astype(int)

    print(f"  Overall: {valid['accelerated'].mean()*100:.1f}% accelerate, "
          f"{valid['decelerated'].mean()*100:.1f}% decelerate")

    print(f"\n  By threat:")
    for T in [0.1, 0.5, 0.9]:
        sub = valid[valid['T_round'] == T]
        print(f"    T={T}: {sub['accelerated'].mean()*100:.1f}% accelerate, "
              f"{sub['decelerated'].mean()*100:.1f}% decelerate")

    # Does threat shift the balance?
    print(f"\n  % accelerating by T (within-subject):")
    subj_accel = valid.groupby(['subj', 'T_round'])['accelerated'].mean().reset_index()
    accel_wide = subj_accel.pivot(index='subj', columns='T_round', values='accelerated').dropna()
    if 0.1 in accel_wide.columns and 0.9 in accel_wide.columns:
        t, p = ttest_rel(accel_wide[0.9], accel_wide[0.1])
        print(f"    T=0.1: {accel_wide[0.1].mean()*100:.1f}%, T=0.9: {accel_wide[0.9].mean()*100:.1f}%")
        print(f"    Paired t: t={t:.3f}, p={p:.4f}")

    # Persistence → survival
    beh_rich = pd.read_csv(Path("/workspace/data/exploratory_350/processed/"
                                 "stage5_filtered_data_20260320_191950/behavior_rich.csv"),
                            low_memory=False, usecols=['subj', 'trial', 'trialEndState', 'isAttackTrial'])
    beh_rich = beh_rich[~beh_rich['subj'].isin([154, 197, 208])]
    beh_rich['survived'] = (beh_rich['trialEndState'] == 'escaped').astype(int)
    atk_surv = beh_rich[beh_rich['isAttackTrial'] == 1]

    valid_surv = valid.merge(atk_surv[['subj', 'trial', 'survived']], on=['subj', 'trial'], how='left')
    r_surv, p_surv = pearsonr(valid_surv['persist_relative_vigor'].dropna(),
                               valid_surv['survived'].loc[valid_surv['persist_relative_vigor'].dropna().index])
    print(f"\n  Persistence → survival (trial-level): r={r_surv:+.3f}, p={p_surv:.2e}")

    print(f"\n  Part 3 complete.")


if __name__ == '__main__':
    run_part3()
