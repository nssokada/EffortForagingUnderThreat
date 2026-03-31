#!/usr/bin/env python3
"""
Part 1: Does threat modulate vigor? (controlling for cookie choice)

Analysis 1.1: Trial-level vigor by threat within cookie type (LMM)
Analysis 1.2: Within-subject paired comparison (T=0.1 vs T=0.9)
Analysis 1.3: Relative vigor 3×3 surface
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel, pearsonr
from pathlib import Path

OUT_DIR = Path("/workspace/results/stats/vigor_analysis")

METRICS = ['median_ipi', 'norm_rate', 'relative_vigor', 'frac_full',
           'press_sd', 'press_cv', 'pause_freq', 'n_presses']
METRIC_LABELS = {
    'median_ipi': 'Median IPI (s)',
    'norm_rate': 'Normalized rate',
    'relative_vigor': 'Relative vigor',
    'frac_full': 'Frac at full speed',
    'press_sd': 'Press rate SD',
    'press_cv': 'Press rate CV',
    'pause_freq': 'Pause frequency',
    'n_presses': 'Press count',
}
# Direction: positive = more vigor
METRIC_DIRECTION = {
    'median_ipi': -1,  # lower IPI = more vigor
    'norm_rate': 1, 'relative_vigor': 1, 'frac_full': 1,
    'press_sd': -1,  # lower variability = more consistent
    'press_cv': -1,
    'pause_freq': -1,  # fewer pauses = more vigor
    'n_presses': 1,
}


def run_part1(df=None):
    if df is None:
        df = pd.read_csv(OUT_DIR / "vigor_metrics.csv")

    # Use full-trial metrics
    full = df[df['epoch'] == 'full'].copy()
    full['predator_probability'] = full['T_round']
    print(f"\n{'='*70}")
    print("PART 1: DOES THREAT MODULATE VIGOR?")
    print(f"{'='*70}")
    print(f"Trials: {len(full)}, Subjects: {full['subj'].nunique()}")

    # ═══════════════════════════════════════════════════════════════
    # 1.1: LMM within cookie type
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 1.1: Trial-level LMM (metric ~ threat + distance + trial | subj) ---")
    print(f"\n  {'Metric':<20} {'Cookie':>6} {'β(threat)':>12} {'z':>8} {'p':>12} {'Direction':>10}")
    print(f"  {'-'*70}")

    full['trial_num'] = full.groupby('subj').cumcount() + 1

    for metric in METRICS:
        for cookie, label in [(1, 'Heavy'), (0, 'Light')]:
            sub = full[(full['cookie'] == cookie)].dropna(subset=[metric])
            if len(sub) < 500:
                print(f"  {METRIC_LABELS[metric]:<20} {label:>6} {'insufficient data':>30}")
                continue
            try:
                m = smf.mixedlm(f"{metric} ~ predator_probability + distance + trial_num",
                                sub, groups=sub["subj"]).fit(reml=False)
                coef = m.fe_params['predator_probability']
                z = m.tvalues['predator_probability']
                p = m.pvalues['predator_probability']
                direction = "↑ vigor" if coef * METRIC_DIRECTION[metric] > 0 else "↓ vigor"
                sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
                print(f"  {METRIC_LABELS[metric]:<20} {label:>6} {coef:>12.4f} {z:>8.2f} {p:>12.2e} {direction:>10} {sig}")
            except Exception as e:
                print(f"  {METRIC_LABELS[metric]:<20} {label:>6} FAILED: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 1.2: Within-subject paired comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 1.2: Paired t-test (T=0.9 vs T=0.1, within subject within cookie) ---")
    print(f"\n  {'Metric':<20} {'Cookie':>6} {'M(T=.1)':>10} {'M(T=.9)':>10} {'Δ':>10} {'t':>8} {'p':>10} {'d':>8}")
    print(f"  {'-'*80}")

    for metric in METRICS:
        for cookie, label in [(1, 'Heavy'), (0, 'Light')]:
            sub = full[(full['cookie'] == cookie)].dropna(subset=[metric])
            subj_lo = sub[sub['T_round'] == 0.1].groupby('subj')[metric].mean()
            subj_hi = sub[sub['T_round'] == 0.9].groupby('subj')[metric].mean()
            shared = sorted(set(subj_lo.index) & set(subj_hi.index))
            if len(shared) < 50:
                continue
            lo = subj_lo.loc[shared].values
            hi = subj_hi.loc[shared].values
            t, p = ttest_rel(hi, lo)
            diff = hi - lo
            d = diff.mean() / diff.std()
            print(f"  {METRIC_LABELS[metric]:<20} {label:>6} {lo.mean():>10.4f} {hi.mean():>10.4f} "
                  f"{diff.mean():>+10.4f} {t:>8.2f} {p:>10.2e} {d:>+8.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 1.3: Relative vigor 3×3 surface
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 1.3: Relative vigor 3×3 surface ---")

    for cookie, label in [(1, 'HEAVY'), (0, 'LIGHT')]:
        sub = full[(full['cookie'] == cookie)].dropna(subset=['relative_vigor'])
        # Per-subject condition means, then average
        subj_cond = sub.groupby(['subj', 'T_round', 'distance'])['relative_vigor'].mean().reset_index()
        cond = subj_cond.groupby(['T_round', 'distance'])['relative_vigor'].agg(['mean', 'sem']).reset_index()

        print(f"\n  {label} cookies (relative vigor, mean ± SE):")
        print(f"  {'':>8} {'D=1':>14} {'D=2':>14} {'D=3':>14}")
        for T in [0.1, 0.5, 0.9]:
            row = f"  T={T:.1f}"
            for D in [1, 2, 3]:
                r = cond[(cond['T_round'] == T) & (cond['distance'] == D)]
                if len(r) > 0:
                    row += f"  {r['mean'].values[0]:.3f}({r['sem'].values[0]:.3f})"
                else:
                    row += f"  {'NA':>14}"
            print(row)

    # Also frac_full surface
    print(f"\n  HEAVY cookies (frac_full):")
    sub = full[(full['cookie'] == 1)].dropna(subset=['frac_full'])
    subj_cond = sub.groupby(['subj', 'T_round', 'distance'])['frac_full'].mean().reset_index()
    cond = subj_cond.groupby(['T_round', 'distance'])['frac_full'].agg(['mean', 'sem']).reset_index()
    print(f"  {'':>8} {'D=1':>14} {'D=2':>14} {'D=3':>14}")
    for T in [0.1, 0.5, 0.9]:
        row = f"  T={T:.1f}"
        for D in [1, 2, 3]:
            r = cond[(cond['T_round'] == T) & (cond['distance'] == D)]
            if len(r) > 0:
                row += f"  {r['mean'].values[0]:.3f}({r['sem'].values[0]:.3f})"
            else:
                row += f"  {'NA':>14}"
        print(row)

    # Pause frequency surface
    print(f"\n  HEAVY cookies (pause frequency):")
    sub = full[(full['cookie'] == 1)].dropna(subset=['pause_freq'])
    subj_cond = sub.groupby(['subj', 'T_round', 'distance'])['pause_freq'].mean().reset_index()
    cond = subj_cond.groupby(['T_round', 'distance'])['pause_freq'].agg(['mean', 'sem']).reset_index()
    print(f"  {'':>8} {'D=1':>14} {'D=2':>14} {'D=3':>14}")
    for T in [0.1, 0.5, 0.9]:
        row = f"  T={T:.1f}"
        for D in [1, 2, 3]:
            r = cond[(cond['T_round'] == T) & (cond['distance'] == D)]
            if len(r) > 0:
                row += f"  {r['mean'].values[0]:.3f}({r['sem'].values[0]:.3f})"
            else:
                row += f"  {'NA':>14}"
        print(row)

    print(f"\n  Part 1 complete.")


if __name__ == '__main__':
    run_part1()
