"""
Robust pooled Bayesian regression: clinical state -> log(omega/kappa).

Tier-1 + Tier-2.1 cleanups:
  1.1  z-score predictors WITHIN each sample, then pool
  1.3  Student-t likelihood (down-weights outlier subjects)
  1.4  subject quality filter (min trials, parameter sanity)
  2.1  univariate headline tests (single predictor per model)
       + multivariate sensitivity (kitchen-sink, factor, DASS, totals)

Output: results/stats/affect_analysis/log_ratio_clinical_robust.csv
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, bambi as bmb, arviz as az

warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO)
sys.path.insert(0, str(REPO / 'notebooks' / 'analysis'))
from load_data import load_both

SAMPLES = {
    "exploratory":  "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
    "confirmatory": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
}

BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)

# All predictors of interest
SUBSCALES = ['DASS21_Anxiety', 'DASS21_Depression', 'DASS21_Stress',
             'STAI_Trait', 'OASIS_Total', 'STICSA_Total', 'PHQ9_Total',
             'MFIS_Total', 'AMI_Behavioural', 'AMI_Social', 'AMI_Emotional']
TOTALS = ['DASS21_Total', 'AMI_Total', 'STAI_Trait', 'MFIS_Total', 'OASIS_Total']
FACTORS = ['F1', 'F2']

MULTIVARIATE_MODELS = {
    'A_subscales':  SUBSCALES,
    'B_factors':    FACTORS,
    'C_dass_only':  ['DASS21_Anxiety', 'DASS21_Depression', 'DASS21_Stress'],
    'D_totals':     TOTALS,
}

# All scales we'll test univariately (deduped)
UNIVARIATE = sorted(set(SUBSCALES + TOTALS + FACTORS))


def build_pooled_master():
    """Build pooled per-subject table with WITHIN-sample z-scored predictors."""
    exp, conf = load_both()
    rows = []
    for sample_name, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d['master'].reset_index().rename(columns={'index': 'subj'}).copy()
        m['sample'] = sample_name
        # Drop missing/bad params
        m = m.dropna(subset=['omega', 'kappa'])
        m = m[(m['omega'] > 0) & (m['kappa'] > 0)]
        m['log_ratio'] = np.log(m['omega']) - np.log(m['kappa'])
        m['log_sum']   = np.log(m['omega']) + np.log(m['kappa'])
        rows.append(m)
    master = pd.concat(rows, ignore_index=True)

    # Clinical scales already merged inside build_master() from load_data.py.
    # Only need to merge factor scores from EFA.
    factors = pd.read_csv(REPO / 'results/stats/clinical/factor_scores.csv')[
        ['subj', 'sample', 'F1', 'F2']]
    df = master.merge(factors, on=['subj', 'sample'], how='left')

    # ===== TIER 1.4: subject quality filter =====
    # Drop subjects with extreme log_ratio (|z| > 3 within-sample, computed BEFORE drop)
    keep_mask = pd.Series(True, index=df.index)
    for s in df['sample'].unique():
        ss = df[df['sample'] == s]
        z = (ss['log_ratio'] - ss['log_ratio'].mean()) / ss['log_ratio'].std()
        bad = ss.index[z.abs() > 3]
        keep_mask.loc[bad] = False
    n_drop = (~keep_mask).sum()
    df = df[keep_mask].copy()
    print(f'Quality filter: dropped {n_drop} subjects with |log_ratio_z| > 3')

    # ===== TIER 1.1: WITHIN-sample z-scoring of EVERY variable =====
    z_cols = ['log_ratio', 'log_sum'] + [c for c in UNIVARIATE if c in df.columns]
    for c in z_cols:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            df.loc[mask, f'{c}_z'] = (x - x.mean()) / x.std()

    return df


def fit_t(formula, data):
    """Bambi Student-t (nu=3) likelihood fit."""
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={"log_likelihood": False})
    return az.summary(fit, hdi_prob=0.95)


def survives(row):
    return (row['hdi_2.5%'] > 0) or (row['hdi_97.5%'] < 0)


def main():
    df = build_pooled_master()
    print(f'N pooled (after filter): {len(df)}')
    print(f'  exp={(df["sample"]=="exploratory").sum()}  conf={(df["sample"]=="confirmatory").sum()}')

    results = []

    # ===== UNIVARIATE (headline) =====
    print('\n========== UNIVARIATE Student-t models (headline) ==========')
    print(f'{"predictor":<26} {"N":>5} {"β":>8} {"SD":>7} {"95% HDI":>22}  flag')
    for pred in UNIVARIATE:
        col = f'{pred}_z'
        if col not in df.columns:
            continue
        sub = df[['log_ratio_z', col]].dropna()
        if len(sub) < 50:
            continue
        s = fit_t(f'log_ratio_z ~ {col}', sub)
        row = s.loc[col]
        sv = survives(row)
        flag = '★' if sv else ' '
        print(f'{pred:<26} {len(sub):>5} {row["mean"]:+.3f}  {row["sd"]:.3f}  '
              f'[{row["hdi_2.5%"]:+.3f}, {row["hdi_97.5%"]:+.3f}]  {flag}')
        results.append({
            'model': 'univariate', 'predictor': pred, 'N': len(sub),
            'mean': float(row['mean']), 'sd': float(row['sd']),
            'hdi_lo': float(row['hdi_2.5%']), 'hdi_hi': float(row['hdi_97.5%']),
            'survives': bool(sv),
        })

    # ===== MULTIVARIATE (sensitivity) =====
    print('\n========== MULTIVARIATE Student-t models (sensitivity) ==========')
    for model_name, preds in MULTIVARIATE_MODELS.items():
        cols_z = [f'{p}_z' for p in preds if f'{p}_z' in df.columns]
        use_cols = ['log_ratio_z'] + cols_z
        sub = df[use_cols].dropna()
        formula = 'log_ratio_z ~ ' + ' + '.join(cols_z)
        print(f'\n--- {model_name} (Student-t, N={len(sub)}) ---')
        s = fit_t(formula, sub)
        for p, col in zip(preds, cols_z):
            if col not in s.index:
                continue
            row = s.loc[col]
            sv = survives(row)
            flag = '★' if sv else ' '
            print(f'  {p:<24} {row["mean"]:+.3f}  {row["sd"]:.3f}  '
                  f'[{row["hdi_2.5%"]:+.3f}, {row["hdi_97.5%"]:+.3f}]  {flag}')
            results.append({
                'model': model_name, 'predictor': p, 'N': len(sub),
                'mean': float(row['mean']), 'sd': float(row['sd']),
                'hdi_lo': float(row['hdi_2.5%']), 'hdi_hi': float(row['hdi_97.5%']),
                'survives': bool(sv),
            })

    # Save
    out = REPO / 'results' / 'stats' / 'affect_analysis' / 'log_ratio_clinical_robust.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # Summary
    surv = [r for r in results if r['survives']]
    print(f'\n=== Surviving effects: {len(surv)} ===')
    for r in surv:
        print(f"  [{r['model']}] {r['predictor']}: β={r['mean']:+.3f} "
              f"HDI [{r['hdi_lo']:+.3f}, {r['hdi_hi']:+.3f}]")


if __name__ == '__main__':
    main()
