"""
Test AMI_Social and DASS_Stress against 4 different operationalizations of
the vigilance-mobilization balance, on ALL 571 subjects (no outlier drop).

Metrics:
  M1. log(omega/kappa)             — current, unbounded
  M2. z(log omega) - z(log kappa)  — within-sample standardized difference
  M3. omega / (omega + kappa)      — bounded compositional proportion
  M4. arctan(log kappa / log omega) in degrees — angle in log-parameter plane

For each metric:
  - Univariate Student-t Bayesian: metric_z ~ AMI_Social_z
  - Univariate Student-t Bayesian: metric_z ~ DASS_Stress_z
  - Multivariate DASS-only Student-t: metric_z ~ DASS_Anx_z + DASS_Dep_z + DASS_Stress_z
  - Spearman rho with AMI_Social and DASS_Stress (sanity check)

All 571 subjects, within-sample z, no outlier filter.

Output: results/stats/affect_analysis/balance_metric_comparison.csv
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO)
sys.path.insert(0, str(REPO / 'notebooks' / 'analysis'))
from load_data import load_both

BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def build_master():
    exp, conf = load_both()
    rows = []
    for sn, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d['master'].reset_index().rename(columns={'index': 'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega', 'kappa'])
        m = m[(m['omega'] > 0) & (m['kappa'] > 0)]
        m['log_omega'] = np.log(m['omega'])
        m['log_kappa'] = np.log(m['kappa'])
        # M1: log ratio
        m['M1_log_ratio'] = m['log_omega'] - m['log_kappa']
        # M3: bounded proportion
        m['M3_proportion'] = m['omega'] / (m['omega'] + m['kappa'])
        # M4: angle in degrees
        m['M4_angle'] = np.arctan2(m['log_kappa'], m['log_omega']) * 180 / np.pi
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)

    # M2 requires within-sample z of log_omega and log_kappa separately
    df['log_omega_zw'] = np.nan
    df['log_kappa_zw'] = np.nan
    for s in df['sample'].unique():
        mask = df['sample'] == s
        for col in ['log_omega', 'log_kappa']:
            x = df.loc[mask, col]
            df.loc[mask, f'{col}_zw'] = (x - x.mean()) / x.std()
    df['M2_zdiff'] = df['log_omega_zw'] - df['log_kappa_zw']

    # Within-sample z-score every metric AND every predictor
    metrics = ['M1_log_ratio', 'M2_zdiff', 'M3_proportion', 'M4_angle']
    preds = ['AMI_Social', 'DASS21_Anxiety', 'DASS21_Depression', 'DASS21_Stress']
    for c in metrics + preds:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            df.loc[mask, f'{c}_z'] = (x - x.mean()) / x.std()

    return df, metrics


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={"log_likelihood": False})
    return az.summary(fit, hdi_prob=0.95)


def survives(row):
    return (row['hdi_2.5%'] > 0) or (row['hdi_97.5%'] < 0)


def main():
    df, metrics = build_master()
    print(f'N = {len(df)}  (no outlier filter)')
    print(f'  exp={(df["sample"]=="exploratory").sum()}  conf={(df["sample"]=="confirmatory").sum()}')

    # Print raw metric correlations to confirm they capture related but distinct info
    print('\n--- Metric inter-correlations (Spearman) ---')
    print(df[metrics].corr(method='spearman').round(3).to_string())

    rows = []

    for metric in metrics:
        outcome = f'{metric}_z'
        print(f'\n{"="*72}\nOutcome: {metric}\n{"="*72}')

        # ===== Spearman sanity checks =====
        sub = df[[metric, 'AMI_Social', 'DASS21_Stress']].dropna()
        rho_ami, p_ami = spearmanr(sub['AMI_Social'], sub[metric])
        rho_ds, p_ds = spearmanr(sub['DASS21_Stress'], sub[metric])
        print(f'  Spearman AMI_Social: rho={rho_ami:+.3f}  p={p_ami:.4f}')
        print(f'  Spearman DASS_Stress: rho={rho_ds:+.3f}  p={p_ds:.4f}')

        # ===== Univariate AMI_Social =====
        sub = df[[outcome, 'AMI_Social_z']].dropna()
        s_summary = fit_t(f'{outcome} ~ AMI_Social_z', sub)
        r = s_summary.loc['AMI_Social_z']
        sv = survives(r)
        print(f'  Univariate AMI_Social_z:  β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {"★" if sv else ""}')
        rows.append({'metric': metric, 'test': 'univariate', 'predictor': 'AMI_Social',
                     'N': len(sub), 'mean': float(r['mean']), 'sd': float(r['sd']),
                     'hdi_lo': float(r['hdi_2.5%']), 'hdi_hi': float(r['hdi_97.5%']),
                     'survives': bool(sv), 'spearman': float(rho_ami), 'spearman_p': float(p_ami)})

        # ===== Univariate DASS_Stress =====
        sub = df[[outcome, 'DASS21_Stress_z']].dropna()
        s_summary = fit_t(f'{outcome} ~ DASS21_Stress_z', sub)
        r = s_summary.loc['DASS21_Stress_z']
        sv = survives(r)
        print(f'  Univariate DASS_Stress_z: β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {"★" if sv else ""}')
        rows.append({'metric': metric, 'test': 'univariate', 'predictor': 'DASS21_Stress',
                     'N': len(sub), 'mean': float(r['mean']), 'sd': float(r['sd']),
                     'hdi_lo': float(r['hdi_2.5%']), 'hdi_hi': float(r['hdi_97.5%']),
                     'survives': bool(sv), 'spearman': float(rho_ds), 'spearman_p': float(p_ds)})

        # ===== Multivariate DASS-only (suppression check) =====
        sub = df[[outcome, 'DASS21_Anxiety_z', 'DASS21_Depression_z', 'DASS21_Stress_z']].dropna()
        s_summary = fit_t(f'{outcome} ~ DASS21_Anxiety_z + DASS21_Depression_z + DASS21_Stress_z', sub)
        print('  Multivariate DASS:')
        for term in ['DASS21_Anxiety_z', 'DASS21_Depression_z', 'DASS21_Stress_z']:
            r = s_summary.loc[term]
            sv = survives(r)
            print(f'    {term:<24} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {"★" if sv else ""}')
            rows.append({'metric': metric, 'test': 'multivariate_DASS',
                         'predictor': term.replace('_z', ''),
                         'N': len(sub), 'mean': float(r['mean']), 'sd': float(r['sd']),
                         'hdi_lo': float(r['hdi_2.5%']), 'hdi_hi': float(r['hdi_97.5%']),
                         'survives': bool(sv), 'spearman': np.nan, 'spearman_p': np.nan})

    out = REPO / 'results/stats/affect_analysis/balance_metric_comparison.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # Summary
    print('\n' + '='*72)
    print('SUMMARY — surviving effects across 4 metrics')
    print('='*72)
    surv = [r for r in rows if r['survives']]
    if surv:
        for r in surv:
            print(f"  [{r['metric']:<14} | {r['test']:<18}] {r['predictor']:<18} "
                  f"β={r['mean']:+.3f}  HDI [{r['hdi_lo']:+.3f}, {r['hdi_hi']:+.3f}]")
    else:
        print('  None.')


if __name__ == '__main__':
    main()
