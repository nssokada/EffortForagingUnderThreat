"""
Agent 1: Metacognitive sensitivity bridge (omega, kappa) -> clinical.

For each subject, compute Pearson correlation between probe rating
(anxiety / confidence) and the binary trial outcome (escaped=1 / captured=0).
Then test:
  A. anx_sensitivity ~ omega_z + kappa_z
  B. conf_sensitivity ~ omega_z + kappa_z
  C. clinical_z ~ anx_sens_z + omega_z + kappa_z
  D. clinical_z ~ conf_sens_z + omega_z + kappa_z
Cross-sample replication required.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, zscore
import statsmodels.api as sm

sys.path.insert(0, '/workspace/notebooks/analysis')
from load_data import load_both  # noqa: E402

OUT_DIR = Path('/workspace/results/stats/avoid_activate')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL = [
    'DASS21_Anxiety', 'DASS21_Depression', 'AMI_Total', 'AMI_Social',
    'PHQ9_Total', 'STAI_Trait', 'OASIS_Total',
]

MIN_PROBES = 5


def compute_sensitivity(data, restrict_to_attack=False):
    """Return per-subject df with anx_sensitivity, conf_sensitivity, n counts."""
    feel = data['feelings'].copy()
    beh = data['beh'].copy()

    # Binary outcome column from beh
    if 'trialEndState' in beh.columns:
        beh['escaped'] = (beh['trialEndState'] == 'escaped').astype(float)
        beh.loc[~beh['trialEndState'].isin(['escaped', 'captured']), 'escaped'] = np.nan
    else:
        beh['escaped'] = (beh['trialReward'] > 0).astype(float)

    # Merge outcome into feelings on (subj, trialNumber <-> trial)
    # Try a few candidate keys
    trial_key_feel = None
    for k in ['trialNumber', 'trial', 'trial_idx']:
        if k in feel.columns:
            trial_key_feel = k
            break
    trial_key_beh = None
    for k in ['trial', 'trialNumber', 'trial_idx']:
        if k in beh.columns:
            trial_key_beh = k
            break
    if trial_key_feel is None or trial_key_beh is None:
        raise RuntimeError(f'trial key not found; feel cols={feel.columns.tolist()}')

    beh_sub = beh[['subj', trial_key_beh, 'escaped', 'isAttackTrial']].rename(
        columns={trial_key_beh: 'trial_key'}
    )
    feel = feel.rename(columns={trial_key_feel: 'trial_key'})
    merged = feel.merge(beh_sub, on=['subj', 'trial_key'], how='left')

    if restrict_to_attack:
        merged = merged[merged['isAttackTrial'] == 1]

    merged = merged.dropna(subset=['escaped', 'response'])

    rows = []
    for subj, g in merged.groupby('subj'):
        anx = g[g['questionLabel'] == 'anxiety']
        conf = g[g['questionLabel'] == 'confidence']
        row = {'subj': subj, 'n_anx': len(anx), 'n_conf': len(conf)}
        # anxiety: higher rating should predict capture (escaped=0)
        if len(anx) >= MIN_PROBES and anx['response'].std() > 0 and anx['escaped'].std() > 0:
            # We want higher anxiety -> lower escape. Take r(anxiety, escaped)
            # Positive r means higher anxiety predicts escape (bad); flip sign so
            # "anx_sensitivity" = higher means anxiety predicts capture (good).
            r_raw, _ = pearsonr(anx['response'], anx['escaped'])
            row['anx_sensitivity'] = -r_raw
        else:
            row['anx_sensitivity'] = np.nan
        if len(conf) >= MIN_PROBES and conf['response'].std() > 0 and conf['escaped'].std() > 0:
            r_raw, _ = pearsonr(conf['response'], conf['escaped'])
            row['conf_sensitivity'] = r_raw  # higher confidence -> escape
        else:
            row['conf_sensitivity'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('subj')


def ols(y, X, label):
    """Fit OLS and return tidy dict of coefficients."""
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 20:
        return None
    yv = df.iloc[:, 0].values
    Xv = sm.add_constant(df.iloc[:, 1:].values)
    try:
        m = sm.OLS(yv, Xv).fit()
    except Exception as e:
        return {'label': label, 'error': str(e)}
    cols = ['const'] + list(X.columns)
    out = {'label': label, 'n': len(df), 'r2': float(m.rsquared)}
    for i, c in enumerate(cols):
        out[f'{c}_beta'] = float(m.params[i])
        out[f'{c}_se'] = float(m.bse[i])
        out[f'{c}_t'] = float(m.tvalues[i])
        out[f'{c}_p'] = float(m.pvalues[i])
    return out


def run_sample(name, data, restrict_to_attack):
    print(f'\n=== {name} (attack_only={restrict_to_attack}) ===')
    sens = compute_sensitivity(data, restrict_to_attack=restrict_to_attack)
    master = data['master']
    df = master[['omega_z', 'kappa_z']].join(sens, how='left')
    # Join clinical
    for c in CLINICAL:
        if c in master.columns:
            df[c] = master[c]

    # z-score sensitivities
    for c in ['anx_sensitivity', 'conf_sensitivity']:
        vals = df[c].values.astype(float)
        m = ~np.isnan(vals)
        z = np.full_like(vals, np.nan, dtype=float)
        if m.sum() > 1:
            z[m] = (vals[m] - np.nanmean(vals[m])) / np.nanstd(vals[m], ddof=1)
        df[c + '_z'] = z

    # Distribution summary
    dist = {}
    for c in ['anx_sensitivity', 'conf_sensitivity']:
        v = df[c].dropna()
        dist[c] = {
            'n': int(v.shape[0]),
            'median': float(v.median()) if len(v) else np.nan,
            'mean': float(v.mean()) if len(v) else np.nan,
            'std': float(v.std()) if len(v) else np.nan,
            'min': float(v.min()) if len(v) else np.nan,
            'max': float(v.max()) if len(v) else np.nan,
        }
    print('distribution:', json.dumps(dist, indent=2))

    results = []

    # Tests A, B
    for target in ['anx_sensitivity', 'conf_sensitivity']:
        y = df[target]
        X = df[['omega_z', 'kappa_z']]
        r = ols(y, X, f'{target} ~ omega_z + kappa_z')
        if r:
            r['sample'] = name
            r['test'] = 'A' if target == 'anx_sensitivity' else 'B'
            results.append(r)

    # Tests C, D
    for sens_col in ['anx_sensitivity_z', 'conf_sensitivity_z']:
        for clin in CLINICAL:
            if clin not in df.columns:
                continue
            cvals = df[clin].values.astype(float)
            if np.isnan(cvals).all():
                continue
            m = ~np.isnan(cvals)
            cz = np.full_like(cvals, np.nan, dtype=float)
            cz[m] = (cvals[m] - np.nanmean(cvals[m])) / np.nanstd(cvals[m], ddof=1)
            y = pd.Series(cz, index=df.index, name=f'{clin}_z')
            X = df[[sens_col, 'omega_z', 'kappa_z']]
            r = ols(y, X, f'{clin}_z ~ {sens_col} + omega_z + kappa_z')
            if r:
                r['sample'] = name
                r['test'] = 'C' if 'anx' in sens_col else 'D'
                r['clinical'] = clin
                r['sens_col'] = sens_col
                results.append(r)

    return df, sens, results, dist


def main():
    exp, conf = load_both()

    # Use all trials by default (more data); we can also inspect attack-only
    results_all = []
    dists = {}
    for name, data in [('exp', exp), ('conf', conf)]:
        df, sens, res, dist = run_sample(name, data, restrict_to_attack=False)
        sens_out = sens.copy()
        sens_out.to_csv(OUT_DIR / f'agent1_subject_sensitivity_{name}.csv')
        results_all.extend(res)
        dists[name] = dist

    res_df = pd.DataFrame(results_all)
    res_df.to_csv(OUT_DIR / 'agent1_metacog_sensitivity.csv', index=False)

    # Also save distribution summary
    with open(OUT_DIR / 'agent1_distribution_summary.json', 'w') as f:
        json.dump(dists, f, indent=2)

    # Cross-sample replication check for A/B
    print('\n=== CROSS-SAMPLE (A/B: sensitivity ~ omega+kappa) ===')
    for test in ['A', 'B']:
        sub = res_df[res_df['test'] == test]
        if len(sub) == 0:
            continue
        exp_row = sub[sub['sample'] == 'exp'].iloc[0] if (sub['sample'] == 'exp').any() else None
        conf_row = sub[sub['sample'] == 'conf'].iloc[0] if (sub['sample'] == 'conf').any() else None
        print(f'Test {test}: {sub.iloc[0]["label"]}')
        for param in ['omega_z', 'kappa_z']:
            if exp_row is not None and conf_row is not None:
                be = exp_row[f'{param}_beta']
                pe = exp_row[f'{param}_p']
                bc = conf_row[f'{param}_beta']
                pc = conf_row[f'{param}_p']
                rep = 'REPLICATES' if (np.sign(be) == np.sign(bc) and pe < 0.05 and pc < 0.05) else ''
                print(f'  {param}: exp b={be:+.3f} p={pe:.3f} | conf b={bc:+.3f} p={pc:.3f}  {rep}')

    # Cross-sample replication check for C/D
    print('\n=== CROSS-SAMPLE (C/D: clinical ~ sens + omega + kappa) ===')
    for test in ['C', 'D']:
        sub = res_df[res_df['test'] == test]
        if len(sub) == 0:
            continue
        for clin in CLINICAL:
            rows = sub[sub['clinical'] == clin]
            if len(rows) < 2:
                continue
            er = rows[rows['sample'] == 'exp']
            cr = rows[rows['sample'] == 'conf']
            if len(er) == 0 or len(cr) == 0:
                continue
            er = er.iloc[0]
            cr = cr.iloc[0]
            sens_col = er['sens_col']
            be = er[f'{sens_col}_beta']
            pe = er[f'{sens_col}_p']
            bc = cr[f'{sens_col}_beta']
            pc = cr[f'{sens_col}_p']
            rep = 'REPLICATES' if (np.sign(be) == np.sign(bc) and pe < 0.05 and pc < 0.05) else ''
            mark = '*' if (pe < 0.05 or pc < 0.05) else ' '
            print(f'  [{test}] {clin:20s}: exp b={be:+.3f} p={pe:.3f} | conf b={bc:+.3f} p={pc:.3f} {mark} {rep}')


if __name__ == '__main__':
    main()
