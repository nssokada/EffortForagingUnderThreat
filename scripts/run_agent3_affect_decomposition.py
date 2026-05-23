"""
Agent3: State-Trait-Reactivity decomposition of probe affect.

Decomposes each subject's anxiety & confidence probes into:
  trait (mean), reactivity (T and D slopes), state variance (residual SD).
Then tests against (omega_z, kappa_z) and clinical scales, with cross-sample replication.
"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/notebooks/analysis')

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import pearsonr, zscore

from load_data import load_both

OUT_DIR = Path('/workspace/results/stats/avoid_activate')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL = ['DASS21_Anxiety', 'DASS21_Depression', 'AMI_Total', 'AMI_Social',
            'PHQ9_Total', 'STAI_Trait', 'OASIS_Total']

FEATURES = ['anx_trait', 'anx_react_T', 'anx_react_D', 'anx_state_sd',
            'conf_trait', 'conf_react_T', 'conf_react_D', 'conf_state_sd']


def decompose_subject(df):
    """Fit response ~ threat + distance + rewardValue. Return trait, T-slope, D-slope, resid SD."""
    if len(df) < 8:
        return None
    y = df['response'].astype(float).values
    X = df[['threat', 'distance', 'trialCookie_rewardValue']].astype(float).values
    # Check variability
    if np.std(y) == 0 or np.std(X[:, 0]) == 0 or np.std(X[:, 1]) == 0:
        return None
    Xc = sm.add_constant(X, has_constant='add')
    try:
        res = sm.OLS(y, Xc).fit()
    except Exception:
        return None
    trait = float(np.mean(y))
    t_slope = float(res.params[1])
    d_slope = float(res.params[2])
    resid_sd = float(np.std(res.resid, ddof=1))
    return trait, t_slope, d_slope, resid_sd


def build_features(feel):
    rows = []
    for subj, g in feel.groupby('subj'):
        sa = g[g['questionLabel'] == 'anxiety']
        sc = g[g['questionLabel'] == 'confidence']
        row = {'subj': subj}
        a = decompose_subject(sa)
        if a is not None:
            row['anx_trait'], row['anx_react_T'], row['anx_react_D'], row['anx_state_sd'] = a
        c = decompose_subject(sc)
        if c is not None:
            row['conf_trait'], row['conf_react_T'], row['conf_react_D'], row['conf_state_sd'] = c
        rows.append(row)
    return pd.DataFrame(rows).set_index('subj')


def test_A(master, feats):
    """feature ~ omega_z + kappa_z (OLS)."""
    results = []
    df = master[['omega_z', 'kappa_z']].join(feats, how='inner')
    for f in FEATURES:
        sub = df[['omega_z', 'kappa_z', f]].dropna()
        if len(sub) < 20:
            continue
        X = sm.add_constant(sub[['omega_z', 'kappa_z']].values)
        y = zscore(sub[f].values)
        res = sm.OLS(y, X).fit()
        results.append({
            'feature': f, 'n': len(sub),
            'beta_omega': res.params[1], 'p_omega': res.pvalues[1],
            'beta_kappa': res.params[2], 'p_kappa': res.pvalues[2],
            'r2': res.rsquared,
        })
    return pd.DataFrame(results)


def test_B(master, feats):
    """clinical ~ feature + omega_z + kappa_z. Report feature partial beta."""
    results = []
    df = master[['omega_z', 'kappa_z'] + CLINICAL].join(feats, how='inner')
    for f in FEATURES:
        for c in CLINICAL:
            sub = df[['omega_z', 'kappa_z', f, c]].dropna()
            if len(sub) < 20:
                continue
            X = sm.add_constant(np.column_stack([
                zscore(sub[f].values),
                sub['omega_z'].values,
                sub['kappa_z'].values
            ]))
            y = zscore(sub[c].values)
            res = sm.OLS(y, X).fit()
            # Also a baseline model (without feature) for delta r2
            X0 = sm.add_constant(sub[['omega_z', 'kappa_z']].values)
            res0 = sm.OLS(y, X0).fit()
            results.append({
                'feature': f, 'clinical': c, 'n': len(sub),
                'beta_feat': res.params[1], 'p_feat': res.pvalues[1],
                'r2_full': res.rsquared, 'r2_base': res0.rsquared,
                'delta_r2': res.rsquared - res0.rsquared,
            })
    return pd.DataFrame(results)


def main():
    exp_data, conf_data = load_both()

    feats_exp = build_features(exp_data['feelings'])
    feats_conf = build_features(conf_data['feelings'])

    feats_exp.to_csv(OUT_DIR / 'agent3_subject_features_exp.csv')
    feats_conf.to_csv(OUT_DIR / 'agent3_subject_features_conf.csv')

    print('\n=== FEATURE DISTRIBUTIONS (exploratory) ===')
    print(feats_exp[FEATURES].describe().round(3).T[['mean', 'std', 'min', 'max', 'count']])
    print('\n=== FEATURE DISTRIBUTIONS (confirmatory) ===')
    print(feats_conf[FEATURES].describe().round(3).T[['mean', 'std', 'min', 'max', 'count']])

    # Test A
    A_exp = test_A(exp_data['master'], feats_exp); A_exp['sample'] = 'exp'
    A_conf = test_A(conf_data['master'], feats_conf); A_conf['sample'] = 'conf'
    A = pd.concat([A_exp, A_conf], ignore_index=True)
    A['test'] = 'A_feature_vs_params'

    print('\n=== TEST A: feature ~ omega_z + kappa_z ===')
    print('EXP:'); print(A_exp[['feature', 'beta_omega', 'p_omega', 'beta_kappa', 'p_kappa', 'r2']].round(4).to_string(index=False))
    print('CONF:'); print(A_conf[['feature', 'beta_omega', 'p_omega', 'beta_kappa', 'p_kappa', 'r2']].round(4).to_string(index=False))

    # Test A replication: same sign + both p<.05
    print('\n=== TEST A REPLICATION ===')
    rep_rows = []
    for f in FEATURES:
        e = A_exp[A_exp.feature == f]
        c = A_conf[A_conf.feature == f]
        if e.empty or c.empty:
            continue
        e = e.iloc[0]; c = c.iloc[0]
        for par in ['omega', 'kappa']:
            be, pe = e[f'beta_{par}'], e[f'p_{par}']
            bc, pc = c[f'beta_{par}'], c[f'p_{par}']
            same_sign = np.sign(be) == np.sign(bc)
            both_sig = (pe < 0.05) and (pc < 0.05)
            if both_sig and same_sign:
                rep_rows.append({'feature': f, 'param': par,
                                 'beta_exp': be, 'p_exp': pe,
                                 'beta_conf': bc, 'p_conf': pc})
    rep_A = pd.DataFrame(rep_rows)
    if rep_A.empty:
        print('  No replicated feature~param associations at p<.05 both samples.')
    else:
        print(rep_A.round(4).to_string(index=False))

    # Test B
    B_exp = test_B(exp_data['master'], feats_exp); B_exp['sample'] = 'exp'
    B_conf = test_B(conf_data['master'], feats_conf); B_conf['sample'] = 'conf'
    B = pd.concat([B_exp, B_conf], ignore_index=True)
    B['test'] = 'B_clinical_vs_feature_controlling_params'

    # Report significant rows
    print('\n=== TEST B: clinical ~ feature + omega_z + kappa_z (feature significant at p<.05) ===')
    print('EXP significant:')
    sigE = B_exp[B_exp.p_feat < 0.05][['feature', 'clinical', 'beta_feat', 'p_feat', 'delta_r2']].round(4)
    print(sigE.to_string(index=False) if not sigE.empty else '  (none)')
    print('CONF significant:')
    sigC = B_conf[B_conf.p_feat < 0.05][['feature', 'clinical', 'beta_feat', 'p_feat', 'delta_r2']].round(4)
    print(sigC.to_string(index=False) if not sigC.empty else '  (none)')

    # Replication
    print('\n=== TEST B REPLICATION (same feature, same clinical, same sign, both p<.05) ===')
    rep_B_rows = []
    for _, er in B_exp.iterrows():
        m = B_conf[(B_conf.feature == er.feature) & (B_conf.clinical == er.clinical)]
        if m.empty:
            continue
        cr = m.iloc[0]
        if er.p_feat < 0.05 and cr.p_feat < 0.05 and np.sign(er.beta_feat) == np.sign(cr.beta_feat):
            rep_B_rows.append({
                'feature': er.feature, 'clinical': er.clinical,
                'beta_exp': er.beta_feat, 'p_exp': er.p_feat, 'dR2_exp': er.delta_r2,
                'beta_conf': cr.beta_feat, 'p_conf': cr.p_feat, 'dR2_conf': cr.delta_r2,
            })
    rep_B = pd.DataFrame(rep_B_rows)
    if rep_B.empty:
        print('  No replicated feature->clinical effects controlling for (omega, kappa).')
    else:
        print(rep_B.round(4).to_string(index=False))

    # Save combined results
    combined = pd.concat([A, B], ignore_index=True, sort=False)
    combined.to_csv(OUT_DIR / 'agent3_affect_decomposition.csv', index=False)
    print(f'\nSaved: {OUT_DIR}/agent3_affect_decomposition.csv')
    print(f'Saved: {OUT_DIR}/agent3_subject_features_exp.csv')
    print(f'Saved: {OUT_DIR}/agent3_subject_features_conf.csv')


if __name__ == '__main__':
    main()
