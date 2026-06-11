"""
Test whether raw per-subject behavioral regression coefficients predict mental
health as well as (or better than) the (ω, κ) model parameters.

Per-subject behavioral signatures:
  - Choice: P(choose_high) ~ threat + distance_H + threat:distance_H
    → 3 betas per subject: β_T_choice, β_D_choice, β_TxD_choice
  - Vigor: mean_rate ~ T_round + actual_dist
    → 2 betas per subject: β_T_vigor, β_D_vigor

Then:
  - Test each beta univariate vs AMI_Total and ANX+DEP_FIXED composite
  - Joint regression: clinical ~ all 5 behavioral betas
  - Compare R² to model-parameter approach (clinical ~ log_omega + log_kappa)
  - Run for exploratory (user's request) AND pooled (for comparison)
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
import statsmodels.api as sm
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

BKW = dict(draws=2000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def per_subject_choice_betas(choice_df):
    """For each subject, fit P(choose_high) ~ T + D + T×D and return betas."""
    rows = []
    for subj_idx, g in choice_df.groupby('subj_idx'):
        if len(g) < 10: continue
        X = pd.DataFrame({
            'threat': g['threat'].values,
            'distance': g['distance_H'].values,
            'T_x_D': g['threat'].values * g['distance_H'].values,
        })
        X = sm.add_constant(X)
        y = g['choice'].values
        try:
            mdl = sm.OLS(y, X).fit()
            rows.append({
                'subj_idx': subj_idx,
                'beta_T_choice': mdl.params['threat'],
                'beta_D_choice': mdl.params['distance'],
                'beta_TxD_choice': mdl.params['T_x_D'],
                'intercept_choice': mdl.params['const'],
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def per_subject_vigor_betas(vigor_df):
    """For each subject, fit mean_rate ~ T + D and return betas."""
    rows = []
    for subj_idx, g in vigor_df.groupby('subj_idx'):
        if len(g) < 5: continue
        X = pd.DataFrame({
            'threat': g['T_round'].values,
            'distance': g['actual_dist'].values,
        })
        # Weight by n_trials per cell
        X = sm.add_constant(X)
        y = g['mean_rate'].values
        try:
            mdl = sm.WLS(y, X, weights=g['n_trials'].values).fit()
            rows.append({
                'subj_idx': subj_idx,
                'beta_T_vigor': mdl.params['threat'],
                'beta_D_vigor': mdl.params['distance'],
                'intercept_vigor': mdl.params['const'],
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def build_subject_data(sample_name):
    """Build per-subject behavioral betas + model params + clinical for one sample."""
    if sample_name == 'exploratory':
        base = REPO/'data/model_input_exploratory'
    else:
        # confirmatory might not be in model_input/; fallback to behavior.csv
        base = REPO/'data/model_input_confirmatory'
        if not base.exists():
            return None
    if not (base/'choice_trials.csv').exists() or not (base/'vigor_cell_means.csv').exists():
        return None

    ct = pd.read_csv(base/'choice_trials.csv')
    vcm = pd.read_csv(base/'vigor_cell_means.csv')
    sm_map = pd.read_csv(base/'subject_mapping.csv')

    choice_betas = per_subject_choice_betas(ct)
    vigor_betas = per_subject_vigor_betas(vcm)

    # Merge betas with subject mapping
    betas = sm_map.merge(choice_betas, on='subj_idx', how='inner') \
                  .merge(vigor_betas, on='subj_idx', how='inner')
    betas['sample'] = sample_name
    return betas


def main():
    # === Build behavioral betas for exploratory (user's request) ===
    print('='*72)
    print('PHASE A — Extract per-subject behavioral betas (EXPLORATORY sample)')
    print('='*72)
    betas_exp = build_subject_data('exploratory')
    print(f'  Exploratory: extracted betas for {len(betas_exp)} subjects')

    # Try confirmatory too if data available
    betas_conf = build_subject_data('confirmatory')
    if betas_conf is not None:
        print(f'  Confirmatory: extracted betas for {len(betas_conf)} subjects')
        all_betas = pd.concat([betas_exp, betas_conf], ignore_index=True)
    else:
        print(f'  Confirmatory: no model_input_confirmatory/ found — skipping')
        all_betas = betas_exp

    # === Merge with model params + clinical scales ===
    exp, conf = load_both()
    pm_rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        if 'AMI_Total' not in m.columns:
            m['AMI_Total'] = m[['AMI_Social','AMI_Behavioural','AMI_Emotional']].sum(axis=1)
        pm_rows.append(m)
    params = pd.concat(pm_rows, ignore_index=True)

    # FIXED STAI
    csta = []
    for sample in ['exp','con']:
        p = REPO/f'results/stats/affect_analysis/stai_fixed_{sample}.csv'
        if p.exists():
            csta.append(pd.read_csv(p))
    cstai = pd.concat(csta, ignore_index=True)[['subj','sample','STAI_Trait_FIXED']]
    df_clinical = params.merge(cstai, on=['subj','sample'], how='left')

    df = all_betas.merge(df_clinical, on=['subj','sample'], how='inner')
    print(f'\n  Joined N = {len(df)} subjects (with betas + params + clinical)')

    # === Within-sample z-score everything ===
    beta_cols = ['beta_T_choice','beta_D_choice','beta_TxD_choice',
                 'beta_T_vigor','beta_D_vigor']
    other_cols = ['log_omega','log_kappa','AMI_Total','AMI_Social',
                  'DASS21_Anxiety','DASS21_Depression','DASS21_Stress',
                  'STAI_Trait_FIXED','OASIS_Total','STICSA_Total','PHQ9_Total']
    for c in beta_cols + other_cols:
        if c not in df.columns: continue
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    # Build ANX+DEP composite
    anxdep = ['DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
              'STAI_Trait_FIXED_z','OASIS_Total_z','STICSA_Total_z','PHQ9_Total_z']
    df['ANXDEP_FIXED'] = df[anxdep].mean(axis=1)
    df['ANXDEP_FIXED_z'] = np.nan
    for s in df['sample'].unique():
        mask = df['sample']==s
        x = df.loc[mask,'ANXDEP_FIXED']
        df.loc[mask,'ANXDEP_FIXED_z'] = (x-x.mean())/x.std()

    # Show distribution of behavioral betas (sanity check)
    print('\n--- Behavioral betas summary ---')
    print(df[beta_cols].describe().round(3).to_string())

    # Correlations between behavioral betas and model parameters
    print('\n--- Correlations: behavioral betas × (log_omega, log_kappa) ---')
    for b in beta_cols:
        for p in ['log_omega', 'log_kappa']:
            r = df[[b, p]].dropna().corr().iloc[0, 1]
            print(f'  {b:<20} vs {p:<12} r = {r:+.3f}')

    def fit_t(formula, data):
        m = bmb.Model(formula, data=data, family='t')
        fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
        return az.summary(fit, hdi_prob=0.95), fit

    def surv(r):
        return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)

    def report(s, fit, terms, label=''):
        print(f'  {label}')
        for t in terms:
            if t not in s.index: continue
            r = s.loc[t]
            samples = fit.posterior[t].values.flatten()
            p_dir = (samples > 0).mean() if r['mean'] > 0 else (samples < 0).mean()
            flag = '★' if surv(r) else ' '
            print(f'    {t:<22} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

    # Loop over samples: exploratory only, pooled
    for sample_label, data_sub in [('EXPLORATORY (N=' + str(len(df[df['sample']=='exploratory'])) + ')',
                                    df[df['sample']=='exploratory']),
                                   ('POOLED (N=' + str(len(df)) + ')', df)]:
        print('\n' + '='*72)
        print(f'PHASE B — Behavioral betas → mental health  [{sample_label}]')
        print('='*72)

        # B.1 Univariate: each beta vs AMI_Total
        print('\n--- B.1 Univariate: each behavioral beta → AMI_Total_z ---')
        for b in beta_cols:
            sub = data_sub[[f'AMI_Total_z', f'{b}_z']].dropna()
            if len(sub) < 30: continue
            s, fit = fit_t(f'AMI_Total_z ~ {b}_z', sub)
            r = s.loc[f'{b}_z']
            samples = fit.posterior[f'{b}_z'].values.flatten()
            p_dir = (samples > 0).mean() if r['mean'] > 0 else (samples < 0).mean()
            flag = '★' if surv(r) else ' '
            print(f'  {b:<22} → AMI_Total: β={r["mean"]:+.3f} [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

        # B.2 Univariate: each beta vs ANX+DEP_FIXED
        print('\n--- B.2 Univariate: each behavioral beta → ANX+DEP_FIXED_z ---')
        for b in beta_cols:
            sub = data_sub[[f'ANXDEP_FIXED_z', f'{b}_z']].dropna()
            if len(sub) < 30: continue
            s, fit = fit_t(f'ANXDEP_FIXED_z ~ {b}_z', sub)
            r = s.loc[f'{b}_z']
            samples = fit.posterior[f'{b}_z'].values.flatten()
            p_dir = (samples > 0).mean() if r['mean'] > 0 else (samples < 0).mean()
            flag = '★' if surv(r) else ' '
            print(f'  {b:<22} → ANXDEP: β={r["mean"]:+.3f} [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

        # B.3 Multivariate: AMI_Total ~ all 5 behavioral betas
        beta_z = [f'{b}_z' for b in beta_cols]
        sub = data_sub[[f'AMI_Total_z'] + beta_z].dropna()
        if len(sub) >= 30:
            print(f'\n--- B.3 Multivariate: AMI_Total ~ all 5 behavioral betas (N={len(sub)}) ---')
            s, fit = fit_t(f'AMI_Total_z ~ ' + ' + '.join(beta_z), sub)
            report(s, fit, beta_z)

        # B.4 Multivariate: ANXDEP ~ all 5 behavioral betas
        sub = data_sub[[f'ANXDEP_FIXED_z'] + beta_z].dropna()
        if len(sub) >= 30:
            print(f'\n--- B.4 Multivariate: ANXDEP ~ all 5 behavioral betas (N={len(sub)}) ---')
            s, fit = fit_t(f'ANXDEP_FIXED_z ~ ' + ' + '.join(beta_z), sub)
            report(s, fit, beta_z)

        # ====================================================
        # COMPARISON: behavioral betas vs (omega, kappa) as predictors
        # ====================================================
        print(f'\n{"="*72}')
        print(f'PHASE C — Predictive comparison: behavioral betas vs (ω, κ)  [{sample_label}]')
        print('='*72)

        # R² from each approach (frequentist OLS for clean R² comparison)
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        for target in ['AMI_Total_z', 'ANXDEP_FIXED_z']:
            sub = data_sub[[target, 'log_omega_z', 'log_kappa_z'] + beta_z].dropna()
            if len(sub) < 30: continue
            y = sub[target].values

            print(f'\n  Predicting {target} (N={len(sub)}):')
            for label, predictors in [
                ('(log ω, log κ) only', ['log_omega_z','log_kappa_z']),
                ('5 behavioral betas only', beta_z),
                ('Both combined (7 predictors)', ['log_omega_z','log_kappa_z'] + beta_z),
            ]:
                X = sub[predictors].values
                # In-sample R²
                mdl = LinearRegression().fit(X, y)
                r2_in = mdl.score(X, y)
                # 5-fold CV R²
                cv_r2 = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2').mean()
                print(f'    {label:<32} in-sample R² = {r2_in:.3f}  |  5-fold CV R² = {cv_r2:+.3f}  ({len(predictors)} predictors)')


if __name__ == '__main__':
    main()
