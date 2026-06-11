"""
Test each EFA factor (from 3-, 4-, 5-factor solutions) against THREE outcomes:
  - log(ω)
  - log(κ)
  - log(ω/κ)  ← balance — could amplify or dampen
Plus log_omega - log_kappa direct comparison.

For each factor in each solution, show all three coefficients side by side,
so we can see whether the balance metric reveals signals invisible to the
single-parameter regressions.
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
from sklearn.decomposition import FactorAnalysis
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
sys.path.insert(0, str(REPO/'scripts/analysis'))
from item_efa_reduced_factors import load_and_prep

BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95)


def survives(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def main():
    print('Loading data...')
    X_df, item_cols = load_and_prep()
    X_df = X_df.copy()
    X_df['log_ratio'] = X_df['log_omega'] - X_df['log_kappa']

    # Within-sample z all parameter outcomes
    for c in ['log_omega', 'log_kappa', 'log_ratio']:
        X_df[f'{c}_z'] = np.nan
        for s in X_df['sample'].unique():
            mask = X_df['sample'] == s
            x = X_df.loc[mask, c]
            X_df.loc[mask, f'{c}_z'] = (x - x.mean()) / x.std()

    print(f'N = {len(X_df)}')

    all_rows = []
    for n_factors in [3, 4, 5]:
        print('\n' + '='*72)
        print(f'  {n_factors}-FACTOR SOLUTION')
        print('='*72)

        fa = FactorAnalysis(n_components=n_factors, rotation='varimax', random_state=42)
        scores = fa.fit_transform(X_df[item_cols].values)
        factor_names = [f'F{i+1}' for i in range(n_factors)]

        df = X_df.copy()
        for i, fn in enumerate(factor_names):
            df[fn] = scores[:, i]
            # within-sample z
            df[f'{fn}_z'] = np.nan
            for s in df['sample'].unique():
                mask = df['sample'] == s
                x = df.loc[mask, fn]
                if x.std() > 0:
                    df.loc[mask, f'{fn}_z'] = (x - x.mean())/x.std()

        # Show top-loading questionnaire per factor for context
        loadings = pd.DataFrame(fa.components_.T, index=item_cols, columns=factor_names)
        for fn in factor_names:
            s = loadings[fn].sort_values(key=abs, ascending=False).head(5)
            prefixes = [it.split('_item_')[0] for it in s.index]
            dom = pd.Series(prefixes).value_counts().idxmax()
            print(f'  {fn} (top-5 dominated by {dom}): '
                  + ', '.join([f'{it.split("_item_")[0]}_{it.split("_")[-1]}={ld:+.2f}' for it, ld in s.head(3).items()]))

        # Run regressions
        print(f'\n  {"factor":<6} {"β(log_ω)":>16} {"β(log_κ)":>16} {"β(log_ω/κ)":>16}')
        for fn in factor_names:
            sub = df[[f'log_omega_z', f'log_kappa_z', f'log_ratio_z', f'{fn}_z']].dropna()
            row_data = {'n_factors': n_factors, 'factor': fn, 'N': len(sub)}
            for outcome, label in [('log_omega', 'log_ω'),
                                    ('log_kappa', 'log_κ'),
                                    ('log_ratio', 'log_ω/κ')]:
                s = fit_t(f'{outcome}_z ~ {fn}_z', sub)
                r = s.loc[f'{fn}_z']
                sv = survives(r)
                row_data[f'beta_{outcome}'] = float(r['mean'])
                row_data[f'hdi_lo_{outcome}'] = float(r['hdi_2.5%'])
                row_data[f'hdi_hi_{outcome}'] = float(r['hdi_97.5%'])
                row_data[f'survives_{outcome}'] = bool(sv)
            # Print row
            ro_beta = row_data['beta_log_omega']; ro_lo=row_data['hdi_lo_log_omega']; ro_hi=row_data['hdi_hi_log_omega']
            rk_beta = row_data['beta_log_kappa']; rk_lo=row_data['hdi_lo_log_kappa']; rk_hi=row_data['hdi_hi_log_kappa']
            rb_beta = row_data['beta_log_ratio']; rb_lo=row_data['hdi_lo_log_ratio']; rb_hi=row_data['hdi_hi_log_ratio']
            fo = '★' if row_data['survives_log_omega'] else ' '
            fk = '★' if row_data['survives_log_kappa'] else ' '
            fb = '★' if row_data['survives_log_ratio'] else ' '
            print(f'  {fn:<6} {ro_beta:+8.3f} [{ro_lo:+.2f},{ro_hi:+.2f}]{fo}  '
                  f'{rk_beta:+8.3f} [{rk_lo:+.2f},{rk_hi:+.2f}]{fk}  '
                  f'{rb_beta:+8.3f} [{rb_lo:+.2f},{rb_hi:+.2f}]{fb}')
            all_rows.append(row_data)

    out = REPO / 'results/stats/affect_analysis/item_efa_vs_balance.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)

    print('\n' + '='*72)
    print('SUMMARY — surviving balance effects (log ω/κ)')
    print('='*72)
    surv_balance = [r for r in all_rows if r['survives_log_ratio']]
    if not surv_balance:
        print('  None.')
    for r in surv_balance:
        print(f"  n={r['n_factors']}, {r['factor']}: β(log ω/κ) = {r['beta_log_ratio']:+.3f} "
              f"[{r['hdi_lo_log_ratio']:+.3f}, {r['hdi_hi_log_ratio']:+.3f}]")

    print('\n  For reference, baseline AMI_Social → log(ω/κ): β=+0.084 to +0.111 across metrics (§4.65)')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
