"""
Test the (ω, κ) profile typology question:
  Type A: high ω + low κ  ("vigilant + mobilized" — avoids AND activates)
  Type B: low ω + high κ  ("disengaged + immobilized" — passive/freeze)
  vs
  Type C: high ω + high κ  ("vigilant + immobilized" — frozen vigilance)
  Type D: low ω + low κ    ("disengaged + mobilized" — chaotic/exploratory)

Note on κ direction: κ is effort COST weighting. Higher κ = penalize motor
effort more = LESS mobilized. Lower κ = more mobilized.

Approach 1: 2×2 median-split quadrants. ANOVA + pairwise contrasts on clinical.
Approach 2: Polar decomposition — angle in (log ω, log κ) plane (= the
            "balance" axis) and radius (= overall engagement magnitude).
Approach 3: 2D visualization — scatter of (log ω, log κ) colored by AMI_Total.

If clinical differences across quadrants are just driven by the linear ω
effect (no genuine non-additive structure), the "typology" framing isn't
substantively new. If quadrants differ in patterns that the main effects
can't predict, that IS a typology finding.
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def fit_t(formula, data, family='t'):
    m = bmb.Model(formula, data=data, family=family)
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95)


def surv(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def main():
    # Load params
    exp, conf = load_both()
    rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        m['log_ratio'] = m['log_omega'] - m['log_kappa']
        m['log_sum']   = m['log_omega'] + m['log_kappa']
        m['angle']     = np.arctan2(m['log_kappa'], m['log_omega'])  # radians
        m['radius']    = np.sqrt(m['log_omega']**2 + m['log_kappa']**2)
        # AMI_Total via subscale sum
        if 'AMI_Total' not in m.columns:
            m['AMI_Total'] = m[['AMI_Social','AMI_Behavioural','AMI_Emotional']].sum(axis=1)
        if 'DASS21_Total' not in m.columns:
            m['DASS21_Total'] = m[['DASS21_Anxiety','DASS21_Stress','DASS21_Depression']].sum(axis=1)
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)

    # Merge corrected STAI
    csta = []
    for sample in ['exp','con']:
        p = REPO/f'results/stats/affect_analysis/clinical_scores_corrected_{sample}.csv'
        if p.exists():
            csta.append(pd.read_csv(p))
    cstai = pd.concat(csta, ignore_index=True)
    cstai = cstai[['subj','sample','STAI_Trait_corrected']]
    cstai['sample'] = cstai['sample'].replace({'exp':'exploratory','con':'confirmatory'})
    df = df.merge(cstai, on=['subj','sample'], how='inner')
    print(f'N = {len(df)}')

    # Within-sample z all variables
    clinical = ['AMI_Total','AMI_Social','DASS21_Total','OASIS_Total',
                'STICSA_Total','STAI_Trait_corrected','MFIS_Total','PHQ9_Total']
    for c in clinical + ['log_omega','log_kappa','log_ratio','log_sum','angle','radius']:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std() > 0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    # ===== Define quadrants via within-sample median split =====
    df['omega_hi'] = 0
    df['kappa_hi'] = 0
    for s in df['sample'].unique():
        mask = df['sample']==s
        df.loc[mask, 'omega_hi'] = (df.loc[mask,'log_omega'] > df.loc[mask,'log_omega'].median()).astype(int)
        df.loc[mask, 'kappa_hi'] = (df.loc[mask,'log_kappa'] > df.loc[mask,'log_kappa'].median()).astype(int)

    # Profile labels (note κ direction: high κ = MORE effort cost = LESS mobilized)
    def label(r):
        if r['omega_hi']==1 and r['kappa_hi']==0: return 'HighW_LowK_VigilantActive'  # Type A
        if r['omega_hi']==0 and r['kappa_hi']==1: return 'LowW_HighK_PassiveFrozen'   # Type B
        if r['omega_hi']==1 and r['kappa_hi']==1: return 'HighW_HighK_FrozenVigilant'
        return 'LowW_LowK_DisengagedActive'
    df['profile'] = df.apply(label, axis=1)

    print('\nQuadrant sizes (within-sample median split):')
    print(df['profile'].value_counts())

    # ===== 1. Quadrant means on each clinical scale =====
    print('\n' + '='*72)
    print('APPROACH 1: Clinical means per (ω, κ) quadrant')
    print('='*72)
    for c in clinical:
        print(f'\n  {c}_z by profile:')
        g = df.groupby('profile')[f'{c}_z'].agg(['mean','std','count'])
        g['se'] = g['std']/np.sqrt(g['count'])
        print(g[['mean','se','count']].round(3).to_string())

    # ===== 2. Bayesian ANOVA on AMI_Total (the headline) =====
    print('\n' + '='*72)
    print('APPROACH 2: Bayesian ANOVA — AMI_Total_z ~ profile (4 groups)')
    print('='*72)
    sub = df[['AMI_Total_z','profile']].dropna()
    m = bmb.Model('AMI_Total_z ~ profile', data=sub, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    s = az.summary(fit, hdi_prob=0.95)
    print(s[s.index.str.contains('profile')][['mean','sd','hdi_2.5%','hdi_97.5%']].round(3).to_string())

    # Direct contrast: Type A (high ω + low κ) vs Type B (low ω + high κ)
    sub_AB = df[df['profile'].isin(['HighW_LowK_VigilantActive','LowW_HighK_PassiveFrozen'])].copy()
    print(f'\n--- Type A vs Type B: direct contrast on each clinical scale ---')
    print(f'  Type A (high ω, low κ): N={sub_AB[sub_AB["profile"]=="HighW_LowK_VigilantActive"].shape[0]}')
    print(f'  Type B (low ω, high κ): N={sub_AB[sub_AB["profile"]=="LowW_HighK_PassiveFrozen"].shape[0]}')
    rows_results = []
    for c in clinical:
        s_d = sub_AB[['profile', f'{c}_z']].dropna()
        s_d['type_A'] = (s_d['profile']=='HighW_LowK_VigilantActive').astype(int)
        mdl = bmb.Model(f'{c}_z ~ type_A', data=s_d, family='t')
        fit = mdl.fit(**BKW, idata_kwargs={'log_likelihood': False})
        r = az.summary(fit, hdi_prob=0.95).loc['type_A']
        flag = '★' if surv(r) else ' '
        print(f'  {c:<26} A−B: β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        rows_results.append({'comparison':'TypeA_vs_TypeB', 'scale':c,
                             'mean':float(r['mean']), 'hdi_lo':float(r['hdi_2.5%']),
                             'hdi_hi':float(r['hdi_97.5%']), 'survives':bool(surv(r))})

    # ===== 3. Polar decomposition — angle + radius =====
    print('\n' + '='*72)
    print('APPROACH 3: Polar decomposition — clinical ~ angle + radius')
    print('='*72)
    print('  angle = arctan(log_κ / log_ω) — direction in parameter plane')
    print('  radius = sqrt(log_ω² + log_κ²) — overall engagement magnitude')
    for c in clinical:
        sub = df[[f'{c}_z','angle_z','radius_z']].dropna()
        s = fit_t(f'{c}_z ~ angle_z + radius_z', sub)
        ra = s.loc['angle_z']; rr = s.loc['radius_z']
        fa = '★' if surv(ra) else ' '
        fr = '★' if surv(rr) else ' '
        print(f'  {c:<26} angle β={ra["mean"]:+.3f} [{ra["hdi_2.5%"]:+.3f},{ra["hdi_97.5%"]:+.3f}]{fa} | '
              f'radius β={rr["mean"]:+.3f} [{rr["hdi_2.5%"]:+.3f},{rr["hdi_97.5%"]:+.3f}]{fr}')
        rows_results.append({'comparison':'polar_'+c, 'scale':c,
                             'mean':float(ra['mean']), 'hdi_lo':float(ra['hdi_2.5%']),
                             'hdi_hi':float(ra['hdi_97.5%']), 'survives':bool(surv(ra))})

    # ===== 4. Does the quadrant contrast survive when controlling for log(ω) main effect? =====
    print('\n' + '='*72)
    print('APPROACH 4: Does typology add anything BEYOND linear ω effect?')
    print('='*72)
    # Test: AMI_Total_z ~ log_omega_z + log_kappa_z + (type_A indicator)
    # If type_A is null after main effects, typology is just a coarsened ω effect.
    df['is_TypeA'] = (df['profile']=='HighW_LowK_VigilantActive').astype(int)
    df['is_TypeB'] = (df['profile']=='LowW_HighK_PassiveFrozen').astype(int)
    for c in ['AMI_Total','AMI_Social','DASS21_Total']:
        sub = df[[f'{c}_z','log_omega_z','log_kappa_z','is_TypeA','is_TypeB']].dropna()
        s = fit_t(f'{c}_z ~ log_omega_z + log_kappa_z + is_TypeA + is_TypeB', sub)
        print(f'\n  {c}_z model:')
        for term in ['log_omega_z','log_kappa_z','is_TypeA','is_TypeB']:
            r = s.loc[term]
            flag = '★' if surv(r) else ' '
            print(f'    {term:<18} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # ===== Save results + viz =====
    pd.DataFrame(rows_results).to_csv(
        REPO/'results/stats/affect_analysis/omega_kappa_profile_clinical.csv', index=False)

    # 2D scatter colored by AMI_Total
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df['log_omega'], df['log_kappa'], c=df['AMI_Total'],
                    cmap='viridis', s=24, alpha=0.7, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('AMI_Total (apathy)')
    # Medians as quadrant lines
    ax.axvline(df['log_omega'].median(), color='red', lw=1, ls='--', alpha=0.5)
    ax.axhline(df['log_kappa'].median(), color='red', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel('log(ω) — capture cost weighting / vigilance')
    ax.set_ylabel('log(κ) — effort cost weighting (high = less mobilized)')
    ax.set_title('(log ω, log κ) parameter space, colored by AMI_Total\n'
                 'Top-right = vigilant + frozen | Bottom-right = vigilant + active | etc.')
    plt.tight_layout()
    plt.savefig(REPO/'results/figs/affect_analysis/omega_kappa_AMI_scatter.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved scatter: results/figs/affect_analysis/omega_kappa_AMI_scatter.png')


if __name__ == '__main__':
    main()
