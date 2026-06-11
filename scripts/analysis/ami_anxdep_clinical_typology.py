"""
Test whether AMI_Total + ANX+DEP composite together carve out clinical subtypes
that map onto distinct (ω, κ) parameter profiles.

Pooled N=571 (user wants pooled only; split sample is underpowered).

Tests:
1. AMI_Total × ANX+DEP composite correlation — are they independent dimensions or
   highly correlated?
2. Median-split clinical quadrants (4 profiles):
   - High AMI + Low ANXDEP: "pure apathetic" — predict high ω
   - Low AMI + High ANXDEP: "pure distressed" — predict low ω
   - High both: "comorbid apathy+distress" — effects cancel?
   - Low both: "healthy" — baseline
3. Mean ω per profile + pairwise contrasts (Bayesian)
4. Re-test interaction (AMI × ANXDEP) on ω with corrected scales + composite
5. Predicted ω scatter: each subject's predicted ω from AMI + ANXDEP composite
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

BKW = dict(draws=2000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def main():
    # Load
    exp, conf = load_both()
    rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        if 'AMI_Total' not in m.columns:
            m['AMI_Total'] = m[['AMI_Social','AMI_Behavioural','AMI_Emotional']].sum(axis=1)
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)

    csta = []
    for sample in ['exp','con']:
        p = REPO/f'results/stats/affect_analysis/clinical_scores_corrected_{sample}.csv'
        if p.exists():
            csta.append(pd.read_csv(p))
    cstai = pd.concat(csta, ignore_index=True)[['subj','sample','STAI_Trait_corrected']]
    cstai['sample'] = cstai['sample'].replace({'exp':'exploratory','con':'confirmatory'})
    df = df.merge(cstai, on=['subj','sample'], how='inner')

    # Within-sample z
    cols = ['log_omega','log_kappa','AMI_Total','AMI_Social',
            'DASS21_Anxiety','DASS21_Depression','DASS21_Stress',
            'STAI_Trait_corrected','OASIS_Total','STICSA_Total','PHQ9_Total']
    for c in cols:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    # ANX+DEP composite
    anxdep = ['DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
              'STAI_Trait_corrected_z','OASIS_Total_z','STICSA_Total_z','PHQ9_Total_z']
    df['ANXDEP_comp'] = df[anxdep].mean(axis=1)
    df['ANXDEP_comp_z'] = np.nan
    for s in df['sample'].unique():
        mask = df['sample']==s
        x = df.loc[mask,'ANXDEP_comp']
        df.loc[mask,'ANXDEP_comp_z'] = (x-x.mean())/x.std()
    print(f'N = {len(df)}')

    # ============================================================
    # Test 1: Correlation between AMI and ANX+DEP
    # ============================================================
    print('\n' + '='*72)
    print('TEST 1: AMI_Total × ANX+DEP composite correlation')
    print('='*72)
    r_pearson, p_pearson = pearsonr(df['AMI_Total'], df['ANXDEP_comp'])
    r_spear, p_spear = spearmanr(df['AMI_Total'], df['ANXDEP_comp'])
    print(f'  Pearson  r = {r_pearson:+.3f}  p = {p_pearson:.3g}')
    print(f'  Spearman ρ = {r_spear:+.3f}  p = {p_spear:.3g}')
    print(f'  Interpretation: r²={r_pearson**2:.1%} variance shared')
    print(f'    Low r (<0.3) = independent dimensions')
    print(f'    Moderate r (0.3-0.6) = correlated but distinguishable')
    print(f'    High r (>0.6) = mostly redundant constructs')

    # Subscale-level correlations
    print('\n  AMI_Total correlations with individual anxiety/depression scales:')
    for sc in ['DASS21_Anxiety','DASS21_Depression','DASS21_Stress',
              'STAI_Trait_corrected','OASIS_Total','STICSA_Total','PHQ9_Total']:
        r,_ = pearsonr(df['AMI_Total'], df[sc])
        print(f'    {sc:<26} r = {r:+.3f}')

    # ============================================================
    # Test 2: Clinical quadrant analysis (median split on AMI and ANX+DEP)
    # ============================================================
    print('\n' + '='*72)
    print('TEST 2: Clinical quadrants (median split on AMI and ANX+DEP)')
    print('='*72)
    df['AMI_hi'] = 0; df['ANXDEP_hi'] = 0
    for s in df['sample'].unique():
        mask = df['sample']==s
        df.loc[mask,'AMI_hi'] = (df.loc[mask,'AMI_Total'] > df.loc[mask,'AMI_Total'].median()).astype(int)
        df.loc[mask,'ANXDEP_hi'] = (df.loc[mask,'ANXDEP_comp'] > df.loc[mask,'ANXDEP_comp'].median()).astype(int)

    def label(r):
        if r['AMI_hi']==1 and r['ANXDEP_hi']==0: return '1_PureApathy'
        if r['AMI_hi']==0 and r['ANXDEP_hi']==1: return '2_PureDistress'
        if r['AMI_hi']==1 and r['ANXDEP_hi']==1: return '3_Comorbid'
        return '4_Healthy'
    df['profile'] = df.apply(label, axis=1)

    print('\nProfile cell sizes:')
    print(df['profile'].value_counts().sort_index())

    print('\nMean log(ω)_z by profile:')
    g = df.groupby('profile').agg(
        n=('subj','count'),
        log_omega_z_mean=('log_omega_z','mean'),
        log_omega_z_std=('log_omega_z','std'),
        log_kappa_z_mean=('log_kappa_z','mean'),
    )
    g['SE_omega'] = g['log_omega_z_std']/np.sqrt(g['n'])
    print(g[['n','log_omega_z_mean','SE_omega','log_kappa_z_mean']].round(3).to_string())

    # STAI diagnostic — check direction
    print('\n--- STAI diagnostic: does STAI_corrected correlate POSITIVELY with other anxiety scales? ---')
    for sc in ['DASS21_Anxiety','OASIS_Total','STICSA_Total']:
        r, _ = pearsonr(df['STAI_Trait_corrected'], df[sc])
        verdict = '✓' if r > 0 else '✗ WRONG DIRECTION'
        print(f'    STAI_corrected vs {sc:<26} r = {r:+.3f}  {verdict}')

    # Bayesian ANOVA-style: log_omega_z ~ profile
    print('\nBayesian regression: log(ω)_z ~ profile + log(κ)_z')
    sub = df[['log_omega_z','log_kappa_z','profile']].dropna().copy()
    m = bmb.Model('log_omega_z ~ profile + log_kappa_z', data=sub, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    s = az.summary(fit, hdi_prob=0.95)
    print('\n  Contrasts (reference = first alphabetic profile):')
    for term in s.index:
        if term.startswith('profile'):
            r = s.loc[term]
            sv = '★' if (r['hdi_2.5%']>0 or r['hdi_97.5%']<0) else ' '
            print(f'    {term:<32} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {sv}')

    # Key contrast: Pure Apathy vs Pure Distress (test the OPPOSITE-DIRECTION hypothesis)
    print('\n  KEY CONTRAST: Pure Apathy vs Pure Distress (expected opposite ω)')
    sub_AB = df[df['profile'].isin(['1_PureApathy','2_PureDistress'])].copy()
    sub_AB['is_apathy'] = (sub_AB['profile']=='1_PureApathy').astype(int)
    m = bmb.Model('log_omega_z ~ is_apathy + log_kappa_z', data=sub_AB, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    s = az.summary(fit, hdi_prob=0.95).loc['is_apathy']
    samples = fit.posterior['is_apathy'].values.flatten()
    p_pos = (samples>0).mean()
    sv = '★' if (s['hdi_2.5%']>0 or s['hdi_97.5%']<0) else ' '
    print(f'    PureApathy − PureDistress: β={s["mean"]:+.3f}  HDI [{s["hdi_2.5%"]:+.3f}, {s["hdi_97.5%"]:+.3f}]  P(β>0)={p_pos:.3f}  {sv}')

    # ============================================================
    # Test 3: Re-test interaction with the composite
    # ============================================================
    print('\n' + '='*72)
    print('TEST 3: AMI × ANX+DEP interaction on log(ω)')
    print('='*72)
    sub = df[['log_omega_z','log_kappa_z','AMI_Total_z','ANXDEP_comp_z']].dropna()
    m = bmb.Model('log_omega_z ~ AMI_Total_z * ANXDEP_comp_z + log_kappa_z',
                  data=sub, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    s = az.summary(fit, hdi_prob=0.95)
    for term in ['AMI_Total_z','ANXDEP_comp_z','AMI_Total_z:ANXDEP_comp_z','log_kappa_z']:
        if term not in s.index: continue
        r = s.loc[term]
        sv = '★' if (r['hdi_2.5%']>0 or r['hdi_97.5%']<0) else ' '
        print(f'  {term:<32} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {sv}')

    # ============================================================
    # Test 4: Predicted ω from AMI + ANX+DEP — characterize subjects
    # ============================================================
    print('\n' + '='*72)
    print('TEST 4: Predicted log(ω) from AMI + ANX+DEP main effects')
    print('='*72)
    # Simple linear prediction
    AMI_z = df['AMI_Total_z'].values
    AXD_z = df['ANXDEP_comp_z'].values
    # Use the pooled-sample coefficients from §4.75/§4.76
    pred_omega_z = +0.134 * AMI_z + (-0.092) * AXD_z
    df['pred_omega_z'] = pred_omega_z
    df['pred_omega_quintile'] = pd.qcut(df['pred_omega_z'], 5, labels=False)

    print('\n  Subjects ranked into quintiles by predicted ω (from AMI + ANX+DEP):')
    g = df.groupby('pred_omega_quintile').agg(
        n=('subj','count'),
        pred_omega_z=('pred_omega_z','mean'),
        actual_omega_z=('log_omega_z','mean'),
        AMI_z=('AMI_Total_z','mean'),
        ANXDEP_z=('ANXDEP_comp_z','mean'),
    )
    g = g.round(3)
    print(g.to_string())

    # Effect size of prediction
    r_pred, p_pred = pearsonr(df['pred_omega_z'].dropna(),
                              df.loc[df['pred_omega_z'].notna(),'log_omega_z'])
    print(f'\n  Predicted ω correlates with actual log(ω): r = {r_pred:+.3f}, p = {p_pred:.3g}')
    print(f'  Variance explained: R² = {r_pred**2:.1%}')

    # ============================================================
    # Viz: 2D clinical space colored by ω
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(df['AMI_Total_z'], df['ANXDEP_comp_z'], c=df['log_omega_z'],
                    cmap='RdBu_r', s=30, alpha=0.7, edgecolors='none', vmin=-2, vmax=2)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axvline(0, color='k', lw=0.5, ls='--')
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('log(ω) z-score')
    ax.set_xlabel('AMI_Total (apathy) — within-sample z')
    ax.set_ylabel('ANX+DEP composite — within-sample z')
    ax.set_title('Clinical 2D space colored by ω\n'
                 'Pure Apathy (lower-right) = high ω | Pure Distress (upper-left) = low ω')
    # Annotate quadrants
    ax.text(2, -2, 'PureApathy\n(high ω)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(-2, 2, 'PureDistress\n(low ω)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(2, 2, 'Comorbid\n(mid ω)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(-2, -2, 'Healthy\n(mid ω)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax = axes[1]
    ax.scatter(df['pred_omega_z'], df['log_omega_z'], s=20, alpha=0.5, c='steelblue')
    ax.plot([-1,1], [-1,1], 'r--', lw=1, alpha=0.6)
    ax.set_xlabel('Predicted log(ω) from AMI + ANX+DEP')
    ax.set_ylabel('Actual log(ω)')
    ax.set_title(f'Predicted vs actual log(ω)\nr={r_pred:.3f}, R²={r_pred**2:.1%}')

    plt.tight_layout()
    out = REPO/'results/figs/affect_analysis/clinical_typology_omega.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved figure: {out}')


if __name__ == '__main__':
    main()
