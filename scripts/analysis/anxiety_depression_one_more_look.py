"""
One more honest look at whether anxiety/depression has any effect on (ω, κ):

1. POSTERIOR PROBABILITY OF DIRECTION (instead of 95% HDI threshold)
   For each clinical scale, compute P(β < 0). HDI threshold is binary
   (significant or not); posterior probability is continuous.
   Effects with P(β < 0) > 0.95 = one-sided 95% credible (less stringent than
   two-sided 95% HDI but still defensible if direction is pre-specified).

2. COMBINED ANXIETY+DEPRESSION COMPOSITE
   Not just an anxiety composite (we did that in §4.64 — null). Combine ALL
   anxiety AND depression scales into one transdiagnostic distress composite
   and test alongside AMI_Total.

3. BAYESIAN MODEL COMPARISON
   Does removing all anxiety+depression scales hurt the model? Compare
   kitchen-sink with vs without those scales using WAIC.
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

BKW = dict(draws=2000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)  # 2000 draws for cleaner posterior tail


def main():
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
        if 'DASS21_Total' not in m.columns:
            m['DASS21_Total'] = m[['DASS21_Anxiety','DASS21_Stress','DASS21_Depression']].sum(axis=1)
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
    cols = ['log_omega','log_kappa',
            'AMI_Total','AMI_Social',
            'DASS21_Anxiety','DASS21_Stress','DASS21_Depression','DASS21_Total',
            'OASIS_Total','STICSA_Total','STAI_Trait_corrected',
            'MFIS_Total','PHQ9_Total']
    for c in cols:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()
    print(f'N = {len(df)}')

    # ===== Composite ANX+DEP =====
    # Combine 6 scales (3 anxiety + 2 depression + 1 stress) into a single distress composite
    anx_dep_z = ['DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
                 'STAI_Trait_corrected_z','OASIS_Total_z','STICSA_Total_z','PHQ9_Total_z']
    df['ANXDEP_composite'] = df[anx_dep_z].mean(axis=1)
    for s in df['sample'].unique():
        mask = df['sample']==s
        x = df.loc[mask,'ANXDEP_composite']
        df.loc[mask,'ANXDEP_composite_z'] = (x-x.mean())/x.std()

    # ============================================================
    # Test 1: Posterior probability of direction for each scale
    # ============================================================
    print('\n' + '='*72)
    print('TEST 1: Posterior P(β < 0) for each anxiety/depression scale (kitchen-sink totals)')
    print('='*72)
    totals_z = ['AMI_Total_z','DASS21_Total_z','OASIS_Total_z','STICSA_Total_z',
                'STAI_Trait_corrected_z','MFIS_Total_z','PHQ9_Total_z']
    sub = df[['log_omega_z','log_kappa_z'] + totals_z].dropna()
    formula = 'log_omega_z ~ ' + ' + '.join(totals_z) + ' + log_kappa_z'
    m = bmb.Model(formula, data=sub, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    posterior = fit.posterior

    print(f'{"scale":<26} {"β mean":>8} {"95% HDI":>22} {"P(β<0)":>10} {"P(β>0)":>10}  verdict')
    for term in totals_z + ['log_kappa_z']:
        samples = posterior[term].values.flatten()
        mean = samples.mean()
        lo, hi = np.percentile(samples, [2.5, 97.5])
        p_neg = (samples < 0).mean()
        p_pos = (samples > 0).mean()
        if max(p_neg, p_pos) > 0.95:
            verdict = '★ 95% directional'
        elif max(p_neg, p_pos) > 0.90:
            verdict = '· 90% directional'
        elif max(p_neg, p_pos) > 0.80:
            verdict = '· 80% directional'
        else:
            verdict = ''
        print(f'  {term:<26} {mean:+8.3f} [{lo:+.3f},{hi:+.3f}]  {p_neg:>8.3f}  {p_pos:>8.3f}  {verdict}')

    # ============================================================
    # Test 2: Combined ANX+DEP composite alongside AMI
    # ============================================================
    print('\n' + '='*72)
    print('TEST 2: ANX+DEP composite (7 scales) alongside AMI_Total + κ')
    print('='*72)
    sub = df[['log_omega_z','log_kappa_z','AMI_Total_z','ANXDEP_composite_z']].dropna()
    m = bmb.Model('log_omega_z ~ AMI_Total_z + ANXDEP_composite_z + log_kappa_z',
                  data=sub, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    posterior = fit.posterior
    print(f'{"term":<26} {"β":>8} {"95% HDI":>22} {"P(β<0)":>10}')
    for term in ['AMI_Total_z', 'ANXDEP_composite_z', 'log_kappa_z']:
        samples = posterior[term].values.flatten()
        mean = samples.mean()
        lo, hi = np.percentile(samples, [2.5, 97.5])
        p_neg = (samples < 0).mean()
        print(f'  {term:<26} {mean:+8.3f} [{lo:+.3f},{hi:+.3f}]  {p_neg:>8.3f}')

    # ============================================================
    # Test 3: Bayesian model comparison via WAIC
    # ============================================================
    print('\n' + '='*72)
    print('TEST 3: WAIC comparison — kitchen-sink with vs without anx+dep scales')
    print('='*72)

    # Model A: full kitchen-sink (AMI + 6 anx/dep + κ)
    sub = df[['log_omega_z','log_kappa_z'] + totals_z].dropna()
    m_full = bmb.Model('log_omega_z ~ ' + ' + '.join(totals_z) + ' + log_kappa_z',
                       data=sub, family='t')
    fit_full = m_full.fit(**BKW, idata_kwargs={'log_likelihood': True})

    # Model B: AMI + MFIS + κ only (drop all anxiety + depression scales)
    minimal = ['AMI_Total_z','MFIS_Total_z']
    m_min = bmb.Model('log_omega_z ~ ' + ' + '.join(minimal) + ' + log_kappa_z',
                      data=sub, family='t')
    fit_min = m_min.fit(**BKW, idata_kwargs={'log_likelihood': True})

    # Model C: AMI + κ only (most parsimonious)
    m_ami = bmb.Model('log_omega_z ~ AMI_Total_z + log_kappa_z', data=sub, family='t')
    fit_ami = m_ami.fit(**BKW, idata_kwargs={'log_likelihood': True})

    comp = az.compare({
        'C_AMI_only':  fit_ami,
        'B_AMI_MFIS':  fit_min,
        'A_full_kitchen_sink': fit_full,
    }, ic='waic')
    print(comp[['rank','elpd_waic','p_waic','elpd_diff','dse','weight']].round(2).to_string())

    print('\nInterpretation:')
    print('  Lower rank = better model. If full model is best, anxiety/depression help.')
    print('  If AMI_only wins, anxiety/depression scales add nothing.')

    # ============================================================
    # Test 4: Within "comorbid" subset, does AMI_Social still drive ω?
    # ============================================================
    print('\n' + '='*72)
    print('TEST 4: Within-subgroup — is AMI_Social effect strongest among COMORBID subjects?')
    print('='*72)
    # Define comorbid using median splits on DASS_Anxiety and DASS_Depression
    df['anx_hi'] = 0; df['dep_hi'] = 0
    for s in df['sample'].unique():
        mask = df['sample']==s
        df.loc[mask,'anx_hi'] = (df.loc[mask,'DASS21_Anxiety'] > df.loc[mask,'DASS21_Anxiety'].median()).astype(int)
        df.loc[mask,'dep_hi'] = (df.loc[mask,'DASS21_Depression'] > df.loc[mask,'DASS21_Depression'].median()).astype(int)
    df['comorbid_label'] = df.apply(lambda r:
        'comorbid' if (r['anx_hi']==1 and r['dep_hi']==1)
        else 'pure_anx' if (r['anx_hi']==1 and r['dep_hi']==0)
        else 'pure_dep' if (r['anx_hi']==0 and r['dep_hi']==1)
        else 'healthy', axis=1)
    print('Subgroup sizes:')
    print(df['comorbid_label'].value_counts())

    print('\n  AMI_Social → log_omega within each subgroup:')
    print(f'  {"subgroup":<14} {"N":>4} {"β":>8} {"95% HDI":>22} {"P(β>0)":>10}')
    for sg in ['comorbid', 'pure_anx', 'pure_dep', 'healthy']:
        sub = df[df['comorbid_label']==sg].copy()
        if len(sub) < 20: continue
        # re-z within this subgroup, by sample
        for c in ['log_omega', 'AMI_Social']:
            for s in sub['sample'].unique():
                mask = sub['sample']==s
                x = sub.loc[mask, c]
                if x.std()>0:
                    sub.loc[mask, f'{c}_z'] = (x-x.mean())/x.std()
        sub = sub.dropna(subset=['log_omega_z','AMI_Social_z'])
        m = bmb.Model('log_omega_z ~ AMI_Social_z', data=sub, family='t')
        fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
        samples = fit.posterior['AMI_Social_z'].values.flatten()
        mean = samples.mean()
        lo, hi = np.percentile(samples, [2.5, 97.5])
        p_pos = (samples>0).mean()
        survives = '★' if (lo>0 or hi<0) else ' '
        print(f'  {sg:<14} {len(sub):>4} {mean:+8.3f} [{lo:+.3f},{hi:+.3f}]{survives}  {p_pos:>8.3f}')


if __name__ == '__main__':
    main()
