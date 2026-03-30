#!/usr/bin/env python3
"""Part 4: Optimal Policy, Suboptimality, and Clinical Mechanism (Steps 4A–4I)"""

import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *


def run_part4(p1=None, p2=None, p3=None):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PART 4: OPTIMAL POLICY, SUBOPTIMALITY, CLINICAL")
    print("=" * 70)

    data = p1['data'] if p1 else load_choice_data()
    param_df = p1['param_df'] if p1 else pd.read_csv(OUT_DIR / 'part1_params_full.csv')
    psych = load_psych()
    feelings = load_feelings()

    # ── Step 4A: Optimal policy ──
    print("\n--- Step 4A: Optimal policy ---")
    data['T_H'] = data['distance_H'].map({1: 5.0, 2: 7.0, 3: 9.0})
    data['S_H'] = np.exp(-data['threat'] * data['T_H'])
    data['S_L'] = np.exp(-data['threat'] * 5.0)
    data['EV_H'] = data['S_H'] * 5 - (1 - data['S_H']) * 5 - data['effort_reqT']
    data['EV_L'] = data['S_L'] * 1 - (1 - data['S_L']) * 5 - (0.4 * 5)
    data['optimal'] = (data['EV_H'] > data['EV_L']).astype(int)

    print(f"\n  {'':>8} {'D=1':>10} {'D=2':>10} {'D=3':>10}")
    for T in [0.1, 0.5, 0.9]:
        row = f"  T={T:.0%}"
        for D in [1, 2, 3]:
            sub = data[(data['T_round'] == T) & (data['distance_H'] == D)]
            opt = sub['optimal'].iloc[0]
            margin = (sub['EV_H'] - sub['EV_L']).iloc[0]
            row += f"  {'H' if opt else 'L'} ({margin:+.2f})"
        print(row)

    n_heavy_opt = data.groupby(['T_round', 'distance_H'])['optimal'].first().sum()
    print(f"\n  Cells favoring heavy: {n_heavy_opt}/9")

    # ── Step 4B: Individual deviations ──
    print("\n--- Step 4B: Individual deviations ---")
    data['overcautious'] = ((data['choice'] == 0) & (data['optimal'] == 1)).astype(int)
    data['overrisky'] = ((data['choice'] == 1) & (data['optimal'] == 0)).astype(int)
    data['is_optimal'] = (data['choice'] == data['optimal']).astype(int)

    subj_dev = data.groupby('subj').agg(
        optimality_rate=('is_optimal', 'mean'),
        overcautious_rate=('overcautious', 'mean'),
        overrisky_rate=('overrisky', 'mean'),
    ).reset_index()
    print(f"  Optimality: M={subj_dev['optimality_rate'].mean():.3f}, SD={subj_dev['optimality_rate'].std():.3f}")
    print(f"  Overcautious: M={subj_dev['overcautious_rate'].mean():.3f}, SD={subj_dev['overcautious_rate'].std():.3f}")
    print(f"  Overrisky: M={subj_dev['overrisky_rate'].mean():.3f}, SD={subj_dev['overrisky_rate'].std():.3f}")

    # ── Step 4C: Affective measures ──
    print("\n--- Step 4C: Affective measures ---")
    anx = feelings[feelings['questionLabel'] == 'anxiety']

    # Calibration (affect-tracking): r(anxiety, danger=1-S)
    # danger = 1 - exp(-p*T) where T depends on distance of the probe trial
    anx = anx.copy()
    anx['T_dur'] = anx['distanceFromSafety'].map(lambda x: {5: 5.0, 7: 7.0, 9: 9.0}.get(int(round(x)), 5.0)
                                                  if pd.notna(x) else 5.0)
    anx['danger'] = 1 - np.exp(-anx['threat'] * anx['T_dur'])

    # Population regression for discrepancy
    slope, intercept = np.polyfit(anx['danger'].values, anx['response'].values, 1)

    calib_list = []
    for s, sdf in anx.groupby('subj'):
        danger = sdf['danger'].values
        resp = sdf['response'].values
        cal_at = pearsonr(resp, danger)[0] if np.std(danger) > 0 and np.std(resp) > 0 else np.nan
        pred = slope * danger + intercept
        disc = (resp - pred).mean()
        calib_list.append({'subj': s, 'calibration_at': cal_at, 'discrepancy': disc})

    # Calibration (confidence-accuracy): r(confidence, optimal_match)
    conf = feelings[feelings['questionLabel'] == 'confidence'].copy()
    # For each probe trial, was the forced cookie the optimal one?
    conf['T_dur'] = conf['distanceFromSafety'].map(lambda x: {5: 5.0, 7: 7.0, 9: 9.0}.get(int(round(x)), 5.0)
                                                    if pd.notna(x) else 5.0)
    conf['S_forced'] = np.exp(-conf['threat'] * conf['T_dur'])
    conf['is_heavy'] = (conf['trialCookie_weight'] == 3.0).astype(int)
    # EV of forced cookie
    conf['EV_forced'] = np.where(conf['is_heavy'] == 1,
                                  conf['S_forced'] * 5 - (1 - conf['S_forced']) * 5,
                                  conf['S_forced'] * 1 - (1 - conf['S_forced']) * 5)

    for s, sdf in conf.groupby('subj'):
        resp = sdf['response'].values
        ev = sdf['EV_forced'].values
        cal_ca = pearsonr(resp, ev)[0] if np.std(ev) > 0 and np.std(resp) > 0 else np.nan
        match = [c for c in calib_list if c['subj'] == s]
        if match:
            match[0]['calibration_ca'] = cal_ca

    calib_df = pd.DataFrame(calib_list).dropna(subset=['calibration_at', 'discrepancy'])
    r_cd, p_cd = pearsonr(calib_df['calibration_at'], calib_df['discrepancy'])
    print(f"  Calibration (affect-tracking) ⊥ Discrepancy: r={r_cd:.3f}, p={p_cd:.4f}")
    if 'calibration_ca' in calib_df.columns:
        r_cc, _ = pearsonr(calib_df['calibration_at'].dropna(), calib_df['calibration_ca'].dropna().reindex(calib_df.index))
        print(f"  Calibration AT ↔ CA: r={r_cc:.3f}")
    save_df(calib_df, 'part4_affect')

    # ── Step 4D: Model comparison for overcaution ──
    print("\n--- Step 4D: Model comparison for overcautious rate ---")
    merged = subj_dev.merge(param_df, on='subj').merge(calib_df, on='subj', how='left')
    for c in ['log_k', 'log_beta', 'log_cd']:
        if c in merged.columns:
            merged[f'{c}_z'] = (merged[c] - merged[c].mean()) / merged[c].std()
    if 'discrepancy' in merged.columns:
        merged['disc_z'] = (merged['discrepancy'] - merged['discrepancy'].mean()) / merged['discrepancy'].std()
    if 'calibration_at' in merged.columns:
        merged['cal_z'] = (merged['calibration_at'] - merged['calibration_at'].mean()) / merged['calibration_at'].std()

    m_valid = merged.dropna(subset=['overcautious_rate', 'log_k_z', 'log_beta_z', 'disc_z', 'cal_z'])
    y = m_valid['overcautious_rate'].values

    models = {
        'M1 null': sm.add_constant(np.ones((len(m_valid), 1))),
        'M2 k': sm.add_constant(m_valid[['log_k_z']].values),
        'M3 β': sm.add_constant(m_valid[['log_beta_z']].values),
        'M4 k+β': sm.add_constant(m_valid[['log_k_z', 'log_beta_z']].values),
        'M5 k×β': sm.add_constant(np.column_stack([m_valid['log_k_z'], m_valid['log_beta_z'],
                                                     m_valid['log_k_z'] * m_valid['log_beta_z']])),
        'M6 disc': sm.add_constant(m_valid[['disc_z']].values),
        'M7 full': sm.add_constant(np.column_stack([m_valid['log_k_z'], m_valid['log_beta_z'],
                                                      m_valid['log_k_z'] * m_valid['log_beta_z'],
                                                      m_valid['disc_z'], m_valid['cal_z']])),
    }

    print(f"\n  {'Model':<12} {'R²':>6} {'Adj R²':>8} {'AIC':>8} {'BIC':>8} {'ΔBIC':>8}")
    print(f"  {'-' * 52}")
    best_bic = float('inf')
    for name, X in models.items():
        ols = sm.OLS(y, X).fit()
        if ols.bic < best_bic:
            best_bic = ols.bic
    for name, X in models.items():
        ols = sm.OLS(y, X).fit()
        print(f"  {name:<12} {ols.rsquared:>6.3f} {ols.rsquared_adj:>8.3f} {ols.aic:>8.1f} {ols.bic:>8.1f} {ols.bic - best_bic:>+8.1f}")

    # Full model details
    X_full = models['M7 full']
    ols_full = sm.OLS(y, X_full).fit()
    print(f"\n  M7 full details (R²={ols_full.rsquared:.3f}):")
    for i, nm in enumerate(['Intercept', 'k_z', 'β_z', 'k×β', 'disc_z', 'cal_z']):
        sig = "***" if ols_full.pvalues[i] < .001 else ("**" if ols_full.pvalues[i] < .01 else ("*" if ols_full.pvalues[i] < .05 else ""))
        print(f"    {nm:<12}: β={ols_full.params[i]:>7.3f}, t={ols_full.tvalues[i]:>6.3f}, p={ols_full.pvalues[i]:.4f} {sig}")

    # ΔR² for discrepancy
    ols_m5 = sm.OLS(y, models['M5 k×β']).fit()
    dr2 = ols_full.rsquared - ols_m5.rsquared
    print(f"\n  ΔR² (disc+cal above k×β): {dr2:.4f}")

    # ── Step 4F: Residual suboptimality ──
    print("\n--- Step 4F: Residual suboptimality ---")
    m_valid['oc_resid'] = ols_m5.resid
    X_resid = sm.add_constant(m_valid[['disc_z', 'cal_z']].values)
    ols_resid = sm.OLS(m_valid['oc_resid'].values, X_resid).fit()
    print(f"  overcautious_resid ~ disc + cal (R²={ols_resid.rsquared:.4f}):")
    for i, nm in enumerate(['Intercept', 'Discrepancy', 'Calibration']):
        sig = "***" if ols_resid.pvalues[i] < .001 else ("**" if ols_resid.pvalues[i] < .01 else ("*" if ols_resid.pvalues[i] < .05 else ""))
        print(f"    {nm:<15}: β={ols_resid.params[i]:>7.3f}, t={ols_resid.tvalues[i]:>6.3f}, p={ols_resid.pvalues[i]:.4f} {sig}")

    # ── Step 4G: Clinical ──
    print("\n--- Step 4G: Clinical mechanism ---")
    clinical = merged.merge(psych, on='subj', how='left')
    for clin in ['STAI_State', 'OASIS_Total', 'AMI_Total']:
        cm = clinical.dropna(subset=[clin, 'log_k_z', 'log_beta_z', 'disc_z'])
        cm[f'{clin}_z'] = (cm[clin] - cm[clin].mean()) / cm[clin].std()

        # Base: clinical ~ k + β + overcaution
        X1 = sm.add_constant(cm[['log_k_z', 'log_beta_z', 'overcautious_rate']].values)
        ols1 = sm.OLS(cm[f'{clin}_z'].values, X1).fit()

        # + discrepancy
        X2 = sm.add_constant(cm[['log_k_z', 'log_beta_z', 'overcautious_rate', 'disc_z']].values)
        ols2 = sm.OLS(cm[f'{clin}_z'].values, X2).fit()

        dr2 = ols2.rsquared - ols1.rsquared
        print(f"\n  {clin} (N={len(cm)}):")
        print(f"    Base R² (k+β+OC): {ols1.rsquared:.4f}")
        print(f"    + disc R²: {ols2.rsquared:.4f} (ΔR²={dr2:.4f})")
        for i, nm in enumerate(['Intercept', 'k_z', 'β_z', 'OC_rate', 'Discrepancy']):
            sig = "***" if ols2.pvalues[i] < .001 else ("**" if ols2.pvalues[i] < .01 else ("*" if ols2.pvalues[i] < .05 else ""))
            print(f"    {nm:<15}: β={ols2.params[i]:>7.3f}, t={ols2.tvalues[i]:>6.3f}, p={ols2.pvalues[i]:.4f} {sig}")

    # ── Step 4H: Anticipatory vigor → suboptimality ──
    print("\n--- Step 4H: Anticipatory vigor → suboptimality ---")
    if p2 and 'df' in p2:
        vigor_df = p2['df']
        # Per-subject anticipatory vigor slope on threat
        from scipy.stats import linregress
        antic_slopes = []
        for s, sdf in vigor_df.dropna(subset=['antic_vigor_resid']).groupby('subj'):
            if len(sdf) > 10:
                sl, _, _, _, _ = linregress(sdf['predator_probability'], sdf['antic_vigor_resid'])
                antic_slopes.append({'subj': s, 'antic_slope': sl})
        antic_df = pd.DataFrame(antic_slopes)
        merged_h = merged.merge(antic_df, on='subj', how='left').dropna(subset=['antic_slope', 'log_k_z', 'log_beta_z'])
        merged_h['antic_slope_z'] = (merged_h['antic_slope'] - merged_h['antic_slope'].mean()) / merged_h['antic_slope'].std()

        X_h = sm.add_constant(merged_h[['antic_slope_z', 'log_k_z', 'log_beta_z']].values)
        ols_h = sm.OLS(merged_h['overcautious_rate'].values, X_h).fit()
        print(f"  overcautious ~ antic_slope + k + β (N={len(merged_h)}, R²={ols_h.rsquared:.3f}):")
        for i, nm in enumerate(['Intercept', 'Antic_slope', 'k_z', 'β_z']):
            sig = "***" if ols_h.pvalues[i] < .001 else ("**" if ols_h.pvalues[i] < .01 else ("*" if ols_h.pvalues[i] < .05 else ""))
            print(f"    {nm:<15}: β={ols_h.params[i]:>7.3f}, t={ols_h.tvalues[i]:>6.3f}, p={ols_h.pvalues[i]:.4f} {sig}")

    # ── Step 4I: Behavioral profiles ──
    print("\n--- Step 4I: Behavioral profiles ---")
    merged['k_group'] = np.where(merged['log_k_z'] >= 0, 'High k', 'Low k')
    merged['b_group'] = np.where(merged['log_beta_z'] >= 0, 'High β', 'Low β')
    merged['profile'] = merged['k_group'] + ' / ' + merged['b_group']

    profiles = {
        'Low k / Low β': 'Vigilant',
        'High k / Low β': 'Effort-averse',
        'Low k / High β': 'Threat-sensitive',
        'High k / High β': 'Overcautious',
    }

    clin_merged = merged.merge(psych[['subj', 'STAI_State']], on='subj', how='left')
    print(f"\n  {'Profile':<20} {'Label':<18} {'N':>4} {'Optimal':>8} {'Overcaut':>9} {'Disc':>7} {'STAI':>6}")
    for prof, label in profiles.items():
        sub = clin_merged[clin_merged['profile'] == prof]
        if len(sub) > 0:
            print(f"  {prof:<20} {label:<18} {len(sub):>4} "
                  f"{sub['optimality_rate'].mean():>8.3f} {sub['overcautious_rate'].mean():>9.3f} "
                  f"{sub['discrepancy'].mean():>7.3f} {sub['STAI_State'].mean():>6.1f}")

    save_df(merged, 'part4_merged')

    elapsed = (time.time() - t0) / 60
    print(f"\n  Part 4 complete ({elapsed:.1f} min)")
    return {}


if __name__ == '__main__':
    p1 = {'param_df': pd.read_csv(OUT_DIR / 'part1_params_full.csv'),
           'data': load_choice_data()}
    run_part4(p1)
