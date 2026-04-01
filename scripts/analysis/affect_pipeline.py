"""
Affect Modeling Pipeline — Anxiety × Confidence Analysis

Stage 1: Per-subject affect regressions (anxiety/confidence ~ T + D + cookie)
Stage 2:
  A — Dissociation structure (are anxiety and confidence informationally separable?)
  B — Computational parameters → affect slopes (ω→confidence, κ→distance sensitivity)
  C — Anxiety-confidence gap → overcaution and clinical symptoms
  D — Metacognitive accuracy (confidence vs model-predicted survival)
"""

import numpy as np, pandas as pd
from scipy.stats import pearsonr, zscore
from scipy.special import expit
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')

EXCLUDE = [154, 197, 208]
DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("results/stats/affect_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CM2 population params for survival computation
CM2_POP = {'gamma': 0.76, 'hazard': 0.481, 'sigma_sp': 0.25}
C_PEN = 5.0


def load_data():
    """Load all data sources."""
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]
    feelings = pd.read_csv(DATA_DIR / "feelings.csv")
    feelings = feelings[~feelings['subj'].isin(EXCLUDE)]
    psych = pd.read_csv(DATA_DIR / "psych.csv")
    psych = psych[~psych['subj'].isin(EXCLUDE)].set_index('subj')
    master = pd.read_csv("results/stats/joint_optimal/master_subject_df.csv", index_col='subj')
    return beh, feelings, psych, master


def stage1_affect_regressions(feelings):
    """Stage 1: Per-subject affect parameters from probe trial regressions.

    rating ~ b0 + b1*T + b2*D + b3*is_heavy
    Extract per-subject intercept, threat slope, distance slope, cookie slope.
    """
    print("=" * 70)
    print("STAGE 1: Per-Subject Affect Regressions")
    print("=" * 70)

    anx = feelings[feelings['questionLabel'] == 'anxiety'].copy()
    conf = feelings[feelings['questionLabel'] == 'confidence'].copy()

    # Check probe design
    print(f"\n  Anxiety probes: {len(anx)}, Confidence probes: {len(conf)}")
    print(f"  Threats: {sorted(anx['threat'].round(1).unique())}")
    print(f"  Distances: {sorted(anx['distance'].unique())}")

    # Check cookie variation on probes
    for label, df in [('Anxiety', anx), ('Confidence', conf)]:
        cookie_vals = df['trialCookie_weight'].unique()
        print(f"  {label} cookie weights: {sorted(cookie_vals)}")

    results = {}
    for label, df in [('anx', anx), ('conf', conf)]:
        df = df.copy()
        df['T'] = df['threat']
        df['D'] = df['distance']
        df['is_heavy'] = (df['trialCookie_weight'] == 3.0).astype(float)

        # Check if cookie varies on probes
        cookie_varies = df.groupby('subj')['is_heavy'].std().mean() > 0.01

        subject_params = []
        for s, sdf in df.groupby('subj'):
            if len(sdf) < 8:
                continue

            # OLS per subject
            if cookie_varies:
                X = sm.add_constant(sdf[['T', 'D', 'is_heavy']].values)
                col_names = ['intercept', 'b_threat', 'b_distance', 'b_cookie']
            else:
                X = sm.add_constant(sdf[['T', 'D']].values)
                col_names = ['intercept', 'b_threat', 'b_distance']

            y = sdf['response'].values
            valid = np.isfinite(y)
            if valid.sum() < 5:
                continue

            try:
                ols = sm.OLS(y[valid], X[valid]).fit()
                params = {col_names[i]: ols.params[i] for i in range(len(col_names))}
                params['subj'] = s
                params['r2'] = ols.rsquared
                if not cookie_varies:
                    params['b_cookie'] = 0.0
                subject_params.append(params)
            except:
                continue

        param_df = pd.DataFrame(subject_params).set_index('subj')
        results[label] = param_df

        print(f"\n  {label.upper()} parameters ({len(param_df)} subjects):")
        for col in ['intercept', 'b_threat', 'b_distance', 'b_cookie', 'r2']:
            if col in param_df.columns:
                print(f"    {col:<14}: mean={param_df[col].mean():+.3f}, SD={param_df[col].std():.3f}")

    return results['anx'], results['conf']


def analysis_a(anx_params, conf_params, master):
    """Analysis A: Are anxiety and confidence informationally dissociable?"""
    print(f"\n{'=' * 70}")
    print("ANALYSIS A: Dissociation Structure")
    print("=" * 70)

    shared = sorted(set(anx_params.index) & set(conf_params.index))
    ap = anx_params.loc[shared]
    cp = conf_params.loc[shared]
    print(f"  N = {len(shared)}")

    # A1: Slope correlation structure
    print(f"\n  A1: Parameter correlations (anxiety vs confidence):")
    print(f"  {'Parameter':<16} {'r':>8} {'p':>10} {'Interpretation'}")
    print(f"  {'-' * 60}")

    pairs = [
        ('intercept', 'Baseline alignment'),
        ('b_threat', 'Threat sensitivity alignment'),
        ('b_distance', 'Distance sensitivity alignment'),
    ]
    for param, interp in pairs:
        r, p = pearsonr(ap[param], cp[param])
        sig = '*' if p < .05 else ' '
        print(f"  {param:<16} {r:>+8.3f} {p:>10.4f}{sig} {interp}")

    # A2: Unique variance in behavior
    print(f"\n  A2: Unique variance in outcomes:")
    m = master.loc[shared]

    for outcome, label in [('escape_rate', 'Escape'), ('pct_opt', 'Optimality'), ('earnings', 'Earnings')]:
        if outcome not in m.columns:
            continue
        y = m[outcome].values
        valid = np.isfinite(y)
        if valid.sum() < 50:
            continue

        # Model 1: anxiety slope only
        X1 = sm.add_constant(ap['b_threat'].values[valid])
        m1 = sm.OLS(y[valid], X1).fit()

        # Model 2: confidence slope only
        X2 = sm.add_constant(cp['b_threat'].values[valid])
        m2 = sm.OLS(y[valid], X2).fit()

        # Model 3: both
        X3 = sm.add_constant(np.column_stack([ap['b_threat'].values[valid],
                                               cp['b_threat'].values[valid]]))
        m3 = sm.OLS(y[valid], X3).fit()

        print(f"\n    {label}:")
        print(f"      Anx slope only: R²={m1.rsquared:.4f}")
        print(f"      Conf slope only: R²={m2.rsquared:.4f}")
        print(f"      Both: R²={m3.rsquared:.4f} (anx β={m3.params[1]:+.4f} p={m3.pvalues[1]:.4f}, "
              f"conf β={m3.params[2]:+.4f} p={m3.pvalues[2]:.4f})")


def analysis_b(anx_params, conf_params, master):
    """Analysis B: Do ω and κ predict specific affect slopes?"""
    print(f"\n{'=' * 70}")
    print("ANALYSIS B: Computational Parameters → Affect Slopes")
    print("=" * 70)

    shared = sorted(set(anx_params.index) & set(conf_params.index) & set(master.index))
    ap = anx_params.loc[shared]
    cp = conf_params.loc[shared]
    m = master.loc[shared]

    om_z = zscore(np.log(m['omega'].values))
    kap_z = zscore(np.log(m['kappa'].values))

    # B1: ω → confidence threat slope (not anxiety threat slope)
    print(f"\n  B1: ω → threat slopes")
    for param_z, param_name in [(om_z, 'ω'), (kap_z, 'κ')]:
        for label, slope_vals in [('anx b_threat', ap['b_threat'].values),
                                   ('conf b_threat', cp['b_threat'].values)]:
            r, p = pearsonr(param_z, slope_vals)
            sig = '*' if p < .05 else ' '
            print(f"    {param_name} → {label}: r={r:+.3f} (p={p:.4f}){sig}")

    # B2: κ → distance sensitivity of confidence
    print(f"\n  B2: κ → distance slopes")
    for param_z, param_name in [(om_z, 'ω'), (kap_z, 'κ')]:
        for label, slope_vals in [('anx b_distance', ap['b_distance'].values),
                                   ('conf b_distance', cp['b_distance'].values)]:
            r, p = pearsonr(param_z, slope_vals)
            sig = '*' if p < .05 else ' '
            print(f"    {param_name} → {label}: r={r:+.3f} (p={p:.4f}){sig}")

    # B summary: full regression table
    print(f"\n  B summary: Full regression (affect slope ~ ω + κ):")
    X = sm.add_constant(np.column_stack([om_z, kap_z]))
    for label, slope_vals in [('anx_b_threat', ap['b_threat'].values),
                               ('conf_b_threat', cp['b_threat'].values),
                               ('anx_b_distance', ap['b_distance'].values),
                               ('conf_b_distance', cp['b_distance'].values),
                               ('anx_intercept', ap['intercept'].values),
                               ('conf_intercept', cp['intercept'].values)]:
        ols = sm.OLS(slope_vals, X).fit()
        print(f"    {label:<18} R²={ols.rsquared:.4f}  "
              f"ω β={ols.params[1]:+.3f}(p={ols.pvalues[1]:.4f})  "
              f"κ β={ols.params[2]:+.3f}(p={ols.pvalues[2]:.4f})")


def analysis_c(anx_params, conf_params, master, psych):
    """Analysis C: Anxiety-confidence gap → overcaution and clinical."""
    print(f"\n{'=' * 70}")
    print("ANALYSIS C: The Anxiety-Confidence Gap")
    print("=" * 70)

    shared = sorted(set(anx_params.index) & set(conf_params.index) &
                    set(master.index) & set(psych.index))
    ap = anx_params.loc[shared]
    cp = conf_params.loc[shared]
    m = master.loc[shared]
    ps = psych.loc[shared]

    # C1: Compute gap parameters
    # gap(T) = (b0_anx + b1_anx*T) - (b0_conf + b1_conf*T)
    # gap_baseline = gap at T=0.1
    # gap_slope = b1_anx - b1_conf (how fast gap widens)
    gap_baseline = (ap['intercept'] + ap['b_threat'] * 0.1) - \
                   (cp['intercept'] + cp['b_threat'] * 0.1)
    gap_slope = ap['b_threat'].values - cp['b_threat'].values
    gap_at_09 = (ap['intercept'] + ap['b_threat'] * 0.9) - \
                (cp['intercept'] + cp['b_threat'] * 0.9)

    print(f"  N = {len(shared)}")
    print(f"\n  Gap baseline (T=0.1): mean={gap_baseline.mean():+.3f}, SD={gap_baseline.std():.3f}")
    print(f"  Gap slope (b1_anx - b1_conf): mean={gap_slope.mean():+.3f}, SD={np.std(gap_slope):.3f}")
    print(f"  Gap at T=0.9: mean={gap_at_09.mean():+.3f}, SD={gap_at_09.std():.3f}")

    gb_z = zscore(gap_baseline.values)
    gs_z = zscore(gap_slope)

    # C2: Gap → overcaution
    print(f"\n  C2: Gap predicts overcaution:")
    if 'oc_ratio' in m.columns and 'omega_z' in m.columns:
        om_z = m['omega_z'].values
        kap_z = m['kappa_z'].values

        # Model 1: just ω + κ
        X1 = sm.add_constant(np.column_stack([om_z, kap_z]))
        m1 = sm.OLS(m['oc_ratio'].values, X1).fit()

        # Model 2: ω + κ + gap_baseline
        X2 = sm.add_constant(np.column_stack([om_z, kap_z, gb_z]))
        m2 = sm.OLS(m['oc_ratio'].values, X2).fit()

        # Model 3: ω + κ + gap_baseline + gap_slope
        X3 = sm.add_constant(np.column_stack([om_z, kap_z, gb_z, gs_z]))
        m3 = sm.OLS(m['oc_ratio'].values, X3).fit()

        print(f"    ω + κ only:          R²={m1.rsquared:.4f}")
        print(f"    + gap_baseline:      R²={m2.rsquared:.4f} (ΔR²={m2.rsquared-m1.rsquared:+.4f}, "
              f"gap β={m2.params[3]:+.4f} p={m2.pvalues[3]:.4f})")
        print(f"    + gap_baseline+slope: R²={m3.rsquared:.4f}")
        for i, lab in enumerate(['const', 'ω', 'κ', 'gap_base', 'gap_slope']):
            sig = '*' if m3.pvalues[i] < .05 else ' '
            print(f"      {lab:<12} β={m3.params[i]:>+8.4f} p={m3.pvalues[i]:.4f}{sig}")

    # C2b: Gap → escape and earnings
    print(f"\n  C2b: Gap predicts outcomes:")
    for outcome, label in [('escape_rate', 'Escape'), ('earnings', 'Earnings'), ('pct_opt', 'Optimality')]:
        if outcome not in m.columns:
            continue
        y = m[outcome].values
        X = sm.add_constant(np.column_stack([om_z, kap_z, gb_z, gs_z]))
        ols = sm.OLS(y, X).fit()
        print(f"    {label}: R²={ols.rsquared:.4f}  gap_base β={ols.params[3]:+.4f}(p={ols.pvalues[3]:.4f})  "
              f"gap_slope β={ols.params[4]:+.4f}(p={ols.pvalues[4]:.4f})")

    # C3: Gap → clinical
    print(f"\n  C3: Gap predicts clinical symptoms:")
    clin_targets = ['STAI_State', 'STAI_Trait', 'OASIS_Total', 'DASS21_Anxiety',
                    'DASS21_Total', 'PHQ9_Total', 'STICSA_Total']

    print(f"  {'Measure':<18} {'R²(gap)':>8} {'β(base)':>8} {'p(base)':>8} {'β(slope)':>9} {'p(slope)':>9}")
    print(f"  {'-' * 60}")
    for clin in clin_targets:
        if clin not in ps.columns:
            continue
        y = ps[clin].values
        valid = np.isfinite(y)
        if valid.sum() < 50:
            continue
        X = sm.add_constant(np.column_stack([gb_z[valid], gs_z[valid]]))
        ols = sm.OLS(y[valid], X).fit()
        s1 = '*' if ols.pvalues[1] < .05 else ' '
        s2 = '*' if ols.pvalues[2] < .05 else ' '
        print(f"  {clin:<18} {ols.rsquared:>8.4f} {ols.params[1]:>+8.3f} {ols.pvalues[1]:>8.4f}{s1} "
              f"{ols.params[2]:>+9.3f} {ols.pvalues[2]:>9.4f}{s2}")

    # C4: Gap vs calibration head-to-head
    print(f"\n  C4: Gap vs calibration (head-to-head):")
    # Calibration = within-subject r(anxiety, threat)
    feelings = pd.read_csv(DATA_DIR / "feelings.csv")
    feelings = feelings[~feelings['subj'].isin(EXCLUDE)]
    anx_raw = feelings[feelings['questionLabel'] == 'anxiety']
    cal = {}
    for s in shared:
        sa = anx_raw[anx_raw['subj'] == s]
        if len(sa) >= 5:
            cal[s] = pearsonr(sa['threat'], sa['response'])[0]
    cal_vals = np.array([cal.get(s, np.nan) for s in shared])
    cal_valid = np.isfinite(cal_vals)

    for outcome, label in [('pct_opt', 'Optimality'), ('escape_rate', 'Escape'), ('earnings', 'Earnings')]:
        if outcome not in m.columns:
            continue
        y = m[outcome].values
        v = cal_valid & np.isfinite(y)

        # Calibration only
        X_c = sm.add_constant(cal_vals[v].reshape(-1, 1))
        m_c = sm.OLS(y[v], X_c).fit()

        # Gap only
        X_g = sm.add_constant(gb_z[v].reshape(-1, 1))
        m_g = sm.OLS(y[v], X_g).fit()

        # Both
        X_b = sm.add_constant(np.column_stack([cal_vals[v], gb_z[v]]))
        m_b = sm.OLS(y[v], X_b).fit()

        print(f"    {label}: cal R²={m_c.rsquared:.4f}, gap R²={m_g.rsquared:.4f}, "
              f"both R²={m_b.rsquared:.4f} (cal p={m_b.pvalues[1]:.4f}, gap p={m_b.pvalues[2]:.4f})")


def analysis_d(anx_params, conf_params, master, feelings):
    """Analysis D: Metacognitive accuracy."""
    print(f"\n{'=' * 70}")
    print("ANALYSIS D: Metacognitive Accuracy")
    print("=" * 70)

    shared = sorted(set(conf_params.index) & set(master.index))
    m = master.loc[shared]

    # Get per-probe model-predicted survival
    conf_raw = feelings[feelings['questionLabel'] == 'confidence'].copy()
    conf_raw = conf_raw[conf_raw['subj'].isin(shared)]

    gamma = CM2_POP['gamma']
    hazard = CM2_POP['hazard']
    sp = CM2_POP['sigma_sp']

    # For each probe trial, compute model-predicted survival
    conf_raw['req'] = np.where(conf_raw['trialCookie_weight'] == 3.0, 0.9, 0.4)
    conf_raw['R'] = conf_raw['trialCookie_rewardValue']
    conf_raw['D'] = conf_raw['distance']

    # Using population-mean omega for the "objective" survival
    # (or per-subject omega for "subjective" survival)
    speed = expit((conf_raw['req'] - 0.25 * conf_raw['req']) / sp)
    conf_raw['S_pop'] = np.exp(-hazard * conf_raw['threat'] ** gamma *
                                conf_raw['D'] / np.clip(speed, 0.01, None))

    # Per-subject omega for personalized survival
    om_map = m['omega'].to_dict()
    conf_raw['omega'] = conf_raw['subj'].map(om_map)
    conf_raw['S_personal'] = conf_raw['S_pop']  # S doesn't depend on omega directly
    # But expected fitness does:
    conf_raw['EV'] = (conf_raw['S_pop'] * conf_raw['R'] -
                      (1 - conf_raw['S_pop']) * conf_raw['omega'] * (conf_raw['R'] + C_PEN))

    # D1: Per-subject confidence accuracy
    # r(confidence, model-predicted survival)
    accuracy = {}
    accuracy_ev = {}
    for s, sdf in conf_raw.groupby('subj'):
        if len(sdf) >= 5:
            r_s, _ = pearsonr(sdf['response'], sdf['S_pop'])
            accuracy[s] = r_s
            r_ev, _ = pearsonr(sdf['response'], sdf['EV'])
            accuracy_ev[s] = r_ev

    acc_series = pd.Series(accuracy, name='conf_accuracy')
    acc_ev_series = pd.Series(accuracy_ev, name='conf_accuracy_EV')

    print(f"  Confidence accuracy (r with S_pop):")
    print(f"    mean = {acc_series.mean():+.3f}, SD = {acc_series.std():.3f}")
    print(f"    % positive: {(acc_series > 0).mean() * 100:.0f}%")

    print(f"\n  Confidence accuracy (r with EV):")
    print(f"    mean = {acc_ev_series.mean():+.3f}, SD = {acc_ev_series.std():.3f}")

    # Anxiety accuracy (r with threat)
    anx_raw = feelings[feelings['questionLabel'] == 'anxiety']
    anx_raw = anx_raw[anx_raw['subj'].isin(shared)]
    anx_accuracy = {}
    for s, sdf in anx_raw.groupby('subj'):
        if len(sdf) >= 5:
            r, _ = pearsonr(sdf['response'], sdf['threat'])
            anx_accuracy[s] = r
    anx_acc_series = pd.Series(anx_accuracy, name='anx_accuracy')

    print(f"\n  Anxiety accuracy (r with threat):")
    print(f"    mean = {anx_acc_series.mean():+.3f}, SD = {anx_acc_series.std():.3f}")

    # D2: Accuracy predicts error type
    print(f"\n  D2: Accuracy predicts errors:")
    shared2 = sorted(set(acc_series.index) & set(m.index))
    ca = acc_series.loc[shared2].values
    aa = anx_acc_series.reindex(shared2).values
    m2 = m.loc[shared2]

    if 'n_oc' in m2.columns:
        for acc_name, acc_vals in [('Conf accuracy', ca), ('Anx accuracy', aa)]:
            v = np.isfinite(acc_vals)
            if v.sum() < 30:
                continue
            r_oc, p_oc = pearsonr(acc_vals[v], m2['n_oc'].values[v])
            r_rk, p_rk = pearsonr(acc_vals[v], m2['n_rk'].values[v])
            print(f"    {acc_name} → n_overcautious: r={r_oc:+.3f} (p={p_oc:.4f})")
            print(f"    {acc_name} → n_reckless:     r={r_rk:+.3f} (p={p_rk:.4f})")

    # D3: Compare conf_accuracy and anx_accuracy as predictors
    print(f"\n  D3: Conf vs anx accuracy as predictors:")
    for outcome, label in [('escape_rate', 'Escape'), ('pct_opt', 'Optimality'), ('earnings', 'Earnings')]:
        if outcome not in m2.columns:
            continue
        y = m2[outcome].values
        v = np.isfinite(ca) & np.isfinite(aa) & np.isfinite(y)
        if v.sum() < 30:
            continue
        X = sm.add_constant(np.column_stack([ca[v], aa[v]]))
        ols = sm.OLS(y[v], X).fit()
        print(f"    {label}: R²={ols.rsquared:.4f}  "
              f"conf_acc β={ols.params[1]:+.4f}(p={ols.pvalues[1]:.4f})  "
              f"anx_acc β={ols.params[2]:+.4f}(p={ols.pvalues[2]:.4f})")

    # D4: Are the two accuracies correlated?
    v = np.isfinite(ca) & np.isfinite(aa)
    r_accs, p_accs = pearsonr(ca[v], aa[v])
    print(f"\n  r(conf_accuracy, anx_accuracy) = {r_accs:+.3f} (p={p_accs:.4f})")

    # Save accuracy measures
    acc_df = pd.DataFrame({
        'conf_accuracy': acc_series,
        'conf_accuracy_EV': acc_ev_series,
        'anx_accuracy': anx_acc_series
    })
    acc_df.to_csv(OUT_DIR / "metacognitive_accuracy.csv")

    return acc_series, anx_acc_series


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import time
    t0 = time.time()

    print("Loading data...")
    beh, feelings, psych, master = load_data()

    # Stage 1
    anx_params, conf_params = stage1_affect_regressions(feelings)

    # Save Stage 1
    anx_params.to_csv(OUT_DIR / "anx_params.csv")
    conf_params.to_csv(OUT_DIR / "conf_params.csv")
    print(f"\n  Saved to {OUT_DIR}")

    # Stage 2
    analysis_a(anx_params, conf_params, master)
    analysis_b(anx_params, conf_params, master)
    analysis_c(anx_params, conf_params, master, psych)
    analysis_d(anx_params, conf_params, master, feelings)

    print(f"\n{'=' * 70}")
    print(f"Total time: {time.time() - t0:.1f}s")
