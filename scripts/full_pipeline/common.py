"""
Common utilities, data loading, and constants for the full pipeline.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ast
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, ttest_rel, ttest_1samp, levene
from scipy.special import expit
from pathlib import Path

# ── Paths ──
DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("/workspace/results/stats/full_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Exclusions ──
EXCLUDE = [154, 197, 208]

# ── Epoch encounter times (from data, fixed per distance) ──
ENC_TIMES = {1: 2.5, 2: 3.5, 3: 5.0}


def load_choice_data():
    """Load free choice trials, apply exclusions."""
    beh = pd.read_csv(DATA_DIR / "behavior.csv")
    beh = beh[~beh['subj'].isin(EXCLUDE)].copy()
    beh['T_round'] = beh['threat'].round(1)
    beh['T_H'] = beh['distance_H'].map({1: 5.0, 2: 7.0, 3: 9.0})
    beh['effort_reqT'] = beh['effort_H'] * beh['T_H'] - 0.4 * 5.0
    beh['trial_number'] = beh.groupby('subj').cumcount() + 1
    beh['current_score'] = beh.groupby('subj')['choice'].cumsum().shift(1, fill_value=0)
    return beh


def load_all_trials():
    """Load behavior_rich (all 81 trials), apply exclusions, add epoch columns."""
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)].copy()
    beh['T_round'] = beh['threat'].round(1)
    beh['actual_dist'] = beh['startDistance'].map({5: 1, 7: 2, 9: 3})
    beh['is_heavy'] = (beh['trialCookie_weight'] == 3.0).astype(int)
    beh['actual_req'] = np.where(beh['is_heavy'] == 1, 0.9, 0.4)
    beh['enc_t'] = pd.to_numeric(beh['encounterTime'], errors='coerce')
    beh['strike_t'] = pd.to_numeric(beh['strike_time'], errors='coerce')
    beh['is_attack'] = beh['isAttackTrial'].astype(int)
    return beh


def load_psych():
    """Load psychiatric battery, apply exclusions."""
    psych = pd.read_csv(DATA_DIR / "psych.csv")
    return psych[~psych['subj'].isin(EXCLUDE)].copy()


def load_feelings():
    """Load probe ratings, apply exclusions."""
    feelings = pd.read_csv(DATA_DIR / "feelings.csv")
    return feelings[~feelings['subj'].isin(EXCLUDE)].copy()


def compute_epoch_vigor(beh_rich):
    """Compute normalized epoch-level vigor for all trials.
    Returns DataFrame with one row per trial, columns for each epoch's vigor.
    """
    records = []
    for _, row in beh_rich.iterrows():
        try:
            pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
            if len(pt) < 5:
                continue
            enc = row['enc_t']
            strike = row['strike_t']
            cal = row['calibrationMax']
            if pd.isna(enc) or enc <= 0 or cal <= 0:
                continue

            def epoch_vigor(epoch_pts):
                if len(epoch_pts) < 4:
                    return np.nan
                ipis = np.diff(epoch_pts)
                ipis = ipis[ipis > 0.01]
                if len(ipis) < 3:
                    return np.nan
                return np.median((1.0 / ipis) / cal)

            r = {
                'subj': row['subj'], 'trial': row['trial'],
                'T_round': row['T_round'], 'distance': float(row['actual_dist']),
                'cookie_type': float(row['is_heavy']),
                'is_attack': row['is_attack'],
                'predator_probability': row['T_round'],
                'antic_vigor': epoch_vigor(pt[pt < enc]),
                'react_vigor': epoch_vigor(pt[(pt >= enc) & (pt < enc + 2)]),
                'term_vigor': (epoch_vigor(pt[(pt >= strike - 2) & (pt <= strike)])
                               if row['is_attack'] == 1 and not pd.isna(strike) and strike > enc + 2
                               else np.nan),
            }
            records.append(r)
        except:
            pass

    df = pd.DataFrame(records)
    df['trial_number'] = df.groupby('subj').cumcount() + 1
    return df


def residualize_epoch(df, col):
    """Residualize an epoch vigor column on cookie_type + trial_number."""
    edf = df.dropna(subset=[col]).copy()
    edf['tn'] = edf.groupby('subj').cumcount() + 1
    try:
        m = smf.mixedlm(f"{col} ~ cookie_type + tn", edf, groups=edf["subj"],
                        re_formula="~cookie_type").fit(reml=False)
    except:
        m = smf.mixedlm(f"{col} ~ cookie_type + tn", edf, groups=edf["subj"]).fit(reml=False)
    edf['resid'] = m.resid
    return edf[['subj', 'trial', 'resid']].rename(columns={'resid': f'{col}_resid'})


def print_table(title, model, terms=None):
    """Print a fixed-effects table from a statsmodels mixed model."""
    print(f"\n  {title}")
    if terms is None:
        terms = model.fe_params.index
    print(f"  {'Term':<35} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>10}")
    print(f"  {'-' * 73}")
    for t in terms:
        coef = model.fe_params.get(t, np.nan)
        se = model.bse_fe.get(t, np.nan)
        z = model.tvalues.get(t, np.nan)
        p = model.pvalues.get(t, np.nan)
        sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ("." if p < .1 else "")))
        print(f"  {t:<35} {coef:>8.4f} {se:>8.4f} {z:>8.3f} {p:>10.4f} {sig}")


def save_df(df, name):
    """Save a DataFrame to the output directory."""
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
