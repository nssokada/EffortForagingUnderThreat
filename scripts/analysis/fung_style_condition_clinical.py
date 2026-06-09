"""
Fung-style condition-specific behavior × clinical scale interaction.

Fung et al. 2019 found that trait anxiety (STAI-Y) predicted flight-initiation
distance ONLY for the slow predator condition, not for fast or medium. The
condition-specific interaction was the headline.

Analog in our data: does trait anxiety / depression / apathy / fatigue predict
behavior or affect at a SPECIFIC threat level (or specific distance), and not at
others? We test this with a 3-way design:

  subject-level outcome: behavior_in_condition × trait_z

Tests:
  1. Per-subject P(heavy | T) × clinical scale interaction across threat levels
  2. Per-subject vigor | T × clinical scale interaction across threat levels
  3. Per-subject anxiety rating | T × clinical scale interaction (state × trait)
  4. Per-subject confidence rating | T × clinical scale interaction
  5. Affect reactivity slope (anxiety_slope_T, confidence_slope_T) → clinical
  6. Behavioral reactivity (choice shift, vigor shift across threat levels) → clinical

We focus on clinically anxiety-relevant scales (STAI_Trait, DASS21_Anxiety,
OASIS_Total, STICSA_Total) and apathy (AMI_Total, AMI_Behavioural).

Outputs:
  results/stats/clinical/fung_style_condition_clinical.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)
NB_DIR = REPO_ROOT / "notebooks" / "analysis"
sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import zscore

from load_data import load_both  # type: ignore


SAMPLES = {
    "exploratory": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
    "confirmatory": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
}

CLINICAL_SCALES = [
    "STAI_Trait", "DASS21_Anxiety", "OASIS_Total", "STICSA_Total",
    "AMI_Total", "AMI_Behavioural", "AMI_Social", "AMI_Emotional",
    "DASS21_Depression", "PHQ9_Total", "MFIS_Total",
]


def build_pooled():
    """Per-subject table: clinical scales + behavior & affect by threat level."""
    exp, conf = load_both()
    em = exp["master"].reset_index().rename(columns={"index": "subj"}).copy()
    cm = conf["master"].reset_index().rename(columns={"index": "subj"}).copy()
    em["sample"] = "exploratory"
    cm["sample"] = "confirmatory"
    master = pd.concat([em, cm], ignore_index=True)

    # Compute per-subject behavior by threat level from choice + vigor trial data
    rows_beh = []
    for sample, paths in SAMPLES.items():
        # Choice + vigor data lives in vigor table per the loader
        # Get the per-trial behavior from load_both()
        d = exp if sample == "exploratory" else conf
        vigor_df = d["vigor"][["subj", "T_round", "distance", "norm_rate", "is_heavy"]].copy()
        vigor_df["sample"] = sample
        rows_beh.append(vigor_df)
    beh = pd.concat(rows_beh, ignore_index=True)

    # Per-subject × T-level behavioral summaries — MUST include sample in groupby
    grp = beh.groupby(["subj", "sample", "T_round"]).agg(
        p_heavy=("is_heavy", "mean"),
        mean_vigor=("norm_rate", "mean"),
    ).reset_index()
    p_heavy_wide = grp.pivot_table(index=["subj", "sample"], columns="T_round", values="p_heavy").reset_index()
    vigor_wide = grp.pivot_table(index=["subj", "sample"], columns="T_round", values="mean_vigor").reset_index()
    p_heavy_wide.columns = ["subj", "sample"] + [f"p_heavy_T{c}" for c in p_heavy_wide.columns[2:]]
    vigor_wide.columns = ["subj", "sample"] + [f"vigor_T{c}" for c in vigor_wide.columns[2:]]

    # Per-subject affect-by-threat from feelings.csv — also MUST include sample
    rows_aff = []
    for sample, path in SAMPLES.items():
        f = pd.read_csv(Path(path) / "feelings.csv", low_memory=False)
        f = f[["subj", "threat", "questionLabel", "response"]].copy()
        f["sample"] = sample
        rows_aff.append(f)
    aff = pd.concat(rows_aff, ignore_index=True).dropna(subset=["response"])
    aff_grp = aff.groupby(["subj", "sample", "threat", "questionLabel"])["response"].mean().reset_index()
    aff_wide = aff_grp.pivot_table(
        index=["subj", "sample"], columns=["questionLabel", "threat"], values="response", aggfunc="first"
    ).reset_index()
    aff_wide.columns = ["subj", "sample"] + [f"{q}_T{t}" for q, t in aff_wide.columns[2:]]

    out = master.merge(p_heavy_wide, on=["subj", "sample"], how="left")
    out = out.merge(vigor_wide, on=["subj", "sample"], how="left")
    out = out.merge(aff_wide, on=["subj", "sample"], how="left")

    # Compute reactivity shifts (high T - low T): condition-specific behavioral deltas
    out["p_heavy_shift_THighLow"] = out.get("p_heavy_T0.9", np.nan) - out.get("p_heavy_T0.1", np.nan)
    out["vigor_shift_THighLow"] = out.get("vigor_T0.9", np.nan) - out.get("vigor_T0.1", np.nan)
    out["anxiety_shift_THighLow"] = out.get("anxiety_T0.9", np.nan) - out.get("anxiety_T0.1", np.nan)
    out["confidence_shift_THighLow"] = out.get("confidence_T0.9", np.nan) - out.get("confidence_T0.1", np.nan)
    return out


def fit_and_test(df, predictor, scale, sample_dummy=True):
    """Fit clinical ~ predictor (z) + sample. Return β, SE, p."""
    sub = df[[predictor, scale, "sample"]].dropna().copy()
    if len(sub) < 30:
        return dict(n=len(sub), beta=np.nan, se=np.nan, t=np.nan, p=np.nan)
    sub[f"{predictor}_z"] = zscore(sub[predictor].values, nan_policy="omit")
    sub[f"{scale}_z"] = zscore(sub[scale].values, nan_policy="omit")
    sub["sample_dummy"] = (sub["sample"] == "confirmatory").astype(int)
    if sample_dummy:
        X = sm.add_constant(sub[[f"{predictor}_z", "sample_dummy"]].values)
    else:
        X = sm.add_constant(sub[[f"{predictor}_z"]].values)
    res = sm.OLS(sub[f"{scale}_z"].values, X).fit()
    return dict(
        n=len(sub),
        beta=float(res.params[1]),
        se=float(res.bse[1]),
        t=float(res.tvalues[1]),
        p=float(res.pvalues[1]),
    )


def main():
    print("=" * 78)
    print("FUNG-STYLE CONDITION-SPECIFIC BEHAVIOR × CLINICAL ANALYSIS")
    print("=" * 78)
    df = build_pooled()
    print(f"\nPooled N: {len(df)}")
    print(f"Samples: exp {(df['sample']=='exploratory').sum()}, conf {(df['sample']=='confirmatory').sum()}")
    print(f"Clinical scales available: {[s for s in CLINICAL_SCALES if s in df.columns]}")

    rows = []

    # Group 1: behavior at SPECIFIC threat level → clinical
    print("\n" + "#" * 78)
    print("# 1. Per-condition behavior → clinical (does behavior at one threat level predict?)")
    print("#" * 78)
    predictors_per_condition = [
        "p_heavy_T0.1", "p_heavy_T0.5", "p_heavy_T0.9",
        "vigor_T0.1", "vigor_T0.5", "vigor_T0.9",
        "anxiety_T0.1", "anxiety_T0.5", "anxiety_T0.9",
        "confidence_T0.1", "confidence_T0.5", "confidence_T0.9",
    ]
    for pred in predictors_per_condition:
        if pred not in df.columns:
            continue
        for scale in CLINICAL_SCALES:
            if scale not in df.columns:
                continue
            r = fit_and_test(df, pred, scale)
            r.update(dict(group="per_condition", predictor=pred, scale=scale))
            rows.append(r)

    # Group 2: condition shifts (high-T minus low-T) → clinical
    print("\n" + "#" * 78)
    print("# 2. CONDITION SHIFTS (Δ across threat levels) → clinical")
    print("#" * 78)
    shifts = ["p_heavy_shift_THighLow", "vigor_shift_THighLow",
              "anxiety_shift_THighLow", "confidence_shift_THighLow"]
    for pred in shifts:
        if pred not in df.columns:
            continue
        print(f"\n  → predictor: {pred}")
        for scale in CLINICAL_SCALES:
            if scale not in df.columns:
                continue
            r = fit_and_test(df, pred, scale)
            r.update(dict(group="shift", predictor=pred, scale=scale))
            rows.append(r)
            sig = "★" if r["p"] < 0.05 else " "
            print(f"     {scale:24s}  β={r['beta']:+.3f} SE={r['se']:.3f} t={r['t']:+.2f} p={r['p']:.4g} (N={r['n']}) {sig}")

    # Group 3: per-subject affect reactivity slopes → clinical
    # These came from phenotype_metacog_slopes_subjects.csv
    print("\n" + "#" * 78)
    print("# 3. Affect reactivity SLOPES (slope on T, slope on D) → clinical")
    print("#" * 78)
    slopes_csv = REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv"
    if slopes_csv.exists():
        slopes_df = pd.read_csv(slopes_csv)
        # Merge clinical scales from master into slopes_df
        clin_cols = [c for c in CLINICAL_SCALES if c in df.columns]
        merged = slopes_df.merge(df[["subj", "sample"] + clin_cols], on=["subj", "sample"], how="left")
        slope_preds = ["confidence_slope_T", "confidence_slope_D", "confidence_intercept", "confidence_cal_T",
                       "anxiety_slope_T", "anxiety_slope_D", "anxiety_intercept", "anxiety_cal_T"]
        for pred in slope_preds:
            if pred not in merged.columns:
                continue
            print(f"\n  → predictor: {pred}")
            for scale in clin_cols:
                r = fit_and_test(merged, pred, scale)
                r.update(dict(group="reactivity_slope", predictor=pred, scale=scale))
                rows.append(r)
                sig = "★" if r["p"] < 0.05 else " "
                print(f"     {scale:24s}  β={r['beta']:+.3f} SE={r['se']:.3f} t={r['t']:+.2f} p={r['p']:.4g} (N={r['n']}) {sig}")
    else:
        print(f"  Slopes CSV not found: {slopes_csv}")

    # Save
    out_path = REPO_ROOT / "results" / "stats" / "clinical" / "fung_style_condition_clinical.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df = pd.DataFrame(rows)
    full_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Summary: which hits are significant, with FDR ─────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY: hits at p < 0.05 (uncorrected)")
    print("=" * 78)
    hits = full_df[full_df["p"] < 0.05].sort_values("p")
    print(f"\n  total tests: {len(full_df)}")
    print(f"  nominal hits (p < 0.05): {len(hits)}")
    print(f"  expected at α = 0.05 by chance: {0.05 * len(full_df):.1f}")
    bonf_alpha = 0.05 / len(full_df)
    bonf_hits = full_df[full_df["p"] < bonf_alpha]
    print(f"  Bonferroni α = {bonf_alpha:.6f}: {len(bonf_hits)} survive")
    if len(hits) > 0:
        print("\n  Top 20 hits:")
        print(hits[["group", "predictor", "scale", "beta", "p", "n"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
