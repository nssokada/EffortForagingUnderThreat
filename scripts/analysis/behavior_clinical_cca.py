"""
CCA between behavioral response features and mental health + affect features.

The univariate tests have been mostly null. CCA asks: is there a multivariate
LINEAR COMBINATION of behavioral signatures that strongly correlates with a
linear combination of clinical / affect features?

Behavioral features per subject (computed from choice + vigor data):
  Choice GLM (P(heavy) ~ T + D + T*D):
    choice_intercept, choice_β_T, choice_β_D, choice_β_TxD
  Vigor GLM (vigor ~ T + D + T*D + cookie):
    vigor_intercept, vigor_β_T, vigor_β_D, vigor_β_TxD, vigor_β_cookie
  Trial-to-trial:
    choice_autocorr, vigor_autocorr, vigor_sd
  Outcome-conditional:
    p_heavy_after_capture, p_heavy_after_escape,
    vigor_after_capture, vigor_after_escape

Mental health + affect side per subject:
  Clinical: DASS21_Anx/Dep/Stress, OASIS, STICSA, AMI_Total, AMI_Beh,
            AMI_Social, AMI_Emo, MFIS, PHQ9
  Affect: anxiety_intercept, anxiety_slope_T, anxiety_slope_D,
          confidence_intercept, confidence_slope_T, confidence_slope_D

Procedure:
  1. Within each sample, fit CCA on behavior × (clinical+affect)
  2. Report canonical correlations and which features load on the top components
  3. Cross-sample replication: project conf data onto exp canonical components
     and check if correlation magnitudes survive

Output: results/stats/affect_analysis/behavior_clinical_cca.csv
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
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA

from load_data import load_both  # type: ignore


SAMPLES = {
    "exploratory": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
    "confirmatory": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
}

CLINICAL = [
    "DASS21_Anxiety", "DASS21_Depression", "DASS21_Stress",
    "OASIS_Total", "STICSA_Total",
    "AMI_Behavioural", "AMI_Social", "AMI_Emotional",
    "MFIS_Total", "PHQ9_Total",
]
AFFECT = [
    "anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
    "confidence_intercept", "confidence_slope_T", "confidence_slope_D",
]


def per_subject_behavior_features():
    """Compute per-subject behavioral features from choice + vigor data."""
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        v = d["vigor"][["subj", "T_round", "distance", "is_heavy", "norm_rate"]].copy()
        v["sample"] = sample
        beh = v.dropna(subset=["norm_rate"]).copy()

        # Per-subject features
        for subj, g in beh.groupby("subj"):
            row = {"subj": int(subj), "sample": sample}

            # Choice GLM: P(heavy) ~ T + D + T*D
            if g["is_heavy"].nunique() > 1 and len(g) > 10:
                X = pd.DataFrame({
                    "T": g["T_round"].astype(float).values,
                    "D": g["distance"].astype(float).values,
                })
                X["TxD"] = X["T"] * X["D"]
                X = sm.add_constant(X)
                try:
                    # Logistic regression on choice
                    res = sm.GLM(g["is_heavy"].values, X.values,
                                 family=sm.families.Binomial()).fit()
                    row["choice_intercept"] = float(res.params[0])
                    row["choice_b_T"] = float(res.params[1])
                    row["choice_b_D"] = float(res.params[2])
                    row["choice_b_TxD"] = float(res.params[3])
                except Exception:
                    pass

            # Vigor GLM: vigor ~ T + D + T*D + cookie
            if len(g) > 10:
                Xv = pd.DataFrame({
                    "T": g["T_round"].astype(float).values,
                    "D": g["distance"].astype(float).values,
                    "cookie": g["is_heavy"].values.astype(float),
                })
                Xv["TxD"] = Xv["T"] * Xv["D"]
                Xv = sm.add_constant(Xv)
                try:
                    res = sm.OLS(g["norm_rate"].values, Xv.values).fit()
                    row["vigor_intercept"] = float(res.params[0])
                    row["vigor_b_T"] = float(res.params[1])
                    row["vigor_b_D"] = float(res.params[2])
                    row["vigor_b_cookie"] = float(res.params[3])
                    row["vigor_b_TxD"] = float(res.params[4])
                    row["vigor_sd"] = float(g["norm_rate"].std())
                except Exception:
                    pass

            # Trial-to-trial dynamics
            g_sorted = g.sort_index()
            try:
                ch = g_sorted["is_heavy"].astype(float).values
                if len(ch) > 5 and ch.std() > 0:
                    row["choice_autocorr"] = float(np.corrcoef(ch[:-1], ch[1:])[0, 1])
                vg = g_sorted["norm_rate"].values
                if len(vg) > 5 and vg.std() > 0:
                    row["vigor_autocorr"] = float(np.corrcoef(vg[:-1], vg[1:])[0, 1])
            except Exception:
                pass

            rows.append(row)
    return pd.DataFrame(rows)


def get_master_clinical_affect():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows.append(m)
    master = pd.concat(rows, ignore_index=True)

    # Affect slopes from existing pipeline
    slopes = pd.read_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv")
    keep_affect = [c for c in AFFECT if c in slopes.columns]
    out = master.merge(slopes[["subj", "sample"] + keep_affect], on=["subj", "sample"], how="left")
    return out


def cca_within_sample(df_behavior, df_clinical, sample_name, beh_cols, ca_cols):
    """Fit CCA on standardized features within one sample."""
    sub = df_behavior[df_behavior["sample"] == sample_name].merge(
        df_clinical[df_clinical["sample"] == sample_name][["subj", "sample"] + ca_cols],
        on=["subj", "sample"], how="inner")
    sub = sub[["subj"] + beh_cols + ca_cols].dropna()
    if len(sub) < 50:
        print(f"  [{sample_name}] not enough complete cases (N={len(sub)})")
        return None
    X = sub[beh_cols].apply(lambda c: zscore(c, nan_policy="omit"), axis=0).values
    Y = sub[ca_cols].apply(lambda c: zscore(c, nan_policy="omit"), axis=0).values
    # Cap n_components at min(features, samples - 1)
    n_comp = min(len(beh_cols), len(ca_cols), len(sub) - 1, 5)
    cca = CCA(n_components=n_comp, max_iter=2000)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    # Canonical correlations
    rs = [float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(n_comp)]
    return {
        "cca": cca, "X_c": X_c, "Y_c": Y_c, "rs": rs,
        "n": len(sub), "beh_cols": beh_cols, "ca_cols": ca_cols,
        "X_loadings": cca.x_loadings_, "Y_loadings": cca.y_loadings_,
        "subj_index": sub["subj"].values,
    }


def main():
    print("=" * 78)
    print("CCA: behavioral features × (clinical + affect) features")
    print("=" * 78)

    print("\nComputing per-subject behavioral features...")
    behavior = per_subject_behavior_features()
    print(f"  behavior rows: {len(behavior)}")
    print(f"  behavior columns: {[c for c in behavior.columns if c not in ('subj', 'sample')]}")

    print("\nLoading clinical + affect features...")
    clinical_affect = get_master_clinical_affect()
    print(f"  N: {len(clinical_affect)}")

    # Define complete feature sets
    beh_cols = [
        "choice_intercept", "choice_b_T", "choice_b_D", "choice_b_TxD",
        "vigor_intercept", "vigor_b_T", "vigor_b_D", "vigor_b_TxD", "vigor_b_cookie",
        "vigor_sd", "choice_autocorr", "vigor_autocorr",
    ]
    beh_cols = [c for c in beh_cols if c in behavior.columns]
    ca_cols_clin = [c for c in CLINICAL if c in clinical_affect.columns]
    ca_cols_aff = [c for c in AFFECT if c in clinical_affect.columns]
    ca_cols_all = ca_cols_clin + ca_cols_aff

    print(f"\nBehavioral features ({len(beh_cols)}): {beh_cols}")
    print(f"Clinical features ({len(ca_cols_clin)}): {ca_cols_clin}")
    print(f"Affect features ({len(ca_cols_aff)}): {ca_cols_aff}")

    rows_summary = []

    # ── CCA against CLINICAL only ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CCA #1: behavior × CLINICAL ONLY")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        print(f"\n  --- {sample} ---")
        res = cca_within_sample(behavior, clinical_affect, sample, beh_cols, ca_cols_clin)
        if res is None: continue
        print(f"    N={res['n']}, canonical correlations: " +
              ", ".join(f"{r:+.3f}" for r in res["rs"]))
        # Top component loadings
        print(f"    Top component (r = {res['rs'][0]:+.3f}):")
        print(f"      behavior loadings:")
        for i, col in enumerate(beh_cols):
            print(f"        {col:24s}  {res['X_loadings'][i, 0]:+.3f}")
        print(f"      clinical loadings:")
        for i, col in enumerate(ca_cols_clin):
            print(f"        {col:24s}  {res['Y_loadings'][i, 0]:+.3f}")
        rows_summary.append({"analysis": "behavior_clinical", "sample": sample,
                             "n": res["n"], "r1": res["rs"][0],
                             "r2": res["rs"][1] if len(res["rs"]) > 1 else None,
                             "r3": res["rs"][2] if len(res["rs"]) > 2 else None})

    # ── CCA against AFFECT only ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CCA #2: behavior × AFFECT ONLY")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        print(f"\n  --- {sample} ---")
        res = cca_within_sample(behavior, clinical_affect, sample, beh_cols, ca_cols_aff)
        if res is None: continue
        print(f"    N={res['n']}, canonical correlations: " +
              ", ".join(f"{r:+.3f}" for r in res["rs"]))
        print(f"    Top component (r = {res['rs'][0]:+.3f}):")
        print(f"      behavior loadings:")
        for i, col in enumerate(beh_cols):
            print(f"        {col:24s}  {res['X_loadings'][i, 0]:+.3f}")
        print(f"      affect loadings:")
        for i, col in enumerate(ca_cols_aff):
            print(f"        {col:24s}  {res['Y_loadings'][i, 0]:+.3f}")
        rows_summary.append({"analysis": "behavior_affect", "sample": sample,
                             "n": res["n"], "r1": res["rs"][0],
                             "r2": res["rs"][1] if len(res["rs"]) > 1 else None,
                             "r3": res["rs"][2] if len(res["rs"]) > 2 else None})

    # ── CCA against ALL (clinical + affect) ──────────────────────────────
    print("\n" + "=" * 78)
    print("CCA #3: behavior × CLINICAL + AFFECT combined")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        print(f"\n  --- {sample} ---")
        res = cca_within_sample(behavior, clinical_affect, sample, beh_cols, ca_cols_all)
        if res is None: continue
        print(f"    N={res['n']}, canonical correlations: " +
              ", ".join(f"{r:+.3f}" for r in res["rs"]))
        print(f"    Top component (r = {res['rs'][0]:+.3f}):")
        print(f"      top behavior loadings (|loading| > 0.2):")
        for i, col in enumerate(beh_cols):
            if abs(res['X_loadings'][i, 0]) > 0.2:
                print(f"        {col:24s}  {res['X_loadings'][i, 0]:+.3f}")
        print(f"      top clinical/affect loadings (|loading| > 0.2):")
        for i, col in enumerate(ca_cols_all):
            if abs(res['Y_loadings'][i, 0]) > 0.2:
                print(f"        {col:24s}  {res['Y_loadings'][i, 0]:+.3f}")
        rows_summary.append({"analysis": "behavior_clinical_affect", "sample": sample,
                             "n": res["n"], "r1": res["rs"][0],
                             "r2": res["rs"][1] if len(res["rs"]) > 1 else None,
                             "r3": res["rs"][2] if len(res["rs"]) > 2 else None})

    # ── Cross-sample replication for the CLINICAL-only CCA ───────────────
    print("\n" + "=" * 78)
    print("CROSS-SAMPLE REPLICATION (clinical-only)")
    print("=" * 78)
    print("Fit canonical model on exploratory, project confirmatory, check correlations")
    res_exp = cca_within_sample(behavior, clinical_affect, "exploratory", beh_cols, ca_cols_clin)
    if res_exp is not None:
        # Get confirmatory data
        df_conf = behavior[behavior["sample"] == "confirmatory"].merge(
            clinical_affect[clinical_affect["sample"] == "confirmatory"][["subj", "sample"] + ca_cols_clin],
            on=["subj", "sample"], how="inner")
        df_conf = df_conf[["subj"] + beh_cols + ca_cols_clin].dropna()
        X_conf = df_conf[beh_cols].apply(lambda c: zscore(c, nan_policy="omit"), axis=0).values
        Y_conf = df_conf[ca_cols_clin].apply(lambda c: zscore(c, nan_policy="omit"), axis=0).values
        X_c_conf, Y_c_conf = res_exp["cca"].transform(X_conf, Y_conf)
        cross_rs = [float(np.corrcoef(X_c_conf[:, i], Y_c_conf[:, i])[0, 1])
                    for i in range(X_c_conf.shape[1])]
        print(f"  Exploratory in-sample r's: " + ", ".join(f"{r:+.3f}" for r in res_exp["rs"]))
        print(f"  Confirmatory projected r's: " + ", ".join(f"{r:+.3f}" for r in cross_rs))
        rows_summary.append({"analysis": "behavior_clinical_xval",
                             "sample": "confirmatory_projection", "n": len(df_conf),
                             "r1": cross_rs[0],
                             "r2": cross_rs[1] if len(cross_rs) > 1 else None,
                             "r3": cross_rs[2] if len(cross_rs) > 2 else None})

    # ── Cross-sample replication for the AFFECT-only CCA ───────────────
    print("\n" + "=" * 78)
    print("CROSS-SAMPLE REPLICATION (affect-only)")
    print("=" * 78)
    res_exp = cca_within_sample(behavior, clinical_affect, "exploratory", beh_cols, ca_cols_aff)
    if res_exp is not None:
        df_conf = behavior[behavior["sample"] == "confirmatory"].merge(
            clinical_affect[clinical_affect["sample"] == "confirmatory"][["subj", "sample"] + ca_cols_aff],
            on=["subj", "sample"], how="inner")
        df_conf = df_conf[["subj"] + beh_cols + ca_cols_aff].dropna()
        X_conf = df_conf[beh_cols].apply(lambda c: zscore(c, nan_policy="omit"), axis=0).values
        Y_conf = df_conf[ca_cols_aff].apply(lambda c: zscore(c, nan_policy="omit"), axis=0).values
        X_c_conf, Y_c_conf = res_exp["cca"].transform(X_conf, Y_conf)
        cross_rs = [float(np.corrcoef(X_c_conf[:, i], Y_c_conf[:, i])[0, 1])
                    for i in range(X_c_conf.shape[1])]
        print(f"  Exploratory in-sample r's: " + ", ".join(f"{r:+.3f}" for r in res_exp["rs"]))
        print(f"  Confirmatory projected r's: " + ", ".join(f"{r:+.3f}" for r in cross_rs))
        rows_summary.append({"analysis": "behavior_affect_xval",
                             "sample": "confirmatory_projection", "n": len(df_conf),
                             "r1": cross_rs[0],
                             "r2": cross_rs[1] if len(cross_rs) > 1 else None,
                             "r3": cross_rs[2] if len(cross_rs) > 2 else None})

    out = REPO_ROOT / "results" / "stats" / "affect_analysis" / "behavior_clinical_cca.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_summary).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
