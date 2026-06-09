"""
Multivariate analysis: do anxiety+confidence (or clinical scales) JOINTLY
explain (ω, κ) configuration?

Three sets of tests, each within sample with cross-sample replication checks:

  TEST 1: MMR — (ω_z, κ_z) ~ affect features
    Tests Pillai's trace / Wilks' lambda for joint significance.
    Reports per-predictor effects on each of ω and κ.

  TEST 2: MMR — (ω_z, κ_z) ~ clinical scales
    Same multivariate inference.

  TEST 3: CCA with (ω, κ) as one set
    Canonical components between parameter pair and affect/clinical features.
    Identifies the direction in (ω, κ) space that the predictor set best
    explains, and which predictors load on that direction.

Cross-sample replication:
  - Fit MMR on exp, test confirmatory predictions
  - CCA: fit on exp, project to conf, check correlations

Output: results/stats/affect_analysis/multivariate_omega_kappa.csv
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


def build_df():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows.append(m)
    master = pd.concat(rows, ignore_index=True)
    master["omega_z"] = np.nan
    master["kappa_z"] = np.nan
    for s in master["sample"].unique():
        mask = master["sample"] == s
        master.loc[mask, "omega_z"] = zscore(np.log(master.loc[mask, "omega"]).values)
        master.loc[mask, "kappa_z"] = zscore(np.log(master.loc[mask, "kappa"]).values)

    slopes = pd.read_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv")
    aff_cols = [c for c in AFFECT if c in slopes.columns]
    df = master.merge(slopes[["subj", "sample"] + aff_cols], on=["subj", "sample"], how="left")
    return df


def mmr_with_pillai(df, predictors, sample_name):
    """Multivariate multiple regression (ω_z, κ_z) ~ predictors.

    Computes Pillai's trace, Wilks' lambda, Hotelling's T2 manually from
    the cross-product matrices. Tests the OVERALL joint significance of the
    full predictor set (not per-predictor), since per-predictor multivariate
    tests with small Y (2-dim) are essentially just per-parameter univariate.
    """
    sub = df[df["sample"] == sample_name].copy()
    sub = sub[["omega_z", "kappa_z"] + predictors].dropna()
    if len(sub) < 30:
        return None
    for p in predictors:
        sub[p] = zscore(sub[p].values, nan_policy="omit")

    Y = sub[["omega_z", "kappa_z"]].values
    X = sm.add_constant(sub[predictors].values)
    n = len(sub); p = X.shape[1] - 1; m = Y.shape[1]  # m = 2 outcomes

    # Fit full and reduced (intercept-only) models
    B_full = np.linalg.lstsq(X, Y, rcond=None)[0]
    Y_pred_full = X @ B_full
    E = (Y - Y_pred_full).T @ (Y - Y_pred_full)  # error SSCP

    X0 = X[:, [0]]  # intercept only
    B0 = np.linalg.lstsq(X0, Y, rcond=None)[0]
    Y_pred0 = X0 @ B0
    H_total = (Y_pred_full - Y_pred0).T @ (Y_pred_full - Y_pred0)  # hypothesis SSCP

    # Multivariate statistics from eigenvalues of E^-1 H
    inv_E = np.linalg.pinv(E)
    eigvals = np.linalg.eigvals(inv_E @ H_total).real
    eigvals = np.clip(eigvals, 0, None)
    pillai = np.sum(eigvals / (1 + eigvals))
    wilks = np.prod(1 / (1 + eigvals))
    hotelling = np.sum(eigvals)

    # Convert Pillai to F-stat (Rao approximation)
    s = min(p, m); df1 = max(p, m); df2 = n - max(p, m) - 1
    try:
        F_pillai = (pillai / s) / ((df1 - pillai) / df2) if (df1 - pillai) > 0 else np.nan
        # df numerator and denominator approx (Pillai)
        num_df = s * df1
        den_df = s * df2
        from scipy.stats import f as f_dist
        p_pillai = 1 - f_dist.cdf(F_pillai, num_df, den_df)
    except Exception:
        F_pillai, num_df, den_df, p_pillai = np.nan, np.nan, np.nan, np.nan

    # Per-parameter univariate
    res_om = sm.OLS(sub["omega_z"].values, X).fit()
    res_kp = sm.OLS(sub["kappa_z"].values, X).fit()
    return {
        "n": n, "p_predictors": p,
        "pillai_stat": float(pillai), "pillai_F": float(F_pillai),
        "pillai_num_df": float(num_df), "pillai_den_df": float(den_df),
        "pillai_p": float(p_pillai),
        "wilks": float(wilks), "hotelling": float(hotelling),
        "omega_coefs": list(zip(predictors,
                                 [float(res_om.params[i+1]) for i in range(len(predictors))],
                                 [float(res_om.pvalues[i+1]) for i in range(len(predictors))])),
        "kappa_coefs": list(zip(predictors,
                                 [float(res_kp.params[i+1]) for i in range(len(predictors))],
                                 [float(res_kp.pvalues[i+1]) for i in range(len(predictors))])),
        "omega_R2": float(res_om.rsquared),
        "kappa_R2": float(res_kp.rsquared),
        "omega_F_p": float(res_om.f_pvalue), "kappa_F_p": float(res_kp.f_pvalue),
    }


def cca_params_vs(df, predictors, sample_name):
    sub = df[df["sample"] == sample_name].copy()
    sub = sub[["omega_z", "kappa_z"] + predictors].dropna()
    if len(sub) < 30:
        return None
    X = sub[["omega_z", "kappa_z"]].values
    Y = sub[predictors].apply(lambda c: zscore(c, nan_policy="omit"), axis=0).values
    n_comp = min(2, len(predictors), len(sub) - 1)
    cca = CCA(n_components=n_comp, max_iter=2000)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    rs = [float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(n_comp)]
    return {
        "n": len(sub), "rs": rs,
        "X_loadings": cca.x_loadings_, "Y_loadings": cca.y_loadings_,
        "cca": cca, "predictors": predictors,
        "subj_index": sub.index.values,
    }


def cross_sample_cca(df, predictors):
    """Fit CCA on exploratory, project confirmatory, return cross-sample correlations."""
    res_exp = cca_params_vs(df, predictors, "exploratory")
    if res_exp is None: return None
    sub_c = df[df["sample"] == "confirmatory"].copy()
    sub_c = sub_c[["omega_z", "kappa_z"] + predictors].dropna()
    X = sub_c[["omega_z", "kappa_z"]].values
    Y = sub_c[predictors].apply(lambda c: zscore(c, nan_policy="omit"), axis=0).values
    X_c, Y_c = res_exp["cca"].transform(X, Y)
    cross_rs = [float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1])
                for i in range(X_c.shape[1])]
    return {"exp_rs": res_exp["rs"], "conf_projected_rs": cross_rs, "n_conf": len(sub_c)}


def print_mmr_result(res, label):
    if res is None:
        print(f"  [{label}] no result")
        return
    print(f"  [{label}] N = {res['n']}, p = {res['p_predictors']} predictors")
    print(f"    Univariate R²: ω = {res['omega_R2']:.4f} (F p = {res['omega_F_p']:.4g}), "
          f"κ = {res['kappa_R2']:.4f} (F p = {res['kappa_F_p']:.4g})")
    sig = "★" if res['pillai_p'] < 0.05 else " "
    sig = "★★" if res['pillai_p'] < 0.01 else sig
    sig = "★★★" if res['pillai_p'] < 0.001 else sig
    print(f"    JOINT MULTIVARIATE test (Pillai's trace):")
    print(f"      Pillai = {res['pillai_stat']:.4f}, F({res['pillai_num_df']:.0f}, {res['pillai_den_df']:.0f}) = "
          f"{res['pillai_F']:.2f}, p = {res['pillai_p']:.4g} {sig}")
    print(f"      Wilks' lambda = {res['wilks']:.4f}, Hotelling = {res['hotelling']:.4f}")
    print("    Per-parameter coefficients (univariate):")
    for (pred, b, pv) in res["omega_coefs"]:
        sig = " ★" if pv < 0.05 else ""
        print(f"      ω: {pred:32s} β = {b:+.3f} p = {pv:.4g}{sig}")
    for (pred, b, pv) in res["kappa_coefs"]:
        sig = " ★" if pv < 0.05 else ""
        print(f"      κ: {pred:32s} β = {b:+.3f} p = {pv:.4g}{sig}")


def main():
    print("=" * 78)
    print("MULTIVARIATE (ω, κ) ANALYSIS — MMR + CCA")
    print("=" * 78)

    df = build_df()
    print(f"\nN: {len(df)} (exp {(df['sample']=='exploratory').sum()}, conf {(df['sample']=='confirmatory').sum()})")

    rows = []

    # ── TEST 1: MMR — (ω, κ) ~ affect features ───────────────────────────
    print("\n" + "=" * 78)
    print("TEST 1: MMR — (ω, κ) ~ AFFECT features")
    print("=" * 78)
    aff_preds = [c for c in AFFECT if c in df.columns]
    for sample in ["exploratory", "confirmatory"]:
        print(f"\n--- {sample} ---")
        res = mmr_with_pillai(df, aff_preds, sample)
        print_mmr_result(res, sample)
        if res is not None:
            rows.append({"test": "mmr_affect", "sample": sample, "n": res["n"],
                         "pillai_stat": res["pillai_stat"], "pillai_F": res["pillai_F"],
                         "pillai_p": res["pillai_p"], "wilks": res["wilks"],
                         "omega_R2": res["omega_R2"], "kappa_R2": res["kappa_R2"],
                         "omega_F_p": res["omega_F_p"], "kappa_F_p": res["kappa_F_p"]})

    # ── TEST 2: MMR — (ω, κ) ~ clinical scales ───────────────────────────
    print("\n" + "=" * 78)
    print("TEST 2: MMR — (ω, κ) ~ CLINICAL scales")
    print("=" * 78)
    clin_preds = [c for c in CLINICAL if c in df.columns]
    for sample in ["exploratory", "confirmatory"]:
        print(f"\n--- {sample} ---")
        res = mmr_with_pillai(df, clin_preds, sample)
        print_mmr_result(res, sample)
        if res is not None:
            rows.append({"test": "mmr_clinical", "sample": sample, "n": res["n"],
                         "pillai_stat": res["pillai_stat"], "pillai_F": res["pillai_F"],
                         "pillai_p": res["pillai_p"], "wilks": res["wilks"],
                         "omega_R2": res["omega_R2"], "kappa_R2": res["kappa_R2"],
                         "omega_F_p": res["omega_F_p"], "kappa_F_p": res["kappa_F_p"]})

    # ── TEST 3: CCA with (ω, κ) as outcome set ───────────────────────────
    print("\n" + "=" * 78)
    print("TEST 3: CCA — (ω, κ) ↔ predictor sets, cross-sample replication")
    print("=" * 78)

    print("\n  --- CCA: (ω, κ) ↔ AFFECT features ---")
    for sample in ["exploratory", "confirmatory"]:
        res = cca_params_vs(df, aff_preds, sample)
        if res is None: continue
        print(f"\n    [{sample}] N={res['n']}, canonical r's: {res['rs']}")
        for k in range(len(res['rs'])):
            print(f"    Component {k+1} (r = {res['rs'][k]:+.3f}):")
            print(f"      (ω, κ) loadings:")
            for i, name in enumerate(["omega_z", "kappa_z"]):
                print(f"        {name:12s} {res['X_loadings'][i, k]:+.3f}")
            print(f"      affect loadings:")
            for i, name in enumerate(aff_preds):
                print(f"        {name:28s} {res['Y_loadings'][i, k]:+.3f}")
    print("\n  Cross-sample replication (affect):")
    cs = cross_sample_cca(df, aff_preds)
    if cs is not None:
        print(f"    Exp in-sample r's: {cs['exp_rs']}")
        print(f"    Conf projected r's: {cs['conf_projected_rs']}")

    print("\n  --- CCA: (ω, κ) ↔ CLINICAL scales ---")
    for sample in ["exploratory", "confirmatory"]:
        res = cca_params_vs(df, clin_preds, sample)
        if res is None: continue
        print(f"\n    [{sample}] N={res['n']}, canonical r's: {res['rs']}")
    print("\n  Cross-sample replication (clinical):")
    cs = cross_sample_cca(df, clin_preds)
    if cs is not None:
        print(f"    Exp in-sample r's: {cs['exp_rs']}")
        print(f"    Conf projected r's: {cs['conf_projected_rs']}")

    out = REPO_ROOT / "results" / "stats" / "affect_analysis" / "multivariate_omega_kappa.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
