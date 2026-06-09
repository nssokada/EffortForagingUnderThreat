"""
Controlled regressions: affect → parameter (with other parameter as covariate)
and clinical → parameter (controlling for affect contrasts).

Three tests per sample:

Test A: ω_z ~ affect_contrasts + κ_z
        — does the heavy-vs-light confidence contrast survive controlling for κ?

Test B: κ_z ~ affect_contrasts + ω_z
        — same logic the other way.

Test C: clinical_scale_z ~ ω_z + κ_z + affect_contrasts (per clinical scale)
        — does ω or κ have unique clinical-relevant variance beyond affect?

Affect contrasts: heavy-minus-light intercept, slope_T, slope_D for anxiety and
confidence (6 contrasts total).

Output: results/stats/affect_analysis/affect_clinical_controlled.csv
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

from load_data import load_both  # type: ignore


SAMPLES = {
    "exploratory": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
    "confirmatory": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
}
MIN_OBS = 4
CLINICAL_SCALES = [
    "DASS21_Anxiety", "DASS21_Depression", "DASS21_Stress",
    "OASIS_Total", "STICSA_Total",
    "AMI_Total", "AMI_Behavioural", "AMI_Social", "AMI_Emotional",
    "MFIS_Total", "PHQ9_Total",
]
CONTRASTS = [
    "anxiety_intercept_HvL", "anxiety_slopeT_HvL", "anxiety_slopeD_HvL",
    "confidence_intercept_HvL", "confidence_slopeT_HvL", "confidence_slopeD_HvL",
]


def per_subject_cookie_stratified():
    rows = []
    for sample, path in SAMPLES.items():
        f = pd.read_csv(Path(path) / "feelings.csv", low_memory=False)
        f["cookie"] = np.where(f["trialCookie_rewardValue"] == 5.0, "heavy", "light")
        for q in ["anxiety", "confidence"]:
            for cookie in ["heavy", "light"]:
                sub_qc = f[(f["questionLabel"] == q) & (f["cookie"] == cookie)].dropna(subset=["response"])
                for subj, g in sub_qc.groupby("subj"):
                    if len(g) < MIN_OBS:
                        continue
                    preds = []
                    for c in ["threat", "distance"]:
                        if c in g.columns and g[c].nunique() > 1:
                            preds.append(c)
                    if len(preds) < 1:
                        continue
                    X = sm.add_constant(g[preds].values)
                    try:
                        res = sm.OLS(g["response"].values, X).fit()
                        row = {
                            "subj": int(subj), "sample": sample,
                            "question": q, "cookie": cookie,
                            "intercept": float(res.params[0]),
                        }
                        for i, p in enumerate(preds):
                            row[f"slope_{p}"] = float(res.params[i + 1])
                        rows.append(row)
                    except Exception:
                        pass
    return pd.DataFrame(rows)


def build_contrasts():
    long = per_subject_cookie_stratified()
    wide = long.pivot_table(
        index=["subj", "sample"], columns=["question", "cookie"],
        values=["intercept", "slope_threat", "slope_distance"],
    )
    wide.columns = [f"{q}_{c}_{kind}".replace("slope_threat", "slopeT").replace("slope_distance", "slopeD")
                    for kind, q, c in wide.columns]
    wide = wide.reset_index()
    for q in ["anxiety", "confidence"]:
        for feat in ["intercept", "slopeT", "slopeD"]:
            heavy_col = f"{q}_heavy_{feat}"
            light_col = f"{q}_light_{feat}"
            if heavy_col in wide.columns and light_col in wide.columns:
                wide[f"{q}_{feat}_HvL"] = wide[heavy_col] - wide[light_col]
    return wide


def build_master():
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
    return master


def fit_z(df, lhs, rhs, sample_name):
    sub = df[df["sample"] == sample_name].copy()
    sub = sub[[lhs] + rhs].dropna()
    if len(sub) < 30:
        return None
    sub[lhs + "_z"] = zscore(sub[lhs].values, nan_policy="omit")
    for r in rhs:
        sub[r] = zscore(sub[r].values, nan_policy="omit")
    X = sm.add_constant(sub[rhs].values)
    return sm.OLS(sub[lhs + "_z"].values, X).fit(), len(sub)


def main():
    print("=" * 78)
    print("CONTROLLED REGRESSIONS: parameter substrate + clinical residual variance")
    print("=" * 78)

    master = build_master()
    contrasts = build_contrasts()
    df = master.merge(contrasts, on=["subj", "sample"], how="inner")
    print(f"\nMerged N: {len(df)} (exp {(df['sample']=='exploratory').sum()}, "
          f"conf {(df['sample']=='confirmatory').sum()})")

    rows = []

    # ── TEST A: ω ~ affect contrasts + κ ──────────────────────────────────
    print("\n" + "=" * 78)
    print("TEST A: ω ~ affect_contrasts + κ (does substrate survive controlling κ?)")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        res_pack = fit_z(df, "omega", CONTRASTS + ["kappa_z"], sample)
        if res_pack is None: continue
        res, N = res_pack
        print(f"\n  [{sample}] N={N}  R²={res.rsquared:.4f}  F p={res.f_pvalue:.4g}")
        preds = CONTRASTS + ["kappa_z"]
        for i, name in enumerate(preds):
            beta = res.params[i + 1]
            p = res.pvalues[i + 1]
            sig = "★" if p < 0.05 else " "
            sig = "★★" if p < 0.01 else sig
            sig = "★★★" if p < 0.001 else sig
            print(f"    {name:32s} β={beta:+.3f}  p={p:.4g} {sig}")
            rows.append({"test": "A_omega_with_kappa_control", "sample": sample, "N": N,
                         "R2": res.rsquared, "predictor": name, "beta": float(beta),
                         "p": float(p)})

    # ── TEST B: κ ~ affect contrasts + ω ──────────────────────────────────
    print("\n" + "=" * 78)
    print("TEST B: κ ~ affect_contrasts + ω")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        res_pack = fit_z(df, "kappa", CONTRASTS + ["omega_z"], sample)
        if res_pack is None: continue
        res, N = res_pack
        print(f"\n  [{sample}] N={N}  R²={res.rsquared:.4f}  F p={res.f_pvalue:.4g}")
        preds = CONTRASTS + ["omega_z"]
        for i, name in enumerate(preds):
            beta = res.params[i + 1]
            p = res.pvalues[i + 1]
            sig = "★" if p < 0.05 else " "
            sig = "★★" if p < 0.01 else sig
            sig = "★★★" if p < 0.001 else sig
            print(f"    {name:32s} β={beta:+.3f}  p={p:.4g} {sig}")
            rows.append({"test": "B_kappa_with_omega_control", "sample": sample, "N": N,
                         "R2": res.rsquared, "predictor": name, "beta": float(beta),
                         "p": float(p)})

    # ── TEST C: clinical ~ ω + κ + affect_contrasts ──────────────────────
    print("\n" + "=" * 78)
    print("TEST C: clinical ~ ω + κ + affect_contrasts (per clinical scale, per sample)")
    print("=" * 78)
    for scale in CLINICAL_SCALES:
        if scale not in df.columns: continue
        print(f"\n--- Scale: {scale} ---")
        for sample in ["exploratory", "confirmatory"]:
            preds = ["omega_z", "kappa_z"] + CONTRASTS
            res_pack = fit_z(df, scale, preds, sample)
            if res_pack is None: continue
            res, N = res_pack
            print(f"  [{sample}] N={N}  R²={res.rsquared:.4f}  F p={res.f_pvalue:.4g}")
            for i, name in enumerate(preds):
                beta = res.params[i + 1]
                p = res.pvalues[i + 1]
                if p < 0.05:
                    sig = "★"
                    if p < 0.01: sig = "★★"
                    if p < 0.001: sig = "★★★"
                    print(f"    {name:32s} β={beta:+.3f}  p={p:.4g} {sig}")
                rows.append({"test": "C_clinical_full", "sample": sample, "scale": scale,
                             "N": N, "R2": res.rsquared, "predictor": name,
                             "beta": float(beta), "p": float(p)})

    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "affect_clinical_controlled.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Replication summaries ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY")
    print("=" * 78)

    rdf = pd.DataFrame(rows)
    for test_label, outcome_label in [("A_omega_with_kappa_control", "ω"),
                                       ("B_kappa_with_omega_control", "κ")]:
        print(f"\n  Test {test_label[0]}: {outcome_label} ~ affect + other param")
        sub = rdf[rdf["test"] == test_label]
        for predictor in CONTRASTS + (["kappa_z"] if test_label.startswith("A") else ["omega_z"]):
            er = sub[(sub["sample"] == "exploratory") & (sub["predictor"] == predictor)]
            cr = sub[(sub["sample"] == "confirmatory") & (sub["predictor"] == predictor)]
            if len(er) == 0 or len(cr) == 0: continue
            eb, ep = er["beta"].iloc[0], er["p"].iloc[0]
            cb, cp = cr["beta"].iloc[0], cr["p"].iloc[0]
            replicates = (ep < 0.05) and (cp < 0.05) and (eb * cb > 0)
            tag = "★ REPLICATES" if replicates else ""
            print(f"    {predictor:32s}  exp β={eb:+.3f} p={ep:.4g}    "
                  f"conf β={cb:+.3f} p={cp:.4g}    {tag}")

    # Test C replication summary: focus on whether ω or κ themselves hit
    print(f"\n  Test C: clinical ~ ω + κ + affect (replicating param effects per scale)")
    sub_c = rdf[rdf["test"] == "C_clinical_full"]
    for scale in sub_c["scale"].unique():
        sub_s = sub_c[sub_c["scale"] == scale]
        for predictor in ["omega_z", "kappa_z"]:
            er = sub_s[(sub_s["sample"] == "exploratory") & (sub_s["predictor"] == predictor)]
            cr = sub_s[(sub_s["sample"] == "confirmatory") & (sub_s["predictor"] == predictor)]
            if len(er) == 0 or len(cr) == 0: continue
            eb, ep = er["beta"].iloc[0], er["p"].iloc[0]
            cb, cp = cr["beta"].iloc[0], cr["p"].iloc[0]
            replicates = (ep < 0.05) and (cp < 0.05) and (eb * cb > 0)
            if replicates or (ep < 0.05) or (cp < 0.05):
                tag = "★ REPLICATES" if replicates else ""
                print(f"    {scale:24s} {predictor:12s}  exp β={eb:+.3f} p={ep:.4g}    "
                      f"conf β={cb:+.3f} p={cp:.4g}    {tag}")


if __name__ == "__main__":
    main()
