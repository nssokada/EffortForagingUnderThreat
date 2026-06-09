"""
Heavy-vs-light affect contrasts as predictors of (ω, κ).

Per subject × question (anxiety/confidence), compute:
  intercept_HvL_diff  = intercept_heavy  − intercept_light
  slope_T_HvL_diff    = slope_T_heavy    − slope_T_light
  slope_D_HvL_diff    = slope_D_heavy    − slope_D_light

These three contrasts capture: "how does affect on heavy cookie trials
differ from affect on light cookie trials?"

Test two models per outcome × sample:

  Model A — DIFFERENCES ONLY (6 predictors: 3 contrasts × 2 questions):
    ω_z ~ anx_int_diff + anx_slope_T_diff + anx_slope_D_diff +
          conf_int_diff + conf_slope_T_diff + conf_slope_D_diff

  Model B — DIFFERENCES + LIGHT BASELINES (12 predictors total):
    Adds the 6 light-only features as a baseline reference.
    Tests whether the heavy-vs-light contrast carries variance beyond the
    light-cookie level itself.

Output: results/stats/affect_analysis/affect_heavy_minus_light_predict_params.csv
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


def per_subject_cookie_stratified():
    """Compute per-cookie intercept, slope_T, slope_D per (subj, question)."""
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


def build_contrasts(long):
    """Wide-pivot then compute heavy-minus-light contrasts.

    Returns DataFrame with one row per (subj, sample) and columns:
      For each question (anxiety, confidence):
        {q}_intercept_light, {q}_slopeT_light, {q}_slopeD_light  (baselines)
        {q}_intercept_HvL,   {q}_slopeT_HvL,   {q}_slopeD_HvL    (contrasts)
    """
    wide = long.pivot_table(
        index=["subj", "sample"],
        columns=["question", "cookie"],
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
        rows.append(m[["subj", "sample", "omega", "kappa"]])
    master = pd.concat(rows, ignore_index=True)
    master["omega_z"] = np.nan
    master["kappa_z"] = np.nan
    for s in master["sample"].unique():
        mask = master["sample"] == s
        master.loc[mask, "omega_z"] = zscore(np.log(master.loc[mask, "omega"]).values)
        master.loc[mask, "kappa_z"] = zscore(np.log(master.loc[mask, "kappa"]).values)
    return master


def fit(df, outcome, predictors, sample_name):
    sub = df[df["sample"] == sample_name].copy()
    sub = sub[[outcome] + predictors].dropna()
    if len(sub) < 30:
        return None
    for p in predictors:
        sub[p] = zscore(sub[p].values, nan_policy="omit")
    X = sm.add_constant(sub[predictors].values)
    return sm.OLS(sub[outcome].values, X).fit(), len(sub)


CONTRASTS = [
    "anxiety_intercept_HvL", "anxiety_slopeT_HvL", "anxiety_slopeD_HvL",
    "confidence_intercept_HvL", "confidence_slopeT_HvL", "confidence_slopeD_HvL",
]
LIGHTS = [
    "anxiety_light_intercept", "anxiety_light_slopeT", "anxiety_light_slopeD",
    "confidence_light_intercept", "confidence_light_slopeT", "confidence_light_slopeD",
]


def main():
    print("=" * 78)
    print("HEAVY-MINUS-LIGHT AFFECT CONTRASTS → ω, κ")
    print("=" * 78)

    long = per_subject_cookie_stratified()
    wide = build_contrasts(long)
    print(f"\nWide affect rows: {len(wide)}")
    master = build_master()
    df = master.merge(wide, on=["subj", "sample"], how="inner")
    print(f"Merged N: {len(df)}  (exp {(df['sample']=='exploratory').sum()}, "
          f"conf {(df['sample']=='confirmatory').sum()})")

    print("\n[Sanity] mean of each contrast (heavy − light, pooled):")
    for c in CONTRASTS:
        if c in df.columns:
            v = df[c].dropna()
            print(f"  {c:30s} mean = {v.mean():+.3f}    median = {v.median():+.3f}")

    rows = []
    for outcome in ["omega_z", "kappa_z"]:
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)

        for model_label, predictors in [
            ("contrasts_only", CONTRASTS),
            ("contrasts_plus_light", CONTRASTS + LIGHTS),
        ]:
            print(f"\n--- Model: {model_label} ({len(predictors)} predictors) ---")
            for sample in ["exploratory", "confirmatory"]:
                res_pack = fit(df, outcome, predictors, sample)
                if res_pack is None: continue
                res, N = res_pack
                print(f"\n  [{sample}] N={N}  R² = {res.rsquared:.4f}  "
                      f"adj R² = {res.rsquared_adj:.4f}  F p = {res.f_pvalue:.4g}")
                for i, name in enumerate(predictors):
                    beta = res.params[i + 1]
                    p = res.pvalues[i + 1]
                    sig = "★" if p < 0.05 else " "
                    sig = "★★" if p < 0.01 else sig
                    sig = "★★★" if p < 0.001 else sig
                    print(f"    {name:32s} β={beta:+.3f}  p={p:.4g} {sig}")
                    rows.append({
                        "outcome": outcome, "sample": sample, "model": model_label,
                        "predictor": name, "N": N, "R2": res.rsquared,
                        "beta": float(beta), "se": float(res.bse[i+1]),
                        "t": float(res.tvalues[i+1]), "p": float(p),
                    })

    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "affect_heavy_minus_light_predict_params.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Replication summary
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY (both samples, p < 0.05, same sign)")
    print("=" * 78)
    rdf = pd.DataFrame(rows)
    for outcome in ["omega_z", "kappa_z"]:
        for model_label in ["contrasts_only", "contrasts_plus_light"]:
            print(f"\n  [{outcome}] [{model_label}]")
            sub = rdf[(rdf["outcome"] == outcome) & (rdf["model"] == model_label)]
            preds = sub["predictor"].unique()
            for predictor in preds:
                er = sub[(sub["sample"] == "exploratory") & (sub["predictor"] == predictor)]
                cr = sub[(sub["sample"] == "confirmatory") & (sub["predictor"] == predictor)]
                if len(er) == 0 or len(cr) == 0:
                    continue
                eb, ep = er["beta"].iloc[0], er["p"].iloc[0]
                cb, cp = cr["beta"].iloc[0], cr["p"].iloc[0]
                replicates = (ep < 0.05) and (cp < 0.05) and (eb * cb > 0)
                tag = "★ REPLICATES" if replicates else ""
                print(f"    {predictor:32s}  exp β={eb:+.3f} p={ep:.4g}    "
                      f"conf β={cb:+.3f} p={cp:.4g}    {tag}")


if __name__ == "__main__":
    main()
