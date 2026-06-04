"""
result_604 re-analysis using TRUE pooled-z (the prereg-compliant approach).

The original result_604 z-scored predictors and outcomes WITHIN sample, which
removes between-sample variance. The prereg's "Other Planned Analyses" #6
calls for pooling for power. This version:

  - Concatenates both samples into one dataset (N ≈ 571)
  - Z-scores ω, κ, and each scale on the POOLED distribution
  - Does NOT include sample as a covariate (treat as one population)
  - Runs the same Stage 1 (subscale regressions with interaction) and
    Stage 2 (comorbidity-group contrasts) but on pooled-z data

Comparison with result_604's within-sample-z analysis tells us whether the
effects we found are robust to standardisation choice.
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
import bambi as bmb
import arviz as az
from scipy.stats import zscore

from config import BKW  # type: ignore


SAMPLES = {
    "exploratory": {
        "psych": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425/psych.csv",
        "m4_params": "results/stats/joint_optimal/exploratory/mcmc_m4_params.csv",
    },
    "confirmatory": {
        "psych": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413/psych.csv",
        "m4_params": "results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv",
    },
}

SCALES = [
    "DASS21_Anxiety", "OASIS_Total", "STAI_Trait", "STAI_State", "STICSA_Total",
    "DASS21_Depression", "PHQ9_Total",
    "DASS21_Stress", "DASS21_Total",
    "AMI_Behavioural", "AMI_Social", "AMI_Emotional", "AMI_Total",
    "MFIS_Physical", "MFIS_Cognitive", "MFIS_Psychosocial", "MFIS_Total",
]

SCALE_TYPE = {
    "DASS21_Anxiety": "anxiety", "OASIS_Total": "anxiety", "STAI_Trait": "anxiety",
    "STAI_State": "anxiety", "STICSA_Total": "anxiety",
    "DASS21_Depression": "depression", "PHQ9_Total": "depression",
    "DASS21_Stress": "stress", "DASS21_Total": "composite",
    "AMI_Behavioural": "apathy", "AMI_Social": "apathy", "AMI_Emotional": "apathy",
    "AMI_Total": "apathy",
    "MFIS_Physical": "fatigue", "MFIS_Cognitive": "fatigue",
    "MFIS_Psychosocial": "fatigue", "MFIS_Total": "fatigue",
}


def load_pooled() -> pd.DataFrame:
    """Concatenate both samples, then pooled-z (no within-sample standardisation)."""
    rows = []
    for sample, paths in SAMPLES.items():
        psych = pd.read_csv(paths["psych"])
        m4 = pd.read_csv(paths["m4_params"])
        df = psych.merge(m4, on="subj", how="inner")
        df["sample"] = sample
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    pooled["log_omega"] = np.log(pooled["omega"])
    pooled["log_kappa"] = np.log(pooled["kappa"])
    # POOLED z-scoring
    pooled["omega_z"] = zscore(pooled["log_omega"].values, nan_policy="omit")
    pooled["kappa_z"] = zscore(pooled["log_kappa"].values, nan_policy="omit")
    return pooled


def stage1_pooled(pooled: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scale in SCALES:
        if scale not in pooled.columns:
            continue
        df = pooled[[scale, "omega_z", "kappa_z", "sample"]].dropna().copy()
        if len(df) < 50:
            continue
        # POOLED z-scoring of outcome
        df[f"{scale}_z"] = zscore(df[scale].values, nan_policy="omit")
        # Two model variants: (a) main effects + interaction, (b) main effects only
        for spec_name, formula in [
            ("with_interaction", f"{scale}_z ~ omega_z + kappa_z + omega_z:kappa_z"),
            ("main_effects_only", f"{scale}_z ~ omega_z + kappa_z"),
        ]:
            model = bmb.Model(formula, data=df)
            result = model.fit(**BKW)
            s = az.summary(result, hdi_prob=0.95)
            terms = ["omega_z", "kappa_z"] + (["omega_z:kappa_z"] if spec_name == "with_interaction" else [])
            for term in terms:
                if term not in s.index:
                    continue
                beta = s.loc[term, "mean"]
                lo = s.loc[term, "hdi_2.5%"]
                hi = s.loc[term, "hdi_97.5%"]
                sig = (lo > 0) or (hi < 0)
                records.append({
                    "scale": scale, "scale_type": SCALE_TYPE.get(scale, "unknown"),
                    "spec": spec_name, "term": term, "n": len(df),
                    "beta": beta, "hdi_lo": lo, "hdi_hi": hi, "sig": sig,
                })
        # Print the with-interaction summary
        sub = [r for r in records if r["scale"] == scale and r["spec"] == "with_interaction"]
        print(f"\n--- {scale} (n={len(df)}) [with_interaction] ---")
        for r in sub:
            star = "★" if r["sig"] else " "
            print(f"  {r['term']:20s} β = {r['beta']:+.4f} [{r['hdi_lo']:+.4f}, {r['hdi_hi']:+.4f}] {star}")
    return pd.DataFrame(records)


def stage2_pooled(pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pooled[["sample", "subj", "DASS21_Anxiety", "DASS21_Depression",
                  "omega_z", "kappa_z"]].dropna().copy()
    # Use POOLED median splits this time (treat as one population)
    m_anx = df["DASS21_Anxiety"].median()
    m_dep = df["DASS21_Depression"].median()
    df["high_anx"] = (df["DASS21_Anxiety"] > m_anx).astype(int)
    df["high_dep"] = (df["DASS21_Depression"] > m_dep).astype(int)

    def label(row):
        if row["high_anx"] == 1 and row["high_dep"] == 1:
            return "comorbid"
        if row["high_anx"] == 1 and row["high_dep"] == 0:
            return "anxious_only"
        if row["high_anx"] == 0 and row["high_dep"] == 1:
            return "depressed_only"
        return "neither"

    df["group"] = df.apply(label, axis=1)
    print("\n--- Comorbidity groups (POOLED median splits) ---")
    sizes = df.groupby("group").size()
    print(sizes)
    print("\nGroup means (raw):")
    print(df.groupby("group")[["DASS21_Anxiety", "DASS21_Depression", "omega_z", "kappa_z"]].mean())

    records = []
    for outcome in ["omega_z", "kappa_z"]:
        print(f"\n--- {outcome} ~ group + sample (pooled) ---")
        df["group_factor"] = pd.Categorical(df["group"], categories=["neither", "anxious_only", "depressed_only", "comorbid"])
        model = bmb.Model(f"{outcome} ~ group_factor + sample", data=df)
        result = model.fit(**BKW)
        s = az.summary(result, hdi_prob=0.95)
        for term in s.index:
            if term in ["Intercept", "sigma"]:
                continue
            beta = s.loc[term, "mean"]
            lo = s.loc[term, "hdi_2.5%"]
            hi = s.loc[term, "hdi_97.5%"]
            sig = (lo > 0) or (hi < 0)
            star = "★" if sig else " "
            print(f"  {term:40s} β = {beta:+.4f} [{lo:+.4f}, {hi:+.4f}] {star}")
            records.append({
                "outcome": outcome, "contrast": term, "beta": beta,
                "hdi_lo": lo, "hdi_hi": hi, "sig": sig,
            })
    return pd.DataFrame(records), sizes.reset_index().rename(columns={0: "n"})


def main():
    print("Loading pooled data with pooled-z standardisation...")
    pooled = load_pooled()
    print(f"Pooled N = {len(pooled)}")
    print(f"ω log-z: mean={pooled['omega_z'].mean():.4f}, sd={pooled['omega_z'].std():.4f}")
    print(f"κ log-z: mean={pooled['kappa_z'].mean():.4f}, sd={pooled['kappa_z'].std():.4f}")

    print("\n" + "=" * 70)
    print("STAGE 1 — Subscale regressions (POOLED-Z)")
    print("=" * 70)
    stage1 = stage1_pooled(pooled)

    print("\n" + "=" * 70)
    print("STAGE 2 — Comorbidity-group analysis (POOLED median splits)")
    print("=" * 70)
    stage2, sizes = stage2_pooled(pooled)

    out_dir = REPO_ROOT / "results" / "stats" / "clinical"
    stage1.to_csv(out_dir / "embodied_subscale_regressions_pooled.csv", index=False)
    stage2.to_csv(out_dir / "embodied_comorbidity_group_params_pooled.csv", index=False)
    sizes.to_csv(out_dir / "embodied_comorbidity_groups_pooled.csv", index=False)
    print(f"\nSaved pooled-z outputs.")

    print("\n" + "=" * 70)
    print("HEADLINE — Stage 1 (POOLED-Z) main effects + interaction")
    print("=" * 70)
    pivot = stage1[stage1["spec"] == "with_interaction"].pivot_table(
        index=["scale_type", "scale"], columns="term", values="beta", aggfunc="first")
    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("HEADLINE — Stage 1 (POOLED-Z) significant terms only")
    print("=" * 70)
    sig = stage1[(stage1["sig"]) & (stage1["spec"] == "with_interaction")]
    if len(sig) == 0:
        print("  NO significant terms in the with-interaction spec.")
    else:
        print(sig[["scale", "term", "beta", "hdi_lo", "hdi_hi"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("HEADLINE — Stage 1 (POOLED-Z) main-effects-only significant terms")
    print("=" * 70)
    sig_main = stage1[(stage1["sig"]) & (stage1["spec"] == "main_effects_only")]
    if len(sig_main) == 0:
        print("  NO significant terms in the main-effects-only spec.")
    else:
        print(sig_main[["scale", "term", "beta", "hdi_lo", "hdi_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
