"""
Embodied clinical decomposition — do (ω, κ) joint parameters dissociate the
clinical phenotypes that broad-stroke symptom regressions missed?

Builds on result_601 (broad scales on ω, κ — essentially null) by running:

  STAGE 1 — Subscale-specific regressions with ω × κ interaction
    For each clinical subscale, fit:
        scale_z ~ omega_z + kappa_z + omega_z:kappa_z + sample_fixed
    Pooled across samples (N ≈ 570), with sample-level z-scoring.
    Tests:
      - Do anxiety-loading subscales preferentially load on ω?
      - Do depression / apathy / fatigue subscales preferentially load on κ?
      - Does the ω × κ interaction predict comorbid presentations?

  STAGE 2 — Comorbidity-group analysis
    Using DASS21-Anxiety and DASS21-Depression median splits within sample:
      anxious_only   = high anx, low dep
      depressed_only = low anx, high dep
      comorbid       = high both
      neither        = low both
    Test:
      omega_z ~ group + sample
      kappa_z ~ group + sample
    Specific predictions:
      - ω elevated in anxious_only and comorbid
      - κ elevated in depressed_only and comorbid
      - Both elevated in comorbid (the embodied prediction)

Outputs:
  results/stats/clinical/embodied_subscale_regressions.csv
  results/stats/clinical/embodied_comorbidity_groups.csv
  results/stats/clinical/embodied_comorbidity_group_params.csv
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
    # Anxiety scales (predicted to load on ω)
    "DASS21_Anxiety",
    "OASIS_Total",
    "STAI_Trait",
    "STAI_State",
    "STICSA_Total",
    # Depression scales (mixed predicted loading)
    "DASS21_Depression",
    "PHQ9_Total",
    # Stress (mixed)
    "DASS21_Stress",
    "DASS21_Total",
    # Apathy (predicted to load on κ)
    "AMI_Behavioural",
    "AMI_Social",
    "AMI_Emotional",
    "AMI_Total",
    # Fatigue (predicted to load on κ)
    "MFIS_Physical",
    "MFIS_Cognitive",
    "MFIS_Psychosocial",
    "MFIS_Total",
]

# Theoretical predictions for each scale (for the summary table)
SCALE_TYPE = {
    "DASS21_Anxiety": "anxiety",
    "OASIS_Total": "anxiety",
    "STAI_Trait": "anxiety",
    "STAI_State": "anxiety",
    "STICSA_Total": "anxiety",
    "DASS21_Depression": "depression",
    "PHQ9_Total": "depression",
    "DASS21_Stress": "stress",
    "DASS21_Total": "composite",
    "AMI_Behavioural": "apathy",
    "AMI_Social": "apathy",
    "AMI_Emotional": "apathy",
    "AMI_Total": "apathy",
    "MFIS_Physical": "fatigue",
    "MFIS_Cognitive": "fatigue",
    "MFIS_Psychosocial": "fatigue",
    "MFIS_Total": "fatigue",
}


def load_subject_table(sample_name: str) -> pd.DataFrame:
    """Per-subject ω, κ + clinical scales for one sample."""
    psych = pd.read_csv(SAMPLES[sample_name]["psych"])
    m4 = pd.read_csv(SAMPLES[sample_name]["m4_params"])
    df = psych.merge(m4, on="subj", how="inner")
    df["sample"] = sample_name
    df["log_omega"] = np.log(df["omega"])
    df["log_kappa"] = np.log(df["kappa"])
    df["omega_z"] = zscore(df["log_omega"].values, nan_policy="omit")
    df["kappa_z"] = zscore(df["log_kappa"].values, nan_policy="omit")
    return df


def stage1_subscale_regression(pooled: pd.DataFrame) -> pd.DataFrame:
    """Fit `scale_z ~ omega_z + kappa_z + omega_z:kappa_z + sample` for each scale.

    Sample-level z-scoring of the outcome to control for any cross-sample shift in
    scale means/SDs."""
    records = []
    for scale in SCALES:
        if scale not in pooled.columns:
            print(f"  SKIP {scale}: not in data")
            continue

        # Z-score outcome within sample, then pool
        df = pooled[[scale, "omega_z", "kappa_z", "sample"]].dropna().copy()
        if len(df) < 50:
            print(f"  SKIP {scale}: too few rows ({len(df)})")
            continue
        df[f"{scale}_z"] = df.groupby("sample")[scale].transform(lambda x: zscore(x.values, nan_policy="omit"))

        # Drop rows with missing parameters
        df = df.dropna(subset=[f"{scale}_z", "omega_z", "kappa_z"])

        print(f"\n--- {scale} (n={len(df)}) ---")
        model = bmb.Model(
            f"{scale}_z ~ omega_z + kappa_z + omega_z:kappa_z + sample",
            data=df,
        )
        result = model.fit(**BKW)
        s = az.summary(result, hdi_prob=0.95)

        for term in ["omega_z", "kappa_z", "omega_z:kappa_z"]:
            if term not in s.index:
                continue
            beta = s.loc[term, "mean"]
            lo = s.loc[term, "hdi_2.5%"]
            hi = s.loc[term, "hdi_97.5%"]
            sig = (lo > 0) or (hi < 0)
            star = "★" if sig else " "
            print(f"  {term:25s} β = {beta:+.4f} [{lo:+.4f}, {hi:+.4f}] {star}")
            records.append({
                "scale": scale,
                "scale_type": SCALE_TYPE.get(scale, "unknown"),
                "term": term,
                "n": len(df),
                "beta": beta,
                "hdi_lo": lo,
                "hdi_hi": hi,
                "sig": sig,
            })
    return pd.DataFrame(records)


def stage2_comorbidity_groups(pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define anxious_only / depressed_only / comorbid / neither via within-sample
    median splits on DASS21_Anxiety and DASS21_Depression. Test ω, κ across groups."""
    df = pooled[["sample", "subj", "DASS21_Anxiety", "DASS21_Depression",
                  "omega_z", "kappa_z"]].dropna().copy()

    # Within-sample median splits
    for sample in df["sample"].unique():
        m_anx = df.loc[df["sample"] == sample, "DASS21_Anxiety"].median()
        m_dep = df.loc[df["sample"] == sample, "DASS21_Depression"].median()
        df.loc[df["sample"] == sample, "high_anx"] = (df.loc[df["sample"] == sample, "DASS21_Anxiety"] > m_anx).astype(int)
        df.loc[df["sample"] == sample, "high_dep"] = (df.loc[df["sample"] == sample, "DASS21_Depression"] > m_dep).astype(int)

    def label(row):
        if row["high_anx"] == 1 and row["high_dep"] == 1:
            return "comorbid"
        if row["high_anx"] == 1 and row["high_dep"] == 0:
            return "anxious_only"
        if row["high_anx"] == 0 and row["high_dep"] == 1:
            return "depressed_only"
        return "neither"

    df["group"] = df.apply(label, axis=1)
    print("\n--- Comorbidity groups (within-sample median splits) ---")
    print(df.groupby(["sample", "group"]).size().unstack(fill_value=0))

    # Group sizes summary
    group_sizes = df.groupby(["sample", "group"]).size().unstack(fill_value=0).reset_index()
    print("\nGroup means (raw, not z-scored):")
    print(df.groupby("group")[["DASS21_Anxiety", "DASS21_Depression", "omega_z", "kappa_z"]].mean())

    # Fit Bayesian ANOVA: omega_z ~ group + sample, kappa_z ~ group + sample
    records = []
    for outcome in ["omega_z", "kappa_z"]:
        print(f"\n--- Bayesian regression: {outcome} ~ group + sample ---")
        # Set 'neither' as reference category
        df["group_factor"] = pd.Categorical(df["group"], categories=["neither", "anxious_only", "depressed_only", "comorbid"])
        model = bmb.Model(f"{outcome} ~ group_factor + sample", data=df)
        result = model.fit(**BKW)
        s = az.summary(result, hdi_prob=0.95)

        # Each term is contrast vs 'neither'
        for term in s.index:
            if term in ["Intercept", "sigma", "sample[T.confirmatory]"]:
                continue
            beta = s.loc[term, "mean"]
            lo = s.loc[term, "hdi_2.5%"]
            hi = s.loc[term, "hdi_97.5%"]
            sig = (lo > 0) or (hi < 0)
            star = "★" if sig else " "
            print(f"  {term:40s} β = {beta:+.4f} [{lo:+.4f}, {hi:+.4f}] {star}")
            records.append({
                "outcome": outcome,
                "contrast": term,
                "beta": beta,
                "hdi_lo": lo,
                "hdi_hi": hi,
                "sig": sig,
            })

    return pd.DataFrame(records), group_sizes


def main():
    # Load both samples
    print("=" * 70)
    print("Loading samples")
    print("=" * 70)
    dfs = []
    for sample in SAMPLES:
        d = load_subject_table(sample)
        print(f"  {sample}: {len(d)} subjects")
        dfs.append(d)
    pooled = pd.concat(dfs, ignore_index=True)
    print(f"Pooled N = {len(pooled)}")

    # Stage 1
    print("\n" + "=" * 70)
    print("STAGE 1 — Subscale-specific regressions (pooled, sample-z-scored)")
    print("=" * 70)
    stage1 = stage1_subscale_regression(pooled)

    # Stage 2
    print("\n" + "=" * 70)
    print("STAGE 2 — Comorbidity-group analysis (within-sample median splits)")
    print("=" * 70)
    stage2_contrasts, stage2_sizes = stage2_comorbidity_groups(pooled)

    # Save
    out_dir = REPO_ROOT / "results" / "stats" / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(out_dir / "embodied_subscale_regressions.csv", index=False)
    stage2_contrasts.to_csv(out_dir / "embodied_comorbidity_group_params.csv", index=False)
    stage2_sizes.to_csv(out_dir / "embodied_comorbidity_groups.csv", index=False)
    print(f"\nSaved: {out_dir / 'embodied_subscale_regressions.csv'}")
    print(f"Saved: {out_dir / 'embodied_comorbidity_groups.csv'}")
    print(f"Saved: {out_dir / 'embodied_comorbidity_group_params.csv'}")

    # Print summary
    print("\n" + "=" * 70)
    print("HEADLINE — Stage 1 (which parameter does each scale prefer?)")
    print("=" * 70)
    pivot = stage1.pivot_table(index=["scale_type", "scale"], columns="term", values="beta", aggfunc="first")
    print(pivot.to_string())
    print("\n" + "=" * 70)
    print("HEADLINE — Stage 2 (which group has elevated ω vs κ?)")
    print("=" * 70)
    pivot2 = stage2_contrasts.pivot_table(index="contrast", columns="outcome", values="beta", aggfunc="first")
    print(pivot2.to_string())


if __name__ == "__main__":
    main()
