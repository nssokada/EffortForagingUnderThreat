"""
Sanity checks on the result_604 clinical decomposition.

Verifies:
  1. AMI scoring direction — should higher AMI = more apathy or less apathy?
     Test: r(AMI_Total, mean_vigor) should be NEGATIVE if AMI higher = more
     apathy (since κ → vigor is negative, and apathy → low engagement).
     If positive, AMI is reverse-scored in our data.

  2. Raw Pearson correlations of ω, κ with clinical scales — verify the
     bambi regression directions match raw correlations.

  3. Population (ω, κ) statistics per sample — check the parameter
     distributions are reasonable.

  4. r(omega, kappa) per sample — should be +0.30 to +0.37 per 208.

  5. Sample-z-scored vs pooled-z-scored regression direction — does
     the result depend on which standardization we use?
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)
NB_DIR = REPO_ROOT / "notebooks" / "analysis"
sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr

from load_data import load_both  # type: ignore


def main():
    print("=" * 70)
    print("SANITY CHECK 1 — AMI scoring direction")
    print("=" * 70)

    exp_data, conf_data = load_both()
    master_exp = exp_data["master"].reset_index().rename(columns={"index": "subj"})
    master_conf = conf_data["master"].reset_index().rename(columns={"index": "subj"})

    print(f"\nExploratory master columns (sample): {[c for c in master_exp.columns if 'AMI' in c or 'PHQ' in c or 'DASS' in c or 'vigor' in c][:15]}")

    # Load psych + m4 directly
    psych_exp = pd.read_csv("data/exploratory_350/processed/stage5_filtered_data_20260403_133425/psych.csv")
    psych_conf = pd.read_csv("data/confirmatory_350/processed/stage5_filtered_data_20260403_142413/psych.csv")
    m4_exp = pd.read_csv("results/stats/joint_optimal/exploratory/mcmc_m4_params.csv")
    m4_conf = pd.read_csv("results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv")

    print("\nAMI_Total distribution (exploratory):")
    print(f"  min={psych_exp['AMI_Total'].min():.1f}, "
          f"25%={psych_exp['AMI_Total'].quantile(0.25):.1f}, "
          f"median={psych_exp['AMI_Total'].median():.1f}, "
          f"75%={psych_exp['AMI_Total'].quantile(0.75):.1f}, "
          f"max={psych_exp['AMI_Total'].max():.1f}")
    print("AMI_Total distribution (confirmatory):")
    print(f"  min={psych_conf['AMI_Total'].min():.1f}, "
          f"25%={psych_conf['AMI_Total'].quantile(0.25):.1f}, "
          f"median={psych_conf['AMI_Total'].median():.1f}, "
          f"75%={psych_conf['AMI_Total'].quantile(0.75):.1f}, "
          f"max={psych_conf['AMI_Total'].max():.1f}")

    print("\n--- KEY CHECK: r(AMI_Total, mean_vigor) per sample ---")
    print("If high AMI = MORE apathy, r should be NEGATIVE (apathetic → low vigor).")
    print("If high AMI = LESS apathy (reverse-scored), r should be POSITIVE.")
    for sample, master, psych in [("Exploratory", master_exp, psych_exp),
                                    ("Confirmatory", master_conf, psych_conf)]:
        if "AMI_Total" not in master.columns and "AMI_Total" in psych.columns:
            # Need to merge
            df = master.merge(psych[["subj", "AMI_Total"]], on="subj", how="inner")
        else:
            df = master
        # Use master's mean_vigor (from load_both — averaged cell-mean) for the check
        if "mean_vigor" not in df.columns:
            print(f"  {sample}: no mean_vigor in master")
            continue
        d = df[["AMI_Total", "mean_vigor"]].dropna()
        r, p = pearsonr(d["AMI_Total"], d["mean_vigor"])
        print(f"  {sample}: r(AMI_Total, mean_vigor) = {r:+.4f}, p = {p:.4g}, n = {len(d)}")

    print("\n=" * 35)
    print("\nSANITY CHECK 2 — raw correlations: ω, κ with key scales")
    print("=" * 70)
    for sample_name, psych, m4 in [("Exploratory", psych_exp, m4_exp),
                                    ("Confirmatory", psych_conf, m4_conf)]:
        df = psych.merge(m4, on="subj", how="inner")
        print(f"\n--- {sample_name} (n = {len(df)}) ---")
        df["log_omega"] = np.log(df["omega"])
        df["log_kappa"] = np.log(df["kappa"])
        for scale in ["AMI_Total", "AMI_Behavioural", "AMI_Social", "AMI_Emotional",
                      "DASS21_Anxiety", "DASS21_Depression", "PHQ9_Total", "STAI_Trait"]:
            if scale not in df.columns:
                continue
            sub = df[[scale, "log_omega", "log_kappa"]].dropna()
            r_om, _ = pearsonr(sub[scale], sub["log_omega"])
            r_ka, _ = pearsonr(sub[scale], sub["log_kappa"])
            print(f"  {scale:25s}  r(log_ω) = {r_om:+.3f}   r(log_κ) = {r_ka:+.3f}")

    print("\n=" * 35)
    print("\nSANITY CHECK 3 — Parameter distributions")
    print("=" * 70)
    for sample_name, m4 in [("Exploratory", m4_exp), ("Confirmatory", m4_conf)]:
        print(f"\n--- {sample_name} ---")
        print(f"  ω: mean = {m4['omega'].mean():.3f}, median = {m4['omega'].median():.3f}, "
              f"sd = {m4['omega'].std():.3f}")
        print(f"  κ: mean = {m4['kappa'].mean():.3f}, median = {m4['kappa'].median():.3f}, "
              f"sd = {m4['kappa'].std():.3f}")
        r_ok, _ = pearsonr(np.log(m4["omega"]), np.log(m4["kappa"]))
        print(f"  r(log_ω, log_κ) = {r_ok:+.3f}  (expect ~+0.30 to +0.37 per result_208)")

    print("\n=" * 35)
    print("\nSANITY CHECK 4 — AMI item-level (if items present)")
    print("=" * 70)
    sample_cols = list(psych_exp.columns)
    ami_items = [c for c in sample_cols if c.startswith("AMI_") and not c.endswith(("Total", "RT", "Behavioural", "Social", "Emotional"))]
    if ami_items:
        print(f"  Found {len(ami_items)} AMI item columns: {ami_items[:5]}...")
    else:
        print("  No AMI item-level columns found in psych.csv — only subscale and total")

    print("\n=" * 35)
    print("\nSANITY CHECK 5 — Pooled-z vs within-sample-z")
    print("=" * 70)
    psych_exp["sample"] = "exploratory"
    psych_conf["sample"] = "confirmatory"
    m4_exp["sample"] = "exploratory"
    m4_conf["sample"] = "confirmatory"
    pooled = pd.concat([
        psych_exp.merge(m4_exp, on=["subj", "sample"], how="inner"),
        psych_conf.merge(m4_conf, on=["subj", "sample"], how="inner"),
    ], ignore_index=True)
    pooled["log_omega"] = np.log(pooled["omega"])
    pooled["log_kappa"] = np.log(pooled["kappa"])

    # Within-sample z
    pooled["omega_z_ws"] = pooled.groupby("sample")["log_omega"].transform(lambda x: zscore(x.values, nan_policy="omit"))
    pooled["kappa_z_ws"] = pooled.groupby("sample")["log_kappa"].transform(lambda x: zscore(x.values, nan_policy="omit"))
    # Pooled z
    pooled["omega_z_pool"] = zscore(pooled["log_omega"].values, nan_policy="omit")
    pooled["kappa_z_pool"] = zscore(pooled["log_kappa"].values, nan_policy="omit")

    print("\nWithin-sample z vs pooled z — Pearson r between standardisations:")
    r_ws_pool_om, _ = pearsonr(pooled["omega_z_ws"], pooled["omega_z_pool"])
    r_ws_pool_ka, _ = pearsonr(pooled["kappa_z_ws"], pooled["kappa_z_pool"])
    print(f"  ω: r(within-sample-z, pooled-z) = {r_ws_pool_om:+.4f}")
    print(f"  κ: r(within-sample-z, pooled-z) = {r_ws_pool_ka:+.4f}")
    print("  (Should be near +1.0 if cross-sample differences are small)")

    print("\nRaw Pearson check on AMI_Total — pooled (no z-scoring needed for r):")
    sub = pooled[["AMI_Total", "log_omega", "log_kappa"]].dropna()
    r_om, p_om = pearsonr(sub["AMI_Total"], sub["log_omega"])
    r_ka, p_ka = pearsonr(sub["AMI_Total"], sub["log_kappa"])
    print(f"  r(AMI_Total, log_ω) = {r_om:+.4f}, p = {p_om:.4g}, n = {len(sub)}")
    print(f"  r(AMI_Total, log_κ) = {r_ka:+.4f}, p = {p_ka:.4g}, n = {len(sub)}")


if __name__ == "__main__":
    main()
