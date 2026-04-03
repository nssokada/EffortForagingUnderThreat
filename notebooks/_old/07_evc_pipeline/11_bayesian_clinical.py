#!/usr/bin/env python3
"""
11_bayesian_clinical.py — Bayesian regression: model params + metacognition → clinical outcomes
Uses bambi (PyMC backend) with NUTS sampling.

Analyses:
1. Individual regressions: clinical_z ~ log_ce + log_cd
2. Full model with metacognition: clinical_z ~ log_ce + log_cd + discrepancy + calibration
3. ROPE analysis for null effects
4. Summary table → results/stats/evc_bayesian_clinical.csv
5. Forest plot → results/figs/paper/fig_bayesian_clinical.png
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy import stats
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
PARAMS_FILE = Path("results/stats/oc_evc_final_params.csv")
FACTOR_FILE = Path("results/stats/psych_factor_scores.csv")
OUT_TABLE = Path("results/stats/evc_bayesian_clinical.csv")
OUT_FIG = Path("results/figs/paper/fig_bayesian_clinical.png")

# ── Clinical subscales to analyze ─────────────────────────────────────────────
CLINICAL_SCALES = [
    ("OASIS_Total", "OASIS"),
    ("STAI_State", "STAI_State"),
    ("PHQ9_Total", "PHQ9"),
    ("DASS21_Anxiety", "DASS_Anxiety"),
    ("DASS21_Stress", "DASS_Stress"),
    ("DASS21_Depression", "DASS_Depression"),
    ("STICSA_Total", "STICSA"),
    ("AMI_Emotional", "AMI_Emotional"),
    ("AMI_Behavioural", "AMI_Behavioural"),
]

ROPE = (-0.1, 0.1)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

params = pd.read_csv(PARAMS_FILE)
psych = pd.read_csv(DATA_DIR / "psych.csv")
feelings = pd.read_csv(DATA_DIR / "feelings.csv")

print(f"Params: {params.shape[0]} subjects")
print(f"Psych:  {psych.shape[0]} subjects")
print(f"Feelings: {feelings.shape[0]} rows")

# ── 2. Compute calibration & discrepancy ──────────────────────────────────────
print("\n" + "=" * 70)
print("COMPUTING CALIBRATION & DISCREPANCY")
print("=" * 70)

# EVC survival: S = (1 - T^gamma) + epsilon * T^gamma * p_esc
# Population params: gamma=0.21, epsilon=0.098, p_esc≈0.6
GAMMA = 0.21
EPSILON = 0.098
P_ESC = 0.6

anx = feelings[feelings["questionLabel"] == "anxiety"].copy()
anx["T"] = anx["threat"]
anx["S_evc"] = (1 - anx["T"] ** GAMMA) + EPSILON * (anx["T"] ** GAMMA) * P_ESC
anx["danger"] = 1 - anx["S_evc"]  # higher = more dangerous

# Calibration: within-subject Pearson r(anxiety_rating, 1-S) = r(anxiety, danger)
calibration_list = []
for subj, grp in anx.groupby("subj"):
    if len(grp) >= 3:
        r, _ = stats.pearsonr(grp["response"], grp["danger"])
        calibration_list.append({"subj": subj, "calibration": r})
    else:
        calibration_list.append({"subj": subj, "calibration": np.nan})
calibration_df = pd.DataFrame(calibration_list)

# Discrepancy: mean residual after regressing anxiety on S at population level
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(anx[["S_evc"]], anx["response"])
anx["resid"] = anx["response"] - lr.predict(anx[["S_evc"]])
discrepancy_df = anx.groupby("subj")["resid"].mean().reset_index()
discrepancy_df.columns = ["subj", "discrepancy"]

print(f"Calibration: mean={calibration_df['calibration'].mean():.3f}, "
      f"std={calibration_df['calibration'].std():.3f}")
print(f"Discrepancy: mean={discrepancy_df['discrepancy'].mean():.3f}, "
      f"std={discrepancy_df['discrepancy'].std():.3f}")

# ── 3. Merge all data ────────────────────────────────────────────────────────
df = params.merge(psych, on="subj").merge(calibration_df, on="subj").merge(discrepancy_df, on="subj")
print(f"\nMerged dataset: {df.shape[0]} subjects")

# Log-transform model parameters (highly skewed)
df["log_ce"] = np.log(df["c_effort"])
df["log_cd"] = np.log(df["c_death"])

# Z-score predictors
for col in ["log_ce", "log_cd", "calibration", "discrepancy"]:
    df[f"{col}_z"] = stats.zscore(df[col], nan_policy="omit")

# Z-score clinical measures
for raw_col, label in CLINICAL_SCALES:
    df[f"{label}_z"] = stats.zscore(df[raw_col], nan_policy="omit")

# Drop rows with NaN calibration
n_before = len(df)
df = df.dropna(subset=["calibration_z", "discrepancy_z"])
print(f"After dropping NaN calibration: {len(df)} subjects (dropped {n_before - len(df)})")

# ── 4. Analysis 1: Individual regressions (params only) ──────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 1: clinical_z ~ log_ce_z + log_cd_z")
print("=" * 70)

results_simple = []

for raw_col, label in CLINICAL_SCALES:
    print(f"\n--- {label} ---")
    outcome_col = f"{label}_z"

    model_df = df[["log_ce_z", "log_cd_z", outcome_col]].dropna().copy()
    model_df.columns = ["log_ce_z", "log_cd_z", "y"]

    model = bmb.Model("y ~ log_ce_z + log_cd_z", data=model_df)
    idata = model.fit(draws=2000, chains=4, target_accept=0.9,
                      init="adapt_diag", random_seed=42)

    summary = az.summary(idata, hdi_prob=0.94, var_names=["log_ce_z", "log_cd_z", "Intercept"])
    print(summary)

    # Extract posteriors
    for pred in ["log_ce_z", "log_cd_z"]:
        post = idata.posterior[pred].values.flatten()
        mean_val = post.mean()
        hdi = az.hdi(post, hdi_prob=0.94)
        p_positive = (post > 0).mean()
        p_negative = (post < 0).mean()
        in_rope = ((post > ROPE[0]) & (post < ROPE[1])).mean()

        results_simple.append({
            "clinical_measure": label,
            "predictor": pred.replace("_z", ""),
            "model": "params_only",
            "posterior_mean": mean_val,
            "hdi_low": hdi[0],
            "hdi_high": hdi[1],
            "P_positive": p_positive,
            "P_negative": p_negative,
            "pct_in_ROPE": in_rope,
        })

        direction = "+" if p_positive > p_negative else "-"
        p_dir = max(p_positive, p_negative)
        print(f"  {pred}: β={mean_val:.3f} [{hdi[0]:.3f}, {hdi[1]:.3f}] "
              f"P(β{direction}0)={p_dir:.3f} | ROPE={in_rope:.1%}")

# ── 5. Analysis 2: Full model with metacognition ─────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 2: clinical_z ~ log_ce_z + log_cd_z + discrepancy_z + calibration_z")
print("=" * 70)

results_full = []
idata_store = {}  # store for forest plot

for raw_col, label in CLINICAL_SCALES:
    print(f"\n--- {label} ---")
    outcome_col = f"{label}_z"

    model_df = df[["log_ce_z", "log_cd_z", "discrepancy_z", "calibration_z", outcome_col]].dropna().copy()
    model_df.columns = ["log_ce_z", "log_cd_z", "discrepancy_z", "calibration_z", "y"]

    model = bmb.Model("y ~ log_ce_z + log_cd_z + discrepancy_z + calibration_z", data=model_df)
    idata = model.fit(draws=2000, chains=4, target_accept=0.9,
                      init="adapt_diag", random_seed=42)

    idata_store[label] = idata

    summary = az.summary(idata, hdi_prob=0.94,
                         var_names=["log_ce_z", "log_cd_z", "discrepancy_z", "calibration_z", "Intercept"])
    print(summary)

    for pred in ["log_ce_z", "log_cd_z", "discrepancy_z", "calibration_z"]:
        post = idata.posterior[pred].values.flatten()
        mean_val = post.mean()
        hdi = az.hdi(post, hdi_prob=0.94)
        p_positive = (post > 0).mean()
        p_negative = (post < 0).mean()
        in_rope = ((post > ROPE[0]) & (post < ROPE[1])).mean()

        results_full.append({
            "clinical_measure": label,
            "predictor": pred.replace("_z", ""),
            "model": "full_metacog",
            "posterior_mean": mean_val,
            "hdi_low": hdi[0],
            "hdi_high": hdi[1],
            "P_positive": p_positive,
            "P_negative": p_negative,
            "pct_in_ROPE": in_rope,
        })

        direction = "+" if p_positive > p_negative else "-"
        p_dir = max(p_positive, p_negative)
        print(f"  {pred}: β={mean_val:.3f} [{hdi[0]:.3f}, {hdi[1]:.3f}] "
              f"P(β{direction}0)={p_dir:.3f} | ROPE={in_rope:.1%}")

# ── 6. Save summary table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

all_results = pd.DataFrame(results_simple + results_full)
all_results.to_csv(OUT_TABLE, index=False)
print(f"Saved table: {OUT_TABLE}")

# Print summary for full model
print("\n" + "=" * 70)
print("SUMMARY TABLE — Full model (clinical_z ~ log_ce + log_cd + discrepancy + calibration)")
print("=" * 70)

full_df = all_results[all_results["model"] == "full_metacog"].copy()
pivot = full_df.pivot(index="clinical_measure", columns="predictor",
                      values=["posterior_mean", "hdi_low", "hdi_high", "pct_in_ROPE"])

for label in [lab for _, lab in CLINICAL_SCALES]:
    row = full_df[full_df["clinical_measure"] == label]
    print(f"\n{label}:")
    for _, r in row.iterrows():
        rope_flag = " *** NULL ***" if r["pct_in_ROPE"] > 0.90 else ""
        sig_flag = " ** CREDIBLE **" if (r["hdi_low"] > 0 or r["hdi_high"] < 0) else ""
        print(f"  {r['predictor']:>15s}: β={r['posterior_mean']:+.3f} "
              f"[{r['hdi_low']:.3f}, {r['hdi_high']:.3f}] "
              f"ROPE={r['pct_in_ROPE']:.1%}{rope_flag}{sig_flag}")

# ── 7. Forest plot ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CREATING FOREST PLOT")
print("=" * 70)

fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)

predictor_names = ["log_ce_z", "log_cd_z", "discrepancy_z", "calibration_z"]
predictor_labels = ["log(c_effort)", "log(c_death)", "Discrepancy", "Calibration"]
colors = ["#2166ac", "#b2182b", "#4dac26", "#984ea3"]

clinical_labels = [lab for _, lab in CLINICAL_SCALES]
y_positions = np.arange(len(clinical_labels))

for ax_idx, (pred_name, pred_label, color) in enumerate(zip(predictor_names, predictor_labels, colors)):
    ax = axes[ax_idx]

    for i, label in enumerate(clinical_labels):
        if label not in idata_store:
            continue
        idata = idata_store[label]
        post = idata.posterior[pred_name].values.flatten()
        mean_val = post.mean()
        hdi_94 = az.hdi(post, hdi_prob=0.94)
        hdi_50 = az.hdi(post, hdi_prob=0.50)

        # Draw 94% HDI (thin line)
        ax.plot([hdi_94[0], hdi_94[1]], [i, i], color=color, linewidth=1.5, alpha=0.6)
        # Draw 50% HDI (thick line)
        ax.plot([hdi_50[0], hdi_50[1]], [i, i], color=color, linewidth=4, alpha=0.8)
        # Draw posterior mean (dot)
        ax.plot(mean_val, i, "o", color=color, markersize=8, zorder=5)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    # ROPE shading
    ax.axvspan(ROPE[0], ROPE[1], color="gray", alpha=0.1, label="ROPE")
    ax.set_xlabel("Posterior β (standardized)", fontsize=11)
    ax.set_title(pred_label, fontsize=13, fontweight="bold")
    ax.set_yticks(y_positions)
    if ax_idx == 0:
        ax.set_yticklabels(clinical_labels, fontsize=10)
    ax.tick_params(axis="x", labelsize=9)

fig.suptitle("Bayesian Regression: Model Parameters & Metacognition → Clinical Outcomes",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved figure: {OUT_FIG}")

# ── 8. Key findings summary ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

full_df = all_results[all_results["model"] == "full_metacog"]

# Params in ROPE
params_rope = full_df[full_df["predictor"].isin(["log_ce", "log_cd"])]
avg_rope = params_rope["pct_in_ROPE"].mean()
print(f"\nModel parameters (log_ce, log_cd) average ROPE containment: {avg_rope:.1%}")

# Discrepancy effects
disc = full_df[full_df["predictor"] == "discrepancy"]
disc_sig = disc[(disc["hdi_low"] > 0) | (disc["hdi_high"] < 0)]
print(f"Discrepancy: credible effect on {len(disc_sig)}/{len(disc)} clinical measures")
for _, r in disc_sig.iterrows():
    print(f"  {r['clinical_measure']}: β={r['posterior_mean']:+.3f} [{r['hdi_low']:.3f}, {r['hdi_high']:.3f}]")

# Calibration effects
cal = full_df[full_df["predictor"] == "calibration"]
cal_sig = cal[(cal["hdi_low"] > 0) | (cal["hdi_high"] < 0)]
print(f"Calibration: credible effect on {len(cal_sig)}/{len(cal)} clinical measures")
for _, r in cal_sig.iterrows():
    print(f"  {r['clinical_measure']}: β={r['posterior_mean']:+.3f} [{r['hdi_low']:.3f}, {r['hdi_high']:.3f}]")

print("\nDone.")
