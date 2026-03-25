"""
Prereg H3: The reallocation strategy approximates the optimal policy.

H3a: Pearson r(reallocation_index, total_earnings) > 0, p < 0.01 one-tailed
H3b: Among suboptimal trials, proportion "too cautious" > 50%, t-test p < 0.05

Outputs:
  /workspace/results/stats/h3_optimality_results.json
  /workspace/results/h3_optimality_results_text.md
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
BEHAVIOR  = DATA_DIR / "behavior.csv"
BEHAVIOR_RICH = DATA_DIR / "behavior_rich.csv"
OUT_JSON  = Path("/workspace/results/stats/h3_optimality_results.json")
OUT_MD    = Path("/workspace/results/h3_optimality_results_text.md")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_p(p):
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}"

def _json_safe(obj):
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
beh = pd.read_csv(BEHAVIOR)
br = pd.read_csv(BEHAVIOR_RICH)

# Compute excess effort
br["effort_chosen"] = np.where(br["choice"] == 1, br["effort_H"], br["effort_L"])
br["excess_effort"] = br["mean_trial_effort"] - br["effort_chosen"]
br = br.dropna(subset=["excess_effort", "threat", "subj"])

N_subj = beh["subj"].nunique()
print(f"  N = {N_subj} subjects")

results = {"dataset": {"N_subjects": int(N_subj)}}

# ══════════════════════════════════════════════════════════════════════════════
# Compute per-subject shift scores and reallocation index
# ══════════════════════════════════════════════════════════════════════════════

# Δchoice = P(choose high | T=0.9) − P(choose high | T=0.1)
choice_by_threat = beh.groupby(["subj", "threat"])["choice"].mean().unstack("threat")
delta_choice = choice_by_threat[0.9] - choice_by_threat[0.1]

# Δvigor = excess_effort(T=0.9) − excess_effort(T=0.1)
vigor_by_threat = br.groupby(["subj", "threat"])["excess_effort"].mean().unstack("threat")
delta_vigor = vigor_by_threat[0.9] - vigor_by_threat[0.1]

# Reallocation index = |Δchoice| + |Δvigor| (both z-scored first)
common = delta_choice.index.intersection(delta_vigor.index)
dc = delta_choice.loc[common]
dv = delta_vigor.loc[common]

dc_z = (np.abs(dc) - np.abs(dc).mean()) / np.abs(dc).std()
dv_z = (np.abs(dv) - np.abs(dv).mean()) / np.abs(dv).std()
reallocation_index = dc_z + dv_z
reallocation_index.name = "reallocation_index"

# Total earnings per subject
earnings = br.groupby("subj")["trialReward"].sum()
earnings = earnings.loc[common]

print(f"  Reallocation index: M = {reallocation_index.mean():.3f}, SD = {reallocation_index.std():.3f}")
print(f"  Total earnings: M = {earnings.mean():.1f}, SD = {earnings.std():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# H3a: Pearson r(reallocation_index, total_earnings) > 0, p < 0.01 one-tailed
# ══════════════════════════════════════════════════════════════════════════════
print("\n── H3a: Reallocation predicts earnings ──")

r_h3a, p_h3a_two = stats.pearsonr(reallocation_index.values, earnings.values)
p_h3a_one = p_h3a_two / 2 if r_h3a > 0 else 1 - p_h3a_two / 2

h3a_supported = (r_h3a > 0 and p_h3a_one < 0.01)
print(f"  r = {r_h3a:.4f}, p(one-tailed) = {p_h3a_one:.4e}")
print(f"  H3a: {'SUPPORTED' if h3a_supported else 'NOT SUPPORTED'}")

results["H3a"] = {
    "r": float(r_h3a),
    "p_two_tailed": float(p_h3a_two),
    "p_one_tailed": float(p_h3a_one),
    "N": int(len(common)),
    "reallocation_mean": float(reallocation_index.mean()),
    "reallocation_sd": float(reallocation_index.std()),
    "earnings_mean": float(earnings.mean()),
    "earnings_sd": float(earnings.std()),
    "supported": bool(h3a_supported),
}

# ══════════════════════════════════════════════════════════════════════════════
# H3b: Among suboptimal trials, proportion "too cautious" > 50%
# ══════════════════════════════════════════════════════════════════════════════
print("\n── H3b: Dominant deviation is excessive caution ──")

# Compute expected value using EMPIRICAL escape rates from attack trials.
# This reflects the actual task dynamics (including effort/speed) rather than
# model-derived S, which with λ=14 makes distance nearly irrelevant.
R_H = 5.0
R_L = 1.0
C = 5.0  # capture penalty

# Empirical escape rate by (threat, chosen_distance) from attack trials
attack = br[br["isAttackTrial"] == 1].copy()
attack["distance_chosen"] = np.where(attack["choice"] == 1, attack["distance_H"], attack["distance_L"])
attack["escaped"] = (attack["trialEndState"] == "escaped").astype(int)
emp_escape = attack.groupby(["threat", "distance_chosen"])["escaped"].mean().to_dict()

print(f"  Empirical escape rates from {len(attack)} attack trials:")
for (t, d), rate in sorted(emp_escape.items()):
    print(f"    T={t}, D={d}: {rate:.3f}")

# EV = (1-T)*R + T*[P_esc*R - (1-P_esc)*C]
# Accounts for: no-attack trials (free reward) + attack trials (empirical escape)
beh_ev = beh.copy()

def compute_ev(T, D, R):
    p_esc = emp_escape.get((T, D), 0.5)
    return (1 - T) * R + T * (p_esc * R - (1 - p_esc) * C)

beh_ev["EV_H"] = beh_ev.apply(lambda r: compute_ev(r["threat"], r["distance_H"], R_H), axis=1)
beh_ev["EV_L"] = beh_ev.apply(lambda r: compute_ev(r["threat"], r["distance_L"], R_L), axis=1)
beh_ev["optimal_choice"] = (beh_ev["EV_H"] > beh_ev["EV_L"]).astype(int)

# Print EV by condition
print("\n  EV by condition:")
for _, row in beh_ev.groupby(["threat", "distance_H"]).first()[["EV_H", "EV_L"]].iterrows():
    t, d = _
    opt = "HIGH" if row["EV_H"] > row["EV_L"] else "LOW"
    print(f"    T={t}, D_H={d}: EV_H={row['EV_H']:.2f}, EV_L={row['EV_L']:.2f} -> {opt}")

# Classify errors
beh_ev["is_optimal"] = (beh_ev["choice"] == beh_ev["optimal_choice"])
beh_ev["error_type"] = np.where(
    beh_ev["is_optimal"], "optimal",
    np.where(
        (beh_ev["choice"] == 0) & (beh_ev["optimal_choice"] == 1),
        "too_cautious",  # chose safe when risky was better
        "too_risky"      # chose risky when safe was better
    )
)

# Overall stats
n_optimal = beh_ev["is_optimal"].sum()
n_suboptimal = (~beh_ev["is_optimal"]).sum()
n_cautious = (beh_ev["error_type"] == "too_cautious").sum()
n_risky = (beh_ev["error_type"] == "too_risky").sum()

print(f"  Optimal trials: {n_optimal} ({n_optimal/len(beh_ev)*100:.1f}%)")
print(f"  Suboptimal trials: {n_suboptimal} ({n_suboptimal/len(beh_ev)*100:.1f}%)")
print(f"    Too cautious: {n_cautious} ({n_cautious/n_suboptimal*100:.1f}% of errors)")
print(f"    Too risky: {n_risky} ({n_risky/n_suboptimal*100:.1f}% of errors)")

# Per-subject proportion of cautious errors (among suboptimal trials)
subopt = beh_ev[~beh_ev["is_optimal"]]
subj_cautious_pct = subopt.groupby("subj").apply(
    lambda x: (x["error_type"] == "too_cautious").mean()
)
# Drop subjects with no suboptimal trials
subj_cautious_pct = subj_cautious_pct.dropna()

mean_cautious_pct = subj_cautious_pct.mean()
print(f"  Per-subject mean % too-cautious errors: {mean_cautious_pct*100:.1f}%")

# One-sample t-test against 50%
t_h3b, p_h3b_two = stats.ttest_1samp(subj_cautious_pct.values, 0.5)
p_h3b_one = p_h3b_two / 2 if t_h3b > 0 else 1 - p_h3b_two / 2

h3b_supported = (mean_cautious_pct > 0.5 and p_h3b_one < 0.05)
print(f"  t = {t_h3b:.2f}, p(one-tailed) = {p_h3b_one:.4e}")
print(f"  H3b: {'SUPPORTED' if h3b_supported else 'NOT SUPPORTED'}")

results["H3b"] = {
    "n_optimal": int(n_optimal),
    "n_suboptimal": int(n_suboptimal),
    "n_too_cautious": int(n_cautious),
    "n_too_risky": int(n_risky),
    "pct_cautious_of_errors": float(n_cautious / n_suboptimal),
    "per_subject_mean_pct_cautious": float(mean_cautious_pct),
    "per_subject_sd_pct_cautious": float(subj_cautious_pct.std()),
    "t": float(t_h3b),
    "p_two_tailed": float(p_h3b_two),
    "p_one_tailed": float(p_h3b_one),
    "N_subjects_with_errors": int(len(subj_cautious_pct)),
    "supported": bool(h3b_supported),
    "method": "empirical_escape_rates",
}

# Also store EV breakdown for reference
ev_by_condition = beh_ev.groupby(["threat", "distance_H"]).agg(
    EV_H=("EV_H", "first"), EV_L=("EV_L", "first"),
    optimal_is_high=("optimal_choice", "mean"),
    actual_chose_high=("choice", "mean"),
).reset_index()
results["ev_by_condition"] = ev_by_condition.to_dict(orient="records")

# ══════════════════════════════════════════════════════════════════════════════
# OVERALL
# ══════════════════════════════════════════════════════════════════════════════
h3_overall = h3a_supported and h3b_supported
results["H3_overall"] = {
    "H3a_supported": bool(h3a_supported),
    "H3b_supported": bool(h3b_supported),
    "overall": bool(h3_overall),
}

print(f"\n{'='*60}")
print("H3 OVERALL VERDICT")
print(f"{'='*60}")
print(f"  H3a (reallocation → earnings): {'SUPPORTED' if h3a_supported else 'NOT SUPPORTED'}")
print(f"  H3b (cautious > 50%):          {'SUPPORTED' if h3b_supported else 'NOT SUPPORTED'}")
print(f"  H3 overall:                     {'SUPPORTED' if h3_overall else 'NOT SUPPORTED'}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=_json_safe)
print(f"\nSaved: {OUT_JSON}")

# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════
md = f"""# H3 Results: Optimality of Reallocation

## Overview

H3 tests whether the threat-driven reallocation (shifting choice toward safety while increasing motor effort) approximates a rational policy. N = {N_subj}.

---

## H3a: Reallocation predicts earnings

Reallocation index = |Δchoice|_z + |Δvigor|_z (sum of z-scored absolute shifts).

Pearson r(reallocation_index, total_earnings) = **{r_h3a:.3f}**, {fmt_p(p_h3a_one)} (one-tailed).

| Measure | Mean | SD |
|---|---|---|
| Reallocation index | {reallocation_index.mean():.3f} | {reallocation_index.std():.3f} |
| Total earnings (pts) | {earnings.mean():.1f} | {earnings.std():.1f} |

Criterion: r > 0, p < 0.01 (one-tailed). **H3a: {'SUPPORTED' if h3a_supported else 'NOT SUPPORTED'}.**

---

## H3b: Dominant deviation is excessive caution

Expected values computed using empirical escape rates from {len(attack)} attack trials, conditioned on threat and chosen distance. EV = (1−T)R + T[P_esc·R − (1−P_esc)·C], where C = 5 (capture cost). This reflects actual task dynamics including effort/speed effects on survival, rather than model-derived S.

| Category | N trials | % |
|---|---|---|
| Optimal | {n_optimal:,} | {n_optimal/len(beh_ev)*100:.1f}% |
| Too cautious | {n_cautious:,} | {n_cautious/len(beh_ev)*100:.1f}% |
| Too risky | {n_risky:,} | {n_risky/len(beh_ev)*100:.1f}% |

Among suboptimal trials: **{n_cautious/n_suboptimal*100:.1f}%** too cautious vs **{n_risky/n_suboptimal*100:.1f}%** too risky.

Per-subject mean % cautious errors: {mean_cautious_pct*100:.1f}% (SD = {subj_cautious_pct.std()*100:.1f}%).
One-sample t-test vs 50%: t = {t_h3b:.2f}, {fmt_p(p_h3b_one)} (one-tailed).

Criterion: > 50%, p < 0.05. **H3b: {'SUPPORTED' if h3b_supported else 'NOT SUPPORTED'}.**

---

## Summary

| Sub-hypothesis | Statistic | p (one-tailed) | Criterion | Result |
|---|---|---|---|---|
| H3a (reallocation → earnings) | r = {r_h3a:.3f} | {fmt_p(p_h3a_one)} | r > 0, p < 0.01 | {'PASS' if h3a_supported else 'FAIL'} |
| H3b (cautious > 50%) | {mean_cautious_pct*100:.1f}%, t = {t_h3b:.2f} | {fmt_p(p_h3b_one)} | > 50%, p < 0.05 | {'PASS' if h3b_supported else 'FAIL'} |

**H3 overall: {'SUPPORTED' if h3_overall else 'NOT SUPPORTED'}.**
"""

with open(OUT_MD, "w") as f:
    f.write(md)
print(f"Saved: {OUT_MD}")
