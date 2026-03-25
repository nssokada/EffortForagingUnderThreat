"""
Prereg H2: Choice shift and vigor shift under threat are coherently coupled.

H2a: Pearson r(Δchoice, Δvigor) < 0, p < 0.01 one-tailed
H2b: Split-half robustness — r(Δchoice_odd, Δvigor_even) and reverse, p < 0.05

Outputs:
  /workspace/results/stats/h2_coupling_results.json
  /workspace/results/h2_coupling_results_text.md
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
OUT_JSON  = Path("/workspace/results/stats/h2_coupling_results.json")
OUT_MD    = Path("/workspace/results/h2_coupling_results_text.md")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_p(p):
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}"

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

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

# Compute excess effort for behavior_rich
br["effort_chosen"] = np.where(br["choice"] == 1, br["effort_H"], br["effort_L"])
br["excess_effort"] = br["mean_trial_effort"] - br["effort_chosen"]
br = br.dropna(subset=["excess_effort", "threat", "subj"])

N_subj = beh["subj"].nunique()
print(f"  behavior.csv: {len(beh)} trials, {N_subj} subjects")
print(f"  behavior_rich.csv: {len(br)} trials (with excess effort)")

results = {"dataset": {"N_subjects": int(N_subj)}}

# ══════════════════════════════════════════════════════════════════════════════
# Compute per-subject shift scores
# ══════════════════════════════════════════════════════════════════════════════

# Δchoice = P(choose high | T=0.9) − P(choose high | T=0.1)
choice_by_threat = beh.groupby(["subj", "threat"])["choice"].mean().unstack("threat")
delta_choice = choice_by_threat[0.9] - choice_by_threat[0.1]
delta_choice.name = "delta_choice"

# Δvigor = mean excess_effort(T=0.9) − mean excess_effort(T=0.1)
vigor_by_threat = br.groupby(["subj", "threat"])["excess_effort"].mean().unstack("threat")
delta_vigor = vigor_by_threat[0.9] - vigor_by_threat[0.1]
delta_vigor.name = "delta_vigor"

# Align
common = delta_choice.index.intersection(delta_vigor.index)
dc = delta_choice.loc[common]
dv = delta_vigor.loc[common]
N = len(common)

print(f"\n  Shift scores computed for N = {N} subjects")
print(f"  Δchoice: M = {dc.mean():.3f}, SD = {dc.std():.3f}")
print(f"  Δvigor:  M = {dv.mean():.4f}, SD = {dv.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# H2a: Pearson r(Δchoice, Δvigor) < 0, p < 0.01 one-tailed
# ══════════════════════════════════════════════════════════════════════════════
print("\n── H2a: Choice-vigor coupling ──")

r_full, p_full_two = stats.pearsonr(dc.values, dv.values)
# One-tailed: we predict r < 0, so p_one = p_two/2 if r < 0, else 1 - p_two/2
p_full_one = p_full_two / 2 if r_full < 0 else 1 - p_full_two / 2

h2a_supported = (r_full < 0 and p_full_one < 0.01)
print(f"  r = {r_full:.4f}, p(two-tailed) = {p_full_two:.4e}, p(one-tailed) = {p_full_one:.4e}")
print(f"  H2a: {'SUPPORTED' if h2a_supported else 'NOT SUPPORTED'}")

results["H2a"] = {
    "r": float(r_full),
    "p_two_tailed": float(p_full_two),
    "p_one_tailed": float(p_full_one),
    "N": int(N),
    "delta_choice_mean": float(dc.mean()),
    "delta_choice_sd": float(dc.std()),
    "delta_vigor_mean": float(dv.mean()),
    "delta_vigor_sd": float(dv.std()),
    "supported": bool(h2a_supported),
}

# ══════════════════════════════════════════════════════════════════════════════
# H2b: Split-half robustness
# ══════════════════════════════════════════════════════════════════════════════
print("\n── H2b: Split-half robustness ──")

# Split trials into odd/even by trial index within each subject
# behavior.csv has 'trial' column (1-indexed)
beh_odd = beh[beh["trial"] % 2 == 1]
beh_even = beh[beh["trial"] % 2 == 0]

# behavior_rich also has 'trial'
br_odd = br[br["trial"] % 2 == 1]
br_even = br[br["trial"] % 2 == 0]

def compute_shift(choice_df, vigor_df):
    """Compute per-subject Δchoice and Δvigor from given trial subsets."""
    ct = choice_df.groupby(["subj", "threat"])["choice"].mean().unstack("threat")
    dc_half = ct[0.9] - ct[0.1]

    vt = vigor_df.groupby(["subj", "threat"])["excess_effort"].mean().unstack("threat")
    dv_half = vt[0.9] - vt[0.1]

    return dc_half, dv_half

dc_odd, dv_odd = compute_shift(beh_odd, br_odd)
dc_even, dv_even = compute_shift(beh_even, br_even)

# Split 1: Δchoice_odd vs Δvigor_even
common_s1 = dc_odd.index.intersection(dv_even.index)
r_s1, p_s1_two = stats.pearsonr(dc_odd.loc[common_s1].values, dv_even.loc[common_s1].values)
p_s1_one = p_s1_two / 2 if r_s1 < 0 else 1 - p_s1_two / 2

# Split 2: Δchoice_even vs Δvigor_odd
common_s2 = dc_even.index.intersection(dv_odd.index)
r_s2, p_s2_two = stats.pearsonr(dc_even.loc[common_s2].values, dv_odd.loc[common_s2].values)
p_s2_one = p_s2_two / 2 if r_s2 < 0 else 1 - p_s2_two / 2

s1_pass = (r_s1 < 0 and p_s1_one < 0.05)
s2_pass = (r_s2 < 0 and p_s2_one < 0.05)
h2b_supported = s1_pass and s2_pass

print(f"  Split 1 (Δchoice_odd vs Δvigor_even): r = {r_s1:.4f}, p(one-tailed) = {p_s1_one:.4e}, N = {len(common_s1)} → {'PASS' if s1_pass else 'FAIL'}")
print(f"  Split 2 (Δchoice_even vs Δvigor_odd): r = {r_s2:.4f}, p(one-tailed) = {p_s2_one:.4e}, N = {len(common_s2)} → {'PASS' if s2_pass else 'FAIL'}")
print(f"  H2b: {'SUPPORTED' if h2b_supported else 'NOT SUPPORTED'}")

results["H2b"] = {
    "split1_choice_odd_vigor_even": {
        "r": float(r_s1), "p_two_tailed": float(p_s1_two),
        "p_one_tailed": float(p_s1_one), "N": int(len(common_s1)),
        "pass": bool(s1_pass),
    },
    "split2_choice_even_vigor_odd": {
        "r": float(r_s2), "p_two_tailed": float(p_s2_two),
        "p_one_tailed": float(p_s2_one), "N": int(len(common_s2)),
        "pass": bool(s2_pass),
    },
    "supported": bool(h2b_supported),
}

# ══════════════════════════════════════════════════════════════════════════════
# OVERALL
# ══════════════════════════════════════════════════════════════════════════════
h2_overall = h2a_supported  # H2b is robustness, not required for support
results["H2_overall"] = {
    "H2a_supported": bool(h2a_supported),
    "H2b_supported": bool(h2b_supported),
    "overall": bool(h2_overall),
}

print(f"\n{'='*60}")
print("H2 OVERALL VERDICT")
print(f"{'='*60}")
print(f"  H2a (coupling):     {'SUPPORTED' if h2a_supported else 'NOT SUPPORTED'}")
print(f"  H2b (split-half):   {'SUPPORTED' if h2b_supported else 'NOT SUPPORTED'}")
print(f"  H2 overall:         {'SUPPORTED' if h2_overall else 'NOT SUPPORTED'}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=_json_safe)
print(f"\nSaved: {OUT_JSON}")

# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════
md = f"""# H2 Results: Choice-Vigor Coupling Under Threat

## Overview

H2 tests whether the shift in choice toward safety under threat is coherently coupled with the shift in motor effort. Participants who avoid high-effort options more under threat should also increase their excess pressing more — a negative correlation between Δchoice and Δvigor.

- **Δchoice** = P(choose high | T=0.9) − P(choose high | T=0.1) (negative = more avoidance)
- **Δvigor** = excess_effort(T=0.9) − excess_effort(T=0.1) (positive = more excess effort)

N = {N} subjects.

---

## H2a: Choice-vigor coupling

Pearson r(Δchoice, Δvigor) = **{r_full:.3f}**, {fmt_p(p_full_one)} (one-tailed).

| Measure | Mean | SD |
|---|---|---|
| Δchoice | {dc.mean():.3f} | {dc.std():.3f} |
| Δvigor | {dv.mean():.4f} | {dv.std():.4f} |

Criterion: r < 0, p < 0.01 (one-tailed). **H2a: {'SUPPORTED' if h2a_supported else 'NOT SUPPORTED'}.**

---

## H2b: Split-half robustness

Shift scores computed from independent trial halves (odd vs. even) to rule out shared condition variance.

| Split | r | p (one-tailed) | N | Result |
|---|---|---|---|---|
| Δchoice(odd) vs Δvigor(even) | {r_s1:.3f} | {fmt_p(p_s1_one)} | {len(common_s1)} | {'PASS' if s1_pass else 'FAIL'} |
| Δchoice(even) vs Δvigor(odd) | {r_s2:.3f} | {fmt_p(p_s2_one)} | {len(common_s2)} | {'PASS' if s2_pass else 'FAIL'} |

Criterion: both r < 0, p < 0.05 (one-tailed). **H2b: {'SUPPORTED' if h2b_supported else 'NOT SUPPORTED'}.**

---

## Summary

| Sub-hypothesis | Test | r | p (one-tailed) | Criterion | Result |
|---|---|---|---|---|---|
| H2a (coupling) | Pearson r | {r_full:.3f} | {fmt_p(p_full_one)} | r < 0, p < 0.01 | {'PASS' if h2a_supported else 'FAIL'} |
| H2b split 1 | Pearson r | {r_s1:.3f} | {fmt_p(p_s1_one)} | r < 0, p < 0.05 | {'PASS' if s1_pass else 'FAIL'} |
| H2b split 2 | Pearson r | {r_s2:.3f} | {fmt_p(p_s2_one)} | r < 0, p < 0.05 | {'PASS' if s2_pass else 'FAIL'} |

**H2 overall: {'SUPPORTED' if h2_overall else 'NOT SUPPORTED'}.**
"""

with open(OUT_MD, "w") as f:
    f.write(md)
print(f"Saved: {OUT_MD}")
