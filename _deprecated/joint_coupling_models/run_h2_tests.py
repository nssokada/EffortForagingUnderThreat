"""
H2 Hypothesis Tests: Survival computation predicts subjective affect.

H2a: S → anxiety (negative) and S → confidence (positive) in LMMs
H2b: k moderates S → affect (S_z × k_z interaction)
H2c: Per-subject threat sensitivity of choice correlates with threat sensitivity of anxiety/confidence

Outputs:
  /workspace/results/stats/h2_results.json
  /workspace/results/h2_results_text.md
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
FEELINGS  = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950/feelings.csv")
BEHAVIOR  = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950/behavior.csv")
PARAMS    = Path("/workspace/results/stats/unified_3param_clean.csv")
OUT_JSON  = Path("/workspace/results/stats/h2_results.json")
OUT_MD    = Path("/workspace/results/h2_results_text.md")

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
feelings = pd.read_csv(FEELINGS)
behavior = pd.read_csv(BEHAVIOR)
params   = pd.read_csv(PARAMS)   # subj, k, beta, alpha

# ── Compute S (survival probability) ─────────────────────────────────────────
# S = (1-T) + T / (1 + λ·D),  λ=2.0,  D = distance + 1
LAMBDA = 2.0
feelings = feelings.copy()
feelings["D"] = feelings["distance"] + 1          # distance in [0,1,2] → D in [1,2,3]
feelings["S"] = (1 - feelings["threat"]) + feelings["threat"] / (1 + LAMBDA * feelings["D"])

# Sanity check
print("S range:", feelings["S"].min().round(3), "-", feelings["S"].max().round(3))

# ── Within-person z-scores ────────────────────────────────────────────────────
def zscore_within(df, col):
    """Z-score col within each subject. Returns series aligned to df index."""
    return df.groupby("subj")[col].transform(lambda x: (x - x.mean()) / x.std(ddof=1))

feelings["S_z"]        = zscore_within(feelings, "S")
feelings["response_z"] = zscore_within(feelings, "response")

# Merge k (log-transformed for moderation) onto feelings
params_sub = params[["subj", "k", "beta"]].copy()
params_sub["k_z"] = (np.log(params_sub["k"]) - np.log(params_sub["k"]).mean()) / np.log(params_sub["k"]).std(ddof=1)
params_sub["beta_z"] = (np.log(params_sub["beta"]) - np.log(params_sub["beta"]).mean()) / np.log(params_sub["beta"]).std(ddof=1)

feelings = feelings.merge(params_sub[["subj", "k", "k_z", "beta", "beta_z"]], on="subj", how="left")

# Split by question type
anx_df  = feelings[feelings["questionType"] == 5].copy()
conf_df = feelings[feelings["questionType"] == 6].copy()

print(f"Anxiety trials: {len(anx_df)}, Confidence trials: {len(conf_df)}")

# ═══════════════════════════════════════════════════════════════════════════════
# H2a: S → anxiety and S → confidence (LMM, random subject intercepts)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── H2a: S → affect LMMs ──")

def fit_lmm_s_affect(df, label):
    """Fit LMM: response_z ~ S_z + (1|subj), return result dict."""
    model = smf.mixedlm("response_z ~ S_z", df, groups=df["subj"])
    result = model.fit(method="lbfgs", maxiter=5000)
    coef   = result.fe_params["S_z"]
    se     = result.bse_fe["S_z"]
    pval   = result.pvalues["S_z"]
    ci_lo, ci_hi = result.conf_int().loc["S_z"]
    n_subj = df["subj"].nunique()
    n_obs  = len(df)
    t_val  = coef / se
    print(f"  {label}: β={coef:.4f}, SE={se:.4f}, t={t_val:.3f}, p={pval:.4e}, "
          f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}], N_subj={n_subj}, N_obs={n_obs}")
    return {
        "beta": round(float(coef), 4),
        "se":   round(float(se), 4),
        "t":    round(float(t_val), 4),
        "p":    float(pval),
        "ci_lower": round(float(ci_lo), 4),
        "ci_upper": round(float(ci_hi), 4),
        "n_subjects": int(n_subj),
        "n_observations": int(n_obs),
    }

h2a_anx  = fit_lmm_s_affect(anx_df,  "S→Anxiety")
h2a_conf = fit_lmm_s_affect(conf_df, "S→Confidence")

# ═══════════════════════════════════════════════════════════════════════════════
# H2b: k moderates S → affect (S_z + S_z:k_z interaction)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── H2b: k moderates S → affect ──")

def fit_lmm_moderation(df, label):
    """Fit LMM: response_z ~ S_z + k_z + S_z:k_z + (1|subj)."""
    model  = smf.mixedlm("response_z ~ S_z + k_z + S_z:k_z", df, groups=df["subj"])
    result = model.fit(method="lbfgs", maxiter=5000)
    coefs  = result.fe_params
    ses    = result.bse_fe
    pvals  = result.pvalues
    cis    = result.conf_int()

    out = {}
    for term in ["S_z", "k_z", "S_z:k_z"]:
        coef  = coefs[term]
        se    = ses[term]
        t_val = coef / se
        pval  = pvals[term]
        ci_lo, ci_hi = cis.loc[term]
        print(f"  {label} [{term}]: β={coef:.4f}, SE={se:.4f}, t={t_val:.3f}, p={pval:.4e}")
        out[term] = {
            "beta": round(float(coef), 4),
            "se":   round(float(se), 4),
            "t":    round(float(t_val), 4),
            "p":    float(pval),
            "ci_lower": round(float(ci_lo), 4),
            "ci_upper": round(float(ci_hi), 4),
        }
    out["n_subjects"]     = int(df["subj"].nunique())
    out["n_observations"] = int(len(df))
    return out

h2b_anx  = fit_lmm_moderation(anx_df,  "S×k→Anxiety")
h2b_conf = fit_lmm_moderation(conf_df, "S×k→Confidence")

# ═══════════════════════════════════════════════════════════════════════════════
# H2c: Cross-domain — per-subject threat slopes (choice vs affect)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── H2c: Cross-domain threat sensitivity correlations ──")

# Per-subject choice threat slope (linear regression of choice ~ threat per subj)
def subj_threat_slope(df, dv, iv="threat"):
    """Return per-subject slope (beta) from OLS dv ~ iv for each subject."""
    slopes = {}
    for subj, grp in df.groupby("subj"):
        if grp[iv].nunique() < 2 or len(grp) < 4:
            continue
        x = grp[iv].values
        y = grp[dv].values
        slope = np.polyfit(x, y, 1)[0]
        slopes[subj] = slope
    return pd.Series(slopes, name="slope")

choice_threat_slopes = subj_threat_slope(behavior, "choice", "threat")

# Per-subject anxiety threat slope
anx_threat_slopes  = subj_threat_slope(anx_df, "response", "threat")
conf_threat_slopes = subj_threat_slope(conf_df, "response", "threat")

print(f"  N subjects with choice slopes: {len(choice_threat_slopes)}")
print(f"  N subjects with anxiety slopes: {len(anx_threat_slopes)}")
print(f"  N subjects with confidence slopes: {len(conf_threat_slopes)}")

def correlate_slopes(s1, s2, label):
    """Pearson correlation between two slope series aligned on index."""
    idx = s1.index.intersection(s2.index)
    x, y = s1[idx].values, s2[idx].values
    r, p = stats.pearsonr(x, y)
    n = len(idx)
    print(f"  {label}: r={r:.4f}, p={p:.4e}, n={n}")
    return {"r": round(float(r), 4), "p": float(p), "n": int(n)}

h2c_choice_anxiety    = correlate_slopes(choice_threat_slopes, anx_threat_slopes,
                                          "choice-threat-slope vs anxiety-threat-slope")
h2c_choice_confidence = correlate_slopes(choice_threat_slopes, conf_threat_slopes,
                                          "choice-threat-slope vs confidence-threat-slope")

# Also report mean slopes for context
print(f"  Mean choice threat slope: {choice_threat_slopes.mean():.4f}")
print(f"  Mean anxiety threat slope: {anx_threat_slopes.mean():.4f}")
print(f"  Mean confidence threat slope: {conf_threat_slopes.mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Save JSON results
# ═══════════════════════════════════════════════════════════════════════════════
results = {
    "hypothesis": "H2",
    "description": "Survival computation predicts subjective affect",
    "survival_formula": "S = (1-T) + T / (1 + lambda * D), lambda=2.0, D=distance+1",
    "lambda": LAMBDA,
    "H2a": {
        "description": "S_z predicts affect (LMM, random subject intercepts, within-person z-scored)",
        "anxiety":    h2a_anx,
        "confidence": h2a_conf,
    },
    "H2b": {
        "description": "k_z moderates S_z → affect (S_z + k_z + S_z:k_z interaction in LMM)",
        "anxiety":    h2b_anx,
        "confidence": h2b_conf,
    },
    "H2c": {
        "description": "Cross-domain: per-subject choice threat sensitivity correlates with affect threat sensitivity",
        "mean_slopes": {
            "choice_threat_slope":     round(float(choice_threat_slopes.mean()), 4),
            "anxiety_threat_slope":    round(float(anx_threat_slopes.mean()), 4),
            "confidence_threat_slope": round(float(conf_threat_slopes.mean()), 4),
        },
        "choice_vs_anxiety":    h2c_choice_anxiety,
        "choice_vs_confidence": h2c_choice_confidence,
    },
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON written to {OUT_JSON}")

# ═══════════════════════════════════════════════════════════════════════════════
# Save prose summary
# ═══════════════════════════════════════════════════════════════════════════════
def fmt_p(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"

def sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "n.s."

prose = f"""# H2 Results: Survival Computation Predicts Subjective Affect

**Date:** 2026-03-20
**Model:** S = (1-T) + T / (1 + λD), λ={LAMBDA}, D = distance + 1 (game units)
**Participants:** N={anx_df['subj'].nunique()} (anxiety), N={conf_df['subj'].nunique()} (confidence)

---

## H2a: S Predicts Anxiety and Confidence

Within-person z-scores of S and response were computed for each participant.
Linear mixed models were fit with random subject intercepts (MixedLM, statsmodels).

### Anxiety (questionType=5, n={h2a_anx['n_observations']} trials, N={h2a_anx['n_subjects']} subjects)

Higher survival probability predicted **lower anxiety**:
β = {h2a_anx['beta']:.4f}, SE = {h2a_anx['se']:.4f}, t = {h2a_anx['t']:.3f}, {fmt_p(h2a_anx['p'])} {sig_label(h2a_anx['p'])}
95% CI: [{h2a_anx['ci_lower']:.4f}, {h2a_anx['ci_upper']:.4f}]

**Interpretation:** As computed survival probability increases (safer conditions), anxiety decreases. Direction is {('consistent with H2' if h2a_anx['beta'] < 0 else 'OPPOSITE to H2 prediction (predicted negative)')} — the model predicts a negative association.

### Confidence (questionType=6, n={h2a_conf['n_observations']} trials, N={h2a_conf['n_subjects']} subjects)

Higher survival probability predicted **higher confidence**:
β = {h2a_conf['beta']:.4f}, SE = {h2a_conf['se']:.4f}, t = {h2a_conf['t']:.3f}, {fmt_p(h2a_conf['p'])} {sig_label(h2a_conf['p'])}
95% CI: [{h2a_conf['ci_lower']:.4f}, {h2a_conf['ci_upper']:.4f}]

**Interpretation:** As survival probability increases, confidence increases. Direction is {('consistent with H2' if h2a_conf['beta'] > 0 else 'OPPOSITE to H2 prediction (predicted positive)')} — the model predicts a positive association.

---

## H2b: Effort Discounting (k) Moderates S → Affect

LMMs including S_z + k_z + S_z:k_z (log-transformed k, population z-scored) with random subject intercepts.
Higher k = stronger effort discounting. A significant S_z:k_z interaction would indicate that
individuals who weight effort more heavily also show stronger coupling between survival and affect.

### Anxiety Moderation (n={h2b_anx['n_observations']} trials, N={h2b_anx['n_subjects']} subjects)

| Term | β | SE | t | p |
|------|---|----|---|---|
| S_z | {h2b_anx['S_z']['beta']:.4f} | {h2b_anx['S_z']['se']:.4f} | {h2b_anx['S_z']['t']:.3f} | {fmt_p(h2b_anx['S_z']['p'])} {sig_label(h2b_anx['S_z']['p'])} |
| k_z | {h2b_anx['k_z']['beta']:.4f} | {h2b_anx['k_z']['se']:.4f} | {h2b_anx['k_z']['t']:.3f} | {fmt_p(h2b_anx['k_z']['p'])} {sig_label(h2b_anx['k_z']['p'])} |
| S_z:k_z | {h2b_anx['S_z:k_z']['beta']:.4f} | {h2b_anx['S_z:k_z']['se']:.4f} | {h2b_anx['S_z:k_z']['t']:.3f} | {fmt_p(h2b_anx['S_z:k_z']['p'])} {sig_label(h2b_anx['S_z:k_z']['p'])} |

### Confidence Moderation (n={h2b_conf['n_observations']} trials, N={h2b_conf['n_subjects']} subjects)

| Term | β | SE | t | p |
|------|---|----|---|---|
| S_z | {h2b_conf['S_z']['beta']:.4f} | {h2b_conf['S_z']['se']:.4f} | {h2b_conf['S_z']['t']:.3f} | {fmt_p(h2b_conf['S_z']['p'])} {sig_label(h2b_conf['S_z']['p'])} |
| k_z | {h2b_conf['k_z']['beta']:.4f} | {h2b_conf['k_z']['se']:.4f} | {h2b_conf['k_z']['t']:.3f} | {fmt_p(h2b_conf['k_z']['p'])} {sig_label(h2b_conf['k_z']['p'])} |
| S_z:k_z | {h2b_conf['S_z:k_z']['beta']:.4f} | {h2b_conf['S_z:k_z']['se']:.4f} | {h2b_conf['S_z:k_z']['t']:.3f} | {fmt_p(h2b_conf['S_z:k_z']['p'])} {sig_label(h2b_conf['S_z:k_z']['p'])} |

---

## H2c: Cross-Domain Threat Sensitivity

Per-subject threat sensitivity estimated as OLS slope of outcome on threat:
- **Choice threat sensitivity:** slope of choice (0=low, 1=high risky option) on threat
- **Anxiety threat sensitivity:** slope of anxiety rating on threat
- **Confidence threat sensitivity:** slope of confidence rating on threat

Mean slopes: choice = {results['H2c']['mean_slopes']['choice_threat_slope']:.4f}, anxiety = {results['H2c']['mean_slopes']['anxiety_threat_slope']:.4f}, confidence = {results['H2c']['mean_slopes']['confidence_threat_slope']:.4f}

### Choice-threat slope vs Anxiety-threat slope
Pearson r = {h2c_choice_anxiety['r']:.4f}, {fmt_p(h2c_choice_anxiety['p'])} {sig_label(h2c_choice_anxiety['p'])}, n = {h2c_choice_anxiety['n']}

{'**Significant positive correlation**: subjects who reduce risky choices more under threat also report larger anxiety increases under threat.' if h2c_choice_anxiety['p'] < 0.05 and h2c_choice_anxiety['r'] > 0 else '**Significant negative correlation**: subjects who reduce risky choices more under threat report smaller anxiety increases.' if h2c_choice_anxiety['p'] < 0.05 and h2c_choice_anxiety['r'] < 0 else '**Null result**: no significant correlation between choice threat sensitivity and anxiety threat sensitivity.'}

### Choice-threat slope vs Confidence-threat slope
Pearson r = {h2c_choice_confidence['r']:.4f}, {fmt_p(h2c_choice_confidence['p'])} {sig_label(h2c_choice_confidence['p'])}, n = {h2c_choice_confidence['n']}

{'**Significant positive correlation**: subjects who reduce risky choices more under threat also show larger confidence drops.' if h2c_choice_confidence['p'] < 0.05 and h2c_choice_confidence['r'] > 0 else '**Significant negative correlation**: subjects who reduce risky choices more under threat show smaller confidence drops.' if h2c_choice_confidence['p'] < 0.05 and h2c_choice_confidence['r'] < 0 else '**Null result**: no significant correlation between choice threat sensitivity and confidence threat sensitivity.'}

---

## Summary

| Test | Outcome | Direction | p |
|------|---------|-----------|---|
| H2a: S → Anxiety | {'Supported' if h2a_anx['p'] < 0.05 and h2a_anx['beta'] < 0 else 'Refuted' if h2a_anx['p'] < 0.05 else 'Null'} | β={h2a_anx['beta']:.3f} (predicted: negative) | {fmt_p(h2a_anx['p'])} |
| H2a: S → Confidence | {'Supported' if h2a_conf['p'] < 0.05 and h2a_conf['beta'] > 0 else 'Refuted' if h2a_conf['p'] < 0.05 else 'Null'} | β={h2a_conf['beta']:.3f} (predicted: positive) | {fmt_p(h2a_conf['p'])} |
| H2b: k × S → Anxiety | {'Significant' if h2b_anx['S_z:k_z']['p'] < 0.05 else 'Null'} | β={h2b_anx['S_z:k_z']['beta']:.3f} | {fmt_p(h2b_anx['S_z:k_z']['p'])} |
| H2b: k × S → Confidence | {'Significant' if h2b_conf['S_z:k_z']['p'] < 0.05 else 'Null'} | β={h2b_conf['S_z:k_z']['beta']:.3f} | {fmt_p(h2b_conf['S_z:k_z']['p'])} |
| H2c: Choice vs Anxiety threat sens. | {'Significant' if h2c_choice_anxiety['p'] < 0.05 else 'Null'} | r={h2c_choice_anxiety['r']:.3f} | {fmt_p(h2c_choice_anxiety['p'])} |
| H2c: Choice vs Confidence threat sens. | {'Significant' if h2c_choice_confidence['p'] < 0.05 else 'Null'} | r={h2c_choice_confidence['r']:.3f} | {fmt_p(h2c_choice_confidence['p'])} |
"""

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write(prose)
print(f"Prose written to {OUT_MD}")
print("\nDone.")
