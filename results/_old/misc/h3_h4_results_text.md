# H3 and H4 Results

**Date:** 2026-03-20
**Sample:** N = 293 (exploratory)
**Model:** L3_add, λ = 2.0
**Data:** behavior.csv (stage5_20260320_191950), smoothed_vigor_ts.parquet, unified_3param_clean.csv

---

## H3: Survival Probability Drives Excess Motor Vigor

We operationalized trial-level vigor as the mean of the normalized keypress rate (vigor_norm) per subject × trial, derived from the 20 Hz kernel-smoothed time series (N = 293 subjects). Demand was defined as the effort fraction associated with the chosen option (effort_L = 0.40 for the low cookie; effort_H ∈ {0.60, 0.80, 1.00} for the high cookie). Excess effort was computed as vigor − demand. Danger was derived from the subjective survival function S = (1−T) + T/(1+λD) with λ = 2.0, as danger = 1 − S, where D ∈ {1, 2, 3} is the chosen distance level and T ∈ {0.1, 0.5, 0.9} is predator threat probability.

### H3(a): Population-level excess effort slope (δ) > 0

Per-subject OLS regression of excess effort on trial-level danger yielded individual slopes δᵢ (the sensitivity of excess vigor to danger). Across subjects, the mean slope was δ = 0.128 (SD = 0.185), with 80.2% of subjects showing positive slopes. A one-sample t-test confirmed that the population mean δ differed significantly from zero, t(292) = 11.85, p < .001 (one-tailed). Delta > 0 supported.

### H3(b): Split-half reliability of δ

To assess the reliability of individual difference estimates in δ, we split trials into odd- and even-numbered halves and computed δ separately for each half. Pearson correlation between halves was r = 0.291 (p < .001); Spearman-Brown corrected reliability was ρSB = 0.451. This indicates moderate reliability of individual excess vigor slopes.

### H3(c): Danger predicts excess effort within choice = 0 (constant demand)

To isolate the effect of danger on vigor from effort-demand confounds, we restricted analysis to low-option trials (choice = 0), where demand is constant at 0.40. A linear mixed-effects model (excess ~ danger, random intercept per subject) yielded a significant danger effect: β = 0.026 (SE = 0.007), z = 3.54, p < .001 (N = 12,678 trials). This confirms that danger modulates vigor even when task demands are held constant.

### H3(c2): Danger predicts excess effort within choice = 1

For high-option trials (choice = 1), where demand varies with effort_H, a parallel LMM yielded: β = 0.036 (SE = 0.008), z = 4.74, p < .001 (N = 10,878 trials), significant.

### H3(d): Threat × Distance interaction on excess effort

A fully specified LMM regressed excess effort on threat, distance, their interaction, and choice (all mean-centered), with random intercepts per subject.
Key findings:
- Threat: β = -0.002, SE = 0.003, z = -0.53, p = 0.594
- Distance: β = -0.162, SE = 0.003, z = -56.07, p < .001
- Threat × Distance: β = -0.017, SE = 0.008, z = -2.12, p = 0.034 (significant)
- Choice: β = -0.148, SE = 0.002, z = -62.91, p < .001

The threat × distance interaction was significant, indicating that the effect of danger on excess effort did depend on distance level.

---

## H4: Coherent Strategy Shift

### Descriptive Statistics

Across subjects (N = 293), the mean threat-based choice shift (P(high|T=0.9) − P(high|T=0.1)) was -0.271 (SD = 0.166), with 90.8% of subjects showing a negative shift (reduced high-option selection under high threat). The mean excess vigor shift (mean_excess at T=0.9 − T=0.1) was 0.084 (SD = 0.096), with 85.3% positive. The coherent shift index (−choice_shift + excess_shift) had mean 0.355 (SD = 0.241).

### H4(a): choice_shift × excess_shift correlation

The correlation between individual choice shifts and excess vigor shifts was r = -0.671, p < .001 (N = 293). A negative correlation indicates that subjects who avoided high-value cookies under threat compensated with higher motor vigor — the hallmark of a coherent defensive strategy. The correlation met the pre-specified threshold of r < −0.5.

### H4(b): Effort-discounting parameter k × vigor slope δ

The correlation between individual effort-discounting rates (k, from L3_add choice model) and individual vigor slopes (δ, from per-subject OLS) was r = -0.195, p < .001 (N = 293), significant. This tests whether subjects who discount effort more in choice also show steeper danger-driven vigor mobilization.

### H4(c): Threat bias parameter β × vigor slope δ

The correlation between threat bias (β) and vigor slope (δ) was r = 0.462, p < .001 (N = 293), significant. β captures residual threat aversion in choice beyond survival-weighted expected value; its relation to δ reveals whether the same threat sensitivity is expressed in both choice and action.

### H4(d): Coherent shift predicts behavioral outcomes

Coherent shift correlated with:
- **Total reward:** r = 0.413, p < .001, significant
- **Escape rate:** r = 0.306, p < .001, significant

### H4(e): Multiple regression: total_reward ~ α_i + δ_i + k + β

Standardized predictors (z-scored) were regressed on total accumulated reward. The model explained R² = 0.321 (adjusted R² = 0.312) of variance, F(34) = 34.04, p < .001 (N = 293). Individual predictor estimates (standardized):

| Predictor | β | SE | t | p |
|-----------|---|-----|---|---|
| α_i (vigor baseline) | 25.065 | 3.476 | 7.21 | p < .001 *** |
| δ_i (danger slope) | 26.263 | 3.915 | 6.71 | p < .001 *** |
| k (effort disc.) | -14.087 | 3.547 | -3.97 | p < .001 *** |
| β (threat bias) | 3.658 | 3.877 | 0.94 | p = 0.346  |

*p < .05 (*), p < .01 (**), p < .001 (***)*

---

*Results generated from scripts/analysis/run_h3_h4_tests.py*
