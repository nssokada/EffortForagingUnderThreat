# Preregistered Hypotheses

## Effort Reallocation Under Threat in Continuous Foraging

Okada, Garg, Wise, Mobbs

---

When foraging under predation risk, how do humans trade off energy expenditure against danger across both choice and motor execution? Does this trade-off approximate the fitness-maximizing policy? Does the accuracy of subjective threat appraisal track the vigor of the defensive motor response?

We will test this using a virtual foraging task where subjects choose between food items varying in reward, effort cost, and predation risk, then physically execute their foraging bout by pressing keys under threat of capture by a virtual predator.

Specific hypotheses are as follows:

---

**People will adjust both what they choose and how hard they press in response to threat.**

1. Participants will choose the high-reward, high-effort option less often as threat increases.
    a. This effect will be modulated by distance: threat will amplify distance-driven avoidance (threat × distance interaction).
    b. Participants will choose the high option less often as distance increases.

2. Participants will press harder than the task demands when danger is high.
    a. Excess effort (pressing rate minus demand of chosen option) will increase with threat level.
    b. This effect will survive within constant-demand trials (i.e., when participants chose the low-effort option), ruling out a demand-driven confound.

3. Participants will report higher anxiety and lower confidence as threat increases.
    a. Trial-level anxiety ratings will increase with threat probability.
    b. Trial-level confidence ratings will decrease with threat probability.

---

**These adjustments will be coherently coupled across individuals — not two independent responses to threat.**

4. Choice shift and vigor shift under threat will be anti-correlated across individuals: participants who shift choices most toward safety will also show the largest increase in excess effort.
    a. This coupling will remain significant when computed from independent trial halves (odd vs. even), controlling for shared condition variance.

5. The reallocation strategy will approximate the expected-value-maximizing policy.
    a. Participants who reallocate more (greater choice shift + greater vigor shift) will achieve higher foraging earnings.
    b. The dominant deviation from optimal will be excessive caution — choosing safe when risky is EV-positive — rather than the reverse.

---

**A survival-weighted value model will best explain choice behavior.**

6. Choices will be best explained by a model in which effort enters as an additive physical cost and survival probability follows a hyperbolic function of distance.
    a. Additive effort will outperform multiplicative effort discounting.
    b. A hyperbolic survival kernel will outperform an exponential kernel.
    c. The survival model will substantially outperform an effort-only baseline.

7. The model-derived survival probability S will predict trial-level anxiety and confidence within subjects.
    a. S will predict anxiety negatively (lower survival → higher anxiety).
    b. S will predict confidence positively (higher survival → higher confidence).

---

**A hierarchical Bayesian model of excess effort will recover individual differences in danger-responsive vigor mobilization.**

8. The population-mean danger mobilization parameter δ will be positive, with more than 80% of participants showing δ > 0.
    a. Individual differences in δ will be recoverable (σ_δ > 0.05).

9. Threat bias in choice (β) will positively correlate with vigor mobilization (δ) across independently estimated Bayesian models, and effort sensitivity (k) will negatively correlate with δ.
    a. The β–δ coupling will have a 95% posterior bootstrap credible interval excluding zero.
    b. The k–δ coupling will have a 95% posterior bootstrap credible interval excluding zero.
    c. β and δ will predict greater closeness to the optimal policy.
    d. A joint hierarchical model with correlated random effects (LKJ prior) will confirm that all pairwise correlation credible intervals exclude zero.

---

**Individuals who mobilize vigor under danger will also show more accurate subjective threat appraisal.**

10. δ will predict steeper within-subject anxiety slopes on S (more negative) and steeper confidence slopes on S (more positive).
    a. Higher δ will be associated with lower mean anxiety — adaptive calibration, not chronic anxiousness.
    b. β will show the same pattern as δ (threat-sensitive people have better calibrated affect).
    c. Effort sensitivity k will not predict calibration accuracy.

---

**The computational parameters governing reallocation will be orthogonal to self-reported psychiatric symptomatology.**

11. No model parameter (k, β, δ) or their coupling will predict psychiatric factor scores (distress, fatigue, apathy) after correction for multiple comparisons.
    a. Tonic baseline vigor (α) will uniquely predict the apathy/amotivation factor.
    b. Participants above clinical cutoffs for depression (PHQ-9 ≥ 10) and anxiety (DASS Anxiety ≥ 8) will show the same β–δ coupling as participants below cutoffs.
