# Preregistered Hypotheses

## Effort Reallocation Under Threat in Continuous Foraging

Okada, Garg, Wise, Mobbs

---

When foraging under predation risk, how do humans trade off energy expenditure against danger across both choice and motor execution? Does this trade-off approximate the fitness-maximizing policy? Does the accuracy of subjective threat appraisal track the vigor of the defensive motor response?

We will test this using a virtual foraging task where subjects choose between food items varying in reward, effort cost, and predation risk, then physically execute their foraging bout by pressing keys under threat of capture by a virtual predator.

The computational framework estimates four per-subject parameters from independently fit hierarchical Bayesian models:

- **k** — effort sensitivity: how strongly effort cost deters choice (from choice model)
- **β** — threat bias: how strongly danger deters choice beyond expected value (from choice model)
- **α** — baseline vigor: tonic pressing rate above task demand (from vigor model)
- **δ** — danger mobilization: how much excess effort increases with model-derived danger, 1 − S (from vigor model)
- **S** — survival probability: S = (1 − T) + T/(1 + λD), integrating threat probability T and distance D (shared across models)

Specific hypotheses are as follows:

---

### I. Threat will shift choice, vigor, and subjective experience

**1.** Threat will reduce high-effort choice, amplify distance-driven avoidance, and increase excess motor effort — while shifting anxiety upward and confidence downward.

1a. High-effort choice will decrease with threat probability and with escape distance, with a threat × distance interaction.

1b. Excess effort (pressing rate minus chosen option's demand) will increase with threat, surviving within constant-demand trials.

1c. Trial-level anxiety will increase and confidence will decrease with threat probability.

---

### II. These adjustments will be coherently coupled across individuals, not independent threat responses

**2**. Choice shift and vigor shift under threat will be inversely correlated across individuals.
2a. Participants who shift choices most toward safety will show the largest increase in excess effort.
2b. This coupling will remain significant when computed from independent trial halves (odd vs. even), ruling out shared condition variance.

**3.** The reallocation strategy will approximate the expected-value-maximizing policy.

3a. Participants who reallocate more will achieve higher foraging earnings.

3b. The dominant deviation from optimal will be excessive caution rather than the reverse.

---

### III. A shared survival computation will link choice and vigor through independently estimated hierarchical models

**4.** Choices will be best explained by a model in which effort enters as an additive physical cost and survival probability follows a hyperbolic function of distance.

4a. Additive effort will outperform multiplicative effort discounting.

4b. A hyperbolic survival kernel will outperform an exponential kernel.

4c. The model-derived survival probability S will predict trial-level anxiety (negatively) and confidence (positively) within subjects.

**5.** Model-derived danger (1 − S) will drive excess motor effort at the population level, with meaningful individual variation in the strength of this response.

5a. The population-mean danger mobilization slope will be positive with a credible interval excluding zero.

**6.** Computational parameters governing the effort–danger trade-off will covary across independently estimated models.

6a. Threat bias in choice (β) will positively correlate with vigor mobilization (δ); effort sensitivity (k) will negatively correlate with δ.

6b. Both couplings will have 95% posterior bootstrap credible intervals excluding zero.

6c. β and δ will jointly predict closer approximation to the optimal policy.

---

### IV. Individuals who mobilize vigor under danger will show more accurate subjective threat appraisal

**7.** Participants whose motor effort is more danger-responsive (higher δ) will also show tighter affective tracking of survival probability S.

7a. δ will predict steeper within-subject anxiety slopes on S (more negative) and confidence slopes on S (more positive).

7b. Threat bias β will show the same pattern, but effort sensitivity k will not — dissociating threat-responsive parameters from effort cost sensitivity.
