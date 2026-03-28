# Review of "Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety"

**Reviewer 1 — Expert in computational modeling of decision-making**

**Manuscript ID:** [Draft 005]
**Journal:** Nature Communications

---

## 1. Summary

This paper presents an Expected Value of Control (EVC) model with linear-quadratic regulator (LQR) cost structure that jointly predicts foraging choice and action vigor in a threat-avoidance paradigm (N = 293). Two subject-level parameters---effort cost (c_e, from choice) and capture aversion (c_d, from vigor)---achieve high fit to behavioral data. The authors then decompose trial-level anxiety into "calibration" (how well anxiety tracks model-derived danger) and "discrepancy" (excess anxiety beyond model predictions), finding that discrepancy predicts clinical symptom severity while the computational parameters themselves do not. The claimed contribution is that metacognitive bias, not threat computation per se, bridges adaptive foraging to clinical anxiety.

---

## 2. Significance and Novelty

The paper addresses a genuine gap: integrating effort-based decision-making with defensive behavior under threat within a single computational framework, and then connecting this framework to clinical variation through a metacognitive lens. This is timely work that bridges foraging ecology, optimal control theory, and computational psychiatry. If the claims hold, this would be a meaningful contribution.

However, the novelty is somewhat overstated. The model is a relatively straightforward expected utility model with softmax choice and a grid-search vigor optimization---not a full EVC implementation in the sense of Shenhav et al. (2013, 2017). The "LQR" label is aspirational rather than earned (see Major Concern 2). The metacognitive decomposition, while well-executed, is essentially a regression residual analysis with a new label.

---

## 3. Strengths

**S1. Clever task design.** The structural feature that c_d cancels from choice (because capture penalty affects both options equally) is an elegant design property that provides clean parameter identification. Probe trials that anchor vigor estimation across all conditions without selection bias are well-conceived.

**S2. Honest reporting.** The authors deserve credit for reporting negative cross-validated R-squared values, acknowledging modest effect sizes (3--11% variance), and including the M4 and M6 comparisons that arguably embarrass the preferred model. This level of transparency exceeds the norm.

**S3. Simpson's paradox insight.** The identification and resolution of the Simpson's paradox in vigor data is a genuine analytical contribution. The conditional analysis revealing threat-driven vigor increases (d = 0.42--0.49) where unconditional analyses show near-null effects is methodologically instructive.

**S4. Parameter recovery.** Recovery correlations of r = 0.92 and r = 0.94 for the two subject-level parameters are strong and adequately demonstrate identifiability of the model.

**S5. Orthogonality of metacognitive dimensions.** The finding that calibration and discrepancy are essentially uncorrelated (r = 0.019) provides clean interpretive leverage for the double dissociation.

---

## 4. Major Concerns

### MC1. The claim that c_d "cancels" from choice requires more rigorous justification

The paper claims that c_d does not enter the choice equation because "its contribution is collinear with the reward differential." This is the load-bearing identification claim for the entire model, yet the mathematical argument is gestured at rather than proven. Let me work through it.

The expected utility for a single option i is:

EU_i = S * R_i - (1 - S) * c_d * (R_i + C)

For the choice comparison:

Delta EU = EU_H - EU_L = S * (R_H - R_L) - (1-S) * c_d * ((R_H + C) - (R_L + C))
         = S * (R_H - R_L) - (1-S) * c_d * (R_H - R_L)
         = (R_H - R_L) * [S - (1-S) * c_d]

This shows that c_d does NOT cancel from choice. It multiplies (R_H - R_L) through the (1-S)*c_d term. What cancels is the fixed penalty C (since R_H + C minus R_L + C = R_H - R_L), not c_d itself. In the equation above, higher c_d would make the heavy option less attractive when S < 1, because the expected loss from capture is greater for the higher-reward option.

The authors' choice equation (Delta EU = S * 4 - c_e * effort_term) omits the (1-S)*c_d*4 term entirely. This is only valid if either: (a) c_d is absorbed into the survival function somehow, or (b) there is an implicit assumption that c_d = 0 in the choice model.

Looking at the code (line 185-186 of evc_final_81trials.py), the choice equation is:

```
delta_reward = S_ch * (ch_R_H - ch_R_L)
delta_effort = ce[ch_subj] * (ch_req_H**2 * ch_D_H - ch_req_L**2 * ch_D_L)
delta_eu = delta_reward - delta_effort
```

This confirms that only the S * delta_R term appears, with no capture-loss term. The missing term is -(1-S) * c_d * delta_R, which is NOT zero (delta_R = 4). The claim of "collinearity" appears to mean that S * delta_R and (1-S) * c_d * delta_R both scale with delta_R and S, making them difficult to separate---but this is a practical identifiability issue, not a mathematical cancellation. The paper should state this correctly: c_d is *omitted* from choice on the grounds that it is poorly identified (collinear with the survival-reward interaction), not that it "cancels."

This matters because if c_d did enter choice, it would attenuate the heavy-option advantage under high threat, and individual differences in c_d would contribute to choice heterogeneity. By omitting it, the model may misattribute some c_d-driven choice variation to c_e. The authors must either (a) present the full derivation showing why omission is justified, (b) fit a version where c_d enters both equations and show it cannot be recovered, or (c) rewrite the claim more carefully as a simplifying assumption rather than a mathematical identity.

### MC2. The LQR label is misleading

Linear-quadratic regulators (Todorov & Jordan, 2002) involve state-space dynamics, feedback control laws, and optimization of a trajectory over time via the Riccati equation. The model here has no state dynamics, no feedback law, and no trajectory optimization. What the authors call "LQR cost structure" is simply two different quadratic cost terms: req^2 * D for choice and (u - req)^2 * D for vigor. These are standard quadratic effort costs applied at two different decision stages.

The paper itself acknowledges that M6 (standard u^2 cost) is empirically indistinguishable (Delta BIC = -142, meaning M6 is slightly *better*). The Discussion appropriately notes this but then argues the LQR framework provides "a principled connection to optimal control theory." This is circular: the connection is principled only if the model actually implements LQR, which it does not.

I strongly recommend either: (a) dropping the LQR framing entirely and describing the model as having separate quadratic costs for commitment and execution, or (b) implementing an actual state-space formulation with dynamics. The current framing will invite criticism from the motor control community who will rightly view this as appropriating terminology without substance.

### MC3. SVI with AutoNormal guide is not Bayesian inference

The Methods state the model is fit with "NumPyro stochastic variational inference (SVI) with an AutoNormal guide (mean-field approximation)." This is variational inference with a factored Gaussian approximation---not MCMC. Yet the Abstract and Results sections use language suggesting full Bayesian posteriors ("94% HDIs," "posterior mass within ROPE").

Several concerns:

1. **Mean-field assumption:** AutoNormal assumes all parameters are independent in the posterior. For a hierarchical model with 586+ correlated parameters, this is a strong assumption that will underestimate posterior uncertainty and may produce biased point estimates. The correlation structure between mu_ce, sigma_ce, and the individual ce_raw values is completely ignored.

2. **BIC calculation is non-standard.** The BIC formula used (2 * loss + k * log(n)) treats the ELBO loss as a negative log-likelihood, but the ELBO is a lower bound on the marginal likelihood, not the log-likelihood. BIC = -2 * log L + k * log(n), where L is the maximized likelihood. Using the ELBO in place of log L conflates the variational approximation error with model fit. This undermines the model comparison entirely.

3. **Parameter count in BIC.** The model counts 2 * N_subjects + 11 = 597 parameters. But in a hierarchical model, the effective number of parameters is far less than the total count because the shrinkage prior constrains subject-level parameters. Using raw parameter count in BIC penalizes hierarchical models unfairly relative to models with fewer levels. This is why M4 (population c_e, 303 params) achieves lower BIC despite catastrophic choice fit (r^2 = 0.001): the BIC penalty for 294 extra parameters outweighs the likelihood improvement. The authors should use WAIC, LOO-CV, or at minimum DIC, which properly account for effective parameter count.

4. **The Bayesian clinical regressions** (Table 2) use PyMC with actual MCMC (bambi, 2000 draws * 4 chains). So the paper uses two different inference engines (SVI for the main model, MCMC for clinical regressions) without clearly distinguishing between them or discussing whether the SVI-derived point estimates introduce noise into the downstream Bayesian analyses.

### MC4. The "double dissociation" is predominantly a single dissociation

The authors claim a "predominant double dissociation" between calibration-performance and discrepancy-symptoms. But:

- Calibration -> performance: r = 0.179--0.230 (significant)
- Calibration -> symptoms: mostly null (6/7 p > .10), *but* STAI-State r = 0.121 (p = .04)
- Discrepancy -> symptoms: beta = 0.18--0.34 (significant)
- Discrepancy -> performance: not clearly reported in the main text (survival r = -0.15, p = .009, mentioned only in Discussion)

So discrepancy *does* predict performance (survival: r = -0.15), and calibration *does* predict one symptom measure (STAI). This is not a double dissociation---it is a pattern of differential association strength with some cross-contamination. The term "double dissociation" has a specific meaning in neuropsychology (Patient A shows deficit X but not Y; Patient B shows Y but not X). The pattern here is better described as "differential prediction" or "relative specificity." Calling it a "predominant double dissociation" with the qualifier "predominant" does not resolve the issue; it just acknowledges the term is being misused.

### MC5. Population-level epsilon may mask important individual differences

Epsilon (effort efficacy, 0.098) is estimated at the population level because individual recovery r is approximately 0. The authors acknowledge this in the Limitations but understate the consequences:

1. Epsilon captures how much people believe effort improves survival. Individual differences in this belief are theoretically critical for understanding threat-related motivation. A population-level estimate forces all subjects to share the same (very low) belief about effort efficacy, meaning the model assumes nobody thinks effort helps much.

2. The low epsilon (0.098) means the survival function is nearly effort-independent: S is approximately (1 - T^gamma), and the epsilon * T^gamma * p_esc term adds very little. This raises the question of whether the vigor model is really driven by survival optimization or whether c_d is simply a free parameter that captures any threat-vigor association. If S(u) is nearly flat in u, then the EU(u) function is dominated by the quadratic cost term, and c_d scales the threat-dependent offset. This is a much weaker claim than "vigor reflects survival-optimal motor execution."

3. The non-recoverability of epsilon may reflect model misspecification rather than inherent non-identifiability. If the true generative process involves heterogeneous effort-efficacy beliefs, a population-level epsilon will soak up the mean while inflating residual noise attributed to sigma_v.

### MC6. Confirmatory sample is missing

The Abstract states "confirmatory sample, N = XXX, preregistered." The Discussion mentions "A preregistered confirmatory study (N = 350 recruited)." For Nature Communications, I would expect at least the confirmatory results to be reported, or at minimum, a clear timeline and preregistration link. Without replication, all findings are exploratory, and the clinical associations in particular (Table 2) could reflect overfitting to sample-specific covariance structures.

---

## 5. Minor Concerns

**m1.** The per-subject choice r^2 of 0.951 is computed on 9 condition cells (3 threat x 3 distance) aggregated per subject. With only 5 trials per cell and 293 subjects, each condition mean is noisy. The r^2 should be reported with a clear statement of what is being correlated (subject-level mean P(heavy) predicted vs. observed across 9 cells, or across subjects at fixed conditions?). The current description is ambiguous.

**m2.** The probability weighting parameter gamma = 0.209 is described as "substantially stronger" than monetary gamble estimates. But gamma in this model enters via T^gamma, where T is a probability. For T = 0.5, T^0.209 = 0.86 while T^0.65 = 0.64. This is a dramatic difference. The authors should discuss whether this extreme compression reflects genuine perceptual distortion of stated probabilities or whether gamma is absorbing variance from other misspecified model components (e.g., the missing c_d term in choice).

**m3.** The vigor r^2 of 0.511 is trial-level. What fraction of this is between-subject vs. within-subject? If most of the variance is between-subject (some people press harder), the model may be capturing individual differences in motor capacity rather than threat-driven vigor modulation.

**m4.** The choice equation uses the term (0.81*D_H - 0.16) as the "LQR commitment cost difference." Where do these constants come from? I can infer: req_H^2 * D_H - req_L^2 * D_L = 0.81*D_H - 0.16 (since 0.9^2 = 0.81 and 0.4^2 * 1 = 0.16). This should be spelled out explicitly for readers to follow the derivation.

**m5.** The softmax temperature for vigor (multiply EU grid by 10.0 in the code) is a hardcoded hyperparameter, not a fitted parameter. This arbitrary choice affects how peaked the vigor predictions are. It should be acknowledged and sensitivity-tested.

**m6.** Block-to-block correlations of discrepancy (r = 0.48--0.68) are described as "moderate-to-good test-retest stability." With only 6 anxiety probes per block (18 total / 3 blocks), each block's discrepancy estimate is extremely noisy. The reliability should be assessed with Spearman-Brown correction or split-half reliability to account for measurement error.

**m7.** Reference 23 (Wise et al., 2020) appears to be about COVID risk perception, not about confidence tracking threat models. I believe the intended reference is Wise et al. (2023, Cell Reports), which is reference 33. The in-text citation in the Introduction ("In Wise and colleagues' work on interactive threat^33^") uses the correct number, but the earlier mention ("Wise and colleagues' finding that confidence tracks the quality of internal models of threat^23^" in the Discussion) cites the wrong paper.

**m8.** The manuscript does not report any goodness-of-fit diagnostics for the SVI procedure: no convergence plots, no ELBO traces, no comparison across random seeds. Given that SVI can get stuck in local optima, some evidence of convergence stability is needed.

**m9.** The N_S = 293 is from N = 350 recruited with 83.7% retention. Were the 57 excluded participants different on any observable demographic or psychiatric measures? Differential attrition could bias the sample toward higher engagement or cognitive capacity.

---

## 6. Questions for the Authors

**Q1.** Can you provide a formal derivation showing exactly which terms cancel in the Delta EU calculation and which are omitted by assumption? As written, the claim that c_d "cancels" is mathematically incorrect.

**Q2.** Have you tried fitting the model with full MCMC (e.g., NUTS in NumPyro) rather than SVI? Given that the Bayesian clinical regressions use MCMC, why was SVI chosen for the main model? Is this a computational constraint, and if so, what is the wall-clock time for SVI vs. MCMC?

**Q3.** What happens to the model comparison if you use WAIC or LOO-CV instead of the nonstandard BIC? In particular, does M4 still "win" on information criteria when effective parameter count is used?

**Q4.** If epsilon is set to zero (i.e., effort has no survival benefit at all), how much does model fit degrade? This would test whether the vigor component is really about survival optimization or just a c_d-scaled threat effect.

**Q5.** The discrepancy measure is the residual from a population-level regression of anxiety on S. Does this regress on S directly, on 1-S, on conditions (T, D), or on the model-derived S using population gamma and epsilon? The Methods say "model-derived danger (1-S)" but this requires choosing whose parameters to use for S. Please clarify.

**Q6.** Can you decompose the vigor r^2 = 0.511 into between-subject and within-subject components? This would clarify whether the model captures threat-driven within-subject vigor modulation or primarily individual differences in overall press rate.

**Q7.** The four behavioral profiles (Cautious, Lazy, Vigilant, Bold) are based on a median split. Have you tested whether this typology replicates in the confirmatory sample, or is it purely descriptive?

---

## 7. Recommendation

**Major Revisions.**

The paper addresses an interesting question with a creative experimental design and generally transparent reporting. However, several issues must be resolved before publication at a venue of this caliber:

1. The mathematical claim about c_d cancelling from choice is incorrect as stated and must be corrected or properly justified (MC1).
2. The LQR framing is misleading and should be revised or dropped (MC2).
3. The inference procedure (SVI with mean-field guide) and model comparison metric (nonstandard BIC) are problematic and should be supplemented or replaced with proper Bayesian model comparison (MC3).
4. The "double dissociation" language should be tempered (MC4).
5. The implications of population-level epsilon for the interpretation of vigor as "survival optimization" need to be addressed (MC5).
6. The confirmatory sample results are needed (MC6).

If these concerns are adequately addressed---particularly the mathematical derivation (MC1), the inference method (MC3), and the replication (MC6)---the paper could make a solid contribution to the field. The core idea that metacognitive discrepancy (not computational parameters) predicts clinical variation is compelling and well-supported within the current sample, but the modeling framework needs to be described more precisely and fit more rigorously.
