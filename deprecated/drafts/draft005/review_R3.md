# Reviewer 3 — Nature Communications

**Manuscript:** "Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety"
**Authors:** Okada, Garg, Wise, & Mobbs

---

## 1. Summary

The authors present a virtual foraging task in which human participants choose between high-effort/high-reward and low-effort/low-reward resources under parametric predation risk (three threat probabilities x three distances). They develop an Expected Value of Control model with LQR cost structure, fitted to both choice and keypress vigor, yielding two subject-level parameters (effort cost c_e and capture aversion c_d). The model's survival signal predicts trial-level anxiety and confidence ratings, and a metacognitive decomposition into "calibration" (accuracy of anxiety tracking danger) and "discrepancy" (excess anxiety over model-predicted danger) reveals a dissociation: calibration predicts task performance while discrepancy predicts clinical symptom severity across multiple psychiatric instruments.

---

## 2. Significance and Novelty

The paper attempts something genuinely ambitious: unifying effort-based choice, motor vigor, subjective affect, and clinical symptomatology within a single computational framework grounded in foraging ecology. If successful, this would be a meaningful contribution. The metacognitive decomposition is the strongest conceptual contribution --- separating *how well* anxiety tracks danger from *how much* it exceeds danger is a clean and useful distinction that advances computational psychiatry.

However, the ecological framing is overstated relative to the actual task. The paper positions itself within the Lima & Dill tradition of foraging under predation risk, but the task departs from this ecology in several fundamental ways that the authors do not adequately address (see Major Concerns). The modeling contribution, while technically competent, is more incremental than the framing suggests: it is essentially a two-parameter utility model fitted to choice proportions and mean press rates. The "EVC-LQR" label imports considerable theoretical weight from optimal control theory that the data do not actually test.

---

## 3. Strengths

1. **Large, well-powered sample with rigorous exclusions.** N = 293 after five-stage screening of 350 is commendable. The 83.7% retention rate is good, and the preregistered confirmatory sample (though not yet completed) is the right approach.

2. **Clean task design for the core question.** The factorial crossing of threat probability and distance, with probe trials providing forced-choice conditions that remove selection bias from vigor estimation, is well thought out. The identification strategy --- c_e from choice, c_d from vigor --- is structurally enforced rather than post-hoc, which is a genuine strength.

3. **Simpson's paradox in vigor.** The identification and resolution of the Simpson's paradox (threat appears to have no effect on vigor until conditioned on cookie type) is an excellent demonstration of why model-based analysis matters. This alone is pedagogically valuable.

4. **Metacognitive decomposition.** The orthogonality of calibration and discrepancy (r = 0.019) and the predominant double dissociation with performance and clinical symptoms is the paper's strongest result. The conceptual connection to Paulus & Stein's interoceptive prediction error framework is well drawn.

5. **Honest effect-size reporting.** The acknowledgment that cross-validated R-squared values are negative and that associations are group-level patterns rather than biomarkers is refreshingly honest for a computational psychiatry paper.

6. **Model comparison.** The ablation approach (Table 1) is more informative than the typical "horse race" between theoretically unrelated models.

---

## 4. Major Concerns

### 4.1 The ecological framing is substantially overstated

The paper opens with Lima & Dill, birds provisioning nestlings near raptors, and sticklebacks venturing from shelter. But the task departs from foraging ecology in ways that undermine this framing:

- **No patch dynamics.** Real foraging under threat involves patch-leaving decisions governed by diminishing returns (Charnov's MVT). Here, the choice is binary and one-shot within each trial. There is no depletion, no marginal value calculation, and no patch residence time decision. The paper cites Charnov (1976) but the model has no connection to the marginal value theorem whatsoever.

- **Explicit probability vs. learned risk.** Real predation risk is never displayed as a number. Animals estimate it from cues --- presence of conspecific alarm calls, predator scent, habitat structure, time of day, encounter history. The use of stated T in {0.1, 0.5, 0.9} reduces the threat manipulation to a framing effect on an economic gamble. The authors acknowledge this in the Limitations but it deserves more prominence given how central the ecological framing is to the paper's pitch.

- **No energy budget or state dependence.** A foundational insight from behavioral ecology (McNamara & Houston, 1986; Lima, 1998) is that foraging-under-predation decisions are state-dependent: a hungry animal takes more risks than a satiated one. The task has no energy state, no depletion, and no metabolic urgency. The 5-point and 1-point rewards are abstract tokens, not caloric intake. This matters because the entire theoretical apparatus of "survival-weighted value" in ecology derives from fitness consequences that depend on an organism's state.

- **The predator is unlearnable.** The predator behavior is scripted (appears at encounter time, strikes at Gaussian-distributed time, unavoidable if it appears). There is no predator learning, no evasion strategy, no predator-prey interaction. In real foraging ecology, much of the behavioral richness comes from the dynamic interaction between predator and prey (see Lima, 2002; Cresswell, 2008). The task reduces this to a Bernoulli coin flip with a scripted animation.

I do not think the paper needs to literally replicate field foraging to be valuable, but the Introduction and Discussion need to be much more honest about what the task captures and what it does not. Currently, the ecological language does real work in the framing --- it is not merely cosmetic --- and readers from ecology will find it misleading. I recommend the authors reframe the contribution as a *decision-making under threat and effort* paradigm inspired by foraging ecology, rather than claiming to study "foraging under predation risk" per se.

### 4.2 The EVC-LQR label imports unjustified theoretical weight

The model is presented as extending the Expected Value of Control framework with Linear-Quadratic Regulator cost structure. But:

- The LQR framework (Todorov & Jordan, 2002) involves state-dependent feedback control with a continuous state-space, cost-to-go functions, and optimal control laws derived from Riccati equations. None of this apparatus is present in the model. What the authors actually do is use a *quadratic cost function* for effort, which is a standard assumption in effort discounting (Hartmann et al., 2013) and does not require LQR theory to motivate.

- The authors themselves acknowledge that the LQR formulation is empirically indistinguishable from a standard u-squared cost (M6, delta-BIC = -142). If the LQR framing adds no explanatory power and requires no new mathematics, why use it? It creates the impression that optimal control theory is being tested when it is not.

- The EVC framework (Shenhav et al., 2013) involves computing expected value of allocating *cognitive* control effort, with explicit reference to ACC function. The extension to physical keypressing under virtual predation is not theoretically developed --- it is asserted. What is the control signal? What is the "expected value of control" here, as opposed to simply "expected utility"?

I recommend either (a) developing the EVC-LQR connection formally (showing that the model's equations actually follow from LQR optimality conditions) or (b) presenting the model more modestly as a two-parameter expected utility model with quadratic effort costs, which is what it functionally is.

### 4.3 "Vigor" is doing double duty as a term

The paper uses "vigor" to describe excess keypress rate relative to the required minimum. In the motor control literature (Niv et al., 2007; Shadmehr et al., 2016), vigor refers to the speed or intensity of movement, modulated by the marginal value of time under the current motivational state. This theoretical framework predicts that vigor increases when the average reward rate is high (opportunity cost of time).

In this task, "vigor" is measured as how much faster than the minimum requirement participants press keys. But: (a) participants must press above certain thresholds to move at all (full speed requires >= 100% of required rate), so excess pressing has discontinuous returns; (b) the marginal survival benefit of pressing faster is modeled as a sigmoid, meaning there is an optimal press rate that balances effort cost against escape probability; (c) the task has no opportunity cost structure --- there is no background reward rate that makes time valuable.

The authors should clarify whether their "vigor" measure reflects the same construct as in the Niv/Shadmehr tradition. If pressing harder directly increases escape probability (as the model implies), then the "excess" pressing is not vigor in the classical sense --- it is a rational survival strategy. True vigor effects would be observed in pressing speed *after* accounting for the survival benefit, i.e., in the residual of what the model cannot explain. Currently, the model "explains" vigor, but this is circular: the model was fitted to vigor, so the survival benefit is already captured in c_d.

### 4.4 SVI instead of full MCMC raises concerns about posterior accuracy

The Methods state that the model was fitted using Stochastic Variational Inference (SVI) with a mean-field AutoNormal guide. This is a significant concern for several reasons:

- Mean-field SVI assumes posterior independence between all parameters, which is unrealistic for hierarchical models where population and subject-level parameters are correlated by definition.
- SVI provides a lower bound on the log marginal likelihood, not exact posterior samples. The BIC values reported are computed from SVI loss, which may not accurately reflect model evidence.
- The Abstract states "hierarchical Bayesian model" and the earlier draft appears to have used MCMC (NumPyro HMC/NUTS). The switch to SVI is not justified or discussed.
- Uncertainty quantification (posterior widths, credible intervals for population parameters) is largely absent. We get point estimates (gamma = 0.209, epsilon = 0.098, tau = 0.476) but no posterior distributions.

The authors should either (a) re-fit with full MCMC and report proper posteriors or (b) explicitly justify SVI, report ELBO convergence diagnostics, and acknowledge the approximation in the interpretation. For a Nature Communications paper claiming to develop a computational model, approximate inference without diagnostics is insufficient.

### 4.5 The confirmatory sample is not yet run

The paper repeatedly references a preregistered confirmatory sample (N = 350, "N = XXX" placeholder throughout). For Nature Communications, which requires robust replication or strong pre-registration, submitting without the confirmatory data substantially weakens the paper. The discovery sample results are interesting but:

- The metacognitive decomposition is the paper's strongest claim, and it is entirely data-driven (calibration and discrepancy were not preregistered in the original study).
- Effect sizes are modest (beta = 0.18-0.34), and it is well known that discovery effect sizes are upward-biased.
- The probability weighting estimate (gamma = 0.209) is unusually extreme relative to the Prospect Theory literature and needs independent confirmation.

I strongly recommend that the confirmatory data be collected and analyzed before (re)submission.

### 4.6 The c_e/c_d dissociation needs stronger ecological grounding

The authors claim that c_e (from choice) and c_d (from vigor) map onto "strategic vs. reactive" defensive modes from the threat-imminence literature (Fanselow, 1994; Mobbs et al., 2020). But:

- The choice is made *before* the predator appears. It is not "strategic defense" in the threat-imminence sense --- it is economic choice under stated risk.
- Vigor occurs during transport, when the predator may or may not appear. If the predator appears, the pressing does become reactive, but the vigor measure includes both attack and non-attack trials. On non-attack trials, excess pressing has *no* survival value (the predator never comes), yet participants still press harder under higher stated threat. This is anticipatory arousal, not reactive defense.
- The threat-imminence continuum distinguishes behaviors by the *temporal/spatial proximity* of danger (circa-strike vs. pre-encounter vs. post-encounter). The task does not manipulate proximity --- it manipulates stated probability. These are fundamentally different constructs in the ethological literature.

The mapping from c_e/c_d to strategic/reactive defense is suggestive but unsupported by the current design. I recommend the authors either provide a more careful theoretical argument or soften this claim substantially.

---

## 5. Minor Concerns

1. **Reference 23 (Wise et al., 2020) is misattributed.** The text discusses "Wise and colleagues' work on interactive threat" and "confidence ratings track the quality of cognitive models of threat," but Reference 23 is a COVID-19 risk perception paper, not the interactive threat mapping paper. The authors likely mean Wise et al., Cell Reports, 2023 (Reference 33).

2. **Table 1 interpretation of M4.** M4 (population c_e) achieves *lower* BIC than the full model (delta-BIC = -1,274), yet the authors retain the full model. The justification (M4 "fails to predict individual choice") is reasonable but awkward: it means the full model is selected on the basis of interpretability rather than parsimony, while BIC formally favors M4. This tension should be discussed more transparently.

3. **Calibration measured with only 18 trials.** Per-subject Pearson r across 18 anxiety probe trials is noisy (expected SE of r approximately 0.24 for r = 0.47). This measurement noise in calibration will attenuate correlations with external criteria and inflate the apparent orthogonality with discrepancy. The authors should report split-half reliability of calibration and discuss attenuation.

4. **The distance manipulation confounds effort and exposure.** The authors acknowledge this in Limitations but it is more serious than presented. Because distance simultaneously increases effort cost and predator exposure time, the model's ability to jointly predict choice and vigor from a single framework could partly reflect this built-in correlation rather than a deep structural insight.

5. **Block-to-block stability of discrepancy (r = 0.48-0.68).** While described as "moderate-to-good," r = 0.48 means 77% of the variance changes across blocks. For a construct proposed as a trait-like individual difference that predicts clinical symptoms, this is concerning. The authors should compare this stability to the stability of the clinical instruments themselves (test-retest of STAI-S is approximately 0.70-0.76 over similar intervals).

6. **"Per-subject choice r-squared = 0.951" is misleading.** This appears to be computed by correlating per-subject observed and predicted P(heavy) across the 9 conditions. With N = 9 data points per subject, each of which is a proportion from approximately 5 trials, this r-squared is inflated by the small number of conditions and the large range of the design. The degrees of freedom are minuscule. Report per-trial accuracy or log-likelihood instead.

7. **Missing literature.** The paper does not cite:
   - Brown, Laundre, & Gurung (1999) on the "ecology of fear" and giving-up densities, which directly formalizes the foraging-under-predation trade-off the paper claims to study.
   - McNamara & Houston (1986, 1992) on state-dependent optimization in foraging, which is the foundational framework for survival-weighted value in ecology.
   - Fanselow (1994) on threat imminence, which is cited only indirectly through secondary references.
   - Hare et al. (2011) or similar dissociations between decision value and vigor in neuroeconomics.

8. **Probability weighting.** The estimate gamma = 0.209 is *far* below standard Prospect Theory estimates (gamma approximately 0.65-0.70 from Tversky & Kahneman, 1992). The authors attribute this to "embodied virtual predation engaging defensive circuitry." But an equally plausible explanation is that participants simply did not believe the stated probabilities, or that the probability weighting function is absorbing other model misspecification (e.g., nonlinear distance effects, risk aversion not captured by the model). This deserves more careful analysis --- e.g., does gamma vary systematically with risk attitude or task comprehension?

9. **No learning analysis.** The authors fit a static model and acknowledge this in Limitations, but 81 trials is enough to detect learning effects. Do P(heavy) or vigor change across blocks? If participants learn about the predator (e.g., realizing that fast pressing does not actually help much, given epsilon = 0.098), this would affect the interpretation of c_d.

---

## 6. Questions for the Authors

1. What would the model predict for a task with learnable predation risk (e.g., uncertain T that must be inferred from experience)? Would c_d and discrepancy still dissociate, or would learning confound the metacognitive decomposition?

2. How sensitive is the discrepancy measure to the specification of the survival function? If gamma is subject-level rather than population-level, does discrepancy change? In other words, is "excess anxiety" possibly just "different probability weighting"?

3. On non-attack trials, pressing faster has literally zero survival value (the predator never appears). How much of the vigor effect is driven by attack vs. non-attack trials? If the effect is present on non-attack trials, it cannot be rational survival optimization and must reflect something else (arousal? preparedness?).

4. What is the mean and variance of the effort calibration (f_max)? How variable are participants in their maximum press rates, and does this variability interact with the model's predictions?

5. The model predicts that c_d should correlate with the *slope* of the vigor-threat relationship. Is this borne out? Is c_d simply a reparameterization of the threat effect on vigor, or does it predict additional variance?

6. Why was SVI used instead of MCMC? Was MCMC attempted and found to be problematic (divergences, poor mixing)? This choice has significant implications for posterior accuracy.

7. Given that cross-validated R-squared is negative for clinical prediction, what is the practical utility of the discrepancy measure? Is it more informative than simply asking people "how anxious are you generally?" (i.e., a single STAI item)?

---

## 7. Recommendation

**Major Revisions.**

The paper addresses an important question --- how foraging computation, motor vigor, and clinical anxiety relate --- and the metacognitive decomposition is a genuine conceptual contribution. However, the ecological framing is substantially overstated, the EVC-LQR theoretical apparatus is not justified by the actual model, the inference method (SVI) is inadequately reported, and the confirmatory sample has not been collected. These issues are addressable but require significant revision.

Specifically, I would require:

1. Honest reframing of the ecological claims (not cosmetic --- substantive revision of Introduction and Discussion)
2. Either formal derivation of the EVC-LQR connection or reframing as a two-parameter EU model
3. Full MCMC fits or thorough justification and diagnostics for SVI
4. Confirmatory sample data, or at minimum a clear editorial commitment to staged review
5. More careful treatment of "vigor" and its relationship to the motor control literature
6. Expanded ecological literature coverage (Brown et al., McNamara & Houston, Fanselow)

If these issues are addressed, I believe the paper could make a solid contribution to Nature Communications. The metacognitive calibration-discrepancy decomposition, if replicated, would be a meaningful advance for computational psychiatry. But the current draft oversells the ecological and theoretical framework in ways that will not survive scrutiny from readers in behavioral ecology or motor control.
