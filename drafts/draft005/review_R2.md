# Review: "Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety"

## Reviewer 2

---

## 1. Summary

This paper develops an Expected Value of Control (EVC) model with LQR cost structure to jointly explain foraging choice and action vigor under parametric predation threat. Two subject-level parameters (effort cost and capture aversion) explain choice and vigor through separable channels. The authors then decompose subjective anxiety into "calibration" (how well anxiety tracks model-derived danger) and "discrepancy" (systematic excess anxiety beyond model-warranted levels), reporting a predominant double dissociation: calibration predicts task performance while discrepancy predicts clinical symptom severity across seven psychiatric instruments. The computational parameters themselves show no clinical associations.

---

## 2. Significance and Novelty

The paper addresses a genuine gap at the intersection of effort-based decision-making, defensive behavior, and computational psychiatry. The joint modeling of choice and vigor within a single cost framework is a meaningful technical contribution, and the Simpson's paradox demonstration is a useful methodological lesson for the vigor literature. The attempt to bridge computation and clinical symptoms via metacognitive decomposition is theoretically ambitious.

However, the novelty of the metacognitive contribution is less clear than presented. The "discrepancy" measure is, at its core, a residualized measure of mean anxiety after removing variance explained by objective danger. The finding that people who report higher anxiety (after controlling for situation) score higher on clinical anxiety measures is, while computationally dressed, not particularly surprising. The paper needs to do more work to distinguish this from the simpler interpretation that "anxious people report more anxiety on both task probes and questionnaires."

---

## 3. Strengths

**S1. Clean task design.** The factorial crossing of threat probability, distance, and effort demand is well thought out. The use of probe trials (forced-choice, identical options) to anchor vigor estimation without selection bias is clever and addresses a real confound. The five-stage quality screening pipeline is thorough.

**S2. Joint modeling.** The demonstration that c_e is identified exclusively from choice and c_d exclusively from vigor, by virtue of the model's structural constraints rather than post-hoc decomposition, is a genuine strength. The near-independence of these parameters (r = -0.14) and strong recovery (r = 0.92 and 0.94) are reassuring.

**S3. Simpson's paradox.** The identification and resolution of the Simpson's paradox in vigor is an important contribution that other researchers studying motor vigor under threat should attend to. This is clearly and convincingly presented.

**S4. Effect-size honesty.** The authors are commendably transparent about the modest clinical effect sizes, the negative cross-validated R-squared values, and the limitations of single-task parameters for individual prediction. This candor is appreciated and too rare in computational psychiatry.

**S5. Strong within-subject affect tracking.** The survival-anxiety and survival-confidence associations (beta = -0.557 and +0.575) are large and convincing. This validates the model's survival signal as a psychologically meaningful quantity.

**S6. Probability weighting.** The gamma = 0.209 estimate is striking and the comparison to monetary gamble estimates (gamma ~ 0.65-0.70) raises interesting questions about domain-specificity of probability distortion under embodied threat.

---

## 4. Major Concerns

**M1. "Metacognition" is doing too much theoretical work for what is measured.** The paper's central claim rests on characterizing calibration and discrepancy as "metacognitive" dimensions. But what is actually measured? Calibration is a within-subject correlation between self-reported anxiety and model-derived danger. Discrepancy is mean residual anxiety after removing the population danger-anxiety slope. Neither of these requires metacognitive processing in any standard sense. Metacognition involves monitoring and regulating one's own cognitive processes -- "thinking about thinking" (Flavell, 1979) or, in the decision-making literature, confidence in one's own judgments (Fleming & Dolan, 2012). What the authors measure is better described as *affective accuracy* and *affective bias* relative to a normative model. The participant is not introspecting on their threat computation; they are reporting how anxious they feel. This is affect measurement, not metacognitive measurement.

The connection to Wells' metacognitive therapy is particularly strained. Wells' concept of meta-worry ("worry about worry") involves second-order beliefs about one's own cognitive processes. The discrepancy measure captures first-order affective reporting that exceeds model predictions -- this is closer to Paulus and Stein's interoceptive prediction error than to Wells' metacognition, and the paper should either reframe accordingly or provide a much more rigorous justification for the metacognitive label. As it stands, the title claim ("Metacognitive bias... bridges foraging decisions to clinical anxiety") overpromises relative to what is demonstrated.

**M2. The double dissociation claim is not adequately supported.** A true double dissociation requires that each variable predicts one outcome *and not the other*. The authors themselves note leakage: calibration shows a weak association with STAI-State (r = 0.121), and discrepancy shows a modest association with survival (r = -0.15). These "cross-associations" are acknowledged but minimized as "small relative to the primary effects." However, this is precisely the pattern one would expect from two correlated-but-not-identical measures of anxiety sensitivity, rather than from genuinely dissociated mechanisms. The paper describes this as a "predominant" double dissociation, which is a hedge that weakens the central claim. The appropriate statistical test for a double dissociation is a formal interaction analysis (e.g., Dunn and Kirsner, 2003) or Steiger's test for dependent correlations comparing the two paths. The paper mentions Steiger's test in the methods but does not report its results for the dissociation claim.

**M3. Discrepancy may simply index trait negative affectivity.** The strongest predictor in Table 2 is STAI-State (beta = 0.338), and discrepancy also predicts DASS-Depression, PHQ-9, and AMI -- measures spanning anxiety, depression, stress, and apathy. This breadth of association is more consistent with discrepancy capturing a general negative affectivity / neuroticism factor than a specific metacognitive anxiety mechanism. If discrepancy were truly about threat-specific metacognitive bias, one would expect stronger specificity for anxiety measures over depression/stress measures. Instead, the effect sizes are remarkably similar across all instruments (beta = 0.22-0.34), suggesting a non-specific factor. The authors should test whether discrepancy predicts anxiety measures *over and above* depression (and vice versa), or whether it simply indexes a shared internalizing dimension.

**M4. The direction of causality is not addressed.** The paper implies (particularly in the Discussion) that discrepancy causes or "bridges to" clinical symptoms. But the reverse direction is equally plausible and arguably more parsimonious: people with elevated trait anxiety report more anxiety on task probes, producing higher discrepancy scores, and also score higher on clinical instruments because both reflect the same underlying disposition. This is a shared-method-variance problem: both discrepancy and clinical measures rely on self-report. The authors need to explicitly discuss this alternative and explain what evidence would distinguish the two accounts. The current framing, particularly in the title and abstract, implies a directionality that the cross-sectional data cannot support.

**M5. The between-subject confidence null is inadequately addressed.** The paper reports that mean confidence does not predict task performance (r = -0.05 to -0.08), which the authors frame as motivation for the metacognitive decomposition. But this null result is potentially damaging to the metacognition story. If confidence is a metacognitive signal about one's own performance, between-subject variation in confidence *should* predict performance differences. The null suggests that confidence ratings reflect something other than metacognitive monitoring -- perhaps demand characteristics, scale usage idiosyncrasies, or mood. If confidence is not metacognitive, why should anxiety (measured on the same scale, in the same probe format) be treated as metacognitive? The authors need to engage with this tension more seriously.

**M6. Non-clinical sample limits clinical generalizability.** All clinical measures reflect dimensional variation in a Prolific sample. The authors acknowledge this in the Limitations but the paper's framing -- including the title -- makes clinical claims that the sample cannot support. Phrases like "bridges foraging decisions to clinical anxiety" and "predicts psychiatric vulnerability" are too strong for a non-clinical sample with self-report symptom scales. The discrepancy-symptom association could change qualitatively in clinical populations (e.g., floor/ceiling effects, medication effects, or qualitative differences in threat processing). The paper should soften clinical language throughout or provide a stronger argument for why dimensional variation in non-clinical samples is informative about clinical populations.

---

## 5. Minor Concerns

**m1. SVI vs. MCMC.** The methods describe SVI with a mean-field AutoNormal guide, but the abstract and introduction reference "hierarchical Bayesian" modeling, which typically implies MCMC sampling. SVI with mean-field approximation can underestimate posterior uncertainty, particularly for hierarchical models. Was MCMC attempted? If SVI was chosen for computational reasons, the authors should report diagnostics (ELBO convergence, posterior predictive checks) and discuss whether the mean-field assumption is appropriate for the hierarchical structure.

**m2. Model comparison concerns.** M4 (population c_e) achieves *lower* BIC than the full model (delta-BIC = -1,274) but is dismissed because it fails to predict individual choice. This is an unusual move -- BIC is being overridden by a secondary criterion not built into the model comparison framework. The authors should either justify this formally (e.g., by using a criterion that penalizes poor individual-level prediction) or acknowledge that by standard model comparison, the simpler model is preferred and the individual-c_e model is retained for theoretical rather than statistical reasons.

**m3. Calibration is computed on only 18 anxiety probe trials.** Within-subject correlations based on 18 observations have substantial sampling noise. The reliability of a Pearson r with n=18 is low, and the distribution of such correlations will be heavily influenced by a few extreme trials (e.g., T=0.9 vs T=0.1). The authors should report the reliability of the calibration measure (e.g., split-half or odd-even) and discuss whether 18 observations is sufficient for stable individual differences.

**m4. The "effort efficacy" parameter epsilon = 0.098 and its non-recoverability.** The authors note that epsilon is not individually recoverable (r approximately 0), which means the model assumes everyone has the same belief about effort's survival benefit. This is a strong assumption that could interact with the discrepancy measure: if some participants actually believe effort helps survival more than others, their "excess anxiety" on low-effort trials might be rational rather than biased.

**m5. Reference 23 (Wise et al., 2020) is a COVID risk perception paper, not the threat confidence paper described in the Introduction.** The text states "In Wise and colleagues' work on interactive threat, confidence ratings track the quality of cognitive models of threat with remarkable fidelity." But reference 23 is about self-reported protective behavior during COVID-19. The intended reference appears to be Wise et al. (2023, Cell Reports, ref 33), which is about interactive cognitive maps under threat. This citation error should be corrected.

**m6. Missing confirmatory sample.** The paper references a preregistered confirmatory sample (N = 350) with "N = XXX" placeholders. While the authors are transparent about this, the paper's claims currently rest entirely on the discovery sample. For Nature Communications, a complete confirmatory replication would substantially strengthen the paper's contribution.

**m7. The connection to Paulus and Stein.** The paper draws heavily on the interoceptive prediction error framework, but discrepancy as measured here is not an interoceptive prediction error. It is the residual between a self-reported affective state and a model-derived environmental quantity. True interoceptive prediction error involves mismatch between predicted and actual body states. The authors should be more precise about which aspect of Paulus and Stein's framework they are operationalizing and which they are borrowing metaphorically.

**m8. Table 2 ROPE interpretation.** The paper states that ROPE containment of 53-93% provides "positive evidence for the null." However, values at the lower end of this range (53-56% for OASIS and AMI) are ambiguous at best -- they indicate approximately equal posterior mass inside and outside the ROPE. The authors should adopt a more conservative threshold for claiming "evidence for the null" (e.g., >80% or >90% in ROPE) and present the lower-ROPE cases as inconclusive rather than null.

**m9. The LQR framing.** The paper makes a significant theoretical investment in the LQR framework but then demonstrates that it is empirically equivalent to a standard u-squared cost (M6: delta-BIC = -142). This raises the question of whether the LQR framing adds explanatory value or merely rhetorical prestige. The theoretical argument for LQR (commitment vs. deviation costs mapping onto choice vs. vigor) is interesting but would be stronger if it generated distinguishing predictions that were confirmed.

---

## 6. Questions for the Authors

1. If you replace the discrepancy measure with simple mean anxiety (without residualizing against the model's danger signal), how do the clinical correlations change? If they are similar, this would suggest that the model-based decomposition adds little beyond raw affect measurement.

2. Can you report Steiger's test comparing the calibration-performance and calibration-symptom correlations (and the analogous test for discrepancy)? This would provide a formal test of the double dissociation.

3. What is the correlation between discrepancy and the random intercept from the anxiety LMM? These seem conceptually similar, and if they are highly correlated, the LMM random intercept would be a simpler operationalization.

4. Have you examined whether discrepancy is driven primarily by specific threat conditions? For instance, does discrepancy manifest mainly at low threat (where anxious individuals fail to down-regulate) or at high threat (where they over-react)?

5. You report that within-subject anxiety and confidence are weakly correlated (r = -0.25) and between-subject means are independent (r = -0.01). Have you computed an analogous confidence-discrepancy measure and tested its clinical associations? If confidence-discrepancy shows the same clinical pattern as anxiety-discrepancy, this would suggest a general affective bias rather than threat-specific metacognitive dysfunction.

6. The gamma = 0.209 probability weighting is very aggressive. At T = 0.1, perceived threat is 0.1^0.209 = 0.62 -- participants treat a 10% chance as if it were 62%. Is this estimate stable across model variants? Could it be absorbing other forms of threat overweighting (e.g., loss aversion)?

---

## 7. Recommendation

**Major Revisions.**

This paper presents a competent computational model and an interesting empirical pattern, but the theoretical interpretation overreaches in several important ways. The "metacognition" framing is not adequately justified, the double dissociation claim requires stronger statistical support, the discrepancy measure needs to be distinguished from simple trait negative affectivity, and the direction of causality requires explicit discussion. The clinical claims are too strong for a non-clinical sample.

The core ingredients are promising: the joint choice-vigor model, the Simpson's paradox demonstration, the calibration-discrepancy decomposition, and the honesty about effect sizes. But the paper needs to either (a) reframe the contribution as about *affective bias* relative to a normative model (dropping or substantially softening the metacognition claim) and demonstrate that the model-based decomposition adds value over raw affect measures, or (b) provide additional evidence that what is measured is genuinely metacognitive (e.g., through correlation with established metacognitive tasks, or demonstration that the discrepancy measure captures something distinct from trait affect). The confirmatory sample should be included before publication at Nature Communications.

I would be enthusiastic about a revised version that addresses these concerns.

---

*Reviewer expertise: metacognition, computational psychiatry, clinical anxiety, interoceptive processing.*
