# §2.5 Draft: Anxiety and confidence independently modulate the foraging decision boundary

**Status:** Post-hoc trial-level extension of preregistered H5a–d. Replicated in both samples.

---

### 2.5 Anxiety and confidence independently modulate the foraging decision boundary

The joint fitness model predicts choice through the value difference between heavy and light options and vigor through the optimal pressing rate. It does not specify a role for what participants reported feeling on probe trials — the trial-by-trial anxiety and confidence ratings collected before each action. We tested whether these affective signals carry behaviorally relevant information beyond the computational parameters, and if so, how they operate on the decision process.

**Between-subject tests (preregistered).** Four preregistered predictions were confirmed in both samples (Table 2). Anxiety calibration to threat — the within-subject correlation between trial-level anxiety and stated threat probability T — improved out-of-sample prediction of all three foraging outcomes when added to a base model of omega and kappa (H5a: delta-ELPD = 4.8, 3.5, and 3.1 for optimality, escape rate, and earnings, respectively; SE excluding zero in each case). Participants whose anxiety more closely tracked the displayed threat level foraged more adaptively, and this adaptive variance was not absorbed by the cost-weight parameters. Within-subject anxiety reactivity — the slope of anxiety on T — predicted the magnitude of threat-driven choice shift (H5b: beta = +0.099, 95% HDI [+0.065, +0.134]): participants whose anxiety responded more steeply to changing threat also shifted their choices more across threat levels. Avoidance sensitivity omega was negatively associated with mean confidence (H5c: beta = -0.181, 95% HDI [-0.340, -0.037]) but showed no reliable association with mean anxiety (beta = -0.067, 78% of posterior within the prespecified ROPE of [-0.10, +0.10]; see Table 2 footnote). Finally, participants with higher mean confidence made fewer overcautious errors (H5d: beta = -1.48, 95% HDI [-2.39, -0.54]) and more reckless errors (beta = +0.29, 95% HDI [+0.07, +0.52]) — a shift in the direction of mistakes rather than their overall rate.

These results establish that anxiety and confidence are behaviourally meaningful and dissociable: anxiety functions as a threat-tracking signal coupled to adaptive behaviour, while confidence reflects the individual's cost weighting and shifts the type of errors they make. The between-subject analyses, however, cannot determine whether these signals modulate the decision process on individual trials or merely correlate with stable traits. We therefore turned to a trial-level analysis.

**Trial-level decision-boundary modulation (post-hoc).** On probe trials, each participant rated either anxiety or confidence for a specific combination of threat and distance. We used these ratings to fit per-subject affective response functions (linear models of anxiety and confidence as functions of threat and distance) and projected the fitted functions onto all choice trials, yielding imputed anxiety and imputed confidence for each of approximately 13,000 decisions per sample. This approach preserves each participant's idiosyncratic affective mapping while extending from the ~18 probe trials per affect type to the full set of ~45 choice trials per participant.

We first asked whether imputed affect predicts choice beyond the model's value signal. In Bayesian linear mixed models with subject random intercepts predicting choice (heavy = 1, light = 0) from the model-derived value difference V_diff = W*(heavy) - W*(light), adding imputed confidence and anxiety jointly improved out-of-sample fit substantially (delta-ELPD = +1,080 in exploratory; +1,039 in confirmatory, relative to V_diff alone). Higher imputed confidence on a given trial pushed toward the high-reward option (beta = +0.168, 95% HDI [+0.155, +0.181]; confirmatory: beta = +0.202, 95% HDI [+0.189, +0.214]), while higher imputed anxiety pushed away from it (beta = -0.167, 95% HDI [-0.180, -0.154]; confirmatory: beta = -0.117, 95% HDI [-0.130, -0.105]). Both effects survived in the joint model with V_diff (all 95% HDIs excluding zero; R-hat = 1.000, ESS > 4,000 for all parameters), indicating that they capture variance the computational model does not absorb (Fig. 5a).

We next asked whether imputed affect predicts the specific errors participants make on each trial. On trials where the model prescribed the heavy cookie (V_diff > 0; approximately 8,700 / 7,000 trials in exploratory / confirmatory), participants nonetheless chose light on 39% / 35% of occasions — overcautious errors. Lower imputed confidence and higher imputed anxiety independently predicted these errors in both samples (confidence: beta = -0.165, 95% HDI [-0.184, -0.147]; confirmatory: beta = -0.159, 95% HDI [-0.177, -0.140]; anxiety: beta = +0.160, 95% HDI [+0.144, +0.179]; confirmatory: beta = +0.106, 95% HDI [+0.087, +0.125]; Bayesian linear mixed models with subject random intercepts). On trials where the model prescribed light (V_diff < 0), higher confidence and lower anxiety predicted the rarer reckless errors (~7% / 13% rate), though with smaller effect sizes (confidence: beta = +0.022, 95% HDI [+0.011, +0.032]; confirmatory: beta = +0.068, 95% HDI [+0.053, +0.081]; anxiety: beta = -0.013, 95% HDI [-0.023, -0.002]; confirmatory: beta = -0.039, 95% HDI [-0.053, -0.025]) (Fig. 5b).

Anxiety and confidence therefore act as independent, opposing modulators of the decision boundary. The computational model specifies the value difference between options on each trial; confidence lowers the threshold for acting on that signal, and anxiety raises it. On trials where the model favours the heavy option but the participant's imputed confidence is low and imputed anxiety is high, the participant is substantially more likely to make an overcautious error — choosing the safe option despite its lower expected value. Neither signal tracks the model's internal latent quantities: when model-derived capture probability and expected fitness are entered alongside the raw task variables (threat, distance, cookie type) in multilevel models predicting self-reported affect, they do not improve fit in either sample (Supplementary Section S5). Confidence and anxiety instead appear to read the observable task structure through subject-specific affective lenses and independently shift the decision criterion applied to the computational output.

![Figure 5. Anxiety and confidence modulate the foraging decision boundary.](data/figures/fig5_decision_boundary.png)

**Fig. 5 | Anxiety and confidence modulate the foraging decision boundary.** **a**, Posterior mean coefficients (95% HDI) from Bayesian linear mixed models predicting trial-level choice (heavy = 1) from the model's value difference (V_diff), imputed confidence, and imputed anxiety, fitted separately and jointly. Both affect signals add to choice prediction beyond V_diff and survive in the joint model. Black = exploratory, grey = confirmatory. **b**, Trial-level error prediction. Left: on trials where the model prescribes heavy (V_diff > 0), lower imputed confidence and higher imputed anxiety independently predict overcautious errors (posterior mean and 95% HDI). Right: on trials where the model prescribes light (V_diff < 0), higher confidence and lower anxiety predict reckless errors. **c**, Preregistered between-subject effects (H5a–d) shown as a forest plot for completeness; all four predictions confirmed in both samples. All models fitted with bambi (4 chains × 2,000 draws + 1,000 tuning; R-hat = 1.000, ESS > 4,000 for all parameters).

---

## Notes for integration

**What changed from current §2.5:**
- Section title changed from "computationally distinct metacognitive signals" to "independently modulate the foraging decision boundary" — mechanistic, not descriptive.
- Preregistered H5a–d compressed into a single paragraph, reported as confirmed, but no longer the climax.
- Trial-level imputation analysis is the new centrepiece: ~13,000 trials per sample, both signals predict choice beyond V_diff, and they predict the specific error type on each trial.
- Bridge-test null (model latent quantities do not predict affect) reported honestly in final paragraph and directed to supplement.
- Figure 5 redesigned around the trial-level results with H5a–d as a supporting forest plot.

**What stays from preregistration:**
- All four H5 predictions reported and confirmed. No H5 result is dropped.
- ROPE caveat for H5c retained (Table 2 footnote, unchanged).
- LOO comparison for H5a retained with exact ΔELPD values.

**What is post-hoc:**
- The imputation method (fitting per-subject affect functions from probes, projecting onto choice trials).
- The trial-level choice prediction (Analysis 1).
- Anxiety predicting error type (Analysis 2 — H5d only preregistered confidence).
- The "decision-boundary modulation" framing.

**Implications for abstract:**
The affect sentence becomes: "Trial-by-trial anxiety and confidence independently modulated the decision boundary beyond the model's value signal, with confidence lowering and anxiety raising the threshold for accepting high-reward options — predicting the type of errors participants made on individual trials."

**Implications for discussion:**
The affect paragraph should note that (a) the decision-boundary interpretation connects to signal detection theory and criterion-shifting accounts; (b) the signals read the stimulus surface, not the model's internal computation, which constrains mechanism; (c) the imputation approach should be validated with designs that collect affect on every trial.
