# Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety

Noah Okada^1^, Ketika Garg^1^, Toby Wise^2^, Dean Mobbs^1,3^

^1^ Division of the Humanities and Social Sciences, California Institute of Technology, Pasadena, CA, USA
^2^ Department of Neuroimaging, King's College London, London, UK
^3^ Computation and Neural Systems Program, California Institute of Technology, Pasadena, CA, USA

---

## Abstract

Foraging under predation risk requires integrating energetic cost and survival probability into a unified decision variable, yet how a single computation governs both the decision of what to pursue and the vigor of pursuit---and how breakdowns in that computation relate to anxiety---remains unknown. Here we develop an Expected Value of Control (EVC) model with a cost structure inspired by linear-quadratic optimal control and test it in a large online effort-foraging task under parametric threat (N = 293). Two subject-level parameters---effort cost (*c*~e~, identified from choice) and capture aversion (*c*~d~, identified from vigor)---plus population-level probability weighting ($\gamma$ = 0.209) jointly predict foraging decisions (per-subject *r*^2^ = 0.951) and press-rate vigor (*r*^2^ = 0.511). Both parameters are recoverable in simulation (*c*~e~: *r* = 0.92; *c*~d~: *r* = 0.94) and are approximately independent (*r* = -0.14). The model's survival signal predicts trial-level anxiety ($\beta$ = -0.557, *t* = -14.0) and confidence ($\beta$ = 0.575, *t* = 13.5). We decompose the affect-danger relationship into calibration (how accurately anxiety tracks danger) and discrepancy (how much anxiety exceeds danger). These dimensions are orthogonal (*r* = 0.019) and show a differential prediction pattern: calibration predicts task performance (*r* = 0.179--0.230 with choice quality and survival), while discrepancy predicts anxiety symptoms (STAI: $\beta$ = 0.338; STICSA: $\beta$ = 0.285; DASS-Anxiety: $\beta$ = 0.275; all 94% HDIs excluding zero). The computational parameters show no reliable associations with symptom severity (~77% posterior mass within the region of practical equivalence). The bridge from adaptive foraging computation to psychiatric vulnerability runs through metacognition---through how people feel about danger, not how they compute it.

---

## Introduction

Animals foraging under predation risk face a fundamental optimization problem: the energy gained from distant, high-value resources must be weighed against increased exposure to capture during transport^1,2^. This trade-off shapes behavior across species, from birds provisioning nestlings near raptor territories^3^ to fish venturing from shelter to access food patches^4^. In humans, analogous computations arise whenever pursuing goals demands sustained effort under threat---commuting through dangerous neighborhoods, persisting at hazardous work, or investing cognitive effort when the costs of failure loom large. Theoretical ecology has long modeled this trade-off using reproductive value as the common currency^1,5^, but translating these models into a computational framework that specifies how the same cost-benefit calculation governs both what an organism chooses and how vigorously it acts has remained an open challenge.

Two research traditions have addressed the components of this problem in isolation. Work on effort-based decision-making has established that humans discount reward value by the physical or cognitive effort required to obtain it^6--9^, with individual differences in effort cost sensitivity linked to apathy, fatigue, and motivational disorders^10,11^. Separately, research on defensive behavior has characterized how organisms modulate responses to threat across the predatory imminence continuum---from strategic avoidance at a distance to reactive flight when danger is immediate^5,12--14^. The vigor of motor execution, in particular, is thought to reflect the marginal value of time under the current motivational state^15,16^, suggesting that threat should intensify physical effort even after the decision to forage has been made. Yet no computational model has formally specified how a single cost function generates both the discrete choice of what to pursue and the continuous regulation of how hard to pursue it.

The Expected Value of Control (EVC) framework^17,18^ provides a candidate architecture for this integration. EVC proposes that the brain computes the expected payoff of allocating control effort, selecting the intensity that maximizes reward minus cost. Because EVC treats effort allocation as a continuous optimization rather than a binary decision, it naturally extends to predict both discrete choices and graded action vigor within a unified framework. However, EVC has been applied primarily to cognitive control in abstract laboratory tasks^19,20^ and has not been extended to physical effort under ecological threat, where the cost function must jointly encode energetic expenditure and survival probability.

We designed a reductionist probe of this effort-threat integration problem. Our task captures the core foraging-under-threat trade-off---choosing between options that differ in reward, effort, and risk---but deliberately abstracts away several features of natural foraging ecology: there are no patch dynamics or marginal value calculations (cf. Charnov^2^), no energy state or metabolic urgency (cf. McNamara & Houston^38^), and threat probabilities are stated rather than learned. By fixing the information environment, we can identify how individuals differ in effort sensitivity and threat aversion without confounding these parameters with learning or state-dependent variation. This comes at a cost: our results speak to the computational structure of effort-threat integration, not to the full complexity of ecological foraging.

A separate gap concerns the relationship between threat computation and clinical anxiety. People experiencing pathological anxiety do not simply overestimate threat; they exhibit a dysfunctional relationship between threat appraisal and affective response^21^. Paulus and Stein^22^ formalize this as interoceptive prediction error: clinical anxiety involves systematic mismatch between expected body states and actual somatic experience under threat, producing affect that is decoupled from the environment. In Wise and colleagues' work on interactive threat^33^, confidence ratings track the quality of cognitive models of threat with remarkable fidelity---participants know when their predictions are improving and adjust behavior accordingly. This raises a question: when people compute danger normatively but feel anxious anyway, is it the computation or the feeling that predicts psychiatric vulnerability? Computational psychiatry has sought to ground psychiatric constructs in formal decision models^24--26^, but no study has decomposed the metacognitive relationship between computed danger and experienced affect within a normative foraging framework where both quantities are measured simultaneously.

Here we address both gaps. We developed an EVC model with a cost structure motivated by linear-quadratic optimal control^27,28^ and tested it in a virtual effort-foraging task under parametric threat (N = 293). Participants chose between high-effort, high-reward and low-effort, low-reward resources while facing predation risk at three probability levels and three distances, then physically executed their foraging bout by pressing keys. Two subject-level parameters---effort cost (*c*~e~) and capture aversion (*c*~d~)---jointly predicted choice and vigor through distinct channels. We then decomposed each participant's subjective anxiety into metacognitive calibration (the fidelity with which anxiety tracks model-derived danger) and discrepancy (the systematic bias by which anxiety exceeds danger), revealing a differential prediction pattern: calibration predicts who performs well, while discrepancy---not the computational parameters themselves---predicts who reports elevated anxiety symptoms. A preregistered confirmatory replication (N = 350 recruited, preregistration available at [AsPredicted URL]) is underway.

---

## Results

### Threat and distance deter high-effort foraging and modulate vigor

We designed an effort-foraging task in which participants (N = 293, after five-stage quality screening of 350 recruits) chose between a high-reward cookie (5 points, requiring sustained keypressing at 60--100% of individually calibrated maximum capacity) and a low-reward cookie (1 point, requiring 40% of maximum) while facing predation risk (threat probability *T* $\in$ {0.1, 0.5, 0.9}) at varying distances (*D* $\in$ {1, 2, 3}; Fig. 1a). On attack trials, a predator appeared and pursued the participant; capture incurred a 5-point penalty and loss of the cookie's value. Participants completed 45 choice trials and 36 probe trials (forced-choice with identical options, paired with affect ratings).

Both threat and distance reduced high-effort choice (Fig. 1b). A logistic mixed-effects model confirmed threat as the dominant deterrent ($\beta$ = -1.28, *z* = -32.0, *p* < 10^-200^), with distance ($\beta$ = -0.65, *z* = -16.3, *p* < 10^-59^) and their interaction ($\beta$ = -0.18, *z* = -4.5, *p* < 10^-5^) also significant. At the extremes, P(heavy) dropped from 0.81 at *T* = 0.1, *D* = 1 to 0.08 at *T* = 0.9, *D* = 3. All adjacent-threat comparisons within each distance were significant (all *t* > 8.3, all *p* < 10^-14^), confirming monotonic deterrence.

Threat also increased motor vigor, but this effect was masked by a Simpson's paradox in unconditional analyses. High threat shifts choice toward light cookies (60% heavy at *T* = 0.1 versus 34% at *T* = 0.9), and light cookies have lower required press rates, so collapsing across cookie type makes average vigor appear flat. Conditioning on chosen cookie type revealed robust threat-driven vigor increases (heavy: *t* = 6.6, *p* < 10^-10^, *d* = 0.42; light: *t* = 7.5, *p* < 10^-13^, *d* = 0.49). Within each cookie type, participants pressed harder when threat was higher---consistent with the prediction that the marginal survival benefit of faster pressing increases with danger.

### An EVC model with LQR-inspired cost structure jointly captures choice and vigor

We formalized the foraging decision as a comparison of expected utilities:

$\Delta EU = S \times 4 - c_{e,i} \times (0.81D_H - 0.16)$

where *S* is the subjective survival probability, *c*~e,i~ is the subject-specific effort cost, and the effort term reflects the commitment cost inspired by LQR theory---the squared required press rate scaled by distance (specifically, 0.9^2^ $\times$ *D*~H~ - 0.4^2^ $\times$ 1 = 0.81*D*~H~ - 0.16). The survival function incorporates probability weighting and effort efficacy:

$S = (1 - T^{\gamma}) + \varepsilon \times T^{\gamma} \times p_{\text{esc}}$

where $\gamma$ = 0.209 is the population probability-weighting exponent (indicating substantial compression of threat probabilities: a nominal 50% threat is experienced as *T*^0.209^ = 0.86) and $\varepsilon$ = 0.098 reflects the near-universal tendency to underweight effort's benefit to survival. Choices follow a softmax rule: P(heavy) = sigmoid($\Delta$EU / $\tau$), with $\tau$ = 0.476.

With per-subject *c*~e~ (log-normal prior, non-centered parameterization), the model reproduced the full threat-by-distance choice surface (Fig. 2a): at *T* = 0.1, predicted P(heavy) declined from 0.87 at *D* = 1 to 0.53 at *D* = 3; at *T* = 0.9, from 0.33 to 0.10. The model achieved per-subject choice *r*^2^ = 0.951 (subj *r* = 0.976), choice accuracy = 79.3%, and AUC = 0.876 (Fig. 2b). Previous versions with population-level effort cost failed to reproduce the distance gradient that drives approximately 40% of choice variance (Supplementary Fig. S1).

For the vigor component, the model computes optimal press rate *u** by maximizing:

$EU(u) = S(u) \times R - (1 - S(u)) \times c_{d,i} \times (R + C) - c_{e,\text{vigor}} \times (u - \text{req})^2 \times D$

where *c*~d,i~ is the subject-specific capture aversion and *c*~e,vigor~ = 0.003 is the population-level motor deviation cost. The survival function becomes speed-dependent: pressing faster than required improves escape probability. Three features of this architecture merit attention.

First, *c*~d~ is excluded from the choice equation not because it cancels algebraically, but because it is empirically unidentifiable from choice data. In the full expected utility, the capture-loss term for option *i* is -(1-*S*) $\times$ *c*~d~ $\times$ (*R*~i~ + *C*); the fixed penalty *C* cancels in the difference between options, but the residual *c*~d~ term scales with (*R*~H~ - *R*~L~)---the same factor as the reward term---rendering the two inseparable. We therefore estimate *c*~e~ exclusively from choice and *c*~d~ exclusively from vigor, a deliberate modeling choice motivated by identifiability.

Second, the distinction between commitment cost (req^2^ $\times$ *D*, governing choice) and deviation cost ((*u* - req)^2^ $\times$ *D*, governing vigor), inspired by the LQR separation of reference-trajectory and tracking costs^27^, resolves a scaling conflict: commitment costs operate at two orders of magnitude above deviation costs, allowing a single theoretical framework to span both decision stages. We note that this is an analogy to LQR optimal control, not a formal implementation: our model has no state dynamics, no feedback law, and no Riccati equation. The quadratic cost structure is motivated by LQR theory but the optimization is static.

Third, probe trials (forced conditions with identical options) anchor *c*~d~ estimation across all threat-by-distance cells without selection bias.

The model predicted trial-level vigor with *r*^2^ = 0.511 (subj vigor *r* = 0.836; Fig. 2c,d). Higher *c*~d~ drives steeper threat-vigor gradients because the marginal benefit of pressing faster grows with the stakes of failed escape (*c*~d~ $\times$ (*R* + *C*)).

Model comparison across six variants confirmed that every component was necessary (Table 1). Removing individual effort cost (M1, effort-only: $\Delta$BIC = +18,659), eliminating individual effort cost while retaining only threat (M2: $\Delta$BIC = +2,094, choice *r*^2^ = 0.006), separating choice and vigor into independent models (M3: $\Delta$BIC = +10,393), or dropping probability weighting (M5: $\Delta$BIC = +2,071) all substantially degraded fit. The LQR-inspired deviation cost performed comparably to a standard *u*^2^ cost (M6: $\Delta$BIC = -142), confirming that the two formulations are empirically indistinguishable in our data. We retain the LQR-inspired framing for its theoretical motivation rather than empirical superiority.

**Table 1. Model comparison.**

| Model | Description | BIC | $\Delta$BIC | Choice *r*^2^ | Vigor *r*^2^ |
|-------|------------|-----|------------|-------------|------------|
| **EVC 2+2** | **Full model** | **32,133** | **---** | **0.951** | **0.511** |
| M1 | Effort only (no threat) | 50,792 | +18,659 | 0.950 | 0.000 |
| M2 | Threat only (no individual effort) | 34,227 | +2,094 | 0.006 | 0.513 |
| M3 | Separate choice + vigor | 42,563 | +10,430 | 0.955 | 0.441 |
| M4 | Population *c*~e~ | 30,860 | -1,274 | 0.001 | 0.512 |
| M5 | No $\gamma$ ($\gamma$ = 1) | 34,204 | +2,071 | 0.955 | 0.425 |
| M6 | Standard *u*^2^ cost | 31,991 | -142 | 0.952 | 0.508 |

*Note.* M4 achieves lower BIC but fails to predict individual choice (choice *r*^2^ = 0.001), sacrificing the primary behavioral target for marginal vigor improvement. M6 is nearly equivalent to the full model, confirming that the LQR-inspired and standard motor cost formulations are empirically indistinguishable.

Parameter recovery confirmed identifiability. Simulating 293 synthetic subjects at the fitted population distribution and refitting yielded correlations (in log space) of *r* = 0.92 for log(*c*~e~) and *r* = 0.94 for log(*c*~d~) (Fig. 2e,f). The two parameters were approximately independent (*r* = -0.14), confirming that they capture distinct dimensions of individual variation.

### The model's survival signal predicts trial-level anxiety and confidence

On probe trials, participants rated either anxiety ("How anxious are you about being captured?") or confidence ("How confident are you about reaching safety?") on a 0--7 scale, after choosing but before pressing. The model's survival probability *S* strongly predicted both ratings via linear mixed models with random intercepts and slopes by subject:

- Anxiety: $\beta$ = -0.557, SE = 0.040, *t* = -14.04, *p* = 8.8 $\times$ 10^-45^ (N~obs~ = 5,274)
- Confidence: $\beta$ = +0.575, SE = 0.043, *t* = +13.48, *p* = 2.1 $\times$ 10^-41^ (N~obs~ = 5,272)

These effects translate to approximately 2 points on the 0--7 scale when moving from the safest (*T* = 0.1, *S* $\approx$ 0.85) to the most dangerous (*T* = 0.9, *S* $\approx$ 0.15) condition. Substantial random slope variance (anxiety: 0.302; confidence: 0.375) indicates meaningful individual differences in how tightly affect tracks survival---some participants show steep affect-survival gradients, while others show flat gradients. This individual variation forms the basis of the decomposition below.

Task-derived affect showed convergent validity with validated clinical instruments: mean task anxiety correlated with STAI (*r* = 0.31) and STICSA (*r* = 0.27), while mean task confidence was negatively associated with AMI (*r* = -0.25). Within-subject task anxiety and task confidence were only weakly correlated (*r* = -0.25), and their between-subject means were essentially independent (*r* = -0.01), indicating that anxiety and confidence function as partially separable affective channels rather than opposite poles of a single dimension.

Despite strong within-subject tracking, between-subject mean confidence did not reliably predict task performance: confident participants did not survive more often (*r* = -0.05, *p* = .41) nor make more EV-optimal choices (*r* = -0.08, *p* = .16). This null finding motivated us to decompose the affect-danger relationship into finer-grained components.

### Affective calibration predicts performance; discrepancy predicts anxiety symptoms

We decomposed each participant's affect-danger relationship into two dimensions:

**Calibration:** the within-subject Pearson correlation between anxiety ratings and model-derived danger (1 - *S*), computed across each participant's 18 anxiety probe trials. Higher calibration indicates that anxiety more accurately tracks the computational danger signal.

**Discrepancy:** the mean residual of a participant's anxiety ratings after removing the population-level anxiety-danger relationship. Positive discrepancy indicates systematically elevated anxiety beyond what the danger signal warrants---an affective signal decoupled from the environment.

These dimensions were orthogonal (*r* = 0.019, *p* = .75; Fig. 3a), confirming that the accuracy of threat monitoring and the magnitude of affective bias are genuinely independent. Most participants (85%) showed positive calibration, indicating that anxiety generally increased with danger, but with wide individual variation (*M* = 0.47, *SD* = 0.32).

A differential prediction pattern emerged (Fig. 3b,c). Calibration predicted adaptive performance: participants whose anxiety accurately tracked danger made higher-quality foraging decisions (*r* = 0.230, *p* < .001) and survived at higher rates (*r* = 0.179, *p* = .002), consistent with the proposal that accurate anxiety functions as an adaptive alarm calibrated to genuine threat^22^. However, calibration was largely unrelated to clinical symptoms (6 of 7 psychiatric measures *p* > .10; only STAI showed a weak positive association, *r* = 0.121, *p* = .04).

Discrepancy predicted anxiety symptoms across the full spectrum of anxiety, depression, and stress measures. Bayesian regression controlling for the model's computational parameters (log(*c*~e~), log(*c*~d~)) confirmed that discrepancy was a robust predictor with all 94% highest density intervals excluding zero (Table 2).

**Table 2. Bayesian regression: discrepancy predicts anxiety symptoms.**

| Measure | $\beta$(discrepancy) | 94% HDI | $\beta$(log *c*~e~) % in ROPE | $\beta$(log *c*~d~) % in ROPE |
|---------|---------------------|---------|-------------------------------|-------------------------------|
| STAI-State | 0.338 | [0.23, 0.45] | 91% | 93% |
| STICSA | 0.285 | [0.17, 0.39] | 79% | 87% |
| DASS-Anxiety | 0.275 | [0.16, 0.40] | 60% | 90% |
| DASS-Stress | 0.255 | [0.13, 0.37] | 62% | 91% |
| DASS-Depression | 0.228 | [0.11, 0.35] | 78% | 81% |
| PHQ-9 | 0.225 | [0.11, 0.34] | 71% | 89% |
| OASIS | 0.228 | [0.12, 0.35] | 56% | 77% |
| AMI-Emotional | -0.222 | [-0.34, -0.11] | 53% | 54% |

*Note.* ROPE = region of practical equivalence, |$\beta$| < 0.10. High percentage in ROPE (>80%) provides Bayesian evidence for the null; values in the 53--60% range are better characterized as inconclusive. All discrepancy HDIs exclude zero.

The model's computational parameters (*c*~e~ and *c*~d~) showed no reliable associations with symptom severity in either frequentist (all FDR-corrected *p* > .70) or Bayesian analyses (ROPE containment 53--93%; Table 2). For high-ROPE cases (e.g., *c*~e~ for STAI-State at 91%), this constitutes positive Bayesian evidence for the null; for lower-ROPE cases (e.g., AMI at 53--54%), the evidence is inconclusive. Overall, the computational parameters governing foraging behavior are predominantly dissociated from the affective dimension governing symptom severity.

The negative association between discrepancy and emotional apathy (AMI-Emotional: $\beta$ = -0.222) adds specificity: individuals who overestimate danger are the opposite of affectively disengaged---they are motivationally aroused, perhaps excessively so. This echoes the proposal that anxiety and apathy occupy opposite poles of motivational dysfunction^10^.

Cross-validated machine learning (elastic net and ridge regression with nested 10-fold CV, 5 repeats) confirmed that no combination of model parameters and affective dimensions could predict individual symptom outcomes: all cross-validated *R*^2^ values were negative (range: -0.03 to -0.06). These associations are group-level patterns---useful for identifying mechanistic dimensions of variation---rather than individually predictive biomarkers.

### Within-trial encounter dynamics reveal a trait-like defensive reflex dissociated from strategic threat processing

We next examined the 20 Hz vigor timeseries aligned to predator encounter events. Encounter reactivity---defined as the difference between post-encounter and pre-encounter excess vigor (cookie-centered)---showed no reliable mean effect across participants (*M* = -0.019, *t* = -1.15, *p* = .25), but massive individual variation (*SD* = 0.28). Critically, encounter reactivity was highly stable across task blocks (*r* = 0.78), identifying it as a trait-like individual difference. Reactivity correlated strongly with the model's capture aversion parameter (*r* = 0.47 with log(*c*~d~)) but not with effort cost (*r* $\approx$ 0 with log(*c*~e~)). Threat probability did not modulate the encounter response (one-way ANOVA: *F* = 0.04, *p* = .96). This dissociation---threat modulates strategic choice but not the phasic encounter reflex---is consistent with the distinction between strategic and reactive defensive modes^5,14^.

Encounter reactivity was selectively associated with lower apathy (AMI: *r* = -0.17, *p* = .004), with incremental variance explained ($\Delta$*R*^2^ = 0.048) beyond the static model parameters. No other clinical measure showed a reliable association (STAI, OASIS, DASS, PHQ-9: all *p* > .05). Combined with the finding that discrepancy---not *c*~d~---predicts anxiety symptoms, this suggests two dissociable clinical dimensions: vigor reactivity relates to motivational engagement (apathy), while affective bias relates to threat processing (anxiety). Piecewise regression confirmed a qualitative shift in vigor dynamics around the encounter: pre-encounter vigor declined (slope = -0.021) while post-encounter vigor increased (slope = +0.029; change *t* = 5.91, *p* < 10^-8^; Supplementary Fig. S2).

### Discrepancy is stable across blocks and shows convergent validity

To evaluate whether discrepancy reflects a stable individual trait rather than transient state fluctuation, we computed within-subject discrepancy separately for each of the three task blocks. Block-to-block correlations ranged from *r* = 0.48 to *r* = 0.68, indicating moderate-to-good test-retest stability across approximately 20 minutes of task performance. This stability is consistent with discrepancy reflecting a trait-like tendency toward affective overestimation^22^.

---

## Discussion

We developed an EVC model with LQR-inspired cost structure that jointly captures foraging choice and action vigor under parametric threat. Two subject-level parameters---effort cost (*c*~e~) and capture aversion (*c*~d~)---explain 95.1% of between-subject choice variance and 51.1% of trial-level vigor variance through a single cost function with distinct channels for decision and execution. The model's survival signal predicts moment-to-moment anxiety and confidence, and decomposing the affect-danger relationship reveals that calibration predicts adaptive performance while discrepancy predicts symptom severity across anxiety, depression, and stress measures. The computational parameters themselves show no reliable associations with psychiatric symptoms.

### An EVC framework for physical effort under threat

The central modeling contribution is extending the EVC framework^17,18^ from cognitive control to physical effort under threat. The key theoretical move is adopting a cost structure inspired by linear-quadratic optimal control^27^, which distinguishes commitment costs (the effort implied by choosing a distant option: req^2^ $\times$ *D*) from deviation costs (the additional effort of pressing faster than required: (*u* - req)^2^ $\times$ *D*). This maps onto a natural division in foraging behavior: the strategic decision of what to pursue versus the tactical execution of how vigorously to pursue it. We emphasize that this is an analogy to LQR theory, not a formal implementation: our model uses the quadratic cost structure motivated by optimal control but does not include state dynamics, feedback laws, or Riccati equations. Although the LQR-inspired and standard motor cost formulations proved empirically equivalent (M6: $\Delta$BIC = -142), the LQR-inspired framework generates the qualitative prediction that choice and vigor should be governed by separable cost channels---a prediction confirmed by the near-independence of *c*~e~ and *c*~d~ (*r* = -0.14).

The probability weighting parameter ($\gamma$ = 0.209) indicates dramatic compression of threat probabilities, consistent with Kahneman and Tversky's^29^ loss-domain distortion but substantially stronger than estimates from monetary gambles ($\gamma$ ~ 0.65--0.70)^30^. This amplification may reflect the embodied nature of virtual predation, which engages defensive circuitry^5,13^ more powerfully than abstract monetary losses. The dissociation of *c*~e~ and *c*~d~ maps onto the distinction between strategic and reactive defensive modes identified in the threat-imminence literature^5,14,31^: choice reflects strategic assessment, while vigor reflects reactive mobilization. These modes may engage partially separable neural circuits, consistent with evidence for prefrontal-mediated strategic defense and subcortical-mediated reactive defense^13,14^.

The within-trial encounter dynamics (R2b) sharpen this strategic-reactive distinction. The threat-independence of the encounter response (*F* = 0.04, *p* = .96) suggests it operates at the reactive end of the defensive cascade---an automatic mobilization triggered by predator detection, not a probability-weighted strategic adjustment. That *c*~d~ captures both this phasic reflex and tonic pressing level (*r* = 0.47) supports the interpretation of *c*~d~ as a general defensive motor readiness trait, spanning the continuum from sustained vigilance to acute encounter response^5,14^.

The identification of the two parameters depends on a structural feature of our task design. In the choice comparison, the fixed capture penalty *C* cancels between options, and the residual *c*~d~ term is collinear with the reward differential (*R*~H~ - *R*~L~), making *c*~d~ empirically unidentifiable from choice data. We therefore exclude *c*~d~ from the choice equation---a deliberate simplifying assumption that pins each parameter to its behavioral channel by the model's architecture rather than by post-hoc decomposition. This exclusion means the model may misattribute some *c*~d~-driven choice variation to *c*~e~; however, strong parameter recovery (*c*~e~: *r* = 0.92; *c*~d~: *r* = 0.94) and near-independence (*r* = -0.14) suggest that cross-contamination is minimal.

### Affective calibration as the bridge between computation and psychopathology

The most consequential finding is that the route from normative foraging computation to anxiety symptoms runs through the affect-computation relationship rather than through the decision parameters themselves. Neither *c*~e~ nor *c*~d~ predicted any symptom measure (all FDR-corrected *p* > .70; Bayesian ROPE containment 53--93%). Instead, discrepancy---the systematic excess of experienced anxiety over model-derived danger---predicted symptoms across seven validated instruments ($\beta$ = 0.18--0.34).

We use "metacognitive" to describe calibration and discrepancy because these dimensions capture the alignment between a person's affective response and the computational danger signal their own behavior implicitly reflects. This is not metacognition in the narrow sense of second-order confidence judgments^39^ or meta-worry^21^, but rather the monitoring of one's own internal states relative to environmental demands. Calibration indexes how faithfully anxiety tracks the survival-relevant danger that choices reveal a person has computed; discrepancy indexes systematic bias in that monitoring. This usage is closer to Paulus and Stein's^22^ interoceptive prediction error framework than to the perceptual metacognition tradition, and we acknowledge that "affective calibration" and "affective bias" are equally valid descriptors. We retain "metacognitive" because the clinically relevant dimension is not the computation itself but the person's *relationship to* their computation---a second-order property. We also note that our measure is not a true interoceptive prediction error---it is a residual between self-reported affect and a model-derived environmental quantity, not a mismatch between predicted and actual body states. The connection to Paulus and Stein is analogical rather than literal.

The calibration-performance association (*r* = 0.19--0.23) complements this picture. Participants whose anxiety faithfully tracked computed danger made better foraging decisions and survived more often, suggesting that well-calibrated anxiety functions as an adaptive alarm tuned to actual threat. This aligns with functional accounts of anxiety^32^ and with Wise and colleagues' finding that confidence tracks the quality of internal threat models^33^. We characterize the overall pattern as a differential prediction rather than a double dissociation in the strict neuropsychological sense, because the separation is not absolute: calibration showed a weak positive association with STAI-State (*r* = 0.12, *p* = .04), and discrepancy showed a modest negative association with survival (*r* = -0.15, *p* = .009). These cross-associations are small relative to the primary effects but indicate leakage that a true double dissociation would preclude.

Our findings create a productive tension with the Affective Gradient Hypothesis^32^, which proposes that affect is the sole currency of motivated behavior. Under this account, discrepancy would not represent a metacognitive error but rather a difference in affective accessibility: some individuals have richer or more salient threat-related affective representations, producing higher anxiety at the same objective danger. Whether discrepancy reflects genuine metacognitive bias or an adaptive affective phenotype cannot be resolved by our data alone, but the clinical associations suggest that regardless of interpretation, excess anxiety relative to computed danger is associated with psychiatric vulnerability.

The negative association between discrepancy and emotional apathy (AMI: $\beta$ = -0.222) further constrains the interpretation: overanxious individuals are motivationally engaged but affectively miscalibrated. This is consistent with the proposal that anxiety and apathy represent opposite poles of motivational activation^10,11^, and suggests that excessive threat-related affect may co-opt motivational resources in ways that produce distress without improving performance.

### Associations with symptom severity and effect-size honesty

The associations with anxiety symptoms are statistically robust but modest in magnitude ($\beta$ = 0.18--0.34, corresponding to approximately 3--11% of symptom variance). These effect sizes are consistent with the broader computational psychiatry literature^24,33^ and with the inherent challenge of linking single-task behavioral parameters to multidetermined psychiatric constructs measured by self-report in a non-clinical sample. Four constraints are important to note.

First, all cross-validated *R*^2^ values for predicting individual outcomes were negative, confirming that these are group-level mechanistic patterns rather than individually predictive biomarkers.

Second, discrepancy is model-dependent: it quantifies the residual between experienced anxiety and model-derived danger. If the survival function is misspecified, discrepancy may partly reflect model error rather than affective bias. The strong within-subject anxiety-survival correlations (*t* = -14.0) mitigate this concern but do not eliminate it.

Third, the correlation between discrepancy and clinical measures may be partly inflated by shared method variance, as both involve self-reported anxiety. However, calibration---also computed from anxiety ratings---does not predict symptom severity (6 of 7 measures *p* > .10), and discrepancy predicts not only anxiety measures but also depression (DASS-Depression, PHQ-9) and apathy (AMI), which are conceptually and methodologically distinct from task anxiety ratings. Nonetheless, we cannot rule out the possibility that discrepancy partly indexes general negative affectivity, and future work should test whether discrepancy predicts anxiety symptoms over and above a general internalizing dimension.

Fourth, all results come from the discovery sample. A preregistered confirmatory study (N = 350 recruited, preregistration at [AsPredicted URL]) with identical task design, model specification, and analysis plan will test whether these patterns replicate. The confirmatory study specifies directional hypotheses and significance thresholds derived from discovery effect sizes.

### The dissociation of choice and vigor

The near-independence of *c*~e~ and *c*~d~ (*r* = -0.14) means that knowing how someone decides tells little about how hard they will try. A descriptive median-split typology (Supplementary Table S2) reveals that the profile combining low effort cost with high capture aversion ("Vigilant") earned the most points (29.9 vs. 2.9--10.1 for other profiles), because it combines willingness to pursue high-value resources with vigorous execution during transport. This dissociation may reflect partially separable neural substrates of strategic and reactive defense^13,14^, though our behavioral data cannot speak directly to neural implementation.

### Limitations

Several limitations warrant consideration. The effort-efficacy parameter $\varepsilon$ is estimated at the population level because it is not individually recoverable (recovery *r* $\approx$ 0). Population-level $\varepsilon$ means our model assumes everyone shares the same (very low, $\varepsilon$ = 0.098) belief about effort's survival value. If some participants actually believe effort helps survival more than others, their "excess anxiety" on high-effort trials might be rational rather than biased. Individual differences in effort-efficacy beliefs may be an important source of variation in both vigor and discrepancy that our model cannot capture; identifying such differences would require richer within-subject designs or physiological measures.

Distance confounds effort duration and threat exposure by design: farther cookies require both more sustained pressing and more time in danger. This confound is shared with natural foraging^1^ and is partially addressed by the factorial crossing of distance with effort demand, but a fully orthogonal design would strengthen causal claims about the independent contributions of effort and threat.

The task uses explicit threat probabilities, which differ from real-world threat assessment involving learning and belief updating^34^. Our model assumes stationary threat and does not capture trial-by-trial belief updating that may occur even with stated probabilities.

Our Prolific sample reflects dimensional variation in self-reported psychiatric symptoms in a non-clinical population. All references to "anxiety symptoms" throughout this paper refer to dimensional variation on validated symptom scales, not clinical diagnoses. Whether the discrepancy-symptom association strengthens---or changes qualitatively---in clinical groups with diagnosed anxiety disorders is an important next step. Effect sizes observed in non-clinical samples may not generalize to clinical populations due to floor/ceiling effects, medication, or qualitative differences in threat processing.

Finally, the model is static: it does not capture learning across trials, strategic adjustments across blocks, or the potential for affect to feed back into subsequent decisions. The moderate block-to-block stability of discrepancy (*r* = 0.48--0.68) suggests partial stability but also meaningful within-session change, which dynamic models could characterize.

---

## Methods

### Participants

We recruited 350 participants from Prolific (https://prolific.co) for an online study. After a five-stage quality control pipeline---(1) task completion, (2) comprehension checks, (3) behavioral consistency screening, (4) effort calibration validation (minimum 10 presses in 10 seconds), and (5) outlier removal (escape rate < 35%)---the final discovery sample comprised N = 293 participants (83.7% retention). No post-hoc exclusions were applied based on model fit quality or statistical extremity. The study was approved by the Caltech Institutional Review Board, and all participants provided informed consent.

### Task design

Participants completed an effort-based foraging task implemented in Unity (WebGL) and presented in a desktop browser. The task featured a circular arena viewed from above, with a safe zone at the center and cookie targets at radial distances.

**Effort calibration.** Before the main task, participants completed three 10-second trials pressing the S, D, and F keys simultaneously as fast as possible. The maximum press count defined each participant's calibrated maximum (*f*~max~). All demands were expressed relative to this calibration.

**Trial structure.** On choice trials, two cookies appeared: a heavy cookie (5 points, requiring pressing at 60--100% of *f*~max~ depending on weight condition) at distance *D* $\in$ {1, 2, 3} (5, 7, 9 game units), and a light cookie (1 point, 40% of *f*~max~) always at *D* = 1. Participants clicked to select (irrevocable commitment), then pressed S+D+F repeatedly to transport the cookie to safety. Movement speed followed a step function: full speed at $\geq$100% of required rate, half at $\geq$50%, quarter at $\geq$25%, and zero below 25%.

**Threat manipulation.** Each trial had predation probability *T* $\in$ {0.1, 0.5, 0.9}. On attack trials, a predator spawned at the perimeter position closest to the participant at a pre-determined encounter time (calibrated to half the expected return time at full speed), approached at 0.5 units/sec, then struck at a Gaussian-distributed time at 4$\times$ the participant's maximum speed (unavoidable). Capture cost 5 points plus the cookie value.

**Probe trials.** Twelve of 27 events per block were probe trials with identical options (forced choice). After selection, the game paused for a rating of either anxiety about capture or confidence in reaching safety (0--7 scale). Probes were prospective: collected after commitment but before motor execution.

**Structure.** Three blocks of 27 events (81 total): 15 choice trials and 12 probe trials per block, yielding 45 choice and 36 probe trials. Conditions were fully crossed: 3 threat $\times$ 3 distance $\times$ 5 repetitions (choice); 3 threat $\times$ 3 distance $\times$ 2 cookie types $\times$ 2 rating types (probe).

### Psychiatric assessment

Between blocks, participants completed the DASS-21 (Depression, Anxiety, Stress subscales), PHQ-9, OASIS, STAI-State, AMI (Behavioural, Social, Emotional subscales), MFIS (Physical, Cognitive, Psychosocial subscales), and STICSA. All subscale scores were z-scored across participants.

### EVC model with LQR-inspired cost structure

**Per-subject parameters** (log-normal, non-centered parameterization):
- *c*~e~ (effort cost): governs choice via distance-dependent effort penalty
- *c*~d~ (capture aversion): governs vigor via the survival incentive

**Population parameters:** $\gamma$ = 0.209 (probability weighting); $\varepsilon$ = 0.098 (effort efficacy); *c*~e,vigor~ = 0.003 (deviation cost); $\tau$ = 0.476 (choice temperature); *p*~esc~ (escape probability); $\sigma$~motor~ (motor noise); $\sigma$~v~ (vigor observation noise).

**Choice model.** $\Delta$EU = *S* $\times$ 4 - *c*~e,i~ $\times$ (0.81*D*~H~ - 0.16). The capture aversion *c*~d~ is excluded because its contribution to the option differential is collinear with the reward term, making it empirically unidentifiable from choice data (see Results). P(heavy) = sigmoid($\Delta$EU / $\tau$).

**Vigor model.** EU(*u*) = *S*(*u*) $\times$ *R* - (1 - *S*(*u*)) $\times$ *c*~d,i~ $\times$ (*R* + *C*) - *c*~e,vigor~ $\times$ (*u* - req)^2^ $\times$ *D*. Survival is speed-dependent: *S*(*u*) = (1 - *T*^$\gamma$^) + $\varepsilon$ $\times$ *T*^$\gamma$^ $\times$ *p*~esc~ $\times$ sigmoid((*u* - req) / $\sigma$~motor~). Optimal *u** is computed via softmax-weighted grid search.

**Joint likelihood.** Choice trials contribute a Bernoulli likelihood; all 81 trials contribute a Normal likelihood for vigor. Both are evaluated simultaneously.

### Model fitting

The primary fit used NumPyro stochastic variational inference (SVI) with an AutoNormal guide (mean-field approximation), Adam optimizer (lr = 0.002), 40,000 steps. SVI provides point estimates of posterior means but may underestimate posterior uncertainty due to the assumption of posterior independence between parameters. HDIs and ROPE analyses reported for the clinical regressions (which use full MCMC via bambi/PyMC) are not affected by this approximation, but the population-level parameter estimates ($\gamma$, $\varepsilon$, $\tau$) from the main model should be interpreted as approximate posterior modes rather than full posteriors.

To validate the SVI approximation, we also fitted the model using NumPyro MCMC with the NUTS sampler (4 chains, 200 warmup + 200 sampling iterations, target acceptance probability = 0.8, max tree depth = 10). MCMC produced zero divergent transitions, and per-subject parameter estimates correlated near-perfectly with SVI estimates (log(*c*~e~): *r* = 0.999; log(*c*~d~): *r* = 0.999; see Supplementary Note: MCMC Validation), confirming that the SVI approximation provides reliable point estimates despite its limitations on posterior uncertainty.

BIC = 2 $\times$ loss + *k* $\times$ log(*n*), where *k* = 2 $\times$ *N*~subjects~ + number of population parameters. Using the ELBO loss in place of the log-likelihood for BIC computation is non-standard, as the ELBO is a lower bound on the marginal likelihood. Future work should supplement BIC with WAIC or LOO-CV, which properly account for effective parameter count.

### Parameter recovery

293 synthetic subjects were generated from the fitted population distribution with the same task design. Simulated data were refitted with identical SVI procedure. Recovery assessed as Pearson *r* between true and recovered parameters in log space.

### Model comparison

Six ablation models were fitted, each removing one component: individual effort cost, threat, joint estimation, probability weighting, or the LQR cost structure. All models were evaluated on the same data (81 trials per subject for both likelihoods).

### Vigor computation

Trial-level vigor was computed as the median normalized press rate (median(1/IPI) / *f*~max~) minus the required rate for the chosen cookie, then centered by cookie type (subtracting the population mean excess for heavy and light cookies separately). This cookie-type centering removes the demand confound while preserving between-subject variation.

### Within-trial vigor dynamics

To characterize encounter-evoked vigor changes, we aligned the 20 Hz smoothed vigor timeseries to each trial's predator encounter time. Pre-encounter and post-encounter excess vigor were computed as the mean cookie-centered excess vigor in symmetric windows before and after the encounter event. Encounter reactivity was defined as the post-encounter minus pre-encounter difference, yielding one value per trial per subject. Cross-block stability was assessed as the Pearson correlation between mean reactivity in the first and second halves of the task. Threat modulation was tested via one-way ANOVA on mean reactivity across the three threat levels. Incremental clinical prediction was assessed by adding reactivity to a regression of AMI scores on log(*c*~e~) and log(*c*~d~). Piecewise linear regression with a knot at the encounter time estimated pre- and post-encounter vigor slopes.

### Affect analysis

Linear mixed models (statsmodels MixedLM, REML, L-BFGS optimizer) predicted anxiety and confidence from standardized survival probability (*S*~z~) with random intercepts and slopes by subject.

### Affective calibration and discrepancy decomposition

**Calibration:** Per-subject Pearson *r* between anxiety ratings and model-derived danger (1 - *S*), using population-level $\gamma$ and $\varepsilon$. **Discrepancy:** Per-subject mean residual from the population-level regression of anxiety on *S*.

### Clinical analysis

Frequentist: Pearson correlations between log-transformed parameters and z-scored psychiatric scores, FDR-corrected. Bayesian: linear regression (bambi/PyMC, weakly informative priors, 2,000 draws $\times$ 4 chains), predicting each clinical measure from log(*c*~e~) + log(*c*~d~) + discrepancy + calibration; ROPE = |$\beta$| < 0.10. Machine learning: elastic net and ridge regression with repeated 10-fold CV.

### Statistical analysis

All tests were two-tailed unless specified. Effect sizes reported as Pearson *r*, standardized $\beta$, or *R*^2^. Multiple comparisons corrected by Benjamini-Hochberg FDR. Steiger's test^35^ for comparing dependent correlations. All analyses conducted in Python 3.11 using NumPyro, JAX, statsmodels, scipy, bambi, and PyMC.

---

## References

1. Lima, S. L. & Dill, L. M. Behavioral decisions made under the risk of predation: a review and prospectus. *Can. J. Zool.* **68**, 619--640 (1990).
2. Charnov, E. L. Optimal foraging, the marginal value theorem. *Theor. Popul. Biol.* **9**, 129--136 (1976).
3. Cresswell, W. Predation in bird populations. *J. Ornithol.* **152**, 251--263 (2011).
4. Milinski, M. & Heller, R. Influence of a predator on the optimal foraging behaviour of sticklebacks. *Nature* **275**, 642--644 (1978).
5. Mobbs, D., Trimmer, P. C., Blumstein, D. T. & Dayan, P. Foraging for foundations in decision neuroscience: insights from ethology. *Nat. Rev. Neurosci.* **19**, 419--427 (2018).
6. Pessiglione, M., Vinckier, F., Bouret, S., Daunizeau, J. & Le Boisselier, R. Why not try harder? Computational approach to motivation deficits in neuro-psychiatric diseases. *Brain* **141**, 629--650 (2018).
7. Westbrook, A. & Braver, T. S. Dopamine does double duty in motivating cognitive effort. *Neuron* **89**, 695--710 (2016).
8. Hartmann, M. N., Hager, O. M., Tobler, P. N. & Kaiser, S. Parabolic discounting of monetary rewards by physical effort. *Behav. Process.* **100**, 192--196 (2013).
9. Husain, M. & Roiser, J. P. Neuroscience of apathy and anhedonia: a transdiagnostic approach. *Nat. Rev. Neurosci.* **19**, 470--484 (2018).
10. Le Heron, C. et al. Brain mechanisms underlying apathy. *J. Neurol. Neurosurg. Psychiatry* **90**, 302--312 (2019).
11. Chong, T. T.-J. et al. Neurocomputational mechanisms underlying subjective valuation of effort costs. *PLoS Biol.* **15**, e1002598 (2017).
12. Mobbs, D., Hagan, C. C., Dalgleish, T., Silston, B. & Prevost, C. The ecology of human fear: survival optimization and the nervous system. *Front. Neurosci.* **9**, 55 (2015).
13. Qi, S. et al. How cognitive and reactive fear circuits optimize escape decisions in humans. *Proc. Natl Acad. Sci. USA* **115**, 3186--3191 (2018).
14. Mobbs, D. et al. Space, time, and fear: survival computations along defensive circuits. *Trends Cogn. Sci.* **24**, 228--241 (2020).
15. Niv, Y., Daw, N. D., Joel, D. & Dayan, P. Tonic dopamine: opportunity costs and the control of response vigor. *Psychopharmacology* **191**, 507--520 (2007).
16. Shadmehr, R., Huang, H. J. & Ahmed, A. A. A representation of effort in decision-making and motor control. *Curr. Biol.* **26**, 1929--1934 (2016).
17. Shenhav, A., Botvinick, M. M. & Cohen, J. D. The expected value of control: an integrative theory of anterior cingulate cortex function. *Neuron* **79**, 217--240 (2013).
18. Shenhav, A. et al. Toward a rational and mechanistic account of mental effort. *Annu. Rev. Neurosci.* **40**, 99--124 (2017).
19. Musslick, S. et al. Multitasking capability versus learning efficiency in neural network architectures. In *Proc. 39th Annu. Conf. Cogn. Sci. Soc.* (2017).
20. Lieder, F., Shenhav, A., Musslick, S. & Griffiths, T. L. Rational metareasoning and the plasticity of cognitive control. *PLoS Comput. Biol.* **14**, e1006043 (2018).
21. Wells, A. *Metacognitive Therapy for Anxiety and Depression* (Guilford, 2009).
22. Paulus, M. P. & Stein, M. B. Interoception in anxiety and depression. *Brain Struct. Funct.* **214**, 451--463 (2010).
23. Wise, T., Zbozinek, T. D., Michelini, G., Hagan, C. C. & Mobbs, D. Changes in risk perception and self-reported protective behaviour during the first week of the COVID-19 pandemic in the United States. *R. Soc. Open Sci.* **7**, 200742 (2020).
24. Gillan, C. M., Kosinski, M., Whelan, R., Phelps, E. A. & Daw, N. D. Characterizing a psychiatric symptom dimension related to deficits in goal-directed control. *eLife* **5**, e11305 (2016).
25. Wise, T. & Dolan, R. J. Associations between aversive learning processes and transdiagnostic psychiatric symptoms in a general population sample. *Nat. Commun.* **11**, 4462 (2020).
26. Huys, Q. J. M., Maia, T. V. & Frank, M. J. Computational psychiatry as a bridge from neuroscience to clinical applications. *Nat. Neurosci.* **19**, 404--413 (2016).
27. Todorov, E. & Jordan, M. I. Optimal feedback control as a theory of motor coordination. *Nat. Neurosci.* **5**, 1226--1235 (2002).
28. Shadmehr, R. & Krakauer, J. W. A computational neuroanatomy for motor control. *Exp. Brain Res.* **185**, 359--381 (2008).
29. Kahneman, D. & Tversky, A. Prospect theory: an analysis of decision under risk. *Econometrica* **47**, 263--291 (1979).
30. Tversky, A. & Kahneman, D. Advances in prospect theory: cumulative representation of uncertainty. *J. Risk Uncertain.* **5**, 297--323 (1992).
31. Mobbs, D. & Kim, J. J. Neuroethological studies of fear, anxiety, and risky decision-making in rodents and humans. *Curr. Opin. Behav. Sci.* **5**, 8--15 (2015).
32. Shenhav, A. The affective gradient hypothesis: an affect-centered account of motivated behavior. *Trends Cogn. Sci.* **28**, 1089--1104 (2024).
33. Wise, T. et al. Interactive cognitive maps support flexible behaviour under threat. *Cell Rep.* **42**, 113400 (2023).
34. Browning, M., Behrens, T. E., Jocham, G., O'Reilly, J. X. & Bishop, S. J. Anxious individuals have difficulty learning the causal statistics of aversive environments. *Nat. Neurosci.* **18**, 590--596 (2015).
35. Steiger, J. H. Tests for comparing elements of a correlation matrix. *Psychol. Bull.* **87**, 245--251 (1980).
36. Cisek, P. & Kalaska, J. F. Neural mechanisms for interacting with a world full of action choices. *Annu. Rev. Neurosci.* **33**, 269--298 (2010).
37. Mobbs, D. et al. Foraging under competition: the neural basis of input-matching in food-deprived participants. *J. Neurosci.* **33**, 9866--9872 (2013).
38. McNamara, J. M. & Houston, A. I. The common currency for behavioral decisions. *Am. Nat.* **127**, 358--378 (1986).
39. Fleming, S. M. & Dolan, R. J. The neural basis of metacognitive ability. *Philos. Trans. R. Soc. B* **367**, 1338--1349 (2012).

---

## Data and code availability

All analysis code is available at [repository URL]. Raw data will be made available on OSF upon publication. Preregistration for the confirmatory sample (N = 350) is available at [AsPredicted URL].

## Acknowledgments

This work was supported by [funding sources]. We thank participants recruited through Prolific for their time.

## Author contributions

N.O. designed the study, collected data, developed the computational model, conducted analyses, and wrote the manuscript. K.G. contributed to study design and data collection. T.W. contributed to computational modeling and analysis design. D.M. supervised the project, provided theoretical framing, and edited the manuscript.

## Competing interests

The authors declare no competing interests.

---

## Figure legends

**Figure 1. Task design and behavioral effects.** (a) Schematic of the effort-foraging task. Participants chose between a high-reward, high-effort cookie at varying distances and a low-reward, low-effort cookie near the safe zone, under threat probability *T* $\in$ {0.1, 0.5, 0.9}. (b) P(choose heavy) as a function of threat and distance. Both factors deter high-effort choice (threat: $\beta$ = -1.28; distance: $\beta$ = -0.65; interaction: $\beta$ = -0.18). (c) Excess vigor by threat level, conditioned on cookie type, showing within-choice threat-driven vigor increases (heavy *t* = 6.6; light *t* = 7.5).

**Figure 2. EVC model fit and parameter recovery.** (a) Observed (bars) and predicted (lines) P(heavy) across the 3 $\times$ 3 threat-distance design. (b) Per-subject predicted vs. observed P(heavy) (*r*^2^ = 0.951). (c) Observed and predicted excess vigor by threat and cookie type. (d) Per-subject predicted vs. observed mean vigor (*r* = 0.836). (e) Parameter recovery: log(*c*~e~) (*r* = 0.92). (f) Parameter recovery: log(*c*~d~) (*r* = 0.94).

**Figure 3. Affective calibration bridges computation and symptom severity.** (a) Calibration and discrepancy are orthogonal (*r* = 0.019, *p* = .75). (b) Calibration predicts performance: choice quality (*r* = 0.230) and survival (*r* = 0.179). (c) Discrepancy predicts anxiety symptoms across seven instruments ($\beta$ = 0.18--0.34; all 94% HDIs excluding zero). (d) Model parameters *c*~e~ and *c*~d~ show null symptom associations (~77% ROPE containment). (e) Within-subject: survival *S* predicts trial-level anxiety ($\beta$ = -0.557) and confidence ($\beta$ = 0.575).

---

## Supplementary Information

### Supplementary Table S1. Full model comparison

| Model | Per-subject params | BIC | $\Delta$BIC | Choice *r*^2^ | Vigor *r*^2^ |
|-------|-------------------|-----|------------|-------------|------------|
| **EVC 2+2** | *c*~e~, *c*~d~ | **32,133** | --- | **0.951** | **0.511** |
| M1: Effort only | *c*~e~ | 50,792 | +18,659 | 0.950 | 0.000 |
| M2: Threat only | *c*~d~ | 34,227 | +2,094 | 0.006 | 0.513 |
| M3: Separate | *c*~e~, *c*~d~ (separate) | 42,563 | +10,430 | 0.955 | 0.441 |
| M4: Pop *c*~e~ | *c*~d~ | 30,860 | -1,274 | 0.001 | 0.512 |
| M5: No $\gamma$ | *c*~e~, *c*~d~ | 34,204 | +2,071 | 0.955 | 0.425 |
| M6: Standard *u*^2^ | *c*~e~, *c*~d~ | 31,991 | -142 | 0.952 | 0.508 |

### Supplementary Table S2. Behavioral profiles from median split on *c*~e~ and *c*~d~

| Profile | *c*~e~ | *c*~d~ | Mean points | Description |
|---------|--------|--------|-------------|-------------|
| Cautious | High | High | 10.1 | Avoids effort, presses hard when engaged |
| Lazy | High | Low | 2.9 | Avoids effort, does not compensate with vigor |
| **Vigilant** | **Low** | **High** | **29.9** | **Pursues high-reward options and executes vigorously** |
| Bold | Low | Low | 8.4 | Pursues high-reward options but does not modulate vigor |

*Note.* Median split is descriptive; these profiles are not statistically validated as a typology. The Vigilant profile's earnings advantage demonstrates that combining willingness to forage with vigorous execution yields the best outcomes.

### Supplementary Figure S1. Distance gradient failure with population-level effort cost

With *c*~e~ at the population level, the model achieved lower BIC (30,860) but failed entirely to predict individual choice (*r*^2^ = 0.001). Predicted P(heavy) was nearly constant across subjects, demonstrating that individual effort cost sensitivity is essential for capturing the primary source of behavioral variation.

### Supplementary Figure S2. Within-trial vigor dynamics around predator encounter

Encounter-aligned vigor timeseries showing the phasic response to predator detection. (a) Population mean vigor (cookie-centered excess) aligned to encounter time, with pre-encounter declining slope (-0.021) and post-encounter increasing slope (+0.029; change *t* = 5.91, *p* < 10^-8^). Shaded region indicates $\pm$1 SE. (b) Distribution of individual encounter reactivity scores (*M* = -0.019, *SD* = 0.28), illustrating the large trait-like individual variation. (c) Cross-block stability of encounter reactivity (*r* = 0.78). (d) Encounter reactivity by threat level, showing no threat modulation (*F* = 0.04, *p* = .96). (e) Encounter reactivity vs. log(*c*~d~) (*r* = 0.47), demonstrating that tonic and phasic vigor co-vary. (f) Encounter reactivity vs. AMI apathy (*r* = -0.17, *p* = .004).

### Supplementary Note: Simpson's paradox in vigor

Unconditional (marginal) mean excess vigor shows a weak threat gradient (T = 0.1: -0.016; T = 0.5: +0.001; T = 0.9: +0.015; marginal *d* = 0.28). This is an artifact of choice reallocation: under high threat, participants shift from heavy to light cookies, and light cookies have lower demand thresholds, mechanically lowering apparent vigor. Conditioning on cookie type reveals the true effect (heavy *d* = 0.42; light *d* = 0.49). The joint EVC model naturally resolves this confound because it conditions on the chosen option.

### Supplementary Note: MCMC Validation

To validate the SVI mean-field approximation used for primary model fitting, we refitted the full EVC 2+2 model using Markov chain Monte Carlo (NUTS sampler, 4 chains, 200 warmup + 200 sampling iterations, target acceptance probability = 0.8, max tree depth = 10). The MCMC fit produced zero divergent transitions and converged well for population parameters (all R-hat < 1.05 except $\sigma$~cd~ = 1.12, which would improve with longer chains).

Per-subject parameter estimates from MCMC correlated near-perfectly with SVI estimates: log(*c*~e~) *r* = 0.999, log(*c*~d~) *r* = 0.999. Mean absolute differences in log space were 0.04 for *c*~e~ and 0.19 for *c*~d~. Population parameter estimates were also consistent: $\gamma$ = 0.209 (SVI: 0.209), $\tau$ = 0.477 (SVI: 0.476), $\varepsilon$ = 0.136 (SVI: 0.098). The log(*c*~e~) $\times$ log(*c*~d~) correlation was *r* = -0.142 (SVI: -0.14), confirming approximate independence of the two subject-level parameters.

Subject-level convergence was adequate for *c*~e~ (all R-hat < 1.03) but marginal for *c*~d~ (median R-hat = 1.04, 134/293 subjects > 1.05), reflecting the challenge of estimating 586 subject-level parameters from 200 post-warmup samples. Longer chains (500+ samples) on GPU hardware would improve *c*~d~ convergence. Critically, the near-perfect SVI-MCMC correlation confirms that the SVI approximation provides reliable point estimates of posterior means, validating its use for the primary analyses despite its limitations on posterior uncertainty quantification.
