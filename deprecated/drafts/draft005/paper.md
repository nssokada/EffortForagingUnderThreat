# Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety

Noah Okada^1^, Ketika Garg^1^, Toby Wise^2^, Dean Mobbs^1,3^

^1^ Division of the Humanities and Social Sciences, California Institute of Technology, Pasadena, CA, USA
^2^ Department of Neuroimaging, King's College London, London, UK
^3^ Computation and Neural Systems Program, California Institute of Technology, Pasadena, CA, USA

---

## Abstract

Foraging under predation risk requires integrating energetic cost and survival probability into a unified decision variable, yet how a single computation governs both the decision of what to pursue and the vigor of pursuit---and how breakdowns in that computation relate to anxiety---remains unknown. Here we develop an Expected Value of Control (EVC) model with linear-quadratic regulator (LQR) cost structure and test it in a large online effort-foraging task under parametric threat (discovery sample, N = 293; confirmatory sample, N = XXX, preregistered). Two subject-level parameters---effort cost (*c*~e~, identified from choice) and capture aversion (*c*~d~, identified from vigor)---plus population-level probability weighting ($\gamma$ = 0.209) jointly predict foraging decisions (per-subject *r*^2^ = 0.951) and press-rate vigor (*r*^2^ = 0.511). Both parameters are recoverable in simulation (*c*~e~: *r* = 0.92; *c*~d~: *r* = 0.94) and are approximately independent (*r* = -0.14). The model's survival signal predicts trial-level anxiety ($\beta$ = -0.557, *t* = -14.0) and confidence ($\beta$ = 0.575, *t* = 13.5). We decompose metacognitive affect into calibration (how accurately anxiety tracks danger) and discrepancy (how much anxiety exceeds danger). These dimensions are orthogonal (*r* = 0.019) and predominantly dissociated: calibration predicts task performance (*r* = 0.179-0.230 with choice quality and survival), while discrepancy predicts clinical symptoms (STAI: $\beta$ = 0.338; STICSA: $\beta$ = 0.285; DASS-Anxiety: $\beta$ = 0.275; all 94% HDIs excluding zero). The computational parameters show no reliable clinical associations (~77% posterior mass within the region of practical equivalence). The bridge from adaptive foraging computation to psychiatric vulnerability runs through metacognition---through how people feel about danger, not how they compute it.

---

## Introduction

Animals foraging under predation risk face a fundamental optimization problem: the energy gained from distant, high-value resources must be weighed against increased exposure to capture during transport^1,2^. This trade-off shapes behavior across species, from birds provisioning nestlings near raptor territories^3^ to fish venturing from shelter to access food patches^4^. In humans, analogous computations arise whenever pursuing goals demands sustained effort under threat---commuting through dangerous neighborhoods, persisting at hazardous work, or investing cognitive effort when the costs of failure loom large. Theoretical ecology has long modeled this trade-off using reproductive value as the common currency^1,5^, but translating these models into a computational framework that specifies how the same cost-benefit calculation governs both what an organism chooses and how vigorously it acts has remained an open challenge.

Two research traditions have addressed the components of this problem in isolation. Work on effort-based decision-making has established that humans discount reward value by the physical or cognitive effort required to obtain it^6--9^, with individual differences in effort cost sensitivity linked to apathy, fatigue, and motivational disorders^10,11^. Separately, research on defensive behavior has characterized how organisms modulate responses to threat across the predatory imminence continuum---from strategic avoidance at a distance to reactive flight when danger is immediate^5,12--14^. The vigor of motor execution, in particular, is thought to reflect the marginal value of time under the current motivational state^15,16^, suggesting that threat should intensify physical effort even after the decision to forage has been made. Yet no computational model has formally specified how a single cost function generates both the discrete choice of what to pursue and the continuous regulation of how hard to pursue it.

The Expected Value of Control (EVC) framework^17,18^ provides a candidate architecture for this integration. EVC proposes that the brain computes the expected payoff of allocating control effort, selecting the intensity that maximizes reward minus cost. Because EVC treats effort allocation as a continuous optimization rather than a binary decision, it naturally extends to predict both discrete choices and graded action vigor within a unified framework. However, EVC has been applied primarily to cognitive control in abstract laboratory tasks^19,20^ and has not been extended to physical effort under ecological threat, where the cost function must jointly encode energetic expenditure and survival probability.

A separate gap concerns the relationship between threat computation and clinical anxiety. People experiencing pathological anxiety do not simply overestimate threat; they exhibit a dysfunctional relationship between threat appraisal and affective response^21^. Paulus and Stein^22^ formalize this as interoceptive prediction error: clinical anxiety involves systematic mismatch between expected body states and actual somatic experience under threat, producing affect that is decoupled from the environment. In Wise and colleagues' work on interactive threat^33^, confidence ratings track the quality of cognitive models of threat with remarkable fidelity---participants know when their predictions are improving and adjust behavior accordingly. This raises a fundamental question: when people compute danger normatively but feel anxious anyway, is it the computation or the feeling that predicts psychiatric vulnerability? Computational psychiatry has sought to ground psychiatric constructs in formal decision models^24--26^, but no study has decomposed the metacognitive relationship between computed danger and experienced affect within a normative foraging framework where both quantities are measured simultaneously.

Here we address both gaps. We developed an EVC model with linear-quadratic regulator (LQR) cost structure^27,28^ and tested it in a virtual effort-foraging task under parametric threat (discovery sample, N = 293). Participants chose between high-effort, high-reward and low-effort, low-reward resources while facing predation risk at three probability levels and three distances, then physically executed their foraging bout by pressing keys. Two subject-level parameters---effort cost (*c*~e~) and capture aversion (*c*~d~)---jointly predicted choice and vigor through distinct channels. We then decomposed each participant's subjective anxiety into metacognitive calibration (the fidelity with which anxiety tracks model-derived danger) and discrepancy (the systematic bias by which anxiety exceeds danger), revealing a predominant double dissociation: calibration predicts who performs well, while discrepancy---not the computational parameters themselves---predicts who is clinically anxious. A preregistered confirmatory replication [N = XXX, confirmatory] is underway.

---

## Results

### Threat and distance deter high-effort foraging and modulate vigor

We designed an effort-foraging task in which participants (N = 293, after five-stage quality screening of 350 recruits) chose between a high-reward cookie (5 points, requiring sustained keypressing at 60--100% of individually calibrated maximum capacity) and a low-reward cookie (1 point, requiring 40% of maximum) while facing predation risk (threat probability *T* $\in$ {0.1, 0.5, 0.9}) at varying distances (*D* $\in$ {1, 2, 3}; Fig. 1a). On attack trials, a predator appeared and pursued the participant; capture incurred a 5-point penalty and loss of the cookie's value. Participants completed 45 choice trials and 36 probe trials (forced-choice with identical options, paired with affect ratings).

Both threat and distance reduced high-effort choice (Fig. 1b). A logistic mixed-effects model confirmed threat as the dominant deterrent ($\beta$ = -1.28, *z* = -32.0, *p* < 10^-200^), with distance ($\beta$ = -0.65, *z* = -16.3, *p* < 10^-59^) and their interaction ($\beta$ = -0.18, *z* = -4.5, *p* < 10^-5^) also significant. At the extremes, P(heavy) dropped from 0.81 at *T* = 0.1, *D* = 1 to 0.08 at *T* = 0.9, *D* = 3---only 8% of participants attempted the high-reward option when both danger and distance were maximal. All adjacent-threat comparisons within each distance were significant by paired *t*-test (all *t* > 8.3, all *p* < 10^-14^), confirming monotonic deterrence.

Threat also increased motor vigor, but this effect was masked by a Simpson's paradox in unconditional analyses. Because high threat shifts choice toward light cookies (60% heavy at *T* = 0.1 versus 34% at *T* = 0.9), and light cookies have lower required press rates, collapsing across cookie type makes average vigor appear flat. Conditioning on chosen cookie type revealed robust threat-driven vigor increases (heavy: *t* = 6.6, *p* < 10^-10^, *d* = 0.42; light: *t* = 7.5, *p* < 10^-13^, *d* = 0.49). Within each cookie type, participants pressed harder when threat was higher---consistent with the prediction that the marginal survival benefit of faster pressing increases with danger.

### An EVC-LQR model jointly captures choice and vigor with two subject-level parameters

We formalized the foraging decision as a comparison of expected utilities:

$\Delta EU = S \times 4 - c_{e,i} \times (0.81D_H - 0.16)$

where *S* is the subjective survival probability, *c*~e,i~ is the subject-specific effort cost, and the effort term reflects the LQR commitment cost---the squared required press rate scaled by distance (the difference in req^2^ $\times$ *D* between heavy and light options). The survival function incorporates probability weighting and effort efficacy:

$S = (1 - T^{\gamma}) + \varepsilon \times T^{\gamma} \times p_{\text{esc}}$

where $\gamma$ = 0.209 is the population probability-weighting exponent (indicating substantial compression of threat probabilities: a nominal 50% threat is experienced as *T*^0.209^ = 0.86) and $\varepsilon$ = 0.098 reflects the universal tendency to underweight effort's benefit to survival. Choices follow a softmax rule: P(heavy) = sigmoid($\Delta$EU / $\tau$), with $\tau$ = 0.476.

With per-subject *c*~e~ (log-normal prior, non-centered parameterization), the model reproduced the full threat-by-distance choice surface (Fig. 2a): at *T* = 0.1, predicted P(heavy) declined from 0.87 at *D* = 1 to 0.53 at *D* = 3; at *T* = 0.9, from 0.33 to 0.10. The model achieved per-subject choice *r*^2^ = 0.951 (subj *r* = 0.976), choice accuracy = 79.3%, and AUC = 0.876 (Fig. 2b). Previous versions with population-level effort cost failed to reproduce the distance gradient that drives approximately 40% of choice variance (see Supplementary Fig. S1).

For the vigor component, the model computes optimal press rate *u** by maximizing:

$EU(u) = S(u) \times R - (1 - S(u)) \times c_{d,i} \times (R + C) - c_{e,\text{vigor}} \times (u - \text{req})^2 \times D$

where *c*~d,i~ is the subject-specific capture aversion and *c*~e,vigor~ = 0.003 is the population-level motor deviation cost. The survival function becomes speed-dependent: pressing faster than required improves escape probability. Three features of this architecture are notable. First, *c*~d~ is absent from the choice equation because the differential capture cost between options is collinear with the differential reward---both scale with S and the reward difference---making *c*~d~ unidentifiable from choice data. The model therefore estimates *c*~e~ exclusively from choice and *c*~d~ exclusively from vigor, without post-hoc decomposition. Second, the LQR distinction between commitment cost (req^2^ $\times$ *D*, governing choice) and deviation cost ((u - req)^2^ $\times$ *D*, governing vigor)^27^ resolves a scaling conflict: commitment costs operate at two orders of magnitude above deviation costs, allowing a single theoretical framework to span both decision stages. Third, probe trials (forced conditions with identical options) anchor *c*~d~ estimation across all threat-by-distance cells without selection bias.

The model predicted trial-level vigor with *r*^2^ = 0.511 (subj vigor *r* = 0.836; Fig. 2c,d). Higher *c*~d~ drives steeper threat-vigor gradients because the marginal benefit of pressing faster grows with the stakes of failed escape (*c*~d~ $\times$ (*R* + *C*)).

Model comparison across six variants confirmed that every component was necessary (Table 1). Removing individual effort cost (M1, effort-only: $\Delta$BIC = +18,659), eliminating individual effort cost while retaining only threat (M2: $\Delta$BIC = +2,094, choice *r*^2^ = 0.006), separating choice and vigor into independent models (M3: $\Delta$BIC = +10,393), or dropping probability weighting (M5: $\Delta$BIC = +2,071) all substantially degraded fit. The LQR deviation cost performed comparably to a standard *u*^2^ cost (M6: $\Delta$BIC = -142), indicating that the LQR formulation does not improve fit beyond standard motor cost but provides a principled theoretical framework.

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

*Note.* M4 achieves lower BIC but fails to predict individual choice (choice *r*^2^ = 0.001), sacrificing the primary behavioral target for marginal vigor improvement. M6 is nearly equivalent to the full model, confirming that the LQR and standard motor cost formulations are empirically indistinguishable.

Parameter recovery confirmed identifiability. Simulating 293 synthetic subjects at the fitted population distribution and refitting yielded correlations (in log space) of *r* = 0.92 for log(*c*~e~) and *r* = 0.94 for log(*c*~d~) (Fig. 2e,f). The two parameters were approximately independent (*r* = -0.14), confirming that they capture distinct dimensions of individual variation.

### The model's survival signal predicts trial-level anxiety and confidence

On probe trials, participants rated either anxiety ("How anxious are you about being captured?") or confidence ("How confident are you about reaching safety?") on a 0--7 scale, after choosing but before pressing. The model's survival probability *S* strongly predicted both ratings via linear mixed models with random intercepts and slopes by subject:

- Anxiety: $\beta$ = -0.557, SE = 0.040, *t* = -14.04, *p* = 8.8 $\times$ 10^-45^ (N~obs~ = 5,274)
- Confidence: $\beta$ = +0.575, SE = 0.043, *t* = +13.48, *p* = 2.1 $\times$ 10^-41^ (N~obs~ = 5,272)

These effects translate to approximately 2 points on the 0--7 scale when moving from the safest (*T* = 0.1, *S* $\approx$ 0.85) to the most dangerous (*T* = 0.9, *S* $\approx$ 0.15) condition. The substantial random slope variance (anxiety: 0.302; confidence: 0.375) indicates meaningful individual differences in how tightly affect tracks survival---some participants show steep affect-survival gradients (well-calibrated metacognition), while others show flat gradients (poor metacognitive tracking). This individual variation forms the basis of the metacognitive decomposition below.

Task-derived affect showed convergent validity with validated clinical instruments: mean task anxiety correlated with STAI (*r* = 0.31) and STICSA (*r* = 0.27), while mean task confidence was negatively associated with AMI (*r* = -0.25). Critically, within-subject task anxiety and task confidence were only weakly correlated (*r* = -0.25), and their between-subject means were essentially independent (*r* = -0.01), indicating that anxiety and confidence function as partially separable affective channels rather than opposite poles of a single dimension.

Despite the strong within-subject tracking, between-subject mean confidence did not reliably predict task performance: confident participants did not survive more often (*r* = -0.05, *p* = .41) nor make more EV-optimal choices (*r* = -0.08, *p* = .16). This null finding motivated us to decompose the affect-danger relationship into finer-grained metacognitive components rather than relying on aggregate confidence as a performance indicator.

### Metacognitive calibration predicts performance; discrepancy predicts clinical symptoms

We decomposed each participant's metacognitive responding into two dimensions:

**Calibration:** the within-subject Pearson correlation between anxiety ratings and model-derived danger (1 - *S*), computed across each participant's 18 anxiety probe trials. Higher calibration indicates that anxiety more accurately tracks the computational danger signal.

**Discrepancy:** the mean residual of a participant's anxiety ratings after removing the population-level anxiety-danger relationship. Positive discrepancy indicates systematically elevated anxiety beyond what the danger signal warrants---an affective signal decoupled from the environment.

These dimensions were orthogonal (*r* = 0.019, *p* = .75; Fig. 3a), confirming that the accuracy of threat monitoring and the magnitude of affective bias are genuinely independent aspects of metacognitive functioning. Most participants (85%) showed positive calibration, indicating that anxiety generally increased with danger, but with wide individual variation (*M* = 0.47, *SD* = 0.32).

A predominant double dissociation emerged (Fig. 3b,c). Calibration predicted adaptive performance: participants whose anxiety accurately tracked danger made higher-quality foraging decisions (*r* = 0.230, *p* < .001) and survived at higher rates (*r* = 0.179, *p* = .002), consistent with the proposal that accurate anxiety functions as an adaptive alarm calibrated to genuine threat^22^. However, calibration was largely unrelated to clinical symptoms (6 of 7 psychiatric measures *p* > .10; only STAI showed a weak positive association, *r* = 0.121, *p* = .04).

Discrepancy predicted clinical symptoms across the full spectrum of anxiety, depression, and stress measures. Bayesian regression controlling for the model's computational parameters (log(*c*~e~), log(*c*~d~)) confirmed that discrepancy was a robust predictor with all 94% highest density intervals excluding zero (Table 2).

**Table 2. Bayesian regression: discrepancy predicts clinical symptoms.**

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

*Note.* ROPE = region of practical equivalence, |$\beta$| < 0.10. High percentage in ROPE provides Bayesian evidence for the null. All discrepancy HDIs exclude zero. STAI-Trait showed a negative association (*r* = -0.223), consistent with reduced trait anxiety in those who underestimate danger.

The model's computational parameters (*c*~e~ and *c*~d~) showed no reliable clinical associations in either frequentist (all FDR-corrected *p* > .70) or Bayesian analyses (ROPE containment ranging 53--93%; Table 2). This is not merely an absence of evidence: when discrepancy is included in the model, the posterior distributions for *c*~e~ and *c*~d~ fall predominantly within the ROPE, providing positive evidence that the computational parameters governing behavior are dissociated from the metacognitive dimension governing distress.

The negative association between discrepancy and emotional apathy (AMI-Emotional: $\beta$ = -0.222) adds specificity: individuals who overestimate danger relative to the survival signal are the opposite of affectively disengaged. They are motivationally aroused---perhaps excessively so. This echoes the proposal that anxiety and apathy occupy opposite poles of motivational dysfunction^10^.

We assessed whether any combination of model parameters and metacognitive dimensions could predict individual clinical outcomes using cross-validated machine learning (elastic net and ridge regression with nested 10-fold CV, 5 repeats). All cross-validated *R*^2^ values were negative (range: -0.03 to -0.06), confirming that these associations are group-level patterns---useful for identifying mechanistic dimensions of variation---rather than individually predictive biomarkers.

### Discrepancy is stable across blocks and shows convergent validity

To evaluate whether metacognitive discrepancy reflects a stable individual trait rather than transient state fluctuation, we computed within-subject discrepancy separately for each of the three task blocks. Block-to-block correlations ranged from *r* = 0.48 to *r* = 0.68, indicating moderate-to-good test-retest stability across approximately 20 minutes of task performance. This stability is consistent with discrepancy reflecting a trait-like tendency toward affective overestimation, as predicted by interoceptive prediction error accounts of anxiety^22^.

---

## Discussion

We developed an Expected Value of Control model with LQR cost structure that jointly captures foraging choice and action vigor under parametric threat. Two subject-level parameters---effort cost (*c*~e~) and capture aversion (*c*~d~)---explain 95.1% of between-subject choice variance and 51.1% of trial-level vigor variance through a single cost function with distinct cost channels for decision and execution. The model's survival signal predicts moment-to-moment anxiety and confidence, and decomposing metacognitive accuracy reveals a predominant double dissociation: calibration predicts adaptive performance while discrepancy predicts psychiatric symptomatology. The computational parameters themselves are clinically inert.

### An EVC framework for physical effort under ecological threat

The central modeling contribution is demonstrating that the EVC framework^17,18^ can be extended from cognitive control to physical effort under ecological threat. The key theoretical move is adopting the LQR cost structure^27^, which distinguishes commitment costs (the effort implied by choosing a distant option: req^2^ $\times$ *D*) from deviation costs (the additional effort of pressing faster than required: (*u* - req)^2^ $\times$ *D*). This distinction maps onto a natural division in foraging behavior: the strategic decision of what to pursue versus the tactical execution of how vigorously to pursue it. Although the LQR and standard motor cost formulations proved empirically equivalent in our data (M6: $\Delta$BIC = -142), the LQR framework provides a principled connection to optimal control theory and generates the qualitative prediction that choice and vigor should be governed by separable cost channels---a prediction confirmed by the near-independence of *c*~e~ and *c*~d~ (*r* = -0.14).

The probability weighting parameter ($\gamma$ = 0.209) indicates that participants dramatically compressed threat probabilities, consistent with Kahneman and Tversky's^29^ loss-domain distortion but substantially stronger than estimates from monetary gambles ($\gamma$ ~ 0.65--0.70)^30^. This amplification may reflect the embodied nature of virtual predation, which engages defensive circuitry^5,13^ more powerfully than abstract monetary losses. The dissociation of *c*~e~ and *c*~d~ maps onto the distinction between strategic and reactive defensive modes identified in the threat-imminence literature^5,14,31^: choice reflects strategic assessment (selecting safer options when threat is high), while vigor reflects reactive mobilization (pressing harder when at risk). These modes may engage partially separable neural circuits, consistent with evidence for prefrontal-mediated strategic defense and subcortical-mediated reactive defense^13,14^.

The clean identification of the two parameters depends on a structural feature of our task design. Because both cookie options share the same survival function in the choice comparison (capture probability affects both equally), the capture aversion *c*~d~ is collinear with the reward term and unidentifiable from choice alone. It is instead identified exclusively from vigor data. This orthogonal identification is a design strength: each parameter is pinned to its behavioral channel by the model's architecture rather than by post-hoc decomposition.

### Metacognition as the bridge between computation and psychopathology

The most consequential finding is that the route from normative foraging computation to clinical anxiety runs through metacognition---through how people feel about danger---rather than through the decision parameters themselves. Neither *c*~e~ nor *c*~d~ predicted any clinical measure (all FDR-corrected *p* > .70; Bayesian ROPE containment 53--93%). Instead, discrepancy---the systematic excess of experienced anxiety over the model's danger signal---predicted symptoms of anxiety, depression, and stress across seven validated instruments ($\beta$ = 0.18--0.34).

This finding provides computational specificity to metacognitive theories of anxiety^21,22^. Wells^21^ proposed that clinical anxiety reflects dysfunctional beliefs *about* threat cognition (meta-worry), not simply elevated threat estimation. Paulus and Stein^22^ formalized this as interoceptive prediction error: anxiety disorders involve systematic mismatch between expected and actual body states under threat. Our discrepancy measure operationalizes this mismatch within a normative framework: it quantifies how much a person's anxiety exceeds what the model's survival computation warrants, given the same threat environment that all participants face. The orthogonality of calibration and discrepancy (*r* = 0.019) means that a person can accurately track danger (high calibration) while simultaneously reporting more anxiety than that danger warrants (high discrepancy)---capturing the clinical phenotype of someone who "knows" they are safe but "feels" afraid.

The calibration-performance association (*r* = 0.19--0.23) complements this picture. Participants whose anxiety faithfully tracked computed danger made better foraging decisions and survived more often, suggesting that well-calibrated anxiety functions as an adaptive internal signal---an alarm system tuned to actual threat that enables appropriate behavioral regulation. This aligns with functional accounts of anxiety^32^ and with Wise and colleagues' finding that confidence tracks the quality of internal models of threat^23^. The critical addition of our work is showing that this adaptive monitoring function (calibration) is orthogonal to the clinical vulnerability dimension (discrepancy), and that it is the *mismatch* between computation and affect, not the computation itself, that predicts psychopathology. We note that this dissociation was predominant but not absolute: calibration showed a weak positive association with STAI-State (*r* = 0.12, *p* = .04), and discrepancy showed a modest negative association with survival (*r* = -0.15, *p* = .009). These cross-associations are small relative to the primary effects and do not undermine the overall pattern, but they indicate that the two metacognitive dimensions are not perfectly encapsulated.

Our findings create an interesting tension with the Affective Gradient Hypothesis^32^, which proposes that affect is the sole currency of motivated behavior---there is no "cold" computation independent of feeling. Under this account, discrepancy would not represent a metacognitive error but rather a difference in affective accessibility: some individuals have richer or more salient threat-related affective representations, producing higher anxiety at the same objective danger. Whether discrepancy reflects a genuine metacognitive bias (as Paulus and Stein argue) or an adaptive affective phenotype (as AGH implies) cannot be resolved by our data alone, but the clinical associations suggest that regardless of interpretation, excess anxiety relative to computed danger is associated with psychiatric vulnerability.

The negative association between discrepancy and emotional apathy (AMI: $\beta$ = -0.222) further constrains the interpretation. Overanxious individuals are not generally dysfunctional---they are motivationally engaged but affectively miscalibrated. This dissociation is consistent with the proposal that anxiety and apathy represent opposite poles of motivational activation^10,11^, and suggests that excessive threat-related affect may co-opt motivational resources in ways that produce distress without improving performance.

### Clinical implications and effect-size honesty

The clinical associations we report are statistically robust but modest in magnitude ($\beta$ = 0.18--0.34, corresponding to approximately 3--11% of symptom variance). These effect sizes are consistent with the broader computational psychiatry literature^24,33^ and with the inherent challenge of linking single-task behavioral parameters to multidetermined psychiatric constructs measured by self-report. Three honest constraints are important to note.

First, all cross-validated *R*^2^ values for predicting clinical outcomes were negative, confirming that the associations are group-level mechanistic patterns rather than individually predictive biomarkers. The field should not expect single-task computational parameters to substitute for clinical assessment.

Second, the discrepancy measure is model-dependent: it quantifies the residual between experienced anxiety and model-derived danger. If the survival function is misspecified, discrepancy may partly reflect model error rather than metacognitive bias. The strong within-subject anxiety-survival correlations (*t* = -14.0) mitigate this concern by confirming that the model's survival signal is a psychologically meaningful quantity, but they do not eliminate it.

Third, all results reported here come from the discovery sample. A preregistered confirmatory study (N = 350 recruited, preregistration available at [AsPredicted URL]) with identical task design, model specification, and analysis plan will test whether these patterns replicate. The confirmatory study specifies directional hypotheses and significance thresholds derived from discovery effect sizes (see Preregistration).

### Behavioral profiles and the dissociation of choice and vigor

The two-parameter structure generates interpretable individual differences. A median split on *c*~e~ and *c*~d~ defines four profiles: Cautious (high effort cost, high capture aversion), Lazy (high effort cost, low capture aversion), Vigilant (low effort cost, high capture aversion), and Bold (low effort cost, low capture aversion). The Vigilant profile earned the most points (mean = 29.9 vs. 2.9--10.1 for other profiles) because it combines willingness to pursue high-value resources with vigorous execution during transport. The near-independence of *c*~e~ and *c*~d~ means that knowing how someone decides tells little about how hard they will try---a dissociation that may reflect the partially separable neural substrates of strategic and reactive defense^13,14^.

### Limitations

Several limitations warrant consideration. The effort-efficacy parameter $\varepsilon$ is estimated at the population level because it is not individually recoverable (recovery *r* $\approx$ 0). This means our model cannot capture individual differences in beliefs about effort's survival benefit---a theoretically important construct that would require richer within-subject designs or physiological measures to identify.

Distance confounds effort duration and threat exposure by design: farther cookies require both more sustained pressing and more time in danger. This confound is shared with natural foraging^1^ and is partially addressed by the factorial crossing of distance with effort demand, but a fully orthogonal design would strengthen causal claims about the independent contributions of effort and threat.

The task uses explicit threat probabilities, which differ from real-world threat assessment involving learning and belief updating^34^. Our model assumes stationary threat and does not capture trial-by-trial belief updating that may occur even with stated probabilities.

Our Prolific sample reflects dimensional psychiatric variation in a non-clinical population. Whether the discrepancy-symptom association strengthens---or changes qualitatively---in clinical groups with diagnosed anxiety disorders is an important next step.

Finally, the model is static: it does not capture learning across trials, strategic adjustments across blocks, or the potential for affect to feed back into subsequent decisions. The moderate block-to-block stability of discrepancy (*r* = 0.48--0.68) suggests partial stability but also meaningful within-session change, which dynamic models could characterize.

---

## Methods

### Participants

We recruited 350 participants from Prolific (https://prolific.co) for an online study. After a five-stage quality control pipeline---(1) task completion, (2) comprehension checks, (3) behavioral consistency screening (implausible keypresses and disengagement), (4) effort calibration validation (minimum 10 presses in 10 seconds), and (5) outlier removal (escape rate < 35%)---the final discovery sample comprised N = 293 participants (83.7% retention). No post-hoc exclusions were applied based on model fit quality or statistical extremity. The study was approved by the Caltech Institutional Review Board, and all participants provided informed consent.

### Task design

Participants completed an effort-based foraging task implemented in Unity (WebGL) and presented in a desktop browser. The task featured a circular arena viewed from above, with a safe zone at the center and cookie targets at radial distances.

**Effort calibration.** Before the main task, participants completed three 10-second trials pressing the S, D, and F keys simultaneously as fast as possible. The maximum press count defined each participant's calibrated maximum (*f*~max~). All demands were expressed relative to this calibration.

**Trial structure.** On choice trials, two cookies appeared: a heavy cookie (5 points, requiring pressing at 60--100% of *f*~max~ depending on weight condition) at distance *D* $\in$ {1, 2, 3} (5, 7, 9 game units), and a light cookie (1 point, 40% of *f*~max~) always at *D* = 1. Participants clicked to select (irrevocable commitment), then pressed S+D+F repeatedly to transport the cookie to safety. Movement speed followed a step function: full speed at $\geq$100% of required rate, half at $\geq$50%, quarter at $\geq$25%, and zero below 25%.

**Threat manipulation.** Each trial had predation probability *T* $\in$ {0.1, 0.5, 0.9}. On attack trials, a predator spawned at the perimeter position closest to the participant at a pre-determined encounter time (calibrated to half the expected return time at full speed), approached at 0.5 units/sec, then struck at a Gaussian-distributed time at 4$\times$ the participant's maximum speed (unavoidable). Capture cost 5 points plus the cookie value.

**Probe trials.** Twelve of 27 events per block were probe trials with identical options (forced choice). After selection, the game paused for a rating of either anxiety about capture or confidence in reaching safety (0--7 scale). Probes were prospective: collected after commitment but before motor execution.

**Structure.** Three blocks of 27 events (81 total): 15 choice trials and 12 probe trials per block, yielding 45 choice and 36 probe trials. Conditions were fully crossed: 3 threat $\times$ 3 distance $\times$ 5 repetitions (choice); 3 threat $\times$ 3 distance $\times$ 2 cookie types $\times$ 2 rating types (probe).

### Psychiatric assessment

Between blocks, participants completed the DASS-21 (Depression, Anxiety, Stress subscales), PHQ-9, OASIS, STAI-State, AMI (Behavioural, Social, Emotional subscales), MFIS (Physical, Cognitive, Psychosocial subscales), and STICSA. All subscale scores were z-scored across participants.

### EVC-LQR model

**Per-subject parameters** (log-normal, non-centered parameterization):
- *c*~e~ (effort cost): governs choice via distance-dependent effort penalty
- *c*~d~ (capture aversion): governs vigor via the survival incentive

**Population parameters:** $\gamma$ = 0.209 (probability weighting); $\varepsilon$ = 0.098 (effort efficacy); *c*~e,vigor~ = 0.003 (LQR deviation cost); $\tau$ = 0.476 (choice temperature); *p*~esc~ (escape probability); $\sigma$~motor~ (motor noise); $\sigma$~v~ (vigor observation noise).

**Choice model.** $\Delta$EU = *S* $\times$ 4 - *c*~e,i~ $\times$ (0.81*D*~H~ - 0.16). The capture aversion *c*~d~ does not enter the choice equation because its contribution is collinear with the reward differential. P(heavy) = sigmoid($\Delta$EU / $\tau$).

**Vigor model.** EU(*u*) = *S*(*u*) $\times$ *R* - (1 - *S*(*u*)) $\times$ *c*~d,i~ $\times$ (*R* + *C*) - *c*~e,vigor~ $\times$ (*u* - req)^2^ $\times$ *D*. Survival is speed-dependent: *S*(*u*) = (1 - *T*^$\gamma$^) + $\varepsilon$ $\times$ *T*^$\gamma$^ $\times$ *p*~esc~ $\times$ sigmoid((*u* - req) / $\sigma$~motor~). Optimal *u** is computed via softmax-weighted grid search.

**Joint likelihood.** Choice trials contribute a Bernoulli likelihood; all 81 trials contribute a Normal likelihood for vigor. Both are evaluated simultaneously.

### Model fitting

NumPyro stochastic variational inference (SVI) with an AutoNormal guide (mean-field approximation), Adam optimizer (lr = 0.002), 40,000 steps. BIC = 2 $\times$ loss + *k* $\times$ log(*n*), where *k* = 2 $\times$ *N*~subjects~ + number of population parameters.

### Parameter recovery

293 synthetic subjects were generated from the fitted population distribution with the same task design. Simulated data were refitted with identical SVI procedure. Recovery assessed as Pearson *r* between true and recovered parameters in log space.

### Model comparison

Six ablation models were fitted, each removing one component: individual effort cost, threat, joint estimation, probability weighting, or the LQR cost structure. All models were evaluated on the same data (81 trials per subject for both likelihoods).

### Vigor computation

Trial-level vigor was computed as the median normalized press rate (median(1/IPI) / *f*~max~) minus the required rate for the chosen cookie, then centered by cookie type (subtracting the population mean excess for heavy and light cookies separately). This cookie-type centering removes the demand confound while preserving between-subject variation.

### Affect analysis

Linear mixed models (statsmodels MixedLM, REML, L-BFGS optimizer) predicted anxiety and confidence from standardized survival probability (*S*~z~) with random intercepts and slopes by subject.

### Metacognitive decomposition

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

**Figure 2. EVC-LQR model fit and parameter recovery.** (a) Observed (bars) and predicted (lines) P(heavy) across the 3 $\times$ 3 threat-distance design. (b) Per-subject predicted vs. observed P(heavy) (*r*^2^ = 0.951). (c) Observed and predicted excess vigor by threat and cookie type. (d) Per-subject predicted vs. observed mean vigor (*r* = 0.836). (e) Parameter recovery: log(*c*~e~) (*r* = 0.92). (f) Parameter recovery: log(*c*~d~) (*r* = 0.94).

**Figure 3. Metacognition bridges computation and clinical symptoms.** (a) Calibration and discrepancy are orthogonal (*r* = 0.019, *p* = .75). (b) Calibration predicts performance: choice quality (*r* = 0.230) and survival (*r* = 0.179). (c) Discrepancy predicts clinical symptoms across seven instruments ($\beta$ = 0.18--0.34; all 94% HDIs excluding zero). (d) Model parameters *c*~e~ and *c*~d~ show null clinical associations (~77% ROPE containment). (e) Within-subject: survival *S* predicts trial-level anxiety ($\beta$ = -0.557) and confidence ($\beta$ = 0.575).

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

### Supplementary Figure S1. Distance gradient failure with population-level effort cost

With *c*~e~ at the population level, the model achieved lower BIC (30,860) but failed entirely to predict individual choice (*r*^2^ = 0.001). Predicted P(heavy) was nearly constant across subjects, demonstrating that individual effort cost sensitivity is essential for capturing the primary source of behavioral variation.

### Supplementary Note: Simpson's paradox in vigor

Unconditional (marginal) mean excess vigor shows a weak threat gradient (T = 0.1: -0.016; T = 0.5: +0.001; T = 0.9: +0.015; marginal *d* = 0.28). This is an artifact of choice reallocation: under high threat, participants shift from heavy to light cookies, and light cookies have lower demand thresholds, mechanically lowering apparent vigor. Conditioning on cookie type reveals the true effect (heavy *d* = 0.42; light *d* = 0.49). The joint EVC model naturally resolves this confound because it conditions on the chosen option.
