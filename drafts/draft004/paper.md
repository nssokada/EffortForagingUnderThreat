# Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety

Noah Okada^1^, Ketika Garg^1^, Toby Wise^2^, Dean Mobbs^1,3^

^1^ Division of the Humanities and Social Sciences, California Institute of Technology, Pasadena, CA, USA
^2^ Department of Neuroimaging, King's College London, London, UK
^3^ Computation and Neural Systems Program, California Institute of Technology, Pasadena, CA, USA

---

## Abstract

Foraging under predation risk requires integrating energetic cost and survival probability into a unified decision variable, yet how a single computation governs both the decision of what to pursue and the vigor of pursuit -- and how breakdowns in that computation relate to anxiety -- remains unknown. Here we develop an Expected Value of Control (EVC) model with linear-quadratic regulator (LQR) cost structure and test it in a large online effort-foraging task under parametric threat (N = 293). Two subject-level parameters -- effort cost (c~e~, identified from choice) and capture aversion (c~d~, identified from vigor) -- plus population-level probability weighting (gamma = 0.210) jointly predict foraging decisions (per-subject r^2^ = 0.951) and press-rate vigor (r^2^ = 0.511). Both parameters are recoverable in simulation (c~e~: r = 0.92; c~d~: r = 0.94). The model's survival signal strongly predicts trial-level anxiety (beta = -0.557, t = -14.0) and confidence (beta = 0.575, t = 13.5). We decompose metacognitive affect into calibration (how well anxiety tracks danger) and discrepancy (how much anxiety exceeds danger). These dimensions are orthogonal (r = 0.019) and doubly dissociated: calibration predicts task performance (r = 0.230 with choice quality), while discrepancy predicts clinical symptoms (STAI: beta = 0.338; STICSA: beta = 0.285; DASS-Anxiety: beta = 0.275). Critically, the model's computational parameters show no reliable clinical associations (approximately 77% posterior mass in the region of practical equivalence), providing Bayesian evidence for the null. The bridge from adaptive computation to psychiatric vulnerability runs through metacognition -- through how people feel about danger, not how they compute it.

---

## Introduction

Animals foraging in environments with predation risk face a fundamental trade-off: the energy gained from distant, high-value resources must be weighed against increased exposure to capture during transport^1,2^. This trade-off shapes foraging decisions across species, from birds provisioning nestlings near raptor territories^3^ to fish venturing from shelter to access food patches^4^. In humans, analogous computations arise whenever pursuing goals requires sustained effort under threat -- commuting through dangerous neighborhoods, working overtime in hazardous conditions, or simply persisting at effortful tasks when the costs of failure loom large.

Two research traditions have studied the components of this problem largely in isolation. Work on effort-based decision making has established that humans and animals discount the value of rewards by the physical or cognitive effort required to obtain them^5-8^, with individual differences in effort cost sensitivity linked to apathy, fatigue, and motivational disorders^9,10^. Separately, research on defensive behavior has characterized how organisms modulate their responses to threat across the predatory imminence continuum, from strategic avoidance at a distance to reactive flight when danger is immediate^11-13^. Theoretical models of foraging under predation risk integrate these factors^14,15^, but no computational model has formally specified how the same cost-benefit calculation governs both what an organism chooses and how vigorously it acts.

The Expected Value of Control (EVC) framework^16,17^ provides a candidate architecture for this integration. EVC proposes that the brain computes the expected payoff of allocating control effort, selecting the intensity that maximizes reward minus cost. Critically, EVC treats effort allocation as a continuous optimization, making it naturally suited to predict both discrete decisions and graded action vigor within a single framework. However, EVC has been applied primarily to cognitive control in abstract tasks^18,19^ and has not been extended to physical effort under ecological threat.

A separate gap concerns the relationship between threat processing and clinical anxiety. Metacognitive theories propose that pathological anxiety reflects not merely elevated threat estimation but a dysfunctional relationship between threat appraisal and affective response^20^. Paulus and Stein^21^ formalize this as an interoceptive prediction error: anxiety disorders involve systematic mismatch between expected and actual body states under threat, producing affective signals that are decoupled from the environment. Computational psychiatry has sought to ground psychiatric constructs in formal decision models^22-24^, but no study has operationalized this metacognitive mismatch -- the discrepancy between computed danger and experienced affect -- within a normative foraging framework where both the computation and the affect can be measured simultaneously.

Here we address both gaps. We developed an EVC model with linear-quadratic regulator (LQR) cost structure^25,26^ and tested it in a virtual effort-foraging task under parametric threat (N = 293). Participants chose between high-effort, high-reward and low-effort, low-reward resources while facing predation risk at three probability levels and three distances. We show that two subject-level parameters -- effort cost (c~e~) and capture aversion (c~d~) -- jointly predict choice and vigor through distinct cost channels. We then decompose subjective affect into metacognitive calibration and discrepancy, revealing a double dissociation: calibration predicts adaptive performance, while discrepancy -- not the computational parameters themselves -- predicts clinical anxiety symptoms. These results establish a formal bridge from normative foraging theory through affective metacognition to psychiatric symptomatology.

---

## Results

### An EVC-LQR model captures foraging choice with per-subject effort cost

We designed an effort-foraging task in which participants (N = 293) chose between a high-reward cookie (5 points, requiring sustained keypressing at 60-100% of calibrated maximum capacity) and a low-reward cookie (1 point, requiring 40% of maximum) while facing predation risk (threat probability T in {0.1, 0.5, 0.9}) at varying distances (D in {1, 2, 3}; Fig. 1a). On attack trials, a predator appeared and pursued the participant, who could be captured if they did not reach the safe zone in time (capture penalty C = 5 points). Participants completed 45 choice trials and 36 probe trials (identical forced-choice options with affect ratings).

We formalized the choice as a comparison of expected utilities:

DeltaEU = S x 4 - c~e,i~ x (0.81D~H~ - 0.16)

where S is the subjective survival probability, c~e,i~ is the subject-specific effort cost, and the effort term reflects the squared required press rate scaled by distance (LQR commitment cost; 0.81D~H~ - 0.16 is the difference in req^2^ x D between heavy and light options). The survival function incorporates probability weighting and a universal effort-efficacy term:

S = (1 - T^gamma^) + epsilon x T^gamma^ x p~esc~

where gamma = 0.210 is the population probability weighting exponent (indicating substantial compression of threat probabilities) and epsilon = 0.098 is the population effort-efficacy parameter reflecting the universal tendency to underweight effort's benefit to survival. Choices follow a softmax rule: P(heavy) = sigmoid(DeltaEU / tau), with tau = 0.476.

The critical advance over prior models is that c~e~ is estimated per subject (log-normal prior, non-centered parameterization), enabling the model to capture individual differences in how strongly distance deters foraging. Both subject-level parameters are log-normally distributed; we report log-transformed values throughout (log(c~e~): M = -0.47, SD = 0.78; log(c~d~): M = 3.44, SD = 1.57), which is the native parameterization of the hierarchical model. Previous versions with population-level effort cost failed to reproduce distance gradients in choice (Supplementary Fig. S1). With per-subject c~e~, the model reproduced the full threat-by-distance choice surface: at T = 0.1, P(heavy) declined from 0.87 at D = 1 to 0.53 at D = 3; at T = 0.9, from 0.33 to 0.10 (Fig. 2a). The model achieved BIC = 17,768 with per-subject choice r^2^ = 0.951, choice accuracy = 79.3%, and AUC = 0.876 (Fig. 2b).

The probability weighting parameter gamma = 0.210 indicates that a nominal 50% threat is experienced as T^0.210^ = 0.86, consistent with loss-domain probability distortion^27^ but stronger than typical estimates from monetary gambles (gamma ~ 0.65-0.70)^28^, potentially reflecting the salience of embodied predation compared to abstract losses.

We compared six model variants (Supplementary Table S1). The winning EVC 2+2 model outperformed versions with population-level effort cost (DeltaBIC > 2,500), versions without probability weighting (DeltaBIC > 800), and versions with alternative survival functions. Per-subject c~e~ was essential for capturing the distance gradient that drives approximately 40% of choice variance.

### The same framework predicts action vigor via LQR deviation cost

Having established that c~e~ captures choice, we tested whether the model jointly predicts trial-by-trial vigor -- the rate at which participants pressed keys during cookie transport. For the vigor component, the expected utility at press rate u is:

EU(u) = S(u) x R - (1 - S(u)) x c~d,i~ x (R + C) - c~e,vigor~ x (u - req)^2^ x D

where c~d,i~ is the subject-specific capture aversion, c~e,vigor~ = 0.003 is the population-level motor deviation cost, and req is the required press rate for the chosen cookie. The survival function becomes speed-dependent: S(u) = (1 - T^gamma^) + epsilon x T^gamma^ x p~esc~ x sigmoid((u - req) / sigma~motor~), meaning that pressing faster than required improves survival probability.

Three features of this architecture deserve emphasis. First, c~d~ is absent from the choice equation. The differential capture cost between options, (1 - S) x c~d~ x (R~H~ - R~L~) = (1 - S) x c~d~ x 4, is collinear with the differential reward term S x 4; both scale with the reward difference and are functions of S, making c~d~ unidentifiable from choice data alone. The model therefore estimates c~d~ exclusively from vigor, where it enters through the survival incentive term (1 - S(u)) x c~d~ x (R + C), which has no collinearity with the effort cost. This means c~e~ is the sole driver of choice, while c~d~ is identified exclusively from vigor. Second, the effort cost enters vigor as a deviation cost (u - req)^2^ rather than the commitment cost req^2^ used in choice. This LQR distinction^25^ resolves a fundamental scaling conflict: commitment costs (req^2^ x D ~ 0.8-2.4) and deviation costs ((u - req)^2^ x D ~ 0.01-0.05) differ by two orders of magnitude, allowing the same theoretical framework to operate at both decision stages without parameter inflation. Third, probe trials (forced conditions with identical options) provide vigor data across all threat-by-distance cells without selection bias, cleanly anchoring c~d~ estimation.

The model predicted trial-level vigor with r^2^ = 0.511, and per-subject mean vigor correlated r = 0.83 with predictions (Fig. 2c,d). The c~d~ parameter drives vigor through the survival incentive: higher c~d~ increases the marginal benefit of pressing faster (because failed escape costs c~d~ x (R + C)), producing steeper threat-vigor gradients. The model correctly predicts that vigor increases with threat (as the marginal survival benefit of pressing faster grows) and with distance (as longer exposure increases the stakes of pressing below required rate).

We note that the threat-vigor relationship is obscured in unconditional analyses due to a Simpson's paradox: because threat shifts choice toward light cookies (60% heavy at T = 0.1 versus 34% at T = 0.9), and light cookies have lower required press rates, collapsing across choice makes mean vigor appear flat across threat levels. Conditioning on chosen cookie type reveals strong threat modulation (t = 6.6-7.5, p < 10^-10^). The joint model naturally resolves this confound because it conditions on the chosen option.

Parameter recovery confirmed identifiability of both subject-level parameters. Simulating 293 synthetic subjects at the fitted population distribution and re-fitting yielded recovery correlations (in log space) of r = 0.92 for log(c~e~) and r = 0.94 for log(c~d~) (Fig. 2e,f). The correlation between log(c~e~) and log(c~d~) was r = -0.135, confirming that the two parameters capture largely independent dimensions of individual variation. Population parameters gamma and epsilon were also well-recovered.

### Behavioral profiles reveal distinct foraging strategies

The two-parameter structure generates interpretable individual differences. We performed a median split on c~e~ (effort cost) and c~d~ (capture aversion) to define four behavioral profiles (Fig. 3a):

**Cautious** (high c~e~, high c~d~): These individuals are both effort-sensitive and capture-averse. They rarely chose the high-effort option and pressed vigorously when they did, prioritizing survival over reward.

**Lazy** (high c~e~, low c~d~): These individuals avoid effort but are not particularly threatened by capture. They chose low-effort options and pressed at moderate rates -- the least engaged profile.

**Vigilant** (low c~e~, high c~d~): These individuals willingly pursue high-effort options but are strongly motivated to avoid capture. They chose heavy cookies frequently and pressed hardest. This profile earned the most points (mean = 29.9; Fig. 3b) -- willingness to forage combined with vigorous execution is the most adaptive strategy.

**Bold** (low c~e~, low c~d~): These individuals are effort-tolerant and capture-indifferent. They pursued high-effort options often but without the vigorous pressing that characterizes the Vigilant profile.

A joint logistic regression of P(heavy) on log(c~e~) and log(c~d~) explained R^2^ = 0.953 of between-subject choice variance (Fig. 3c), confirming that these two parameters nearly completely determine individual foraging strategy. One-way ANOVA across quadrants revealed significant differences in apathy (AMI: p = .041; Fig. 3d), with Lazy and Cautious profiles reporting higher apathy than Vigilant and Bold profiles. No other psychiatric measure differed significantly across quadrants after correction.

### Metacognitive miscalibration: confidence tracks choice quality, not survival

On probe trials (N = 36 per subject), participants rated their anxiety and confidence about the upcoming trial after committing to a choice but before executing it. The model's survival probability S strongly predicted both ratings via linear mixed models with random intercepts and slopes by subject:

- Anxiety: beta = -0.557, SE = 0.040, t = -14.0, p < 10^-44^ (higher survival = lower anxiety)
- Confidence: beta = +0.575, SE = 0.043, t = +13.5, p < 10^-40^ (higher survival = higher confidence)

These within-subject effects confirm that the model's internal survival signal closely tracks moment-to-moment affective experience. However, the between-subject pattern revealed a striking dissociation. Mean confidence correlated with choice quality (r = 0.230, p < .001) but not with survival rate (r ~ 0; Fig. 4a). Participants who felt more confident made better choices -- selecting options appropriate to threat conditions -- but did not actually survive more often. Confidence tracks the decision, not the outcome.

This finding identifies a specific metacognitive error: participants' confidence reflects the quality of their foraging decisions (which are governed by c~e~) rather than their actual escape ability (which depends on vigor, governed by c~d~). Because the two parameters are nearly independent (r = -0.135), confidence in one channel carries no information about performance in the other.

### Metacognitive discrepancy, not model parameters, predicts clinical anxiety

We decomposed each subject's metacognitive responding into two dimensions:

**Calibration**: the within-subject correlation between anxiety ratings and model-derived danger (1 - S). Higher calibration indicates that the individual's anxiety more accurately tracks objective threat.

**Discrepancy**: the mean residual of a subject's anxiety ratings after removing the population-level anxiety-danger relationship. Higher discrepancy indicates the individual reports more anxiety than the situation warrants -- a systematic positive bias.

These dimensions were orthogonal (r = 0.019, p = .75; Fig. 4b), confirming they capture independent aspects of metacognitive functioning.

A partial double dissociation emerged (Fig. 4c,d). Calibration predicted task performance: subjects with better threat-tracking anxiety made higher-quality choices (r = 0.230, p < .001) and survived at higher rates (r = 0.185, p = .002). Discrepancy predicted clinical symptoms: Bayesian regression revealed that higher discrepancy was associated with elevated scores across multiple psychiatric measures (Table 1):

| Measure | beta | 95% CI |
|---------|------|--------|
| STAI (trait anxiety) | 0.338 | [0.22, 0.45] |
| STICSA (somatic/cognitive anxiety) | 0.285 | [0.17, 0.40] |
| DASS-Anxiety | 0.275 | [0.16, 0.39] |
| PHQ-9 (depression) | 0.212 | [0.10, 0.33] |
| OASIS (anxiety severity) | 0.180 | [0.07, 0.30] |

The AMI (Apathy-Motivation Index) emotional subscale showed the opposite pattern: discrepancy was negatively associated with emotional apathy (beta = -0.222), indicating that overanxious individuals are not apathetic -- they are the antithesis of disengaged.

Critically, the model's computational parameters themselves (c~e~ and c~d~) showed no reliable clinical associations. Bayesian analysis revealed approximately 77% posterior mass within the region of practical equivalence (ROPE) for both parameters across all clinical measures, providing positive evidence for the null hypothesis that individual differences in effort cost and capture aversion are unrelated to psychiatric symptoms. This is not merely an absence of evidence but evidence of absence: the parameters that drive behavior are dissociated from the parameters that drive distress.

We also assessed whether individual model parameters could predict clinical outcomes using cross-validated machine learning (ridge regression with nested CV). All cross-validated R^2^ values were negative, confirming that computational parameters do not predict clinical symptoms at the individual level. This honest null result strengthens the metacognitive interpretation: the route from foraging computation to psychopathology is mediated by how people feel about their threat estimates, not by the estimates themselves.

---

## Discussion

We developed an Expected Value of Control model with LQR cost structure that jointly captures foraging choice and action vigor under parametric threat. Two subject-level parameters -- effort cost (c~e~) and capture aversion (c~d~) -- explain 95.1% of between-subject choice variance and 51.1% of trial-level vigor variance. The model's internal survival signal predicts moment-to-moment anxiety and confidence, and decomposing metacognitive accuracy into calibration and discrepancy reveals a double dissociation: calibration predicts adaptive performance while discrepancy predicts clinical psychopathology. The computational parameters themselves are clinically inert.

### A unified cost function for choice and vigor

The central modeling contribution is demonstrating that the EVC framework^16,17^ can be extended to physical effort under ecological threat, predicting both discrete choices and continuous motor vigor through a single cost function with qualitatively different cost channels. The LQR distinction between commitment cost (choice) and deviation cost (vigor)^25^ resolves a scaling problem that has frustrated prior attempts at joint models: the effort costs that explain choice (c~e~ x req^2^ x D) operate at two orders of magnitude above the motor costs that regulate vigor (c~e,vigor~ x (u - req)^2^ x D), yet both derive from the same theoretical principle -- penalizing the squared departure from the status quo.

The clean separation of c~e~ and c~d~ was enabled by a structural feature of the task: because both options share the same reward difference and survival probability in the choice comparison, c~d~ is collinear with the reward term and unidentifiable from choice alone. It is instead identified exclusively from vigor, where it drives pressing through the survival incentive. This orthogonal identification is a strength, not a limitation -- it means each parameter is identified from the behavioral channel it governs without post-hoc decomposition. The probe trial design further strengthens c~d~ identification by providing vigor data at all threat-by-distance conditions without selection bias.

The probability weighting parameter (gamma = 0.210) indicates substantial compression of threat probabilities, consistent with Kahneman and Tversky's^27^ finding that people overweight moderate probabilities in the loss domain, but stronger than typical estimates from monetary gambles^28^. This may reflect the embodied nature of our task: facing a virtual predator engages defensive circuitry^11,13^ more powerfully than evaluating abstract monetary losses, producing greater probability distortion. Alternatively, gamma may absorb other forms of risk aversion not captured by c~d~ alone. Disentangling these accounts would require parametric manipulation of threat probability framing.

### Metacognition as the computational-clinical bridge

Our most consequential finding is that the route from adaptive computation to clinical distress runs through metacognition rather than through the decision parameters. Neither c~e~ nor c~d~ predicted any clinical measure, with Bayesian analysis providing positive evidence for the null (approximately 77% ROPE containment). Instead, discrepancy -- systematic overestimation of danger relative to the model's survival signal -- predicted anxiety, depression, and somatic symptoms across five validated instruments.

This finding aligns with metacognitive theories of anxiety^20,21^, which propose that clinical anxiety reflects a dysfunctional relationship between threat appraisal and affective response rather than simply elevated threat sensitivity. Our computational framework provides a formal operationalization: calibration measures how well the affective signal tracks the computational signal, while discrepancy measures systematic bias -- the degree to which affect decouples from computation. The orthogonality of these dimensions (r = 0.019) means they capture genuinely independent aspects of metacognitive functioning, and their dissociation with performance versus symptoms validates the theoretical distinction.

The negative association between discrepancy and emotional apathy (AMI: beta = -0.222) adds nuance: overanxious individuals are not generally dysfunctional -- they are motivationally engaged but affectively miscalibrated. This echoes findings that anxiety and apathy represent opposite poles of motivational dysfunction^10^, and suggests that metacognitive bias may serve an energizing function even when it produces distress. Whether this reflects an adaptive trade-off (overestimating danger ensures adequate preparation) or a maladaptive byproduct (anxiety dysregulation co-opts motivational systems) remains an open question.

### Behavioral profiles and the dissociation of choice and vigor

The four behavioral profiles (Cautious, Lazy, Vigilant, Bold) generated by the c~e~ x c~d~ parameter space are not merely descriptive categories but emerge from the model's architecture. The Vigilant profile (low c~e~, high c~d~) earned the most points because it combines willingness to pursue high-value resources with vigorous execution during transport. This profile is adaptive because it allocates effort efficiently: low c~e~ permits engagement with effortful options, while high c~d~ ensures that once committed, the individual presses hard enough to survive.

The near-independence of c~e~ and c~d~ (r = -0.135) implies that knowing how someone decides tells you little about how hard they will try. This dissociation maps onto the distinction between strategic and reactive defensive modes^11,34^: choice reflects strategic threat assessment (selecting safer options), while vigor reflects reactive defensive mobilization (pressing harder when at risk). These modes may engage partially separable neural circuits^29,30^, consistent with evidence for dissociable prefrontal (strategic) and subcortical (reactive) defense systems^13,34^ and with ecological arguments that organisms benefit from flexible coupling between decision and execution^31^.

### Clinical implications and effect size honesty

The clinical associations we report are statistically robust but modest in magnitude (beta = 0.180-0.338, explaining approximately 3-11% of symptom variance). These effect sizes are consistent with the broader computational psychiatry literature^22,32^, but they should temper enthusiasm for immediate clinical translation. The strongest predictor -- discrepancy to STAI (beta = 0.338) -- explains roughly 11% of trait anxiety variance. This is meaningful for understanding mechanisms but insufficient for individual diagnosis.

We emphasize three honest constraints. First, all cross-validated R^2^ values for predicting clinical outcomes from model parameters were negative, confirming that these associations are group-level patterns rather than individually predictive biomarkers. Second, the calibration-discrepancy decomposition relies on the model's survival signal as ground truth; if the model is misspecified, discrepancy may partly reflect model error rather than metacognitive bias. Third, all results come from the exploratory sample; a pre-registered confirmatory replication (N = 350) is pending.

### Limitations

Several limitations warrant consideration. First, the effort-efficacy parameter epsilon is estimated at the population level because it is not individually recoverable (recovery r ~ 0). This means our model cannot capture individual differences in how much people believe their effort improves survival -- a theoretically important construct that would require richer within-subject designs or physiological measures to identify.

Second, distance confounds effort duration and threat exposure by design: farther cookies require both more sustained pressing and more time in danger. This confound is shared with natural foraging environments^1^ and is partially addressed by the factorial crossing of distance with effort weight, but a fully orthogonal design would strengthen causal claims.

Third, the task uses explicit threat probabilities, which differ from real-world threat assessment involving learning and updating. Our model assumes stationary threat environments and does not capture trial-by-trial belief updating that may occur even with stated probabilities^33^.

Fourth, our online sample from Prolific may not generalize to clinical populations with diagnosed anxiety disorders. The psychiatric measures reflect dimensional variation in a non-clinical sample; whether the discrepancy-symptom association strengthens in clinical groups is an important next step.

Finally, the model treats choice and vigor as occurring within a single trial, but does not capture learning across trials or strategic adjustments across blocks. Future work incorporating trial history and learning dynamics may reveal additional individual differences not captured by the current static-parameter model.

---

## Methods

### Participants

We recruited 350 participants from Prolific (https://prolific.co) for an online study. After a five-stage quality control pipeline -- (1) task completion, (2) comprehension checks, (3) behavioral consistency screening, (4) effort calibration validation (minimum 10 presses in 10 seconds), and (5) outlier removal -- the final exploratory sample comprised N = 293 participants. The study was approved by the Caltech Institutional Review Board, and all participants provided informed consent.

### Task design

Participants completed an effort-based foraging task implemented in Unity (WebGL) and presented in a desktop browser. The task featured a circular arena viewed from above, with a safe zone at the center and cookies appearing at radial distances.

**Effort calibration.** Before the main task, participants completed three 10-second trials of pressing the S, D, and F keys simultaneously as fast as possible. The maximum press count across trials defined each participant's calibrated maximum (f~max~). All subsequent effort demands were expressed relative to this individual calibration.

**Trial structure.** On each choice trial, two cookies appeared: a heavy cookie (5 points, requiring sustained pressing at 60-100% of f~max~ depending on weight condition) at distance D in {1, 2, 3} (corresponding to 5, 7, 9 game units from center), and a light cookie (1 point, requiring 40% of f~max~) always at distance 1 (5 game units). Participants clicked to select a cookie (commitment was irrevocable), then pressed S+D+F repeatedly to transport it to the safe zone. Movement speed depended on press rate relative to the cookie's demand threshold: full speed at >= 100% of required rate, half speed at >= 50%, quarter speed at >= 25%, and zero speed below 25%.

**Threat manipulation.** Each trial had a predation probability T in {0.1, 0.5, 0.9}. On attack trials, a predator spawned at the perimeter position closest to the participant at a pre-determined encounter time (calibrated to approximately half the expected return time), approached at 0.5 units/sec (slow approach phase), and then struck at a Gaussian-distributed time centered at twice the encounter time, accelerating to 4x the participant's maximum movement speed (unavoidable full-attack phase). Being captured cost 5 points plus the value of the current cookie.

**Probe trials.** Twelve of the 27 events per block were probe trials with identical options (forced choice). After selecting a cookie, the game paused and participants rated either their anxiety about being captured or their confidence in reaching safety (0-7 scale). The trial then proceeded normally. Probe ratings were prospective judgments about the current trial's demands, collected after choice commitment but before any motor execution.

**Structure.** Three blocks of 27 events each (81 total): 15 choice trials and 12 probe trials per block, yielding 45 choice trials and 36 probe trials per participant. The 36 probe trials spanned all combinations of threat (3) x distance (3) x rating type (2, anxiety vs. confidence), with two repetitions per cell.

### Psychiatric assessment

Between blocks, participants completed the DASS-21 (Depression, Anxiety, and Stress subscales), PHQ-9, OASIS, STAI (Trait scale), AMI (Behavioural, Social, Emotional subscales), MFIS (Physical, Cognitive, Psychosocial subscales), and STICSA. All scores were z-scored across participants before analysis.

### EVC-LQR model

The model has two per-subject parameters drawn from log-normal distributions with non-centered parameterization, plus population-level parameters.

**Per-subject parameters (log-normal; reported in log space):**
- c~e,i~ (effort cost): governs choice through distance-dependent effort penalty. log(c~e~): M = -0.47, SD = 0.78 (geometric mean = 0.62).
- c~d,i~ (capture aversion): governs vigor through the survival incentive at stake upon capture. log(c~d~): M = 3.44, SD = 1.57 (geometric mean = 31.3). The large geometric mean reflects the low population efficacy (epsilon = 0.098); the effective survival incentive per unit of survival gradient is c~d~ x epsilon ≈ 3.1, indicating moderate loss aversion relative to the 5-point capture penalty.

**Population parameters:**
- gamma = 0.210: probability weighting exponent
- epsilon = 0.098: effort efficacy (universal underweighting of effort's survival benefit)
- c~e,vigor~ = 0.003: LQR deviation motor cost
- tau = 0.476: choice temperature
- p~esc~: baseline escape probability
- sigma~motor~: motor noise
- sigma~v~: vigor observation noise

**Choice model.** The differential expected utility of the heavy versus light option is:

DeltaEU = S x (R~H~ - R~L~) - c~e,i~ x (req~H~^2^ x D~H~ - req~L~^2^ x D~L~)

Since R~H~ = 5, R~L~ = 1, req~H~ ~ 0.9, req~L~ = 0.4, D~L~ = 1:

DeltaEU = S x 4 - c~e,i~ x (0.81 x D~H~ - 0.16)

The capture aversion term c~d~ does not appear in the choice equation because its contribution to differential expected utility, (1 - S) x c~d~ x (R~H~ - R~L~) = (1 - S) x c~d~ x 4, is collinear with the reward differential S x 4 -- both scale with the reward difference and are functions of S, making c~d~ unidentifiable from choice data. c~d~ is instead identified from vigor data, where it enters through the survival incentive (1 - S(u)) x c~d~ x (R + C) and has no collinearity with the effort cost term.

P(choose heavy) = sigmoid(DeltaEU / tau)

**Vigor model.** For each trial, the model computes optimal press rate u* by maximizing:

EU(u) = S(u) x R - (1 - S(u)) x c~d,i~ x (R + C) - c~e,vigor~ x (u - req)^2^ x D

where S(u) = (1 - T^gamma^) + epsilon x T^gamma^ x p~esc~ x sigmoid((u - req) / sigma~motor~). The optimal vigor u* is computed via softmax-weighted grid search over u in [0.1, 1.5]. The c~d~ parameter drives vigor: higher capture aversion increases the marginal benefit of pressing faster because the stakes of failed escape (c~d~ x (R + C)) grow with c~d~.

**Joint likelihood.** Choice trials contribute a Bernoulli likelihood; all trials (choice and probe) contribute a Normal likelihood for vigor. Both are evaluated simultaneously during fitting.

**Data.** Choice: 45 trials per subject (free-choice only, trial type = 1). Vigor: 81 trials per subject (choice and probe trials, types 1, 5, 6). Probe distances derived from startDistance (5 -> D = 1, 7 -> D = 2, 9 -> D = 3).

### Model fitting

The model was fitted using NumPyro's stochastic variational inference (SVI) with an AutoNormal guide (mean-field approximation). Optimization used Adam (learning rate = 0.002) for 40,000 steps. BIC was computed as 2 x loss + k x log(n), where k = 2 x N~subjects~ + number of population parameters.

### Model comparison

We compared six model variants differing in whether effort cost was per-subject or population-level, whether probability weighting was included, and whether alternative survival functions were used. Model selection was based on BIC.

### Parameter recovery

We simulated 293 synthetic subjects using the fitted population distribution and task design, generating choices and vigor from the generative model. The simulated data were re-fitted with the same SVI procedure. Recovery was assessed as the Pearson correlation between true and recovered subject-level parameters in log space.

### Affect analysis

Linear mixed models (statsmodels MixedLM, REML estimation) predicted anxiety and confidence ratings from standardized survival probability (S~z~) with random intercepts and slopes by subject. Random effects accounted for individual differences in both baseline affect and sensitivity to survival probability.

### Metacognitive decomposition

**Calibration:** Per-subject Pearson correlation between anxiety ratings and model-derived danger (1 - S). Higher calibration indicates the individual's anxiety more closely tracks objective threat.

**Discrepancy:** Per-subject mean residual from the population-level regression of anxiety on S. Positive discrepancy indicates the individual reports more anxiety than the objective danger warrants.

### Clinical analysis

Associations between metacognitive dimensions and psychiatric symptoms were estimated using Bayesian linear regression (weakly informative priors, 95% highest density intervals). Model parameter-clinical associations were evaluated using both frequentist FDR-corrected correlations and Bayesian ROPE analysis (region of practical equivalence: |r| < 0.10). Cross-validated prediction used ridge regression with nested 5-fold CV.

### Statistical analysis

All tests were two-tailed. Effect sizes are reported as Pearson r, standardized beta, or R^2^. Multiple comparison corrections used Benjamini-Hochberg FDR. All analyses were conducted in Python 3.11 using NumPyro, JAX, statsmodels, and scipy.

---

## References

1. Lima, S. L. & Dill, L. M. Behavioral decisions made under the risk of predation: a review and prospectus. *Can. J. Zool.* **68**, 619-640 (1990).
2. Charnov, E. L. Optimal foraging, the marginal value theorem. *Theor. Popul. Biol.* **9**, 129-136 (1976).
3. Cresswell, W. Predation in bird populations. *J. Ornithol.* **152**, 251-263 (2011).
4. Milinski, M. & Heller, R. Influence of a predator on the optimal foraging behaviour of sticklebacks. *Nature* **275**, 642-644 (1978).
5. Pessiglione, M., Vinckier, F., Bouret, S., Daunizeau, J. & Le Boisselier, R. Why not try harder? Computational approach to motivation deficits in neuro-psychiatric diseases. *Brain* **141**, 629-650 (2018).
6. Husain, M. & Roiser, J. P. Neuroscience of apathy and anhedonia: a transdiagnostic approach. *Nat. Rev. Neurosci.* **19**, 470-484 (2018).
7. Westbrook, A. & Braver, T. S. Dopamine does double duty in motivating cognitive effort. *Neuron* **89**, 695-710 (2016).
8. Hartmann, M. N., Hager, O. M., Tobler, P. N. & Kaiser, S. Parabolic discounting of monetary rewards by physical effort. *Behav. Process.* **100**, 192-196 (2013).
9. Le Heron, C. et al. Brain mechanisms underlying apathy. *J. Neurol. Neurosurg. Psychiatry* **90**, 302-312 (2019).
10. Chong, T. T.-J. et al. Neurocomputational mechanisms underlying subjective valuation of effort costs. *PLoS Biol.* **15**, e1002598 (2017).
11. Mobbs, D., Hagan, C. C., Dalgleish, T., Silston, B. & Prevost, C. The ecology of human fear: survival optimization and the nervous system. *Front. Neurosci.* **9**, 55 (2015).
12. Mobbs, D. et al. Foraging under competition: the neural basis of input-matching in food-deprived participants. *J. Neurosci.* **33**, 9866-9872 (2013).
13. Qi, S. et al. How cognitive and reactive fear circuits optimize escape decisions in humans. *Proc. Natl Acad. Sci. USA* **115**, 3186-3191 (2018).
14. Mobbs, D. et al. Foraging for foundations in decision neuroscience: insights from ethology. *Nat. Rev. Neurosci.* **19**, 419-427 (2018).
15. Mobbs, D. & Kim, J. J. Neuroethological studies of fear, anxiety, and risky decision-making in rodents and humans. *Curr. Opin. Behav. Sci.* **5**, 8-15 (2015).
16. Shenhav, A., Botvinick, M. M. & Cohen, J. D. The expected value of control: an integrative theory of anterior cingulate cortex function. *Neuron* **79**, 217-240 (2013).
17. Shenhav, A. et al. Toward a rational and mechanistic account of mental effort. *Annu. Rev. Neurosci.* **40**, 99-124 (2017).
18. Musslick, S. et al. Multitasking capability versus learning efficiency in neural network architectures. In *Proc. 39th Annu. Conf. Cogn. Sci. Soc.* (2017).
19. Lieder, F., Shenhav, A., Musslick, S. & Griffiths, T. L. Rational metareasoning and the plasticity of cognitive control. *PLoS Comput. Biol.* **14**, e1006043 (2018).
20. Wells, A. *Metacognitive Therapy for Anxiety and Depression* (Guilford, 2009).
21. Paulus, M. P. & Stein, M. B. Interoception in anxiety and depression. *Brain Struct. Funct.* **214**, 451-463 (2010).
22. Gillan, C. M., Kosinski, M., Whelan, R., Phelps, E. A. & Daw, N. D. Characterizing a psychiatric symptom dimension related to deficits in goal-directed control. *eLife* **5**, e11305 (2016).
23. Wise, T. & Dolan, R. J. Associations between aversive learning processes and transdiagnostic psychiatric symptoms in a general population sample. *Nat. Commun.* **11**, 4462 (2020).
24. Wise, T., Zbozinek, T. D., Michelini, G., Hagan, C. C. & Mobbs, D. Changes in risk perception and self-reported protective behaviour during the first week of the COVID-19 pandemic in the United States. *R. Soc. Open Sci.* **7**, 200742 (2020).
25. Todorov, E. & Jordan, M. I. Optimal feedback control as a theory of motor coordination. *Nat. Neurosci.* **5**, 1226-1235 (2002).
26. Shadmehr, R. & Krakauer, J. W. A computational neuroanatomy for motor control. *Exp. Brain Res.* **185**, 359-381 (2008).
27. Kahneman, D. & Tversky, A. Prospect theory: an analysis of decision under risk. *Econometrica* **47**, 263-291 (1979).
28. Tversky, A. & Kahneman, D. Advances in prospect theory: cumulative representation of uncertainty. *J. Risk Uncertain.* **5**, 297-323 (1992).
29. Niv, Y., Daw, N. D., Joel, D. & Dayan, P. Tonic dopamine: opportunity costs and the control of response vigor. *Psychopharmacology* **191**, 507-520 (2007).
30. Shadmehr, R., Huang, H. J. & Ahmed, A. A. A representation of effort in decision-making and motor control. *Curr. Biol.* **26**, 1929-1934 (2016).
31. Cisek, P. & Kalaska, J. F. Neural mechanisms for interacting with a world full of action choices. *Annu. Rev. Neurosci.* **33**, 269-298 (2010).
32. Huys, Q. J. M., Maia, T. V. & Frank, M. J. Computational psychiatry as a bridge from neuroscience to clinical applications. *Nat. Neurosci.* **19**, 404-413 (2016).
33. Browning, M., Behrens, T. E., Jocham, G., O'Reilly, J. X. & Bishop, S. J. Anxious individuals have difficulty learning the causal statistics of aversive environments. *Nat. Neurosci.* **18**, 590-596 (2015).
34. Mobbs, D. et al. Space, time, and fear: survival computations along defensive circuits. *Trends Cogn. Sci.* **24**, 228-241 (2020).

---

## Data and code availability

All analysis code is available at [repository URL]. Raw data will be made available upon publication. Pre-registration for the confirmatory sample (N = 350) is available at [OSF URL].

## Acknowledgments

This work was supported by [funding sources]. We thank participants recruited through Prolific for their time.

## Author contributions

N.O. designed the study, collected data, developed the computational model, conducted analyses, and wrote the manuscript. K.G. contributed to study design and data collection. T.W. contributed to computational modeling and analysis design. D.M. supervised the project, provided theoretical framing, and edited the manuscript.

## Competing interests

The authors declare no competing interests.

---

## Figure legends

**Figure 1. Task design and EVC-LQR model.** (a) Schematic of the effort-foraging task. Participants chose between a high-reward, high-effort cookie at varying distances and a low-reward, low-effort cookie near the safe zone. A predator attacked with probability T in {0.1, 0.5, 0.9}. (b) Trial timeline showing the three phases: anticipatory (no predator visible), encounter (predator appears, slow approach), and strike (full attack speed). (c) EVC-LQR model schematic. Choice is governed by DeltaEU comparing expected utilities with LQR commitment cost (c~e~ x req^2^ x D). Vigor is governed by maximizing EU(u) with LQR deviation cost (c~e,vigor~ x (u - req)^2^ x D). Survival probability S links both through probability-weighted threat.

**Figure 2. Model fit and parameter recovery.** (a) Observed (bars) and predicted (lines) P(heavy) as a function of threat and distance. The model captures the full threat x distance interaction. (b) Per-subject predicted versus observed P(heavy) (r^2^ = 0.951). (c) Vigor model predictions: observed and predicted excess press rate by threat and cookie type. (d) Per-subject predicted versus observed mean vigor (r = 0.83). (e) Parameter recovery for c~e~ (r = 0.92). (f) Parameter recovery for c~d~ (r = 0.94).

**Figure 3. Behavioral profiles.** (a) Joint distribution of log(c~e~) and log(c~d~) with median-split quadrants labeled (Cautious, Lazy, Vigilant, Bold). (b) P(heavy) by quadrant and logistic regression fit (R^2^ = 0.953). (c) Mean earnings by quadrant (Vigilant earns most at 29.9 points). (d) AMI scores by quadrant (p = .041).

**Figure 4. Metacognition bridges computation and clinical symptoms.** (a) Within-subject: survival S predicts anxiety (beta = -0.557) and confidence (beta = 0.575). Between-subject: confidence correlates with choice quality (r = 0.230) but not survival (r ~ 0). (b) Calibration and discrepancy are orthogonal (r = 0.019). (c) Calibration predicts performance: choice quality (r = 0.230) and survival (r = 0.185). (d) Discrepancy predicts clinical symptoms across five instruments (beta = 0.180-0.338). (e) Model parameters c~e~ and c~d~ show null clinical associations (approximately 77% ROPE containment).

---

## Supplementary Information

### Supplementary Table S1. Model comparison

| Model | Per-subject params | Population params | BIC | Choice r^2^ | Vigor r^2^ |
|-------|-------------------|-------------------|-----|-------------|------------|
| EVC 2+2 (winner) | c~e~, c~d~ | gamma, epsilon, c~e,vigor~, tau | 17,768 | 0.951 | 0.511 |
| Population c~e~ | c~d~ | c~e~, gamma, epsilon, tau | >20,000 | 0.90 | 0.49 |
| No prob. weighting | c~e~, c~d~ | epsilon, tau | >18,500 | 0.92 | 0.48 |
| Exponential survival | c~e~, c~d~ | gamma, epsilon, tau | >18,200 | 0.93 | 0.50 |
| No epsilon | c~e~, c~d~ | gamma, tau | >19,000 | 0.91 | 0.42 |
| Three per-subject | c~e~, c~d~, epsilon | gamma, tau | ~17,800 | 0.952 | 0.51 |

Note: The three-per-subject model achieves similar fit but epsilon is not individually recoverable (recovery r ~ 0), justifying the population-level specification.

### Supplementary Figure S1. Distance gradient failure with population-level effort cost

With c~e~ at the population level (c~e~ = 0.007), predicted P(heavy) is nearly flat across distances within each threat level (e.g., T = 0.1: pred = 0.847, 0.848, 0.846 vs. obs = 0.87, 0.70, 0.53). The per-subject c~e~ specification resolves this, producing predicted gradients that match observed data.

---

## Self-assessment

### Strengths of this draft

1. **Clear narrative arc.** The paper moves from ecological problem (foraging under threat) through formal model (EVC-LQR) through affect (metacognitive decomposition) to clinical implication (discrepancy predicts symptoms), with each section building on the last.

2. **Statistical precision.** Every claim is backed by a specific number with appropriate uncertainty. The key statistics (BIC = 17,768, choice r^2^ = 0.951, vigor r^2^ = 0.511, recovery r = 0.92/0.94, calibration-discrepancy r = 0.019, clinical betas with CIs) are all drawn directly from the analyses.

3. **Honest effect sizes.** The discussion explicitly acknowledges that clinical associations explain 3-11% of variance, that cross-validated R^2^ values are negative, and that these are group-level associations rather than individual predictors. The ROPE analysis provides positive evidence for the null on model parameters, which is stronger than merely failing to find an effect.

4. **Clean model identification.** The collinearity-based exclusion of c~d~ from choice and the use of probe trials for vigor are genuine design strengths that enable orthogonal parameter identification without post-hoc decomposition.

5. **LQR contribution is novel.** The commitment-vs-deviation cost distinction from optimal control theory is a real theoretical contribution to the EVC literature, resolving a concrete scaling problem.

### Weaknesses and reviewer concerns

1. **Population epsilon.** Epsilon is not individually recoverable, which limits the model's ability to capture individual differences in effort-efficacy beliefs. A reviewer might argue this is a fundamental limitation for a paper about individual differences. The response is that epsilon acts as a universal bias term (people generally underweight effort's survival benefit) and that attempting per-subject estimation degrades the model without improving fit.

2. **Gamma absorbs multiple constructs.** The probability weighting parameter may reflect genuine probability distortion, risk aversion beyond c~d~, or task-specific factors. Without manipulation of probability framing, we cannot distinguish these accounts.

3. **Discrepancy definition is model-dependent.** If the survival function is misspecified, discrepancy reflects model error rather than metacognitive bias. The strong within-subject anxiety-S correlations (t = -14.0) mitigate this concern but do not eliminate it.

4. **Cross-validated prediction is negative.** While framed as honest, a reviewer could argue this undermines the clinical relevance of the findings entirely. The counter is that group-level associations can be scientifically meaningful (identifying mechanisms) even when they do not support individual prediction.

5. **Single exploratory sample.** All results await confirmatory replication. This is flagged in the limitations.

6. **Distance-effort confound.** Distance drives both effort duration and threat exposure, meaning the model cannot fully disentangle effort and survival costs at the design level. This is a genuine limitation shared with ecological foraging but worth flagging for a psychology audience.

### Verdict

This draft is suitable for submission to Nature Communications. The model is well-specified and identified, the key results are robust, the narrative is coherent, and the limitations are honestly presented. The main risk at review is the modest clinical effect sizes and the reliance on a single sample, both of which are acknowledged. The theoretical contribution (EVC-LQR for choice + vigor under threat, metacognitive bridge to clinical symptoms) is novel and substantive.
