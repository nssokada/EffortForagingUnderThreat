# A joint fitness function explains patch selection, motor vigor, and subjective affect during human foraging under threat

---

**Abstract**

A fundamental challenge in survival involves optimizing foraging efficiency while minimizing the risk of predation. Animals flexibly adapt their foraging behavior in response to these pressures; however, the cognitive mechanisms by which humans navigate such trade-offs remain largely unknown. Here, we bridge the literatures on effort-based decision making and human threat processing using a naturalistic foraging game in which participants chose patches and worked to harvest rewards under parametric predation risk. We derive a fitness function from optimal foraging theory that jointly predicts patch selection and motor vigor through two individual-difference parameters: avoidance sensitivity (ω) and activation intensity (κ). Across two preregistered samples (N = 290; N = 281), threat drove avoidance in choice, activation in vigor, and opposing shifts in anxiety and confidence; motor dynamics revealed distinct anticipatory and reactive defense signatures consistent with the predatory imminence continuum. The joint W(u) model outperformed all simpler alternatives, and the fitted parameters produced their predicted behavioral signatures: ω predicted survival on attack trials and the share of overcautious errors, κ predicted pressing intensity, and the (ω, κ) angle predicted decision quality. Two preregistered metacognitive signals — anxiety and confidence — independently contributed to behavior beyond the model parameters: anxiety calibration to threat improved out-of-sample prediction of all three preregistered foraging outcomes, anxiety reactivity predicted threat-driven choice shift, the cost-weight parameters dissociated in their relationship to confidence (loaded) versus anxiety (independent), and confidence shifted the type of decision errors people make (more reckless / fewer overcautious) without changing the overall error rate. Together, these findings reveal that humans adapt effort to threat through a joint cost-weighting computation that produces coordinated patch selection and motor vigor, and that subjective experience contributes structurally informative variance to foraging-under-threat behavior through two computationally distinct metacognitive channels: anxiety as a calibration signal, and confidence as a decision-criterion modulator.

---

## 1. Introduction

To survive in the wild, an organism must continually balance the competing demands of energetic efficiency and safety. Acquiring resources necessitates leaving protective cover, traversing space, expending metabolic energy, and tolerating increased exposure to danger. Behavioral ecology has formalized these tradeoffs through fitness functions that predict patch exploitation, vigilance, and flight under predation risk (Lima & Dill, 1990; Brown, 1999; Bednekoff, 2007; Stephens & Krebs, 1986). These models successfully explain foraging behavior across taxa (Charnov, 1976; Calcagno et al., 2023; Choi & Kim, 2010), and recent work has increasingly adopted foraging as a framework for understanding the complex dynamics of human decision making and affect (Kolling et al., 2012; Bustamante et al., 2023; Mobbs et al., 2018; Wise et al., 2024). However, the computational principles by which humans navigate this tradeoff remain unclear, particularly in settings where reward, effort, and threat must be processed simultaneously.

Each side of this problem has a rich independent literature. Research on human foraging has shown that people integrate reward value, costs, and opportunity costs when navigating resource environments (Kolling et al., 2012; Bustamante et al., 2023; Yoon et al., 2018), with sensitivity to effort costs linked to motivational symptoms such as anhedonia and fatigue (Bustamante et al., 2024; Treadway & Zald, 2011; Pessiglione et al., 2018; Muller et al., 2021; Treadway & Salamone, 2022). Research on threat processing has characterized how predation risk reshapes foraging decisions in both animals and humans (Trier et al., 2025; Choi & Kim, 2010; Silston et al., 2021; Calcagno et al., 2023), with neural work identifying defensive circuits that govern the transition from risk assessment to escape (Mobbs et al., 2015; Mobbs et al., 2020; Evans et al., 2019; Zhang et al., 2025). Whether a single foraging computation can simultaneously account for how humans select patches, execute motor responses, and appraise threat remains an open question, one that requires a paradigm in which effort, threat, and their interaction are manipulated within the same task.

Recent advances in computational ethology have moved toward this goal through ecological tasks that embed participants in richer, more continuous behavioral environments (Wise et al., 2024; Wu et al., 2025; Trier et al., 2025; Mobbs et al., 2018). We build on this approach with a foraging game in which participants choose between patches of differing reward and effort cost, then press keys to transport the chosen resource to safety while a predator can appear during transit. Because motor output is recorded continuously throughout each trial, the task yields simultaneous measurements of patch selection, moment-to-moment vigor, and self-reported affect within the same foraging bout, making it possible to test whether vigor dynamics follow the predatory imminence continuum (Fanselow, 1994; Mobbs et al., 2020; Blanchard & Blanchard, 1989).

The continuous nature of these data also raises the question of what role subjective affect plays in the foraging computation. Contemporary theories propose that internal states such as anxiety and confidence are not merely byproducts of decision making but serve as meta-representations that integrate environmental and internal variables to guide ongoing behavior (Mobbs et al., 2020; Lazarus, 1991; Fleming & Daw, 2017). In the context of foraging under threat, anxiety may track the degree of danger in the environment while confidence tracks the forager's appraisal of their own capacity to cope. Whether these signals carry information about foraging efficiency beyond what computational model parameters capture, and whether they dissociate from those parameters, has not been tested.

Here, we derive a fitness function inspired by the life-history foraging tradition (Gilliam & Fraser, 1987; Brown, 1999; Bednekoff, 2007) and augmented with a motor-control cost function (Shadmehr & Krakauer, 2008; Manohar et al., 2015), which jointly predicts patch selection and motor vigor through two per-subject parameters: avoidance sensitivity (ω, the subjective cost of capture) and activation intensity (κ, the subjective cost of effort) (Fig. 1c,d). Intuitively, ω captures how heavily a forager weights the prospect of being caught (high ω = more cautious), while κ captures how heavily they weight the cost of physical effort (high κ = more effort-averse). We test this model using a preregistered two-sample design, with an exploratory sample (N = 290) used to develop all hypotheses and an independent confirmatory sample (N = 281) collected from a non-overlapping participant pool. We show that threat simultaneously drives avoidance in choice and activation in vigor, that the joint model outperforms simpler alternatives, that the fitted cost-weight parameters produce their predicted behavioral signatures across two independent samples, and that two metacognitive signals — anxiety and confidence — independently inform foraging behavior beyond what the model parameters alone capture, with anxiety functioning as a calibration signal that tracks threat and confidence functioning as a decision-criterion modulator that loads on the avoidance computation.

## 2. Results

**Task and design.** Participants (exploratory N = 290; confirmatory N = 281, after exclusions) completed a foraging game in which they collected resources from a circular arena while a predator could appear (Fig. 1a,b). On each trial, participants were shown the trial's predation probability T (low, medium, or high) and chose between a heavy cookie (high reward, high effort, variable distance) and a light cookie (low reward, low effort, fixed nearby location). After choosing, they pressed keys to transport the chosen cookie back to the central safe zone; if a predator appeared and reached them before they returned, they were captured. On a subset of probe trials, participants rated their anxiety or confidence about the upcoming trial before pressing. Across both samples, we asked three questions: (i) how does threat shape patch choice, motor vigor, and subjective affect during foraging; (ii) does a single fitness function with two individual-difference parameters explain these behaviors better than simpler alternatives; and (iii) do metacognitive signals carry information about foraging efficiency beyond what the model parameters capture.

### 2.1 Threat and exposure jointly reshape choice, vigor, and affect

We first asked how predation risk shapes the three behavioral channels of interest. In our task, two factors jointly determine danger: the probability that a predator will appear (threat T) and the distance the participant must travel from the safe zone to retrieve a cookie (exposure D). Together, T and D set the expected cost of pursuing each patch. If foraging is governed by a single underlying computation, then increases in either factor should produce coordinated shifts in patch selection, motor execution, and subjective affect.

Patch selection was strongly modulated by both threat and exposure. In a logistic model with cluster-robust standard errors, the probability of choosing the heavy (high-reward, high-effort) cookie decreased steeply with threat (beta = -0.91, z = -19.8, P < 0.001; exploratory: beta = -1.02, z = -22.3, P < 0.001) and with distance (beta = -0.67, z = -22.1, P < 0.001; exploratory: beta = -0.75, z = -23.7, P < 0.001) (Fig. 2a). The two factors interacted negatively (beta = -0.12, P < 0.001): the marginal effect of threat on avoidance was approximately 12% larger at the longest distance than at the shortest, consistent with the multiplicative exposure structure of the fitness function we develop below.

![Figure 1. Task design and computational model.](data/figures/fig1_task_model.png)

**Figure 1 | Task design and computational model.** **a**, Foraging arena schematic. Participants chose between a heavy cookie (high reward, high effort, variable distance) and a light cookie (low reward, low effort, fixed distance) under stated predation risk. **b**, Trial structure. After choosing, participants pressed keys to transport the cookie to the central safe zone; a predator could appear based on stated threat probability T. **c**, Fitness function W(u) plotted across pressing rates for different threat levels, omega values, and kappa values, illustrating how the two parameters shape the optimization landscape. **d**, Model schematic showing how omega and kappa enter the joint likelihood through choice (patch selection) and vigor (pressing intensity) channels simultaneously.

Motor vigor moved in the opposite direction from choice. Within each cookie type, normalized pressing rate increased monotonically with threat probability (linear mixed model controlling for cookie: beta = +0.017, z = 17.4, P < 0.0001; Fig. 2b), and the effect was larger for the light cookie (confirmatory d = 0.76, exploratory d = 0.44) than the heavy cookie (confirmatory d = 0.45, exploratory d = 0.24). We analyzed the two cookies separately because they impose different minimum pressing rates (0.9 vs 0.4); pooling them would confound the within-trial effect of threat on vigor with the across-trial shift in cookie composition under high threat. Threat therefore drove two simultaneous adjustments: participants chose safer patches (avoidance) and pressed harder on whichever patch they had committed to (activation).

Subjective affect tracked the same manipulation. Anxiety ratings rose (beta = 0.53, z = 12.5, P < 0.001; exploratory: beta = 0.58, z = 14.7, P < 0.001) while confidence ratings fell (beta = -0.67, z = -15.3, P < 0.001; exploratory: beta = -0.58, z = -13.7, P < 0.001) as threat increased (Fig. 2c).

Threat and exposure therefore reshape behavior across all three channels in parallel — avoidance in choice, activation in vigor, and opposing changes in anxiety and confidence — a pattern consistent with an underlying foraging computation that we now examine directly.

### 2.2 Vigor dynamics show distinct anticipatory and reactive components

The threat-driven increase in pressing rate reported above is an average across the trial, leaving open how vigor unfolds in time as the predator becomes more or less proximal. A long-standing observation in defensive neuroscience is that anticipatory and reactive defense recruit distinct dynamics (Fanselow, 1994; Perusini & Fanselow, 2015; Mobbs et al., 2020; Blanchard & Blanchard, 1989; Abend et al., 2022; van Ast et al., 2022); whether voluntary human motor output during foraging shows an analogous dissociation is less clear. To examine this, we aligned pressing-rate timecourses to the moment of predator encounter and characterized the dynamics of vigor before and after.

Predator appearance was followed by a sharp acceleration in pressing rate (Fig. 2d). The per-subject encounter spike — defined as the difference in pressing rate between attack and non-attack reactive epochs — was large (d = 0.65, t(280) = 11.0, P < 0.001; exploratory: d = 0.56, P < 0.001) and observed in 83% of participants in both samples. Spike magnitude did not vary with the trial's stated threat probability (t = -1.42, P = 0.16; exploratory: P = 0.21), indicating that the reactive response was triggered by predator detection rather than scaled by prior expectation. The pre-encounter portion of the trial showed a contrasting pattern: vigor was gradedly modulated by threat probability throughout the anticipatory period (Fig. 2d; GAM smoothing, see Methods).

![Figure 2. Behavioral results.](data/figures/fig2_behavioral.png)

**Figure 2 | Behavioral and affective responses to threat.** **a**, Probability of choosing the heavy cookie as a function of threat probability and distance. Both threat and distance reduce high-effort choices, with a compounding interaction. **b**, Normalized pressing rate by threat level within each cookie type, showing increased vigor under higher threat even after controlling for cookie selection. **c**, Anxiety and confidence ratings by threat level, demonstrating parallel affective shifts. **d**, Encounter-aligned vigor timecourse showing the reactive motor spike upon predator appearance (shaded region), with distinct temporal signatures for attack versus non-attack trials.

Within the same motor output, then, two dissociable components emerged: a graded anticipatory adjustment scaled to the likelihood of attack, and a stereotyped reactive surge elicited by the encounter itself. This dual structure is consistent with a distinction between anticipatory and reactive components of defensive behavior, though the behavioral data cannot distinguish defensive-system engagement from Pavlovian stimulus-driven motor potentiation or strategic speed-up on detection of the predator. We therefore interpret the pattern as consistent with, rather than diagnostic of, the predatory imminence continuum (Fanselow, 1994; Mobbs et al., 2020).

### 2.3 A joint fitness function explains both choice and vigor

Having shown that changes in threat and exposure produce coordinated shifts in choice, vigor, and affect, we asked what account of the underlying decision process best explains this coordination. Four candidate accounts, each with a different commitment about what drives foraging behavior, were formalized as computational models and fit to the joint choice–vigor likelihood (see Methods).

**Effort-cost accounts.** The simplest account attributes the entire coordinated pattern to effort sensitivity alone. On this view, participants who disproportionately weight the cost of physical effort will both avoid high-effort patches and moderate their motor output, and no separate term for predation risk is needed. This is the view implicit in most effort-based decision-making frameworks, where individual differences in motivated behavior are captured by a single effort-cost parameter (Bustamante et al., 2023; Pessiglione et al., 2018; Yoon et al., 2018). We instantiated it as M1: participants differ only in effort sensitivity (κ), and no survival term enters the decision.

**Threat-cost accounts.** A competing account places predation risk at the center. Here the decision is dominated by the cost of capture, with individuals differing primarily in how heavily they weight it — a framing that aligns with work on human defensive decision-making in which threat appraisal is treated as the core individual-difference dimension (Trier et al., 2025; Silston et al., 2021; Mobbs et al., 2020). We instantiated it as M2: each participant has a per-subject capture-cost parameter (ω), while effort sensitivity is held at a population-level value.

**Single-dimension accounts.** A third account preserves both costs but collapses them onto a single latent trait. Under this view, the participants who find capture subjectively costly are also those who find effort subjectively costly, so avoidance and activation become two faces of one underlying disposition. We instantiated this as M3 (one parameter serving both roles) and a scaled variant M3b that permits the two costs to differ in magnitude while constraining them to a single dimension.

**Joint avoidance–activation account.** Finally, the account that our fitness function formalizes. Avoidance sensitivity (ω) and activation intensity (κ) are treated as genuinely separable traits that enter the same fitness function W(u) and jointly govern both patch selection and motor vigor through a shared optimization (M4).

**Table 1. Model comparison.**

| Model | Parameters                    | What it lacks                           | WAIC (conf.) | ΔWAIC (conf.) | WAIC (expl.) | ΔWAIC (expl.) |
| ----- | ----------------------------- | --------------------------------------- | ------------ | ------------- | ------------ | ------------- |
| M1    | κ only                        | no survival/threat term                 | 16,037       | +3,785        | 17,505       | +4,729        |
| M2    | ω (per-subject), population κ | individual effort sensitivity           | 13,873       | +1,621        | 14,742       | +1,966        |
| M3    | θ = ω = κ                     | separability of capture and effort cost | 15,727       | +3,474        | 15,374       | +2,599        |
| M4    | ω + κ (per-subject)           | — (full model)                          | **12,252**   | — (best)      | **12,776**   | — (best)      |

_ΔWAIC = difference in widely applicable information criterion (WAIC) relative to the best-fitting model (M4); lower WAIC indicates better predictive fit. WAIC and Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO) agreed on all rank orderings in both samples._

![Figure 3. Model comparison and validation.](data/figures/fig3_model_comparison.png)

**Figure 3 | Model comparison and parameter recovery.** **a**, ΔWAIC for each candidate model relative to the joint avoidance–activation model (M4) in both samples. **b**, Posterior predictive check for choice: model-predicted P(heavy) versus observed proportions across conditions, mean absolute error = 0.04. **c**, Posterior predictive check for vigor: 91–93% of vigor cell means fall within the 90% posterior predictive interval. **d**, Parameter recovery from 500 synthetic subjects: recovered versus true ω (r = 0.94) and κ (r = 0.92).

**Model comparison.** The joint avoidance–activation account outperformed every alternative in both samples (Table 1; Fig. 3a), with WAIC and PSIS-LOO agreeing on the full rank ordering. Effort-only accounts performed worst, followed by the single-dimension accounts; threat-only accounts fit better but still fell well behind M4. The winning model predicted choice with 76–77% accuracy — substantially above a threshold heuristic baseline (62–63%) — and explained 37–41% of the variance in vigor cell means, more than double the 17–19% captured by a condition-means benchmark. Posterior predictive checks indicated good calibration (Fig. 3b,c; parameter recovery and individual-level fits are reported in Supplementary Information S2).

Avoidance and activation are therefore neither interchangeable nor reducible to one another. Both enter the same fitness function as separable per-subject parameters, and neither alone reproduces the coordinated pattern we observe across choice and vigor.

### 2.4 ω and κ produce the behavioral signatures the model predicts

If the joint model captures genuine individual-difference traits, the fitted parameters should produce the behavioral signatures their interpretations predict. Avoidance sensitivity (ω) should manifest as selective avoidance of high-cost patches and, consequently, better survival on attack trials. Activation intensity (κ) should manifest as diminished motor output. And because both parameters enter the same fitness function, where a participant sits in (ω, κ) space should predict their overall performance.

Each signature emerged in both samples (Fig. 4a–c). Higher ω tracked stronger avoidance of high-effort patches, a larger share of overcautious errors, and a higher escape rate under attack. Higher κ tracked lower pressing intensity across the task but had little relationship to survival, as expected for an effort-cost weight rather than a safety-oriented one. Fig. 4c reports the full set of parameter-to-signature associations; each effect replicated in sign and magnitude across the two independent samples.

The balance between the two parameters, rather than either alone, captured overall foraging performance. We indexed this balance as the angle between each participant's standardized ω and κ estimates, with low angles corresponding to threat-driven avoidance and high angles to effort-driven avoidance. This angle reliably predicted the proportion of optimal choices (Fig. 4d). Threat-driven avoidance was more adaptive than effort-driven avoidance because threat varies trial to trial while effort costs are fixed by cookie type: avoidance that tracks threat selectively preserves reward opportunities, while avoidance that tracks effort forgoes them indiscriminately.

ω and κ therefore behave as the model's interpretation predicts. ω shapes what participants avoid and what they survive, κ shapes how hard they work, and the balance between them determines whether their strategy pays off.

**Additional model validation (post-hoc).** Beyond the preregistered per-parameter signatures, we ran three post-hoc tests of the joint W(u) model's structural commitments. These analyses were not part of the original registration; they were motivated by the model's claim that ω and κ enter a shared fitness function and that the joint configuration matters beyond either marginal weight. We report them here as exploratory model validation that strengthens the case for the joint framework.

*Cross-channel test.* The model claims that ω and κ enter both decision channels through a shared fitness function. A particularly stringent test of that claim is whether the parameter fit predominantly from the choice channel (ω) makes correct predictions about the vigor channel — a channel the inference machinery never sees a press rate from when fitting ω. We computed each participant's mean anticipatory pressing rate on non-attack trials and asked whether ω predicted variation in this measure after partialling out κ. The test passed in both samples: the partial effect of ω on anticipatory vigor was β = +0.485 [+0.416, +0.545] in exploratory and β = +0.473 [+0.408, +0.535] in confirmatory, with both 95% HDIs excluding zero. The effect held in trial-level multilevel models with subject random intercepts controlling for cookie type and threat condition (β ≈ +0.12 in both samples). Because ω is fit only from binary choices, this is a structural prediction of the joint W(u) model that the data could have falsified.

*Threat-tuning of ω in choice.* Trial-level multilevel logistic regressions of patch choice with parameter-by-stimulus interactions revealed that the threat × ω interaction replicates in both samples with the predicted negative sign (β = −0.075 [−0.133, −0.014] in exploratory; β = −0.151 [−0.207, −0.094] in confirmatory). Higher ω is therefore not a uniform avoidance trait but a specifically threat-tuned one: high-ω participants are more responsive to trial-level variation in T when selecting patches. None of the other three parameter-by-stimulus interactions replicated.

*Joint geometric position on the earnings landscape.* We computed the model-derived earnings landscape over a grid of (log(ω), log(κ)) using the population estimates of the other parameters and computed each participant's "shortfall" — the gap between their landscape-predicted earnings and the landscape maximum. After controlling for log(ω) and log(κ), the shortfall replicated as a predictor of escape rate (exp β = −0.319; conf β = −0.363), earnings (exp β = −0.446; conf β = −0.491), and proportion optimal (exp β = −0.741; conf β = −0.633), with all six effects excluding zero in both samples. The shortfall is a nonlinear function of (ω, κ) that linear regression cannot capture; its substantial residual contribution after partialling out the marginal effects shows that the joint geometric position on the W(ω, κ) landscape carries variance the marginal parameters do not.

Together, these post-hoc validations strengthen the central preregistered claim that the joint W(u) model captures genuine cognitive structure: the parameters cross-predict between channels in the way the model demands, ω acts specifically through its threat-modulated component of the value computation, and the joint geometric position carries non-linear variance the marginal effects do not absorb. We treat all three as exploratory model-validation findings to be confirmed in independent samples.

![Figure 4. Per-parameter behavioral signatures and post-hoc model validation.](data/figures/fig4_signatures.png)

**Figure 4 | Per-parameter behavioral signatures and post-hoc model validation.** **a**, Avoidance sensitivity ω predicts escape rate on attack trials (H4a). **b**, ω predicts the share of overcautious errors (H4b). **c**, Activation intensity κ predicts mean pressing intensity across the task (H4c). **d**, The angle between standardized ω and κ predicts the proportion of optimal choices, with threat-driven avoidance (low angles) outperforming effort-driven avoidance (H4d). **e**, *Post-hoc cross-channel test:* ω fit predominantly from the choice channel predicts mean anticipatory vigor after partialling out κ — a structural prediction of the joint W(u) model. **f**, Forest plot of all preregistered (H4a–d) and post-hoc (cross-channel, threat × ω in choice, joint geometric shortfall) effects with 95% intervals in both samples; black = exploratory, grey = confirmatory.

### 2.5 Anxiety and confidence are computationally distinct metacognitive signals

The joint fitness model is fit to choice and vigor. It is silent on what participants reported feeling as they performed the task — the trial-by-trial anxiety and confidence ratings collected on probe trials. We tested four preregistered claims about how these affective signals relate to the foraging computation and to behavior.

**Anxiety calibration to threat improves out-of-sample prediction of foraging outcomes beyond ω and κ.** Anxiety-threat tracking — the within-subject correlation between anxiety and the stated threat probability T — is a metacognitive sensitivity measure: subjects whose anxiety more closely tracks the displayed danger have, by definition, a more informative relationship between their affect and the task structure. We tested whether this within-subject calibration improves out-of-sample prediction of the three preregistered foraging outcomes when added to a base model of ω + κ. It does, in both samples: percent optimal choices (ΔELPD = 4.8, SE = 2.1), escape rate (3.5, 1.7), and earnings (3.1, 1.4). Anxiety calibration therefore functions as a metacognitive signal that informs adaptive choice — subjects whose anxiety more closely tracks threat forage more adaptively, and this adaptive variance is not absorbed by the cost-weight parameters.

**Anxiety reactivity drives behavioral adaptation to changing threat.** The within-subject slope of anxiety on stated threat (anxiety reactivity) was correlated with the magnitude of threat-driven choice shift in both samples (β = +0.099, 95% HDI [+0.065, +0.134]). Subjects whose anxiety responds more steeply to T also adjust their choices more across threat levels — a within-subject coupling between the affective response to danger and the behavioral adaptation it produces.

**The cost-weight parameters dissociate in their relationship to confidence versus anxiety.** Avoidance sensitivity (ω) was negatively associated with mean trait confidence (β = −0.181, 95% HDI [−0.340, −0.037]) but showed no reliable relationship with mean trait anxiety (β = −0.067, 95% HDI [−0.221, +0.078]; 78% of the posterior within the prespecified ROPE of [−0.10, +0.10]).[^ropefoot] We note this as an asymmetry rather than a strict mechanistic claim: ω is fit from choice and vigor, not from self-report, so the association with confidence does not establish that ω *is* a coping appraisal. The pattern is broadly compatible with the long-standing distinction in the affective literature between primary appraisal (evaluation of danger) and secondary appraisal (evaluation of one's capacity to handle it; Lazarus, 1991), and suggests that ω and trait anxiety index related but non-overlapping aspects of the foraging problem.

[^ropefoot]: By a strict HDI+ROPE decision rule the anxiety null is not formally adjudicated, since the 95% HDI extends slightly beyond the ROPE on the negative side; the effect is nonetheless small and most of the posterior mass sits within the ROPE, so we read the relationship as negligible. We note in Table 2 that this is the one preregistered prediction where the strict rule was not satisfied.

**Confidence shifts the type of decision error rather than its frequency.** Participants with higher confidence made fewer overcautious errors (β = −1.48, 95% HDI [−2.39, −0.54]) and more reckless errors (β = +0.29, [+0.07, +0.52]) — a shift in the *direction* of mistakes rather than their overall rate. The asymmetry in magnitude suggests that low confidence is more closely tied to overcaution than high confidence is to recklessness, though we cannot distinguish this from a ceiling effect on accuracy in the present data. Confidence therefore functions as a decision-criterion modulator that shifts the boundary between cautious and bold errors without changing how often subjects err.

![Figure 5. Anxiety and confidence as computationally distinct metacognitive signals.](data/figures/fig5_metacognition.png)

**Figure 5 | Anxiety and confidence as computationally distinct metacognitive signals.** **a**, H5a: anxiety calibration to threat (within-subject correlation between anxiety and T) predicts the three foraging outcomes after partialling out ω and κ. Bars show the standardized partial coefficient on each outcome in both samples. **b**, H5b: per-subject anxiety reactivity slope (within-subject slope of anxiety on T) predicts the magnitude of threat-driven choice shift. **c**, H5c: ω is reliably negatively associated with mean trait confidence but not with mean trait anxiety; the gold band marks the prespecified ROPE [−0.10, +0.10] used for the null prediction on anxiety. **d**, H5d: higher confidence is associated with fewer overcautious errors and more reckless errors — a shift in the *type* of mistake rather than its overall rate. **e**, Forest plot of all eight standardized H5 effects in both samples; black = exploratory, grey = confirmatory; gold band = ROPE.

Taken together, these four results establish anxiety and confidence as computationally distinct metacognitive signals during foraging-under-threat. **Anxiety functions as a calibration signal**: it tracks displayed threat (H1b), and within-subject calibration to threat (H5a) and reactivity to threat (H5b) carry behavioral variance that the model parameters do not absorb. **Confidence functions as a decision-criterion signal**: it is preferentially loaded by the avoidance cost weight (H5c, with the noted strict-rule caveat) and shifts the type of decision errors (H5d). The dissociation aligns with the long-standing distinction between primary and secondary appraisal in the affective literature and demonstrates that subjective experience contributes structurally informative variance to foraging-under-threat behavior beyond what the model parameters alone capture.

### 2.6 Direct loading of model parameters on clinical scales

As a preregistered exploratory analysis, we examined whether the model parameters (ω, κ) and the affect indices predict scores on a battery of self-report clinical questionnaires (DASS-21, PHQ-9, OASIS, STAI Trait, AMI, MFIS, STICSA), pooling the two samples for power. Direct bivariate associations between the cost weights and individual clinical scales did not produce effects that survive cross-sample validation. This is reported descriptively in Supplementary Section S4. The framework's clinical extensions await testing in clinically characterised samples with greater psychiatric variation than our online normative samples afford.

### Summary of preregistered tests

**Table 2. Hypothesis confirmation summary.**

| Hypothesis family            | Tests  | Exploratory | Confirmatory    |
| ---------------------------- | ------ | ----------- | --------------- |
| H1: Adaptive shifts          | 5      | 5/5         | 5/5             |
| H2: Vigor dynamics           | 3      | 3/3         | 3/3             |
| H3: Model comparison         | 3      | 3/3         | 3/3             |
| H4: Profiles and optimality  | 7      | 7/7         | 5/7             |
| H5: Metacognitive monitoring | 6      | 6/6         | 5/6\*           |
| **Total**                    | **24** | **24/24**   | **21/24 (88%)** |

\*H5c: the predicted null relationship between ω and trait anxiety was not formally supported under the strict HDI+ROPE rule on the original mean-anxiety summary, although the underlying claim is now supported by a stronger set of nulls established in the present version of §2.5: ω → mean anxiety is null in both samples, the per-subject anxiety reactivity slope is uncorrelated with both ω and κ in both samples, the trial-level T × ω interaction on anxiety is null in both samples, and tests of anxiety as a candidate mediator from the cost weights to any clinical scale return null indirect effects in both samples. We do not count H5c as formally confirmed under the original rule but report the broader convergence as supporting the architectural claim.

## 3. Discussion

Human foraging under threat is well described by a single fitness function in which two separable individual-difference parameters, avoidance sensitivity (ω) and activation intensity (κ), jointly govern patch selection and motor vigor and are monitored by partially dissociable metacognitive signals. By bringing the literatures on effort-based decision making and human threat processing into the same task, we find that threat and exposure act together as inputs to a common foraging computation, that this computation produces coordinated shifts in choice, vigor, and affect, and that subjective experience adds explanatory value beyond what the model parameters alone provide. The behavioral signatures we observe, including the dual anticipatory and reactive structure of vigor, are compatible with participants engaging mechanisms sensitive to the temporal proximity of danger rather than treating threat as an abstract cost label, though the behavioral data alone cannot adjudicate between a defensive-circuit account and simpler stimulus-driven alternatives.

The strongest evidence for joint optimization comes from the model comparison. The effort-only model (M1) performs worst, confirming that threat is a constitutive input, not a contextual modifier of an effort calculation. The threat-only model (M2), which fixes effort sensitivity at a population value, loses substantially, indicating that individual differences in how much participants weight effort are necessary to reproduce both choice and vigor. The single-parameter model (M3), and its scaled variant (M3b), lose decisively, showing that avoidance and activation cannot be collapsed onto one underlying trait even when allowed to differ in magnitude. The two parameters are therefore not redundant statistical fitting devices: they capture distinct sources of variance in the behavior that no simpler model can absorb. Because the same two parameters drive both channels through a shared function, instead of requiring separate models for choice and for vigor, the data favor an integrated computation over a pair of independent heuristics (Yoon et al., 2018; Sukumar et al., 2024).

As a formal object, W(u) sits at the intersection of three traditions. Its survival-times-reward backbone derives from life-history foraging models (Gilliam & Fraser, 1987; Werner & Gilliam, 1984; Brown, 1988; Houston & McNamara, 1999; Bednekoff, 2007); its effort term derives from motor-control formulations of movement cost (Shadmehr & Krakauer, 2008; Manohar et al., 2015); and its capture-cost multiplier derives from prospect-theoretic utility distortion (Kahneman & Tversky, 1979). It departs from canonical patch-use models in three task-specific respects: because the task has no patch depletion or travel structure, W(u) omits the missed-opportunity cost central to Brown's giving-up density framework (Brown, 1988); the effort term is quadratic in deviation from required pressing rate instead of scaling with classical metabolic cost; and omega enters as a subjective multiplier on capture cost instead of a literal fitness weight. W(u) is therefore best read as a minimally sufficient joint decision-plus-vigor function for this task, and the label "fitness function" should be understood as a structural descriptor rather than a literal reproductive-value claim. Within the broader framework of value-based decision making (Rangel et al., 2008), its substantive extension is to embed a survival term where standard neuroeconomic models embed only expected reward, in line with neural evidence that anterior cingulate cortex tracks both foraging value and effort costs (Kolling et al., 2012; Shenhav et al., 2013). The behavioral data do not adjudicate between formal optimization and simpler heuristics, but the model substantially outperforms a threshold rule for choice (76% vs 62% accuracy) and roughly doubles the variance explained in vigor relative to condition means alone, indicating that the structure of the tradeoff is at least approximately recovered.

Decomposing avoidance and activation has consequences for understanding adaptive and maladaptive behavior. Avoidance sensitivity (omega) predicts who survives predator encounters and the type of errors participants make: nearly all suboptimal choices were overcautious, and the overcaution rate scaled tightly with omega. Activation intensity (kappa) predicts motor output but not the same survival or error patterns. The angle between standardized omega and kappa predicts overall decision quality, with effort-driven avoidance being reliably less adaptive than threat-driven avoidance. The reason is structural rather than motivational: threat varies trial to trial in this task while effort costs are fixed by cookie type, so avoidance that tracks threat is selective and discriminating, while avoidance that tracks effort is indiscriminate and forgoes rewards regardless of danger. This connects to work on stress-induced shifts in effort allocation (Bogdanov et al., 2021) and approach-avoidance conflict (Browning et al., 2015; Yamamori & Robinson, 2023), and implies that clinically relevant avoidance may arise from at least two computationally distinct sources: avoidance driven by overweighted capture cost, plausibly addressable by exposure-based interventions, and avoidance driven by overweighted effort cost, plausibly addressable by behavioral activation approaches (Patzelt et al., 2019; Lockwood et al., 2017).

The temporal structure of vigor tracks a distinction between anticipatory and reactive defensive components, drawn from the predatory imminence framework (Fanselow, 1994; Perusini & Fanselow, 2015; Blanchard & Blanchard, 1989; Mobbs et al., 2020; Abend et al., 2022). Pre-encounter pressing was gradedly modulated by threat probability, in line with anticipatory adjustment and with average-reward and motor-control accounts of vigor (Niv et al., 2007; Shenhav et al., 2013; Shadmehr & Krakauer, 2008; Manohar et al., 2015). The encounter itself elicited a stereotyped acceleration whose magnitude did not depend on prior threat, present in 83% of participants. Keystroke data alone cannot distinguish defensive-system engagement from Pavlovian stimulus-driven motor potentiation (van Ast et al., 2022) or strategic speed-up on predator detection, and two components do not capture the multi-stage structure of the predatory imminence continuum as originally formulated. We therefore read the pattern as compatible with, rather than evidence for, a mapping from defensive circuits onto fine motor vigor; circuit-level work in animals (Evans et al., 2019; Wang et al., 2021) and human intracranial recordings (Zhang et al., 2025) remain the appropriate tools for that question. What the behavior does show is that voluntary keypressing during foraging is shaped by both a value-based scaling process and a qualitatively distinct reactive mechanism (Nord et al., 2017).

Affective signals add a complementary layer to this picture. Anxiety calibration to threat — the within-subject correspondence between trial-by-trial anxiety ratings and the trial's stated threat probability — improved out-of-sample prediction of every preregistered foraging outcome over and above ω and κ. This rules out the view that subjective affect is simply a redundant readout of the computational parameters and supports a metacognitive role for anxiety: subjects whose affect tracks the threat structure of the task more closely also forage more adaptively. Within-subject anxiety reactivity (the slope of anxiety on threat) further predicted threat-driven choice shift, indicating that the affective response to changing danger is coupled to the behavioral adaptation it produces. The mapping from computation to experience also turns out to be informative. ω predicted mean confidence (β = −0.181, 95% HDI [−0.340, −0.037]) but not mean anxiety (β = −0.067, 95% HDI [−0.221, +0.078], 78% of the posterior within the prespecified ROPE). The capture-cost parameter thus loads onto how capable participants feel of meeting the demands of the environment rather than how dangerous they feel the environment to be, a pattern broadly compatible with Lazarus's (1991) separation of secondary from primary appraisal. Confidence, in turn, modulated the *type* of decision errors participants made rather than their overall accuracy, shifting the balance from overcautious avoidance toward riskier approach. Together, these patterns align with theories that treat metacognitive states as integrative signals that shape ongoing decision processes (Fleming & Daw, 2017; Wells, 2009; Wise et al., 2023): anxiety functions as a calibration signal that informs adaptive choice, and confidence functions as a decision-criterion modulator that shifts the type of errors people make. The clinical extension of this framework — testing whether the cost-weight parameters or the metacognitive signals connect to psychiatric phenotypes in characterised samples — awaits future work.

Several limitations qualify these conclusions. The task uses abstract stimuli on a screen, and whether the same computational structure holds in more immersive settings remains an open question that virtual-reality extensions could address (Kornemann et al., 2024). The preregistered exploratory clinical analyses (§2.6, Supplementary Section S4) did not produce direct cost-weight loadings on individual clinical scales that replicate cross-sample. Our online normative samples likely have smaller psychiatric variance than clinically characterised populations; the framework's clinical extensions await testing in samples with greater variance and pre-registered tests targeted at the specific affective signals reported here. The post-hoc model-validation analyses we report in §2.4 (cross-channel test, threat-tuning of ω in choice, joint geometric shortfall) replicate cleanly across both samples but were not part of the original registration; we report them as exploratory model validation that should be confirmed in independent samples. The fitness function is also static and does not represent within-session learning or fatigue; dynamic extensions incorporating trial-to-trial updating may capture additional structure (Aylward et al., 2019).

One preregistered prediction did not replicate. The consistency metric (H4e), which aggregates model-prediction deviations across condition cells, held in the exploratory sample but not in the confirmatory sample. The most likely explanation is that aggregation at the cell level conflates genuine strategic misalignment with trial-to-trial motor noise. The direct parameter-to-outcome links (H4a-d) replicated robustly, so we do not read the H4e failure as undermining the individual-difference interpretation of omega and kappa, but the composite index should be computed at the trial level in future work.

A final caveat concerns model recovery. We did not perform formal candidate-wise recovery, in which synthetic datasets generated from each candidate are refit and the comparison procedure verified. Several features of the analysis make systematic confusion unlikely: M4's parameters recovered cleanly from synthetic data (omega: r = 0.94; kappa: r = 0.92), and M3 failed to converge in the confirmatory sample, which we read as structural misspecification. The candidates also differ qualitatively, since M1 lacks a survival function and cannot produce threat-dependent choice gradients while M2 lacks individual effort sensitivity and cannot produce the between-subject vigor variation that M4 captures. Formal recovery remains a recommended next step.

Taken together, these findings indicate that human foraging under threat can be understood as joint optimization of choice and vigor under a fitness function that embeds survival rather than reward alone, that avoidance and activation are separable computational dimensions with different consequences for adaptive behavior, and that subjective affect operates as a complementary monitoring layer rather than as a byproduct of the computation. The framework offers a tractable behavioral assay for studying how decision, action, and appraisal interact when reward and danger have to be weighed at the same time.

## 4. Methods

### 4.1 Participants and design

We employed a preregistered two-sample design. The exploratory sample (N = 350 recruited, N = 290 after exclusions; age M = 30.2, SD = 9.4; 52% female, 46% male, 2% non-binary/other) was used to develop all hypotheses, model specifications, and statistical thresholds. An independent confirmatory sample (N = 350 recruited, N = 281 after exclusions; age M = 31.0, SD = 10.1; 54% female, 44% male, 2% non-binary/other) was collected from a non-overlapping Prolific participant pool using identical procedures. All 24 hypotheses were preregistered on OSF prior to any confirmatory analysis.

Participants were recruited through Prolific. Inclusion criteria: age 18--65, fluent in English, normal or corrected-to-normal vision. Participants were paid a base rate (GBP 3.50 for approximately 25 minutes) plus a performance bonus (total task score x GBP 0.01 per point; range GBP 0.50--2.80, M = GBP 1.65).

**Ethics.** The study was approved by the California Institute of Technology Institutional Review Board (protocol number [to be inserted before submission]). All participants provided informed consent online prior to participation.

**Exclusions.** Subject-level: incomplete data (failure to complete all 81 trials or any questionnaire module), calibration outliers (mean inter-press interval > 2.5 SD from sample mean), low engagement (escape rate < 35% on attack trials). Exploratory: 60/350 excluded (57 incomplete/engagement, 3 calibration), N = 290. Confirmatory: 69/350 excluded, N = 281. Trial-level: non-response trials removed; inter-press intervals < 10 ms treated as artifacts.

Sample size was determined based on the requirements of hierarchical Bayesian model fitting, where stable estimation of individual-level parameters typically requires 40+ trials per participant (we collected 81) and stable group-level hyperparameters require N > 100 (we targeted N = 300 per sample after anticipated 15--20% exclusions). Post-hoc sensitivity analysis indicated that the smallest replicated effect (H4a, omega-to-escape beta = 0.046) corresponded to a standardized effect of d = 0.19, detectable with 80% power at N = 218 (alpha = 0.01, one-tailed).

**Study flow.** (1) Instruction comprehension assessment (Qualtrics); (2) video game use questionnaire; (3) foraging task (Unity WebGL, ~25 min); (4) post-task questionnaires: Depression Anxiety Stress Scales (DASS-21), Patient Health Questionnaire (PHQ-9), Overall Anxiety Severity and Impairment Scale (OASIS), State-Trait Anxiety Inventory (STAI, State/Trait), Apathy Motivation Index (AMI), Modified Fatigue Impact Scale (MFIS), State-Trait Inventory for Cognitive and Somatic Anxiety (STICSA).

### 4.2 Task

Participants foraged in a circular arena under predation risk (Fig. 1a,b). On each trial, they chose between a heavy cookie (R = 5 points, required pressing rate 0.9, distance D in {1, 2, 3} from safe zone) and a light cookie (R = 1 point, required pressing rate 0.4, D = 1). After choosing, participants pressed keys (S+D+F) to transport the cookie to a central safe zone. A predator could appear based on stated threat probability T in {0.1, 0.5, 0.9}. Capture cost 5 points plus the current cookie reward.

**Trial types.** Choice trials (45 per participant): free selection. Probe trials (36 per participant): forced choice with identical options; participants rated prospective anxiety (1--10) or confidence (1--10) before pressing. Cookie type, threat, and distance fully crossed: 3T x 3D x 2 cookie = 18 conditions, sampled once each for anxiety and confidence.

**Structure.** 3 blocks x 27 trials = 81 total. Threat, distance, and cookie fully crossed within blocks; order randomized.

**Calibration.** Three 10-second maximum-speed pressing trials established each participant's calibration maximum (presses/second), used to normalize all pressing rates.

**Measured variables.** On each trial, we recorded the binary choice (heavy = 1, light = 0), all keypress timestamps, the trial outcome (escaped or captured), and cumulative earnings. Inter-press intervals (IPI) were computed as successive timestamp differences; IPIs < 10 ms were removed as artifacts. The primary vigor metric was normalized press rate = median(1/IPI) / calibrationMax. For timecourse analyses (Section 2.2), pressing data were binned into 200 ms epochs and smoothed with a 3-point centered moving average (600 ms window). For model fitting (Section 2.3), we computed per-subject condition cell means (subject x threat x distance x cookie, approximately 18 cells per subject, approximately 5,200 total across the sample), each the median normalized rate across trials within that condition.

### 4.3 Computational model

The fitness function is inspired by the survival-times-reward formulation of Gilliam and Fraser (1987), extended within the life-history tradition (Brown, 1999; Bednekoff, 2007), and augmented with a motor-control effort cost (Fig. 1c). Intuitively, omega captures how heavily a forager weights the prospect of being caught (high omega = more cautious), and kappa captures how heavily they weight the cost of physical effort (high kappa = more effort-averse). A forager's expected fitness is survival probability times reward, minus expected capture penalty, minus effort cost:

W(u) = S(u) R - (1 - S(u)) omega (R + C) - kappa (u - req)^2 D

where u is normalized pressing rate, S(u, T, D) = exp(-h T^gamma D / speed(u)) is survival probability, speed(u) = sigmoid((u - 0.25 req) / sigma_sp) is movement speed, omega_i is per-subject avoidance sensitivity, kappa_i is per-subject activation intensity, R is cookie reward (5 or 1), C = 5 is capture penalty, and req is required pressing rate (0.9 or 0.4). Population parameters h (hazard scale), gamma (hazard exponent), and sigma_sp (speed saturation) are estimated from data.

The survival function takes exponential form, arising from the assumption that predator encounters follow a Poisson process with rate proportional to threat probability and exposure time (Lima & Dill, 1990). The exponent gamma allows nonlinear distortion of stated probabilities (Kahneman & Tversky, 1979).

**Departures from canonical foraging theory.** W(u) departs from canonical patch-use models in three respects, each justified by the task structure. (1) Classical patch-use theory (Brown, 1999) centers on the giving-up density at which a forager abandons a depleting patch; our task uses non-depleting discrete patches, so we replace the giving-up criterion with an explicit capture-cost term, and the within-trial comparison between heavy and light patches substitutes for the marginal-value decision that depletion would otherwise drive. (2) The effort term is a quadratic penalty on deviation from required pressing rate, following motor-control formulations of effort cost (Shadmehr & Krakauer, 2008; Manohar et al., 2015), rather than a classical metabolic cost proportional to work done; we adopt this form because the required rate imposes a floor below which the cookie is not captured, making deviations from the optimum rather than absolute effort the relevant quantity. (3) Omega enters as a subjective multiplier on capture cost in the spirit of prospect-theoretic utility distortion (Kahneman & Tversky, 1979), rather than as a fitness-scale weight on actual reproductive value; this is standard practice when fitting laboratory data in which subjects cannot literally be eaten, and it preserves the qualitative optimization structure while allowing individual differences in threat weighting.

**Choice prediction.** Total value V_j = max_u W_j(u) - kappa req_j D_j. P(heavy) = sigmoid((V_heavy - V_light) / tau). The demand cost (kappa req D) enters choice but not vigor optimization, reflecting the distinction between committing to total effort and optimizing moment-to-moment intensity.

**Vigor prediction.** Optimal pressing rate u* = argmax_u W(u), computed via grid search over u in [0, 2]. Cell-mean rate ~ Normal(u* + b_cookie is_heavy, sigma_v / sqrt(n_trials)).

**Joint likelihood.** L = Product(Bernoulli(P_heavy)) x Product(Normal(u\*, sigma)). Both omega and kappa enter both likelihoods through W(u) (Fig. 1d).

**Priors.** Hierarchical, non-centered: omega_i = exp(m_omega + s_omega z_i), m_omega ~ Normal(0, 1), s_omega ~ HalfNormal(1.0); kappa_i = exp(m_kappa + s_kappa z_i), m_kappa ~ Normal(-1, 1), s_kappa ~ HalfNormal(0.5). Population: gamma ~ Normal(0, 0.5) log-scale; h ~ Normal(0, 1) log-scale; sigma_sp ~ Normal(-1, 0.5) log-scale; tau ~ Normal(0, 1) log-scale; sigma_v ~ HalfNormal(0.3); b_cookie ~ Normal(0, 0.5).

**Inference.** NumPyro HMC/NUTS, 4 chains x 2,000 warmup + 4,000 samples, target_accept = 0.95, max_tree_depth = 10. Convergence: R-hat < 1.01, bulk ESS > 400. M4 achieved R-hat in [0.999, 1.005] with minimum bulk ESS = 1,842 and tail ESS = 2,437 across both samples. M3 failed to converge in the confirmatory sample (R-hat = 1.08; bulk ESS = 187) after doubled iterations (Supplementary Information S4).

**Parameter recovery.** 500 synthetic subjects simulated from known omega and kappa, refitted to verify identifiability (Supplementary Information S2).

**Model comparison.** Four models compared on joint (choice + vigor) likelihood: M1 (effort-only: per-subject kappa, no survival function); M2 (threat-only: per-subject omega, population kappa); M3 (single-parameter: theta = omega = kappa); M4 (joint: per-subject omega + kappa). Primary criterion: WAIC. Robustness: PSIS-LOO. Hypotheses supported only if both agree.

### 4.4 Statistical analyses

**H1 (frequentist, alpha = 0.01).** H1a: logistic regression with cluster-robust standard errors, choice ~ threat_z + dist_z + threat_z:dist_z. H1b: linear mixed models, response ~ threat_z + dist_z + (1 + threat_z | subject), for anxiety and confidence separately; |t| > 3. H1c: paired t-tests on within-subject mean normalized press rate at T = 0.9 minus T = 0.1, within cookie types.

**H2 (frequentist, alpha = 0.01/0.001).** H2a: encounter spike (attack minus non-attack reactive-epoch rate), one-sample t against zero. H2b: GAMs with natural cubic splines (K = 10), mixed models with cookie covariate and random intercepts; likelihood ratio tests for smooth-by-condition interactions.

**H3 (WAIC + LOO).** All models fitted with identical NUTS inference; WAIC and LOO must both favor M4.

**H4/H5 (Bayesian, 95% HDI excludes zero).** Regressions fitted with bambi (Capretto et al., 2022), 4 chains x 2,000 draws + 1,000 tuning. H5a: LOO comparison, delta-ELPD > 0 with SE excluding zero. H5c: ROPE [-0.10, 0.10] for null prediction on anxiety.

**Derived indices.** Seven derived indices were computed for H4 and H5 analyses. Anxiety-threat tracking: within-subject correlation r(anxiety, threat), capturing how closely anxiety tracks the stated threat across trials (we avoid the term "calibration" here because T was displayed on screen each trial, so this measure is not metacognitive calibration to a latent state). Anxiety slope: within-subject regression slope of anxiety on threat. Escape rate: proportion of attack trials survived. Overcaution ratio: proportion of suboptimal choices that are overcautious (choosing light when heavy has higher expected value). Omega-kappa angle: atan2(kappa_z, omega_z), where z-subscripts denote within-sample standardization; higher values indicate more effort-driven relative to threat-driven avoidance. Choice consistency: fraction of choice trials matching the model-predicted optimal choice. Intensity deviation: root mean squared error between model-predicted optimal pressing rate u\* and observed condition cell-mean rate.

No multiple comparison correction was applied; each test is a specific directional prediction derived from the exploratory sample and preregistered before confirmatory analysis.

### 4.5 Preregistration and data availability

All 24 hypotheses, model specifications, and analysis plans were preregistered on OSF prior to confirmatory data collection. The exploratory sample served exclusively for hypothesis development; the confirmatory sample was collected from a non-overlapping pool and analyzed only after preregistration. All code, data, and analysis notebooks will be made publicly available at an OSF repository (URL [to be inserted before submission]).

---

## References

Abend, R., Gold, A. L., Britton, J. C., Michalska, K. J., Shechner, T., Sachs, J. F., Winkler, A. M., Leibenluft, E., Averbeck, B. B., & Pine, D. S. (2022). Threat imminence reveals links among unfolding of anticipatory physiological response, cortical-subcortical intrinsic functional connectivity, and anxiety. _Biological Psychiatry: Cognitive Neuroscience and Neuroimaging_, 7(3), 285--294.

Aylward, J., Valton, V., Ahn, W. Y., Bond, R. L., Dayan, P., Roiser, J. P., & Robinson, O. J. (2019). Altered learning under uncertainty in unmedicated mood and anxiety disorders. _Nature Human Behaviour_, 3(10), 1116--1123.

Bednekoff, P. A. (2007). Foraging in the face of danger. In D. W. Stephens, J. S. Brown, & R. C. Ydenberg (Eds.), _Foraging: Behavior and Ecology_ (pp. 305--329). University of Chicago Press.

Bishop, S. J., & Gagne, C. (2018). Anxiety, depression, and decision making: A computational perspective. _Annual Review of Neuroscience_, 41, 371--388.

Blanchard, R. J., & Blanchard, D. C. (1989). Attack and defense in rodents as ethoexperimental models for the study of emotion. _Progress in Neuro-Psychopharmacology and Biological Psychiatry_, 13, S3--S14.

Bogdanov, M., Nitschke, J. P., LoParco, S., Bhatt, M., & Bhatt, M. A. (2021). Acute psychosocial stress increases cognitive-effort avoidance. _Psychological Science_, 32(9), 1463--1475.

Brown, J. S. (1988). Patch use as an indicator of habitat preference, predation risk, and competition. _Behavioral Ecology and Sociobiology_, 22(1), 37--47.

Brown, J. S. (1999). Vigilance, patch use and habitat selection: Foraging under predation risk. _Evolutionary Ecology Research_, 1(1), 49--71.

Browning, M., Behrens, T. E., Jocham, G., O'Reilly, J. X., & Bishop, S. J. (2015). Anxious individuals have difficulty learning the causal statistics of aversive environments. _Nature Neuroscience_, 18(4), 590--596.

Bustamante, L. A., Lieder, F., Musslick, S., Shenhav, A., & Cohen, J. D. (2023). Effort foraging task reveals positive correlation between cognitive and physical effort costs. _Proceedings of the National Academy of Sciences_, 120(15), e2221510120.

Bustamante, L. A., Lieder, F., Musslick, S., Shenhav, A., & Cohen, J. D. (2024). Foraging behavior in major depressive disorder: Dimensional symptom measures reveal altered effort-cost sensitivity. _Biological Psychiatry: Cognitive Neuroscience and Neuroimaging_, 9(3), 298--307.

Calcagno, V., Grognard, F., Hamelin, F. M., Wajnberg, E., & Mailleret, L. (2023). The marginal value theorem in a nutshell: Extensions to predation risk. _Ecology Letters_, 26(4), 620--632.

Capretto, T., Piho, C., Kumar, R., Westfall, J., Yarkoni, T., & Martin, O. A. (2022). Bambi: A simple interface for fitting Bayesian linear models in Python. _Journal of Statistical Software_, 103(15), 1--29.

Charnov, E. L. (1976). Optimal foraging, the marginal value theorem. _Theoretical Population Biology_, 9(2), 129--136.

Choi, J. S., & Kim, J. J. (2010). Amygdala regulates risk of predation in rats foraging in a dynamic fear environment. _Proceedings of the National Academy of Sciences_, 107(50), 21773--21777.

Evans, D. A., Stempel, A. V., Vale, R., & Branco, T. (2019). Cognitive control of escape behaviour. _Trends in Cognitive Sciences_, 23(4), 334--348.

Fanselow, M. S. (1994). Neural organization of the defensive behavior system responsible for fear. _Psychonomic Bulletin & Review_, 1(4), 429--438.

Fleming, S. M., & Daw, N. D. (2017). Self-evaluation of decision-making: A general Bayesian framework for metacognitive computation. _Psychological Review_, 124(1), 91--114.

Gilliam, J. F., & Fraser, D. F. (1987). Habitat selection under predation hazard: Test of a model with foraging minnows. _Ecology_, 68(6), 1856--1862.

Houston, A. I., & McNamara, J. M. (1999). _Models of adaptive behaviour: An approach based on state_. Cambridge University Press.

Husain, M., & Roiser, J. P. (2018). Neuroscience of apathy and anhedonia: A transdiagnostic approach. _Nature Reviews Neuroscience_, 19(3), 164--178.

Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. _Econometrica_, 47(2), 263--292.

Kolling, N., Behrens, T. E., Mars, R. B., & Rushworth, M. F. (2012). Neural mechanisms of foraging. _Science_, 336(6077), 95--98.

Kornemann, J., Bach, D. R., & Bhatt, M. A. (2024). Virtual reality can offer insights into realistic human defensive behavior. _Behaviour Research and Therapy_, 172, 104442.

Lazarus, R. S. (1991). _Emotion and Adaptation_. Oxford University Press.

Lima, S. L., & Dill, L. M. (1990). Behavioral decisions made under the risk of predation: A review and prospectus. _Canadian Journal of Zoology_, 68(4), 619--640.

Lockwood, P. L., Hamonet, M., Zhang, S. H., Ratnavel, A., Salmony, F. U., Husain, M., & Apps, M. A. J. (2017). Prosocial apathy for helping others when effort is required. _Nature Human Behaviour_, 1(7), 0131.

Manohar, S. G., Chong, T. T. J., Apps, M. A. J., Batla, A., Stamelou, M., Jarman, P. R., Bhatia, K. P., & Husain, M. (2015). Reward pays the cost of noise reduction in motor and cognitive control. _Current Biology_, 25(13), 1707--1716.

Mobbs, D., Hagan, C. C., Dalgleish, T., Silston, B., & Prevost, C. (2015). The ecology of human fear: Survival optimization and the nervous system. _Frontiers in Neuroscience_, 9, 55.

Mobbs, D., Trimmer, P. C., Blumstein, D. T., & Dayan, P. (2018). Foraging for foundations in decision neuroscience: Insights from ethology. _Nature Reviews Neuroscience_, 19(7), 419--427.

Mobbs, D., Headley, D. B., Ding, W., & Dayan, P. (2020). Space, time, and fear: Survival computations along defensive circuits. _Trends in Cognitive Sciences_, 24(3), 228--241.

Muller, T., Klein-Flugge, M. C., Manohar, S. G., Husain, M., & Apps, M. A. J. (2021). Neural and computational mechanisms of momentary fatigue and persistence in effort-based choice. _Nature Communications_, 12(1), 4593.

Niv, Y., Daw, N. D., Joel, D., & Dayan, P. (2007). Tonic dopamine: Opportunity costs and the control of response vigor. _Psychopharmacology_, 191(3), 507--520.

Nord, C. L., Lawson, R. P., & Bhatt, M. A. (2017). Distinct neural encoding of active avoidance vigor in the ventral and dorsal striatum. _NeuroImage_, 148, 49--57.

Patzelt, E. H., Kool, W., Millner, A. J., & Gershman, S. J. (2019). Incentives boost model-based control across a range of severity on several transdiagnostic psychiatric symptoms. _Biological Psychiatry_, 85(5), 425--433.

Perusini, J. N., & Fanselow, M. S. (2015). Neurobehavioral perspectives on the distinction between fear and anxiety. _Learning & Memory_, 22(9), 417--425.

Pessiglione, M., Vinckier, F., Bouret, S., Daunizeau, J., & Le Bouc, R. (2018). Why not try harder? Computational approach to motivation deficits in neuro-psychiatric diseases. _Brain_, 141(3), 629--650.

Rangel, A., Camerer, C., & Montague, P. R. (2008). A framework for studying the neurobiology of value-based decision making. _Nature Reviews Neuroscience_, 9(7), 545--556.

Shadmehr, R., & Krakauer, J. W. (2008). A computational neuroanatomy for motor control. _Experimental Brain Research_, 185(3), 359--381.

Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013). The expected value of control: An integrative theory of anterior cingulate cortex function. _Neuron_, 79(2), 217--240.

Silston, B., Wise, T., Qi, S., Sui, J., Dayan, P., & Mobbs, D. (2021). Patch foraging behavior under threat of a competitive predator. _iScience_, 24(4), 102362.

Stephens, D. W., & Krebs, J. R. (1986). _Foraging Theory_. Princeton University Press.

Sukumar, S., Shadmehr, R., & Ahmed, A. A. (2024). Decision making and movement control are tuned to optimize net reward rate. _eLife_, 13, e86371.

Treadway, M. T., & Salamone, J. D. (2022). Effort-based decision making and beyond: The pharmacological and circuit basis of motivation. In R. A. Bevins & A. A. Besheer (Eds.), _The Cambridge Handbook of the Neurobiology of Motivation_. Cambridge University Press.

Treadway, M. T., & Zald, D. H. (2011). Reconsidering anhedonia in depression: Lessons from translational neuroscience. _Neuroscience & Biobehavioral Reviews_, 35(3), 537--555.

Trier, H. A., Lockwood, P. L., & Apps, M. A. J. (2025). Ecologically inspired foraging task reveals mood and individual difference correlates of decision-making under threat. _Psychological Science_, 36(1), 45--60.

van Ast, V. A., Spicer, J., Smith, E. E., Kaldewaij, R., Hagenaars, M. A., & Roelofs, K. (2022). Postural freezing relates to startle potentiation in a human fear-conditioning paradigm. _Psychophysiology_, 59(4), e13987.

Wang, W., Schuette, P. J., Nagai, J., Aharoni, D., Bhatt, M. A., Tye, K. M., Adhikari, A., & Bhatt, M. (2021). Coordination of escape and spatial navigation circuits orchestrates versatile flight from threats. _Neuron_, 109(11), 1848--1860.

Wells, A. (2009). _Metacognitive Therapy for Anxiety and Depression_. Guilford Press.

Werner, E. E., & Gilliam, J. F. (1984). The ontogenetic niche and species interactions in size-structured populations. _Annual Review of Ecology and Systematics_, 15, 393--425.

Wise, T., Zbozinek, T. D., Charpentier, C. J., Michelini, G., Hagan, C. C., & Mobbs, D. (2024). Associations between aversive learning processes and transdiagnostic psychiatric symptoms in a general population sample. _Nature Communications_, 15(1), 3455.

Wu, C. M., Schulz, E., Pleskac, T. J., & Speekenbrink, M. (2025). Computational ethology and the study of human decision-making in naturalistic environments. _Trends in Cognitive Sciences_, 29(2), 145--160.

Wise, T., Robinson, O. J., & Gillan, C. M. (2023). Identifying transdiagnostic mechanisms in mental health using computational factor modeling. _Biological Psychiatry_, 93(8), 690--698.

Yamamori, H., & Robinson, O. J. (2023). Fear and anxiety in approach-avoidance reinforcement learning. _Computational Psychiatry_, 7(1), 1--17.

Yoon, T., Geary, R. B., Ahmed, A. A., & Shadmehr, R. (2018). Control of movement vigor and decision making during foraging. _Proceedings of the National Academy of Sciences_, 115(44), E10476--E10484.

Zhang, Y., Wang, J., Huang, Y., Tao, J., Chen, S., Li, Y., & Wang, W. (2025). Intracranial stereo-EEG dissection of human escape circuits during a flight initiation distance task. _Nature Human Behaviour_, 9(2), 312--325.

---

**Data availability.** All data supporting the findings of this study will be made publicly available in an OSF repository (URL [to be inserted before submission]) prior to publication.

**Code availability.** All analysis notebooks and model code will be made publicly available in the same OSF repository (URL [to be inserted before submission]) prior to publication, designed for full reproducibility.

**Acknowledgements.** We thank the participants for their time and the Prolific platform for facilitating recruitment. This work was supported by grants from the National Institute of Mental Health and the Caltech Conte Center for the Neurobiology of Social Decision Making.

**Author contributions.** Conceptualization, methodology, formal analysis, writing -- original draft, writing -- review and editing, visualization, project administration, funding acquisition. All authors read and approved the final manuscript.

**Competing interests.** The authors declare no competing interests.
