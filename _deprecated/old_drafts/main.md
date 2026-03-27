# A Common Computational Structure Integrates Effort and Threat Across Decision, Emotion, and Action

**Authors:** Noah Okada, Ketika Garg, Toby Wise, Dean Mobbs
**Date:** 2026-02-23
**Draft Version:** 2

---

## Abstract

A fundamental challenge in survival is balancing energetic expenditure against exposure to danger. Across species, animals flexibly modulate distance traveled, time spent exposed, and movement vigor to balance resource acquisition against predation risk. Yet how humans integrate these variables remains poorly understood. Drawing on field studies of foraging under predation, we developed a task in which participants chose between options varying in reward, effort, and exposure to threat. Across two large online samples (exploratory N = 293; confirmatory N = 350), choice behavior reflected structured integration of energetic cost and exposure to danger, with stable individual differences in how exposure shaped decisions. A survival-based choice model in which exposure scaled perceived danger while effort discounted reward value best explained behavior, outperforming standard effort-discounting accounts. Crucially, the same computational structure that explained choices also predicted independent measures of action vigor and emotion. Participants behaved, felt, and acted *as if* they computed a unified survival-relevant value signal, with individual differences in model parameters selectively shaping emotional reactivity and vigor regulation. These findings demonstrate that human decisions under threat follow a common computational logic that generalizes across choice, affect, and action domains.

## Introduction

Adaptive behavior requires organisms to integrate multiple dimensions of the environment in order to guide survival-relevant action. In natural settings, acquiring resources necessitates leaving protective cover, traversing space, expending metabolic energy, and tolerating increased exposure to danger (Lima & Dill, 1990; Lima et al., 1985; Bednekoff, 2007). These demands create a mixed-motivation control problem in which appetitive incentives promote action while aversive incentives constrain it (Mobbs et al., 2018). Contemporary theoretical frameworks propose that defensive behavior is governed by dynamically constructed internal states—meta-representations that integrate internal and external variables to support survival (Mobbs et al., 2018; Silston et al., 2021). Such representations are hypothesized to bridge ecological decision-making, affective experience, and motor policy execution, yet the computational structure of these representations remains largely untested in humans.

Converging accounts further suggest that these internal representations are inherently affective: potential future states are evaluated in terms of their anticipated emotional consequences, forming a common currency that organizes motivated behavior (Shenhav, 2024). In parallel, work on aversive motivation and cognitive control demonstrates that aversive incentives do not uniformly suppress behavior, but instead modulate policy allocation depending on motivational context and the bundling of appetitive and aversive incentives (Yee et al., 2022; Prater Fahey et al., 2025; Pavlickova et al., 2024). Real-world decisions therefore require integrating mixed incentives into a unified valuation process that simultaneously shapes subjective affect, choice, and action. Despite substantial theoretical interest, few paradigms permit computational characterization of this integrative process while independently measuring its affective and behavioral correlates.

Ecological foraging provides a natural substrate for examining this problem (Mobbs et al., 2018). Movement toward reward simultaneously incurs energetic cost and modulates exposure-dependent threat, creating a continuous trade-off between resource acquisition and survival (Lima et al., 1985; Newman & Caraco, 1987; Yoon et al., 2018). Classical foraging theory proposes that organisms regulate behavior *as if* they compute survival-relevant variables that integrate these competing pressures (Lima & Dill, 1990; Bednekoff, 2007). However, human studies have typically examined energetic cost (Bustamante et al., 2023; Muller et al., 2021; Klein-Flugge et al., 2016), threat (Silston et al., 2021; Trier et al., 2025), affect, and control in isolation, limiting mechanistic inference about how multidimensional survival-relevant computations are structured and transformed into adaptive action policies.

Here, we introduce a controlled foraging paradigm that parametrically manipulates reward magnitude, energetic cost, and exposure to threat within a unified state space. Participants select movement policies that jointly determine reward acquisition and risk exposure, while providing independent trial-level reports of subjective anxiety and confidence and exhibiting rapid defensive action vigor under imminent threats. Using hierarchical Bayesian modeling, we test whether behavior is best explained by a model in which energetic costs discount reward value while exposure scales inferred danger, yielding survival-weighted subjective values that govern policy selection. We further examine whether this computational structure generalizes beyond choice to predict affective reports and downstream action vigor.

By characterizing a common computational structure linking valuation, subjective affect, and action execution, this work provides empirical support for integrative accounts of defensive behavior and clarifies how mixed appetitive and aversive incentives are combined into adaptive control policies.

## Methods

### Participants

A total of 700 participants were recruited from the online platform Prolific across two independent samples (exploratory: *N* = 350; confirmatory: *N* = 350). All participants completed the experiment in a single session using a desktop or laptop web browser (mobile devices were not permitted). Sessions lasted approximately 40–50 minutes. Participants were compensated at a base rate of $10 per hour, with additional bonus payments contingent on task performance.

Participants were excluded based on preregistered quality-control criteria designed to ensure valid task engagement and data integrity. Exclusion criteria included incomplete task completion, implausible keypress rates during a calibration procedure (indicative of incorrect device usage or technical artifacts), invalid predator dynamics consistent with browser rendering or physics errors, and failure to demonstrate adequate task engagement during foraging (escape rate < 35%). All exclusion thresholds and preprocessing steps were specified prior to data collection and are reported in detail in the Supplementary Methods. After applying these criteria, the final preregistered exploratory sample comprised 293 participants.

The study protocol was approved by the Caltech Institutional Review Board, and all participants provided informed consent electronically prior to participation. The study was preregistered prior to data collection.

### Experimental Task

The task was implemented in the Unity game engine (version 2022.3.4f1) and deployed for online participation via WebGL in a standard desktop web browser. Participants completed a virtual foraging task in a bounded circular arena viewed from a top-down perspective. On each trial, two reward options (cookies) were simultaneously presented, and participants selected one to retrieve and return to a designated safe zone located at the center of the arena.

### Effort Calibration

Prior to the main task, participants completed an effort calibration procedure to normalize effort demands across individuals. Participants were instructed to press three keyboard keys (S, D, and F) simultaneously as rapidly as possible during three 10-second trials. The maximum number of key presses achieved across trials was recorded as each participant's calibrated maximum press rate. Participants who failed to achieve a minimum of 10 presses were excluded as an attention check.

### Effort Manipulation

During the main task, effort was manipulated as a function of cookie weight and distance from safety, with demands defined relative to each participant's calibrated capacity. Cookie weight determined the required press rate to achieve full movement speed: heavy cookies required 100% of the participant's calibrated maximum press rate, while light cookies required 40%. Distance from the safe zone determined the duration over which effort had to be sustained, as farther rewards required prolonged exertion. Low-effort trials always consisted of light cookies placed at close distances, requiring 40% of maximum calibrated effort capacity. High-effort trials varied in intensity through combinations of weight and distance, corresponding to 60%, 80%, and 100% of maximum effort capacity.

### Movement Dynamics

Participants selected their chosen reward by clicking on the corresponding cookie. After selection, participants transported the cookie back to the safe zone by repeatedly pressing the designated keys (S, D, and F simultaneously). This design allowed participants to self-modulate their pressing vigor: maintaining the maximum required press rate for the selected cookie yielded full movement speed, while lower press rates resulted in proportionally slower movement. Press rate was continuously monitored and mapped onto movement speed in real time using discrete speed tiers. Participants achieved full speed when pressing at or above 100% of the required rate for that trial, half speed at ≥50% of the required rate, quarter speed at ≥25%, and zero speed below 25%. When press rate fell below the 25% threshold, movement speed decayed to zero rather than simply remaining static.

### Predator Dynamics

To introduce risk, each trial was associated with a probabilistic predator encounter. On attack trials, threat unfolded across two temporal phases. During the pre-encounter phase, the predator spawned at the closest perimeter position to the participant and approached slowly (0.5 units/second), providing an anticipatory threat cue. The onset of this phase was calibrated to occur at approximately half the expected return time if the participant maintained full movement speed, ensuring adequate opportunity to detect the threat and initiate escape regardless of reward distance. During the post-encounter phase, the predator accelerated to full attack speed and pursued the participant directly. The timing of this acceleration was drawn from a Gaussian distribution centered at twice the pre-encounter onset time, introducing uncertainty about precisely when the predator would strike. Predator speed was set to four times each participant's maximum calibrated movement speed, ensuring that participants could not outrun the predator and had to rely on reaching safety in time to avoid capture. Failure to return the reward to the safe zone before interception resulted in capture and trial termination.

### Trial Structure

This threat structure naturally divided each trial into two epochs: a pre-encounter period during which threat was possible but had not yet manifested, and a post-encounter period beginning at predator appearance and characterized by imminent danger. By jointly manipulating reward magnitude, effort costs, and probabilistic threat within a continuous-action environment, the task was designed to probe how participants integrate prospective effort and reward information under varying levels of threat, and how action vigor is modulated across anticipatory and reactive escape phases.

### Block Structure and Self-Report Probes

The task was divided into three blocks of 27 trials each. Between blocks, participants completed a brief effort recalibration procedure to account for potential fatigue or practice effects. These inter-block intervals also included psychiatric questionnaires. Within each block, a subset of trials served as self-report probes to assess subjective state. On these trials, participants were presented with a forced-choice scenario in which both options were identical—matched on reward magnitude, effort (weight), distance, and threat probability—such that the selection itself was uninformative. After making a selection, participants responded to one of two probe questions. Anxiety probe trials asked: "How anxious are you about being captured on this trial?" Confidence probe trials asked: "How confident are you in your ability to reach safety on this trial?" Each block contained 6 anxiety probes and 6 confidence probes, with probe trials spanning the 9 possible combinations of effort (low, high), threat probability (10%, 50%, 90%), and distance (near, middle, far). This design enabled assessment of how subjective anxiety and confidence varied as a function of objective task demands.

### Measures of Psychiatric Symptoms

To assess individual differences in psychiatric symptom dimensions, participants completed a battery of standardized self-report questionnaires embedded between blocks of the task. Measures were selected to capture transdiagnostic variation in affective distress, anxiety severity, motivational traits, and fatigue-related functional impact previously linked to threat processing, effort allocation, and decision-making under uncertainty.

Specifically, participants completed the Depression Anxiety Stress Scales – 21 item version (DASS-21), the Patient Health Questionnaire–9 (PHQ-9), the Overall Anxiety Severity and Impairment Scale (OASIS), the Trait subscale of the State–Trait Anxiety Inventory (STAI-T), the Apathy Motivation Index (AMI), and the Modified Fatigue Impact Scale (MFIS). The DASS-21 provided continuous measures of depressive symptoms, physiological anxiety, and stress-related arousal. The PHQ-9 indexed depressive symptom severity. The OASIS assessed global anxiety severity and functional impairment, and the STAI-T measured stable individual differences in trait anxiety. The AMI quantified motivational and apathy-related traits across behavioral activation, social motivation, and emotional sensitivity subdomains. The MFIS measured the perceived impact of fatigue on cognitive, physical, and psychosocial functioning.

Questionnaires were administered via an embedded online survey interface and scored according to published guidelines. Where applicable, subscale scores were computed to dissociate symptom dimensions (e.g., depression vs. anxiety vs. stress on the DASS-21; behavioral vs. social vs. emotional apathy on the AMI; physical vs. cognitive fatigue on the MFIS). All questionnaire scores were treated as continuous variables and z-scored across participants prior to analysis. These measures were used to examine associations between psychiatric symptom dimensions and task-derived behavioral and computational parameters.

Questionnaires were not used to screen or exclude participants, and no diagnostic classifications were assigned. Analyses focused on dimensional variation across the full sample rather than categorical diagnosis.

### Computational Modeling of Behavior

We modeled foraging patch choice using a family of hierarchical Bayesian models formalizing how energetic costs and exposure-dependent threat are integrated into option values. On each trial, participants chose between a high-reward option (*H*) and a low-reward option (*L*), each characterized by reward magnitude *R_o*, energetic effort *E_o*, distance traveled *D_o*, and a trial-level threat intensity *T*.

### Survival Function

When the threat component was included, option-specific survival probability was defined as

$$S_{u,o} = \exp(-T \cdot D_o^{z_i}),$$

where *z_i* is a subject-specific parameter governing the nonlinearity of distance-dependent exposure. Values of *z_i* > 1 indicate convex exposure (disproportionate risk at longer distances), while *z_i* < 1 indicates concave exposure (diminishing marginal risk). When the threat component was omitted, survival was set to *S_{u,o}* = 1 for both options.

### Effort Discounting

When the effort component was included, rewards were attenuated by an effort-dependent discount function *f*(·) with subject-specific sensitivity parameter *k_i*:

$$R_o^{\text{eff}} = R_o \cdot f(E_o; k_i).$$

When the effort component was omitted, *R_o*^eff = *R_o*.

We evaluated four candidate discount functions:

- Exponential: *f*(*E*; *k*) = exp(−*kE*)
- Hyperbolic: *f*(*E*; *k*) = 1 / (1 + *kE*)
- Quadratic: *f*(*E*; *k*) = max(0, 1 − *kE*²)
- Linear: *f*(*E*; *k*) = 1 − *kE*

### Subjective Value

Subjective value combined effort-discounted reward and survival probability with a fixed capture penalty *C*:

$$SV_o = R_o^{\text{eff}} \cdot S_{u,o} - (1 - S_{u,o}) \cdot C.$$

This formulation captures the expected value of pursuing option *o* under probabilistic survival: discounted reward is realized with probability *S_{u,o}*, whereas capture incurs penalty *C* with probability 1 − *S_{u,o}*.

### Choice Rule

Choices were modeled using a softmax choice rule based on the difference in subjective value:

$$p(\text{choose } H) = \sigma\left(\frac{SV_H - SV_L}{\tau}\right),$$

where σ(·) is the logistic function and τ is a population-level inverse-temperature parameter controlling choice stochasticity. Observed choices were modeled as Bernoulli draws from this probability.

### Threat-Induced Choice Bias Extension

To test whether threat influences choice beyond its effect on survival probability, we extended the best-fitting model (exponential effort discounting) with a threat-induced bias term. This extension modifies the choice rule as follows:

$$p(\text{choose } H) = \sigma\left(\frac{SV_H - SV_L - \beta_i \cdot T}{\tau}\right),$$

where *β_i* is a subject-specific bias parameter. Positive values of *β_i* indicate that higher threat levels shift choice probability toward the low-reward option, consistent with threat-induced risk aversion beyond what is captured by the survival-weighted value computation. This bias term captures residual threat sensitivity not explained by expected value calculations alone, potentially reflecting affective or heuristic influences on choice under threat.

### Hierarchical Priors

Parameters were estimated hierarchically. Subject-level parameters were modeled on the log scale with normal population-level distributions:

$$\log z_i \sim \mathcal{N}(\mu_z, \sigma_z),$$
$$\log k_i \sim \mathcal{N}(\mu_k, \sigma_k),$$

and exponentiated to enforce positivity. Parameters were constrained to plausible ranges during sampling (*z_i* ∈ [0.1, 3.0], *k_i* ∈ [0.01, 5.0]).

For the bias-extended model, subject-level bias parameters were similarly specified:

$$\log \beta_i \sim \mathcal{N}(\mu_\beta, \sigma_\beta),$$

with a weakly informative prior on the population mean (*μ_β* ~ 𝒩(−0.5, 0.5)) centered on small positive values in the original scale.

The inverse-temperature parameter τ was modeled at the population level only (not hierarchically across subjects) and constrained to τ ∈ [0.1, 10.0].

### Model Variants

We systematically compared the following model classes:

1. **Effort-only models**: Effort discounting with *S_{u,o}* = 1 (four discount functions)
2. **Threat-only model**: Survival-weighted values with *R_o*^eff = *R_o*
3. **Full effort–threat models**: Combined effort discounting and survival weighting (four discount functions)
4. **Bias-extended model**: Full model with exponential discounting plus threat-induced choice bias (*β_i*)

### Inference

Models were estimated using Hamiltonian Monte Carlo with the No-U-Turn Sampler (NUTS) implemented in NumPyro. We ran 4 chains with 1,000 warmup iterations and 1,000 sampling iterations per chain, using a target acceptance probability of 0.95 and maximum tree depth of 10.

### Model Comparison

Predictive performance was assessed using the Widely Applicable Information Criterion (WAIC), computed from pointwise log-likelihoods across posterior draws. Models were ranked by WAIC, with lower values indicating better out-of-sample predictive accuracy. We report ΔWAIC relative to the best-fitting model.

### Posterior Predictive Checks

Posterior predictive checks were performed by sampling parameters from the joint posterior and generating predicted choice probabilities for each trial. We assessed model fit using calibration curves (predicted vs. observed choice frequencies), Brier scores, and McFadden's pseudo-*R*².

### Code and data availability

TODO

### Acknowledgments

TODO

## Results

### Behavioral sensitivity to effort and threat

We first confirmed that participants' choices were systematically modulated by both energetic cost and threat exposure. Mixed-effects logistic regression revealed that increasing effort cost (distance from safety) reduced the probability of selecting the high-reward option, consistent with effort-based discounting. Independently, higher threat probability substantially reduced preference for the high-reward option, indicating prioritization of safety under danger. Critically, a significant interaction showed that the deterrent effect of effort was amplified under high threat, such that participants were especially unlikely to pursue costly options when danger was elevated. These effects exhibited substantial inter-individual variability, motivating a computational analysis of how effort and threat are integrated into choice.

### A Unified Effort–Threat Model Best Explains Foraging Choices

To characterize how effort and threat are integrated into choice, we compared a family of hierarchical Bayesian models, including effort-only models, a threat-only model, combined effort–threat models with multiple discounting functions, and a bias-extended model incorporating residual threat sensitivity.

Model comparison using WAIC revealed that the factored effort–threat model with exponential effort discounting and a threat bias term provided the best predictive performance (Fig. **??**). This model substantially outperformed all effort-only and threat-only models, as well as alternative discounting forms, indicating that participants integrated energetic cost and threat into a unified subjective value computation rather than relying on a single dimension.

Posterior predictive checks confirmed that the winning model accurately captured the graded effects of both effort and threat on choice probability across threat levels (Fig. **??**). Group-level calibration showed close correspondence between predicted and observed choice probabilities (Fig. **??**), and subject-level calibration demonstrated high individual predictive accuracy (mean accuracy ≈ 0.83; Fig. **??**). Trial-level posterior predictions further illustrated that the model tracked moment-to-moment fluctuations in individual choice behavior (Fig. **??**).

Together, these results establish that foraging choices are well-described by a survival-weighted subjective value computation integrating energetic cost and exposure-dependent threat, with an additional bias capturing residual threat sensitivity.

### Model Parameters Are Independently Identifiable and Capture Stable Individual Differences

We next examined the posterior distributions of the three core model parameters: hazard sensitivity (*z*), effort discounting (*k*), and threat bias (*β*). All parameters exhibited well-behaved posterior distributions with substantial inter-individual variability (Fig. **??**), indicating stable individual differences in how participants weighted distance-dependent danger, discounted energetic cost, and expressed threat bias.

Crucially, parameter recovery analyses showed that these parameters were independently identifiable, with minimal posterior correlations between *z*, *k*, and *β* (Fig. **??**). This confirms that the model dissociates distinct computational components rather than collapsing them into a single factor.

### Model-Derived Survival Estimates Predict Subjective Anxiety and Confidence

We next asked whether the survival estimates derived from the fitted model corresponded to participants' subjective states. For each trial, we computed the model-implied survival probability using each participant's fitted parameters and the trial's task features. Trial-by-trial mixed-effects models revealed that higher model-derived survival predicted lower anxiety and higher confidence, with robust and opposite-signed effects across subjective domains (Anxiety: *β* = −1.84, *p* < .001; Confidence: *β* = +2.00, *p* < .001; Survival × Question interaction: *p* < .001; Fig. **??**). Nonparametric correlations showed consistent monotonic relationships across participants (Anxiety: Spearman *ρ* = −0.28; Confidence: *ρ* = +0.30; Fig. **??**).

These findings demonstrate that the computational structure characterizing choice behavior generalizes to predict subjective affective experience. Participants who behaved *as if* a given situation were dangerous also reported feeling more anxious in that situation, consistent with a common underlying computation governing both choice and affect.

### Individual Differences in Hazard Sensitivity and Threat Bias Shape Emotional Reactivity

We next tested whether individual differences in model parameters modulated how emotional responses scaled with task variables.

We tested whether individual differences in hazard sensitivity (*z*) and threat bias (*β*) moderated affective responses to distance and threat intensity respectively.

Higher hazard sensitivity (*z*) was associated with steeper decreases in confidence as distance increased (Distance × *z*: β = −0.056, *p* = .023), consistent with *z* governing how strongly spatial exposure translates into perceived danger. The corresponding effect on anxiety was directionally consistent but did not survive correction for multiple comparisons (β = +0.046, *p* = .051). Neither interaction survived FDR correction across all four moderation tests (all *p*_FDR > .067).

Higher threat bias (*β*) was associated with greater anxiety amplification at higher threat intensities (Threat × *β*: β = +0.050, *p* = .038), though this also did not survive FDR correction (*p*_FDR = .068). The corresponding effect on confidence was null (*p* = .134). Independently of the interaction, *z* predicted a chronic confidence deficit (*β* = −0.199, *p*_FDR = .013), replicating the main-effects result.

Together, these findings suggest directional tendencies consistent with *z* scaling distance-dependent emotional reactivity and *β* amplifying threat-driven anxiety, though these moderation effects are modest and sensitive to correction. The more robust finding is the main effect of *z* on chronic confidence across the task.

### Model-Derived Survival Estimates Regulate Defensive Action Vigor

We next examined whether the same computational structure governing choice and affect also predicted downstream defensive action. Trial-level analyses showed that lower model-derived survival predicted higher action vigor, even after controlling for chosen effort level (*β* = −0.23, *p* < .001; Fig. **??**). High-effort choices were executed with greater vigor, confirming behavioral validity.

Individual differences modulated this coupling. Participants with higher hazard sensitivity (*z*) showed stronger trial-by-trial coupling between model-derived survival and vigor (Survival × *z* interaction *p* < .001; Fig. **??**), indicating that individuals who weighted distance-as-danger more heavily in their choices also showed more flexible adjustment of execution speed. In contrast, higher effort discounting (*k*) predicted lower overall vigor across trials (*p* < .001; Fig. **??**), reflecting energetic constraints on action execution. Threat bias (*β*) did not significantly predict vigor.

### Defensive Encounters Trigger Vigor Mobilization

To validate that vigor reflected defensive mobilization under imminent threat, we examined changes in action vigor across encounter phases. Predator attacks elicited robust increases in vigor from pre- to post-encounter phases (*p* < .001; Fig. **??**). Longer distances and higher-effort choices attenuated ramp-up magnitude, consistent with biomechanical and energetic constraints.

Hazard sensitivity (*z*) predicted stronger baseline vigor and greater phase-dependent changes, whereas effort discounting (*k*) constrained overall vigor and ramp-up magnitude. Threat bias (*β*) did not reliably modulate execution dynamics (Fig. **??**).

## Discussion

In this study, we investigated how humans integrate energetic cost and exposure-dependent threat to regulate adaptive behavior across choice, subjective affect, and defensive action. Using a controlled foraging paradigm and hierarchical Bayesian modeling, we characterized a survival-weighted subjective value computation in which energetic costs discount reward value and exposure scales inferred danger. This computational structure not only explained choice behavior but also generalized to predict trial-by-trial anxiety and confidence as well as downstream defensive action vigor. Individual differences in hazard sensitivity and effort discounting further shaped emotional reactivity and vigor regulation. Together, these findings provide convergent evidence that humans integrate appetitive and aversive incentives according to a common computational logic that governs valuation, feeling, and action.

### A common computational structure links valuation, affect, and action

A central contribution of this work is the demonstration that a single computational structure—survival-weighted, effort-discounted subjective value—describes behavior across multiple domains. The model parameters fitted to choice data predicted independent self-reports of anxiety and confidence: participants who chose *as if* a situation were dangerous also reported feeling more anxious in that situation. The same parameterization also predicted defensive action vigor during rapid responses to threat, linking deliberative valuation to motor policy execution. This cross-domain generalization supports the interpretation that participants apply a coherent computational logic across choice, affect, and action, rather than relying on independent processes for each domain.

This finding aligns with theoretical accounts proposing that adaptive behavior is organized around integrated representations that combine multiple environmental and bodily variables to support survival (Mobbs et al., 2018; Shenhav, 2024). Rather than treating valuation, emotion, and action as separate computational modules, these frameworks posit that a common computational structure underlies all three. The present results provide concrete empirical support for this integrative perspective by showing that the same parameterized model describes behavior across domains.

Importantly, our findings characterize the *structure* of the computation governing behavior, not the mechanism by which it is implemented. The model-derived survival estimates are deterministic transformations of observable task features (distance, threat probability) using fitted parameters—they are not hidden states inferred from noisy observations. What the cross-domain generalization demonstrates is that participants behave *as if* they perform this computation, and that individual differences in how they weight the relevant variables are stable across behavioral modalities. Whether this reflects a unified internal representation, parallel computations with shared parameters, or some other architecture remains an open question for future work.

### Affective grounding of survival-relevant valuation

The observation that model-derived survival probability tracked subjective anxiety and confidence provides evidence that survival-relevant valuation and affective experience share computational structure. Higher model-derived survival was associated with reduced anxiety and increased confidence, consistent with the notion that situations evaluated as safer are also experienced as less threatening. Moreover, individual differences in hazard sensitivity selectively modulated how emotional responses scaled with distance, whereas threat bias selectively amplified anxiety responses to threat intensity. These dissociations indicate that distinct computational components shape different aspects of affective experience under threat.

These results complement accounts proposing that affect functions as a common currency for evaluating future states and guiding behavior (Shenhav, 2024). Rather than treating emotion as an epiphenomenal consequence of decision processes, the present findings support a view in which affect and choice reflect the same underlying computation, even if the functional relationship between them remains to be specified.

### Integration of mixed incentives into adaptive control policies

The behavioral and modeling results demonstrate that participants integrated energetic cost and threat into a unified subjective value computation rather than treating these dimensions independently. The best-fitting model required both effort discounting and survival weighting, along with a residual threat bias capturing non-normative sensitivity. This structure provides a mechanistic account of how mixed appetitive and aversive incentives are combined to regulate behavior.

Importantly, aversive incentives did not simply suppress action. Lower model-derived survival increased defensive action vigor, indicating that threat can invigorate execution depending on context (Pavlickova et al., 2024; Shenhav et al., 2020). At the same time, higher effort sensitivity constrained overall vigor, reflecting energetic limitations on action (Kurzban, 2016; Treadway & Salamone, 2022). These findings resonate with work in aversive motivation and cognitive control emphasizing that aversive signals can both facilitate and inhibit behavior depending on motivational structure (Yee et al., 2022; Prater Fahey et al., 2025). The present results extend this literature by characterizing a computational framework through which these mixed incentives shape policy selection and execution.

### From computational characterization to action execution

Defensive encounters triggered rapid mobilization of action vigor, validating that the task captured genuine defensive dynamics under imminent threat. Individual differences in hazard sensitivity modulated both baseline vigor and phase-dependent changes, suggesting that how strongly individuals weight exposure in their value computations also shapes the flexibility of their action policies. In contrast, effort discounting constrained energetic deployment without selectively modulating threat-driven ramping, and threat bias primarily influenced affect rather than execution. This dissociation supports a functional separation between how value is computed and how it is translated into action.

Together, these results provide empirical support for accounts proposing that survival-relevant computation integrates multidimensional variables into value signals that then govern policy execution (Bailey et al., 2016; Suzuki et al., 2021; Treadway & Salamone, 2022). The present work bridges these levels by showing that a computationally defined value structure predicts both affective experience and motor behavior in humans.

### Relation to ecological and computational foraging theories

Classical foraging theory proposes that organisms regulate behavior *as if* they compute variables that integrate energetic gain and survival risk (Lima & Dill, 1990; Lima et al., 1985; Newman & Caraco, 1987; Bednekoff, 2007). While this principle has been extensively validated in non-human animals (Wernecke et al., 2016), human studies have often examined energetic cost (Bustamante et al., 2023; Vogel et al., 2020; Scholey et al., 2017; Hewitt et al., 2025), threat (Silston et al., 2021; Trier et al., 2025), or affect in isolation. By embedding these dimensions within a unified state space and explicitly modeling their integration, the present results demonstrate that humans likewise behave *as if* they compute a survival-relevant value that governs both decision-making and action.

The form of the recovered value function—multiplicative integration of survival probability with effort-discounted reward—provides a computationally interpretable instantiation of this principle. The presence of a residual threat bias further suggests that affective or heuristic influences modulate valuation beyond normative expected value, consistent with ecological pressures favoring conservative responses under danger.

### Conclusion

By integrating behavioral modeling with independent measures of subjective affect and defensive action, this work characterizes a common computational structure linking valuation, feeling, and action in humans. The findings demonstrate how mixed appetitive and aversive incentives are integrated according to a unified computational logic that generalizes across behavioral domains. More broadly, the results provide a computational bridge between ecological theories of survival (Lima & Dill, 1990; Mobbs et al., 2018), affect-centered accounts of motivation (Shenhav, 2024), and control-oriented perspectives on action selection (Yee et al., 2022; Shenhav et al., 2020), advancing understanding of how organisms navigate environments where reward and danger are inseparable.

## References

<!-- TODO: This section is empty. The PDF references section contained no entries. References will need to be added separately. -->
