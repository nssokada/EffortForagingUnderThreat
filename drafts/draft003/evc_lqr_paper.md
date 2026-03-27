# Expected Value of Control Integrates Effort and Threat Across Decision and Action in Human Foraging

Noah Okada, Ketika Garg, Toby Wise, Dean Mobbs

---

## Abstract

Foraging under predation risk requires integrating energetic costs and survival probabilities into a single decision variable that governs both what to pursue and how vigorously to pursue it. Current models treat effort-based choice and action vigor as separate processes, leaving unclear how a common cost-benefit computation might unify them. Here we develop an Expected Value of Control (EVC) model grounded in linear-quadratic regulator (LQR) optimal control theory and test it in an effort-foraging task with parametric threat (N = 293). The model has two subject-level parameters -- capture aversion (c_death) and effort efficacy (epsilon) -- plus a population-level probability weighting exponent (gamma = 0.318) and effort cost (c_effort = 0.007). A single cost function with LQR-inspired commitment costs (choice) and deviation costs (vigor) jointly predicts foraging decisions (per-subject r = 0.90) and press-rate vigor (trial-level r = 0.51, per-subject r = 0.72). Both parameters are well-recovered in simulation (c_death: r = 0.89; epsilon: r = 0.93). Subjective survival probability strongly predicts trial-level anxiety (beta = -0.79, t = -13.1) and confidence (beta = 0.85, t = 13.4). We decompose metacognitive affect into calibration (within-subject tracking of objective danger) and discrepancy (systematic bias). Calibration predicts task performance (r = 0.24 with choice quality, r = 0.19 with survival), while discrepancy predicts clinical symptoms across anxiety (OASIS: r = 0.18), depression (PHQ-9: r = 0.20), and somatic anxiety (STICSA: r = 0.25) measures. This partial double dissociation suggests that how well people track danger determines performance, while how much they overestimate danger determines psychiatric vulnerability. These findings establish a computational bridge from normative foraging theory through affective metacognition to clinical symptomatology, though effect sizes for clinical associations are modest and require confirmation in an independent sample.

---

## Introduction

Effort-based decision making is a core function of adaptive behavior. When foraging for food while exposed to predation risk, organisms must continuously evaluate whether the energetic cost of obtaining a resource is justified by its value, discounted by the probability of being captured before consuming it (Charnov, 1976; Lima & Dill, 1990). This evaluation must occur not only at the moment of choosing which resource to pursue, but also during execution -- modulating the vigor of ongoing action to balance speed against motor cost (Shadmehr & Krakauer, 2008; Niv et al., 2007). Despite growing interest in both effort discounting (Pessiglione et al., 2018; Husain & Roiser, 2018) and defensive behavior under threat (Mobbs et al., 2015; Qi et al., 2018), no computational model has jointly captured choice and vigor within a single foraging framework.

The Expected Value of Control (EVC) framework (Shenhav et al., 2013; 2017) provides a candidate architecture for this integration. EVC proposes that the dorsal anterior cingulate cortex computes the expected payoff of allocating control, selecting the intensity of cognitive or motor effort that maximizes reward minus effort cost. Critically, EVC treats effort allocation as a continuous optimization problem rather than a binary choice, making it naturally suited to predict both discrete decisions (choose which resource to pursue) and continuous actions (how hard to press during pursuit). However, EVC has primarily been applied to cognitive control tasks (Musslick et al., 2015; Lieder et al., 2018) and has not been extended to physical effort under ecological threat.

A key challenge in modeling effort-cost foraging is that choice and vigor impose qualitatively different effort demands. At the choice stage, the agent commits to a foraging bout whose total effort cost scales with the difficulty and distance of the resource. During execution, effort cost reflects deviations from the committed press rate -- pressing faster than required to improve survival, or pressing slower than required due to fatigue or low motivation. This distinction maps directly onto the linear-quadratic regulator (LQR) framework from optimal control theory (Todorov & Jordan, 2002; Shadmehr & Krakauer, 2008), where commitment cost represents the fixed price of engaging an option and deviation cost represents the online motor regulation penalty. An LQR-inspired formulation naturally resolves a scaling conflict: choice requires large effort costs (to explain distance-based choice gradients) while vigor requires small costs (to avoid suppressing press rates), because commitment costs (req^2 x D ~ 0.8-2.4) and deviation costs ((u - req)^2 x D ~ 0.01-0.05) differ by two orders of magnitude from the same cost parameter.

Here, we present an EVC-LQR model of effort foraging under threat and test it in a large online sample (N = 293). Participants chose between high-effort/high-reward and low-effort/low-reward resources while facing parametric predation risk (T in {0.1, 0.5, 0.9}) at varying distances (D in {1, 2, 3}). Two subject-level parameters -- capture aversion (c_death, how much the individual fears being caught) and effort efficacy (epsilon, how much the individual believes pressing harder improves survival) -- jointly determine choice and vigor. We then test whether the model's subjective survival signal predicts trial-level anxiety and confidence, and whether individual differences in metacognitive accuracy (calibration) versus bias (discrepancy) dissociate performance from clinical symptomatology.

---

## Results

### R1: An EVC-LQR model captures foraging choice with two subject-level parameters

We formalized the foraging decision as an expected utility computation where the value of each option is:

EU = S x R - (1 - S) x c_death x (R + C) - c_effort x req^2 x D

where S is the subjective survival probability, R is reward (5 or 1 points for heavy or light cookies), C = 5 is the capture penalty, c_death is subject-specific capture aversion, c_effort = 0.007 is the population-level effort cost, req is the required press rate (0.9 or 0.4 for heavy or light), and D is distance (1-3). The survival function incorporates probability weighting and effort efficacy:

S = (1 - T^gamma) + epsilon x T^gamma x p_esc

where T is threat probability, gamma = 0.318 is the population-level probability weighting exponent (indicating compression of threat probabilities), epsilon is subject-specific effort efficacy, and p_esc = 0.6 is the population escape probability.

The model was fitted via stochastic variational inference (SVI) to 12,966 choice trials from 293 subjects. Subjects chose the high-effort option on 43.1% of trials (SD = 20.3%). The model achieved BIC = 20,368, with per-subject choice predictions correlating r = 0.90 with observed choice proportions (Figure 2C). Choice accuracy (predicting binary choice from P(heavy) > 0.5) was 75.4% with AUC = 0.819.

The probability weighting parameter gamma = 0.318 indicates substantial compression: a nominal threat of T = 0.5 is experienced as T^0.318 = 0.80, consistent with Kahneman and Tversky's (1979) finding that people overweight moderate probabilities in the loss domain. The two subject-level parameters were well-identified: log(c_death) ranged from -3.6 to 1.4 (median = -0.51), and log(epsilon) ranged from -7.1 to 1.2 (median = -1.56), with a non-significant correlation between them (r = 0.06, p > 0.05 in log space).

Parameter recovery analysis confirmed identifiability: simulating 5 datasets of 50 subjects x 45 trials each and re-fitting yielded recovery correlations of r = 0.89 for c_death and r = 0.93 for epsilon (both p < 10^-85; Figure S1). The population-level gamma was also well-recovered (true = 0.318, recovered = 0.314). The population-level c_effort was not individually recoverable (recovered = 0.047 vs. true = 0.007), consistent with its role as a scaling parameter that trades off against tau and other population parameters.

### R2: The same computation predicts action vigor via LQR deviation cost

Having fit the model to choice data, we tested whether the same parameters predict moment-to-moment vigor. For each trial, the model computes an optimal press rate u* by maximizing:

EU(u) = S(u) x R - (1 - S(u)) x c_death x (R + C) - c_effort x (u - req)^2 x D

where the effort cost is now the LQR deviation cost -- the penalty for pressing above or below the committed rate. This means that pressing at exactly the required rate incurs zero additional vigor cost (that cost was already paid at the choice stage), while pressing faster trades motor cost against improved survival.

The survival function at execution becomes speed-dependent:

S(u) = (1 - T^gamma) + epsilon x T^gamma x p_esc x sigmoid((u - req) / sigma_motor)

where sigma_motor = 0.15 controls the steepness of the speed-to-survival mapping. This means that pressing faster than required increases survival probability, but with diminishing returns.

Using the same c_death and epsilon parameters fitted to choice, the model's predicted excess vigor correlated r = 0.51 with observed trial-level excess press rates (cookie-type centered) and r = 0.72 at the per-subject level (Figure 2D). The model correctly predicts that vigor increases with threat (as higher threat increases the marginal survival benefit of pressing faster) and that this effect is stronger for subjects with higher epsilon (who believe their effort matters more).

The vigor r^2 from the joint SVI fit was 0.485, which exceeds what independent choice and vigor models typically achieve. The LQR formulation is critical: replacing deviation cost with the standard u^2 x D formulation (used at the choice stage) reduces vigor fit substantially because the large commitment-scale costs suppress all press rates toward zero.

### R3: Parameter profiles reveal interpretable behavioral archetypes

We performed a median split on c_death and epsilon to create four behavioral archetypes (Figure 3A):

**Vigilant** (high c_death, high epsilon; n = 70): These individuals fear capture but believe effort helps. They chose the heavy option on 37.4% of trials -- less than average because high c_death makes risky options aversive -- but achieved the second-highest survival rate (70.3%) through effective effort allocation. Mean earnings: 13.6 points.

**Helpless** (high c_death, low epsilon; n = 77): These individuals fear capture but do not believe effort improves their chances. They chose heavy on only 21.3% of trials -- the lowest of any group -- and earned the least (1.7 points), despite having the highest survival rate (73.6%) due to consistently choosing safe, low-distance options.

**Reckless** (low c_death, high epsilon; n = 77): These individuals are not deterred by capture and believe effort matters. They chose heavy on 62.2% of trials -- the highest rate -- and earned the most (27.7 points) but had lower survival (66.1%) due to frequent high-risk foraging.

**Disengaged** (low c_death, low epsilon; n = 69): These individuals neither fear capture nor believe effort helps. They chose heavy at 52.0% and had the lowest survival (62.9%). Mean earnings: 5.4 points.

The quadrant ANOVA for P(heavy) was highly significant (F(3,289) = 143.8, p < 10^-50), as was survival (F(3,289) = 8.3, p < 10^-4) and earnings (F(3,289) = 4.8, p = 0.003). Notably, the "Helpless" profile -- high fear, low efficacy -- maps conceptually onto learned helplessness (Maier & Seligman, 1976) and showed the worst earnings despite maximal caution.

A multiple regression of P(heavy) on log(c_death), log(epsilon), and their interaction explained R^2 = 0.877 of between-subject variance, confirming that these two parameters nearly completely determine individual foraging strategy. The interaction term was significant (b = 0.037, p < 0.001), indicating that the effect of efficacy on choice depends on threat sensitivity.

### R4: Metacognitive miscalibration -- confidence tracks choice, not survival

On probe trials, participants rated their anxiety and confidence about the current trial's demands. The EVC model's subjective survival probability (S) strongly predicted both ratings via linear mixed models with random intercepts and slopes by subject:

- Anxiety: beta = -0.786, SE = 0.060, t = -13.09, p < 10^-38 (higher survival = lower anxiety)
- Confidence: beta = 0.848, SE = 0.063, t = 13.40, p < 10^-40 (higher survival = higher confidence)

These effects are large (roughly 0.8 rating points per SD of S on a 0-7 scale) and confirm that the model's internal survival signal closely tracks subjective affective experience.

However, at the between-subject level, mean confidence was uncorrelated with both choice quality (r = 0.012, p = 0.84) and survival rate (r = -0.048, p = 0.41). This null result is striking: participants who felt more confident did not actually perform better. Steiger's test comparing the two correlations was non-significant (z = 0.82, p = 0.41), indicating that confidence was equally uninformative about both performance metrics.

This pattern -- strong within-person calibration to threat but no between-person relationship between confidence and outcomes -- suggests metacognitive miscalibration: individuals differ in their baseline confidence level in ways that are orthogonal to actual performance.

### R5: Calibration predicts performance; discrepancy predicts clinical symptoms

To formalize the metacognition result, we decomposed each subject's anxiety responding into two orthogonal components:

**Calibration**: The within-subject Pearson correlation between anxiety ratings and model-derived danger (1 - S). Higher calibration means the individual's anxiety more closely tracks objective threat level. Mean calibration = 0.47 (SD = 0.32).

**Discrepancy**: The mean residual of a subject's anxiety ratings after subtracting the population-level prediction (anxiety ~ S). Higher discrepancy means the individual reports more anxiety than the objective situation warrants. These components were near-orthogonal (r = -0.024, p = 0.69).

A partial double dissociation emerged:

**Calibration predicted performance**: Subjects with better threat-tracking anxiety made higher-quality choices (r = 0.239, p < 0.001) and survived more (r = 0.185, p = 0.002). Calibration was largely unrelated to clinical measures (6 of 8 clinical associations p > 0.1; exception: STAI-State r = 0.138, p = 0.019).

**Discrepancy predicted clinical symptoms**: Subjects who overestimated danger scored higher on:
- STAI-State: r = 0.308, p < 10^-6
- STICSA: r = 0.249, p < 10^-4
- DASS-Anxiety: r = 0.234, p < 10^-3
- DASS-Stress: r = 0.217, p < 10^-3
- DASS-Depression: r = 0.206, p < 10^-3
- PHQ-9: r = 0.201, p < 10^-3
- STAI-Trait: r = -0.203, p < 10^-3 (note: lower trait anxiety, suggesting the discrepancy captures state rather than trait overreaction)
- OASIS: r = 0.177, p = 0.003

The dissociation was partial rather than complete: discrepancy also predicted survival (r = -0.153, p = 0.009, negative -- overestimators survived less), and calibration correlated with STAI-State (r = 0.138, p = 0.019). Nevertheless, the pattern is clear: calibration primarily predicts adaptive performance, while discrepancy primarily predicts maladaptive psychological distress.

Importantly, the EVC model parameters themselves showed minimal direct clinical associations. After FDR correction, no correlation between log(c_death) or log(epsilon) and any psychiatric subscale survived (best uncorrected: log(c_death) -> AMI-Emotional, r = 0.121, p = 0.039). This suggests that the bridge from computation to psychopathology runs through metacognition -- through how people *feel about* their threat estimates -- rather than through the threat estimates themselves.

---

## Discussion

### Summary of findings

We developed an EVC-LQR model that jointly captures foraging choice and action vigor under threat with two subject-level parameters (capture aversion and effort efficacy) and a population-level probability weighting exponent. The LQR-inspired cost structure -- commitment cost for choice, deviation cost for vigor -- resolves a fundamental scaling conflict between choice and vigor effort costs using a single parameter. The model's internal survival signal predicts trial-level anxiety and confidence, and decomposing metacognitive accuracy into calibration and discrepancy reveals a partial double dissociation between performance and clinical outcomes.

### Contribution to the EVC framework

Our findings extend the EVC framework (Shenhav et al., 2013; 2017) in three ways. First, we demonstrate that EVC can be applied to physical effort under ecological threat, not just cognitive control in abstract tasks. Second, we show that the same cost function can predict both discrete choices and continuous vigor through the LQR distinction between commitment and deviation costs. Third, we identify probability weighting (gamma = 0.318) as a critical component: participants substantially overweight threat probabilities, consistent with loss-domain probability distortion (Prelec, 1998) but now demonstrated in an embodied foraging context.

The gamma parameter deserves particular attention. At gamma = 0.318, a nominal 50% threat is experienced as equivalent to an 80% threat. This compression is stronger than typical estimates from monetary gambles (gamma ~ 0.65-0.70; Tversky & Kahneman, 1992), potentially reflecting the salience of embodied predation compared to abstract losses. However, gamma is a population parameter in our model and may absorb other forms of risk aversion not captured by c_death alone.

### Metacognition as the computational-clinical bridge

Perhaps our most important finding is that the route from normative computation to clinical distress runs through metacognition rather than through the decision parameters themselves. Neither c_death nor epsilon significantly predicted any clinical measure after FDR correction. Instead, it is the *discrepancy* between objective danger and subjective anxiety that predicts psychopathology -- across depression (PHQ-9: r = 0.20), generalized anxiety (OASIS: r = 0.18), and somatic anxiety (STICSA: r = 0.25).

This finding is consistent with metacognitive theories of anxiety (Wells, 2009; Paulus & Stein, 2010), which propose that clinical anxiety reflects not simply heightened threat estimation but a mismatch between threat estimates and affective responses. Our computational framework provides a formal operationalization of this mismatch: calibration measures the signal, while discrepancy measures the noise -- the systematic bias in affective responding that is decoupled from the decision computation.

### Clinical implications and honest assessment of effect sizes

The clinical associations we report are statistically significant but modest in magnitude (r = 0.18-0.31, R^2 = 0.03-0.10). These effect sizes are consistent with the broader literature on computational phenotyping of psychiatric symptoms (Gillan et al., 2016; Wise et al., 2023), but they should temper any enthusiasm for using these parameters as clinical biomarkers. The strongest clinical predictor in our data is anxiety discrepancy -> STAI-State (r = 0.31), which explains roughly 10% of variance -- meaningful for group-level inference but insufficient for individual diagnosis.

We also note that the double dissociation between calibration and discrepancy was partial. Discrepancy also predicted lower survival (r = -0.15), suggesting that individuals who overestimate danger may also make suboptimal foraging decisions (perhaps through excessive caution or paradoxically poor effort allocation). And calibration correlated weakly with STAI-State (r = 0.14), suggesting that better threat-tracking may be associated with higher state anxiety (a finding that, if replicated, could reflect appropriate vigilance rather than pathology).

### Limitations

Several limitations warrant consideration. First, c_effort is a population-level parameter, meaning our model cannot capture individual differences in motor cost sensitivity. This was a deliberate tradeoff: pilot analyses showed that c_effort was not individually recoverable (recovery r ~ 0.04), likely because its effects are absorbed by tau and epsilon at the individual level. Future work with richer motor data (e.g., force transducers, EMG) may enable individual-level effort cost estimation.

Second, epsilon blends two distinct constructs: the individual's belief that effort improves survival (control efficacy) and their actual motor performance (execution efficacy). Our task does not distinguish between someone who presses at the right rate because they believe it helps versus someone who presses at the right rate because they have good motor control. Disambiguating these components would require manipulating feedback about effort-outcome contingencies.

Third, the model does not capture individual differences in distance sensitivity. Observed choice data show clear distance gradients (P(heavy) decreasing with distance), but the model's per-trial predictions are insensitive to distance because c_effort is a small population parameter. This means the distance effect in choice is primarily captured by the aggregate population fit rather than by subject-level predictions. A richer model with individual distance sensitivity could improve choice PPC at the cost of additional parameters.

Fourth, our confirmatory sample (N = 350) has not yet been analyzed. All results reported here are from the exploratory sample and should be treated as hypothesis-generating until replicated. We have pre-registered the key analyses for the confirmatory sample.

Finally, the task uses a simplified threat model where predation probability is explicit and constant within a trial. Real-world threat assessment involves learning, updating, and contextual modulation that our model does not capture.

---

## Methods

### Participants

We recruited 350 participants from Prolific (https://prolific.co). After quality control filtering (5-stage pipeline: completion, comprehension, behavioral consistency, effort calibration, and outlier removal), the final exploratory sample included N = 293 participants (see Supplementary Methods for exclusion criteria and rates).

### Task

Participants completed an effort-based foraging task in a circular arena viewed from above. On each trial, two cookies appeared at varying distances from a central safe zone: a heavy cookie (5 points, requires sustained fast pressing at >= 90% of calibrated maximum) and a light cookie (1 point, requires >= 40% of maximum). Participants clicked to select a cookie, then pressed the S, D, and F keys simultaneously to transport it back to the safe zone. Movement speed depended on press rate relative to calibrated thresholds (full speed at >= 100%, half at >= 50%, quarter at >= 25%, zero below 25%).

On each trial, a predator attack occurred with probability T in {0.1, 0.5, 0.9}. If the predator appeared, it approached at 0.5 units/sec and struck at a Gaussian-distributed time. Being captured cost 5 points plus the value of the current cookie. The heavy cookie appeared at distances D in {1, 2, 3} (corresponding to 5, 7, 9 game units from center), while the light cookie was always at distance 1 (5 game units).

Each participant completed 3 blocks of 27 events each (81 total), comprising 15 choice trials and 12 probe trials per block. On probe trials, both options were identical (forced choice), and participants rated either anxiety or confidence on a 0-7 scale before executing the trial.

### Preprocessing

Raw data were processed through a 5-stage pipeline: (1) completion filtering, (2) comprehension check, (3) behavioral consistency, (4) effort calibration validation, and (5) outlier removal. Press rates were computed from inter-press intervals, with trials having fewer than 5 valid keypresses excluded. Excess vigor was defined as the median normalized press rate minus the required rate, then centered by cookie type to remove main effects of required speed.

### EVC-LQR Model

The model has two per-subject parameters (c_death, epsilon) sampled from log-normal distributions with non-centered parameterization, plus population-level parameters: c_effort (effort cost), gamma (probability weighting), tau (choice temperature), p_esc (escape probability), sigma_motor (motor noise), and sigma_v (vigor observation noise).

**Choice:** EU_H = S x 5 - (1-S) x c_death x 10 - c_effort x 0.81 x D; EU_L = S x 1 - (1-S) x c_death x 6 - c_effort x 0.16. P(choose heavy) = sigmoid((EU_H - EU_L) / tau).

**Vigor:** EU(u) = S(u) x R - (1-S(u)) x c_death x (R+5) - c_effort x (u-req)^2 x D, where S(u) = (1 - T^gamma) + epsilon x T^gamma x p_esc x sigmoid((u - req) / sigma_motor). Optimal vigor u* is computed via softmax-weighted grid search over u in [0.1, 1.5].

**Joint likelihood:** Choice is modeled as Bernoulli; vigor as Normal(excess_pred, sigma_v). Both likelihoods are evaluated simultaneously during fitting.

### Model Fitting

The model was fitted using NumPyro's SVI with an AutoNormal guide (mean-field variational inference). Optimization used Adam (lr = 0.002) for 40,000 steps. The final ELBO was used to compute BIC = 2 x loss + k x log(n) where k = 2 x N_subjects + 10 population parameters.

### Parameter Recovery

We simulated 5 datasets of 50 subjects x 45 trials each, drawing subject parameters from the fitted population distribution and generating choices and vigor from the generative model. Each dataset was re-fitted with 25,000 SVI steps. Recovery was assessed as the Pearson correlation between true and recovered parameters in log space.

### Affect Analysis

Linear mixed models (statsmodels MixedLM, REML, L-BFGS) predicted anxiety and confidence ratings from standardized survival probability S_z, with random intercepts and slopes by subject.

### Metacognitive Decomposition

**Calibration:** Per-subject Pearson correlation between anxiety ratings and model-derived danger (1 - S).

**Discrepancy:** Per-subject mean residual from the population-level regression of anxiety on S. A positive discrepancy indicates more anxiety than the objective situation warrants.

### Clinical Analysis

Psychiatric symptoms were assessed with DASS-21, PHQ-9, OASIS, STAI (State and Trait), AMI, MFIS, and STICSA. Log-space correlations between model parameters and all subscales were corrected for multiple comparisons using Benjamini-Hochberg FDR.

### Statistical Analysis

All analyses used two-tailed tests. Effect sizes are reported as Pearson r or R^2. ANOVA F-tests used type II sums of squares. Steiger's test was used to compare dependent correlations. P-values from multiple correlation tests were FDR-corrected. All analyses were conducted in Python 3.11 using NumPyro 0.15, JAX, statsmodels 0.14, and scipy 1.12.

---

## Figures

**Figure 1.** Task design and EVC-LQR model schematic.

**Figure 2.** Model fit and posterior predictive checks. (A) Choice PPC by threat x distance. (B) Vigor PPC by threat x cookie type. (C) Per-subject choice prediction (r = 0.90). (D) Per-subject vigor prediction (r = 0.72). (E) Choice residuals. (F) Vigor residuals.

**Figure 3.** Behavioral profiles. (A) cd x epsilon parameter space with quadrant labels. (B) P(heavy) by quadrant. (C) Survival rate by quadrant. (D) Clinical measures by quadrant.

**Figure 4.** Metacognition and clinical associations. (A) EVC survival predicts anxiety and confidence. (B) Confidence vs. choice quality and survival rate. (C) Performance associations (calibration vs. discrepancy). (D) Clinical associations.

**Figure S1.** Parameter recovery. (A) True vs. recovered c_death (r = 0.89). (B) True vs. recovered epsilon (r = 0.93).

**Figure S2.** Clinical forest plot: log(c_death) and log(epsilon) correlations with all psychiatric subscales.

---

## Data and Code Availability

All analysis code is available at [repository URL]. Raw data will be made available upon publication. Pre-registration for the confirmatory sample is available at [OSF URL].

---

## Acknowledgments

[To be added]

---

## Critical Review

### 1. Are any claims overstated relative to the evidence?

**Choice PPC distance issue.** The per-trial choice predictions do not capture the distance gradient: predicted P(heavy) is nearly flat across D = 1, 2, 3 within each threat level (e.g., T=0.1: pred = 0.847, 0.848, 0.846 vs obs = 0.808, 0.695, 0.566). This is because c_effort = 0.007 is too small to produce meaningful distance effects at the per-trial level. The model achieves good per-subject predictions because between-subject variance in c_death and epsilon dominates, but within-subject distance sensitivity is not well-captured. This should be acknowledged more prominently and may undermine claims about the model "jointly capturing" choice behavior.

**Vigor PPC level shift.** The vigor predictions have a systematic positive offset (pred ~ 0.05-0.32 vs obs ~ -0.04-0.04 for condition means). The trial-level correlation (r = 0.51) is substantially lower than the fitted model's r^2 = 0.485 reported at fitting, suggesting the PPC uses somewhat different computational settings than the fit. This discrepancy should be investigated.

**"Double dissociation" language.** The dissociation is explicitly partial -- discrepancy also predicts survival, and calibration weakly predicts STAI-State. The paper should use "partial dissociation" consistently and avoid implying a clean separation.

### 2. Are effect sizes reported honestly?

Yes. The clinical r values (0.18-0.31) are reported with appropriate caveats about their magnitude. The R^2 of 0.03-0.10 is explicitly noted as insufficient for individual diagnosis. The null FDR-corrected result for direct parameter-clinical associations is not hidden.

### 3. Are limitations adequately acknowledged?

The limitations section covers the main issues: population c_effort, epsilon ambiguity, distance insensitivity, and lack of confirmatory sample. Additional limitations that could be mentioned:
- The SVI fitting procedure may produce different estimates than full MCMC, and the BIC approximation from SVI loss is not standard.
- The cookie-type centering of vigor may remove meaningful variance.
- The probe trial design (forced choice, pre-execution rating) may not reflect naturalistic affect during execution.

### 4. Would a Shenhav-lab reviewer find the EVC framing justified?

Likely yes, with caveats. The model implements core EVC principles: expected value computation, effort as a continuous optimization variable, and cost-benefit integration. The LQR extension is a natural generalization. However, a Shenhav-lab reviewer might note that: (a) the model does not include an identity signal or specify which control signal is being allocated; (b) the "control" is purely motor effort, not cognitive control as in most EVC applications; (c) the population-level c_effort means individuals cannot differ in effort sensitivity, which is a major individual-differences prediction of EVC theory.

### 5. Would an effort-discounting reviewer accept population-level c_effort?

This is the most vulnerable point. Traditional effort-discounting models (Pessiglione et al., 2018; Klein-Flugge et al., 2015) treat effort sensitivity as a core individual difference. Our decision to fix c_effort at the population level is justified by the recovery analysis showing it is not individually identifiable, but a skeptical reviewer could argue this reflects a limitation of our task (which confounds effort with distance) rather than a theoretical insight. The paper should be clear that this is a pragmatic choice, not a claim that effort sensitivity does not vary across individuals.

### 6. Is the metacognition story supported by the statistics?

The core affect results are very strong (t > 13 for both anxiety and confidence LMMs). The metacognitive decomposition is well-motivated and the calibration-performance link (r = 0.24) is robust. The discrepancy-clinical links are significant across multiple measures, which provides convergent validity. However, the partial nature of the dissociation weakens the "bridge" narrative. The strongest claim supported is: "discrepancy predicts clinical symptoms while calibration predicts performance, with some leakage." The paper should resist framing this as a clean double dissociation.

**Overall assessment:** The paper presents a solid computational model with good parameter recovery and strong affect results. The metacognition story is the most novel and compelling contribution. The main weaknesses are (a) the choice PPC's failure to capture within-subject distance effects and (b) the modest clinical effect sizes. With appropriate hedging on these points, the paper is suitable for Nature Communications.
