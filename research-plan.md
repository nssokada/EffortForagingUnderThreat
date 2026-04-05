# Research Plan: effort-foraging

> Fill out this document before running `/write-paper`. The more specific
> you are here, the better your paper will be. The paper-writer agent
> reads this file as its primary brief.

---

## 1. Research Question

**Primary question:**
How do humans jointly determine which patches to forage and how intensely to work when foraging under predation risk — and can a single fitness function, grounded in optimal foraging theory, simultaneously explain patch selection (choice), motor execution (vigor), and subjective affect (anxiety/confidence) through two individual-difference parameters?

**Sub-questions:**
- Does threat simultaneously drive avoidance in choice and activation in vigor, and do these map onto distinct anticipatory vs reactive defense dynamics (predatory imminence continuum)?
- Can two per-subject parameters — avoidance sensitivity (omega, the subjective cost of capture) and activation intensity (kappa, the subjective cost of effort) — outperform simpler alternatives and recover ecologically meaningful foraging profiles?
- Do metacognitive signals (anxiety calibration, confidence) monitor the foraging computation and independently predict foraging efficiency beyond what the model parameters capture?
- Do task-elicited affect signals dissociate clinical symptom dimensions (anxiety vs depression vs apathy)?

---

## 2. Hypotheses

**Central hypothesis:**
A single fitness function W(u) = S(u)R - (1-S(u))omega(R+C) - kappa(u-req)^2 D, derived from optimal foraging theory (Bednekoff 2007; Brown 1999), simultaneously predicts patch selection and motor vigor through two separable individual-difference parameters: avoidance sensitivity (omega) and activation intensity (kappa). Metacognitive signals — anxiety and confidence — monitor this computation and bridge to clinical outcomes.

**Alternative hypotheses (tested as M1-M3):**
- M1 (Effort-only): Choice depends only on effort cost, threat is irrelevant, vigor has no systematic structure
- M2 (Threat-only): Only survival matters; individual effort differences are irrelevant
- M3 (Single-parameter): One trait governs both avoidance and activation — they are not separable

**Null hypothesis:**
The joint model (M4) does not outperform simpler alternatives; omega and kappa are not identifiable as separate traits; metacognitive signals do not add information beyond the computational parameters.

### Preregistered Hypotheses (24 tests)

**H1: Threat will reduce high-effort choices, increase motor vigor, and shift anxiety upward and confidence downward.**

- H1a. High-effort choices will decrease with threat probability and distance.
- H1b. Anxiety will increase with threat and distance. Confidence will decrease with threat and distance.
- H1c. Within each chosen effort level, pressing intensity will increase with threat.

**H2: Motor vigor will follow the predatory imminence continuum, with distinct anticipatory and reactive dynamics.**

- H2a. Predator encounter will trigger a rapid motor spike in pressing rate.
- H2b. The temporal shape of the vigor timecourse will differ by encounter status and by threat level.

**H3: A joint fitness model with two per-subject parameters will outperform simpler alternatives.**

- H3a. The joint model will outperform an effort-only model that ignores threat.
- H3b. The joint model will outperform a threat-only model that lacks individual effort sensitivity.
- H3c. The joint model will outperform a single-parameter model, demonstrating that capture cost and effort cost are separable traits.

**H4: The model parameters will predict survival, error patterns, and decision quality.**

- H4a. Higher capture cost (omega) will predict higher escape rates on attack trials.
- H4b. Capture cost will predict the proportion of overcautious errors.
- H4c. Higher effort cost (kappa) will predict lower pressing intensity.
- H4d. The balance between capture cost and effort cost (omega-kappa angle) will predict decision quality: effort-driven avoidance will be less optimal than threat-driven avoidance.
- H4e. Consistency with the joint fitness function — across both patch selection and motor intensity — will predict foraging earnings.

**H5: Anxiety and confidence will independently monitor the foraging computation and predict foraging efficiency beyond the model parameters.**

- H5a. Anxiety calibration (how well anxiety tracks threat) will predict foraging optimality beyond the model parameters. Primary outcome: percent optimal choices. Supporting: escape rate, earnings.
- H5b. Anxiety reactivity (slope on threat) will predict adaptive choice shifting across threat levels.
- H5c. Capture cost will predict subjective confidence but not anxiety. Null prediction for anxiety tested via ROPE [-0.10, +0.10].
- H5d. Confidence will predict the type of errors people make — fewer overcautious errors but more reckless errors.

---

## 3. Key Arguments

### Argument 1: Threat drives simultaneous but dissociable behavioral responses
- **Claim:** Threat shifts choice (avoidance), vigor (activation), and affect (anxiety up, confidence down) simultaneously, and within-cookie vigor effects are masked by Simpson's paradox in unconditional analyses.
- **Evidence:** H1 — all 5 tests pass in both samples (N=290, N=281). Threat: beta = -0.91 to -1.02 on choice (P < 0.001). Vigor within cookie: d = 0.24-0.76. Anxiety: beta = +0.53-0.58, confidence: beta = -0.58 to -0.67 (all P < 0.001).
- **Significance:** Establishes that the task elicits the full defensive repertoire — avoidance, activation, and affective monitoring — not just one channel.

### Argument 2: Vigor follows the predatory imminence continuum
- **Claim:** Motor vigor shows distinct anticipatory (threat-modulated, graded) and reactive (encounter-triggered, all-or-nothing) dynamics, consistent with the predatory imminence continuum (Fanselow 1994; Mobbs et al. 2020).
- **Evidence:** H2 — encounter spike d = 0.56-0.65 (P < 0.001), threat-independent (P = 0.16). GAM LRT: encounter chi-squared = 760-1025, threat chi-squared = 115-292 (all P < 0.001). Both replicate.
- **Significance:** First demonstration of the imminence continuum in motor vigor (not just freezing/flight). Links animal defense theory to human motor control.

### Argument 3: A joint fitness function with two parameters best explains both channels
- **Claim:** The joint model (M4: omega + kappa) decisively outperforms effort-only, threat-only, and single-parameter alternatives on joint (choice + vigor) WAIC and LOO.
- **Evidence:** H3 — M4 wins by DWAIC = 1,621-4,729 across all comparisons in both samples. WAIC and LOO agree on every comparison. Choice accuracy = 0.76-0.77, vigor r-squared = 0.37-0.41. Parameter recovery: omega r = 0.94, kappa r = 0.92.
- **Significance:** The two parameters are identifiable and separable — capture cost and effort cost are genuinely distinct individual traits, not a single dimension.

### Argument 4: The parameters define ecologically meaningful foraging profiles
- **Claim:** Omega predicts who survives and what errors they make; kappa predicts motor output; their balance (angle) predicts decision quality. Effort-driven avoidance is less optimal than threat-driven avoidance.
- **Evidence:** H4 — 5/7 confirmatory (all core tests pass). Omega to escape: beta = 0.046 (HDI excludes zero). Kappa to vigor: beta = -0.196. Angle to optimality: beta = -0.054. 90% of errors are overcautious. H4e (consistency to earnings) did not replicate.
- **Significance:** The avoid-activate decomposition has real ecological consequences — it determines who survives predator attacks and who wastes resources on unnecessary caution.

### Argument 5: Metacognitive signals monitor the computation and bridge to clinical outcomes
- **Claim:** Anxiety calibration, anxiety slope, and confidence are three orthogonal metacognitive monitors that predict foraging efficiency beyond omega and kappa. Omega maps to confidence (coping appraisal) but not anxiety (threat appraisal). Task affect dissociates clinical dimensions: anxiety level to clinical anxiety, confidence to depression/apathy, calibration to apathy.
- **Evidence:** H5 — 7/7 confirmatory. Calibration improves LOO for all 3 outcomes (3/3). Slope to choice shift: beta = 0.099. Omega to confidence: beta = -0.181 (HDI excludes zero); omega to anxiety: beta = -0.067 (within ROPE). Confidence to overcaution: beta = -1.48; to reckless: beta = +0.29. Pooled clinical: confidence to DASS-Depression beta = -0.16; to AMI beta = -0.22 (both HDI exclude zero).
- **Significance:** The metacognitive layer contains information the computation doesn't — and it's the layer that connects to psychopathology. The foraging task produces affective biomarkers that dissociate along clinical dimensions.

---

## 4. Methodology

### 4.1 Design

Preregistered two-sample design. Exploratory sample (N = 350 recruited, N = 290 after exclusions) used to develop all hypotheses, model specifications, and statistical thresholds. Independent confirmatory sample (N = 350 recruited, N = 281 after exclusions) collected from a non-overlapping Prolific participant pool using an identical task. All hypotheses preregistered on OSF prior to any confirmatory analysis. Registration status: "Data exists but the authors have not observed it yet."

### 4.2 Participants

Recruited through Prolific. Inclusion: 18-65 years old, fluent in English, normal or corrected-to-normal vision. Paid base rate plus performance bonus proportional to total task score.

Study flow: (1) instruction comprehension assessment in Qualtrics, (2) brief video game use questionnaire, (3) foraging task (Unity WebGL, ~25 min), (4) post-task questionnaires: DASS-21, PHQ-9, OASIS, STAI (State and Trait), AMI, MFIS, STICSA.

### 4.3 Task

Participants forage in a circular arena under predation risk. On each trial, they choose between:
- **Heavy cookie:** R = 5 points, required pressing rate 0.9, distance D in {1, 2, 3} (5, 7, or 9 game units from safe zone)
- **Light cookie:** R = 1 point, required pressing rate 0.4, always at distance D = 1

After choosing, participants press keys (S+D+F) repeatedly to transport the cookie to the safe zone at the center. A predator may appear based on the stated threat probability T in {0.1, 0.5, 0.9}, displayed before each trial. Capture costs 5 points plus the current cookie reward.

**Trial types:**
- Choice trials (45 per participant): free selection between heavy and light cookies
- Probe trials (36 per participant): both options identical (forced choice); participants rate prospective anxiety (1-10) or confidence (1-10) before pressing. Cookie type, threat, and distance fully crossed: 3T x 3D x 2 cookie = 18 conditions, each sampled once for anxiety and once for confidence. Anxiety and confidence measured on separate probe trials (18 each).

**Structure:** 3 blocks x 27 trials = 81 total events. Threat, distance, and cookie assignment fully crossed within blocks. Trial order randomized within blocks.

**Calibration:** 3 x 10-second maximum-speed pressing trials at task start establish each participant's calibrationMax (presses/second), used to normalize all pressing rates.

### 4.4 Measured variables

**Behavioral:**
- Choice: binary (heavy = 1, light = 0) per choice trial
- Pressing rate: keypresses recorded at native ~5 Hz. Inter-press intervals (IPI) computed as successive timestamp differences; IPIs < 10 ms removed as artifacts. Primary metric: normalized press rate = median(1/IPI) / calibrationMax
- For timecourse analyses (H2): 200 ms bins smoothed with 3-point centered moving average (600 ms window)
- For model fitting (H3): per-subject condition cell means (subject x threat x distance x cookie, ~18 cells per subject, ~5,200 total), each the median normalized rate across trials within that condition
- Trial outcome: escaped or captured
- Total earnings: sum of rewards across all trials (determines bonus)

**Affective (probe trials only):**
- Anxiety rating (1-10): "How anxious are you about this trial?"
- Confidence rating (1-10): "How confident are you that you will succeed?"

**Derived indices:**
- Anxiety calibration: within-subject r(anxiety, threat). Higher = anxiety better tracks danger.
- Anxiety slope: within-subject regression slope of anxiety on threat
- Escape rate: proportion of attack trials survived
- Overcaution ratio: proportion of errors that are overcautious
- Omega-kappa angle: atan2(kappa_z, omega_z). Higher = more effort-driven avoidance.
- Choice consistency: fraction of trials matching model-predicted choice
- Intensity deviation: RMSE between model-predicted u* and observed cell-mean rate

### 4.5 Computational model

**Fitness function (Bednekoff 2007; Brown 1999):**

W(u) = S(u) * R - (1 - S(u)) * omega * (R + C) - kappa * (u - req)^2 * D

where:
- u = pressing rate (normalized by calibration maximum)
- S(u, T, D) = exp(-h * T^gamma * D / speed(u)) is survival probability
- speed(u) = sigmoid((u - 0.25 * req) / sigma_sp) is movement speed, saturating above req
- omega_i = per-subject avoidance sensitivity (subjective cost of capture)
- kappa_i = per-subject activation intensity (subjective cost of effort)
- R = cookie reward (5 or 1), C = 5 (capture penalty), req = required pressing rate (0.9 or 0.4)
- h, gamma, sigma_sp = population parameters (hazard scale, hazard exponent, speed saturation)

**Choice prediction:**
V_j = max_u W_j(u) - kappa * req_j * D_j (total demand cost). P(heavy) = sigmoid((V_H - V_L) / tau).

The total demand cost (kappa * req * D) enters the choice equation but not the vigor optimization. This reflects the distinction between deciding how much effort to commit (total demand) and optimizing moment-to-moment pressing intensity (marginal deviation cost). Both governed by the same kappa.

**Vigor prediction:**
u* = argmax_u W(u) for the chosen cookie. Cell-mean rate ~ Normal(u* + b_cookie * is_heavy, sigma_v / sqrt(n_trials)).

We use cell means rather than trial-level data because W(u) predicts a single optimal rate per condition — trial-to-trial variance is motor noise, not parametric signal. The sqrt(n_trials) denominator ensures cells with fewer observations receive less weight.

**Joint likelihood:**
L = Product(Bernoulli(P_heavy) over choice trials) x Product(Normal(u*, sigma) over vigor cells)

Both omega and kappa enter both likelihoods through the same W function.

**Priors (hierarchical, non-centered):**
- omega_i = exp(m_omega + s_omega * z_i), m_omega ~ Normal(0, 1), s_omega ~ HalfNormal(1.0)
- kappa_i = exp(m_kappa + s_kappa * z_i), m_kappa ~ Normal(-1, 1), s_kappa ~ HalfNormal(0.5)
- Population: gamma ~ Normal(0, 0.5) log-scale; h ~ Normal(0, 1) log-scale; sigma_sp ~ Normal(-1, 0.5) log-scale; tau ~ Normal(0, 1) log-scale; sigma_v ~ HalfNormal(0.3); b_cookie ~ Normal(0, 0.5)

**Inference:** NumPyro HMC/NUTS, 4 chains x 2,000 warmup + 4,000 samples, target_accept = 0.95, max_tree_depth = 10. Convergence: R-hat < 1.01, bulk ESS > 400. If non-convergent, double iterations.

**Parameter recovery:** 500 synthetic subjects simulated from known omega/kappa, refitted to verify identifiability.

### 4.6 Model comparison

Four models compared, all on the same joint (choice + vigor) likelihood:

- **M1 (Effort-only):** kappa per-subject. Choice: delta_V = delta_R - kappa * delta_effort(D). No survival function, no threat. Vigor: intercept-only (no condition structure). Tests whether threat adds anything beyond effort cost.
- **M2 (Threat-only):** omega per-subject, population kappa. Choice and vigor from W(u), but no per-subject effort sensitivity. Tests whether individual effort differences matter.
- **M3 (Single-parameter):** theta = omega = kappa. One parameter enters W(u) as both. Tests whether a single trait can serve both roles.
- **M4 (Joint):** omega + kappa per-subject, both entering W(u). Full model.

Primary criterion: WAIC. Robustness: PSIS-LOO. Hypothesis supported only if both agree.

### 4.7 Statistical analyses

**H1 (frequentist, P < 0.01):**
- H1a: Logistic regression with cluster-robust SE: choice ~ threat_z + dist_z + threat_z:dist_z, clustered by subject
- H1b: Linear mixed models: response ~ threat_z + dist_z + (1 + threat_z | subject), separately for anxiety and confidence. |t| > 3.
- H1c: Paired t-tests: within-subject mean normalized press rate at T=0.9 minus T=0.1, within heavy and light cookies separately

**H2 (frequentist, P < 0.01 / P < 0.001 for spike):**
- H2a: Encounter spike = per-subject mean reactive-epoch rate on attack minus non-attack. One-sample t vs 0.
- H2b: GAMs with natural cubic regression splines (K=10), MixedLM with cookie covariate and random intercepts. LRT for smooth-by-condition interactions.

**H3 (WAIC + LOO):**
- All four models fitted with identical NUTS inference. WAIC and LOO must both favor M4.

**H4/H5 (Bayesian, 95% HDI excludes zero):**
- All regressions: bambi, 4 chains x 2,000 draws + 1,000 tuning, default weakly informative priors
- H5a: LOO comparison, delta-ELPD > 0 with SE excluding zero
- H5c: ROPE [-0.10, +0.10] for null prediction on anxiety

**No multiple comparison correction** — each test is a specific directional prediction from the exploratory sample.

### 4.8 Data exclusion

**Subject-level:**
- Incomplete data: must complete all 81 trials + all modalities
- Calibration outliers: mean IPI > 2.5 SD from sample mean
- Task engagement: escape rate < 35% across attack trials

Exploratory: 60/350 excluded (57 incomplete/engagement, 3 calibration outliers) → N = 290. Confirmatory: 69/350 excluded → N = 281.

**Trial-level:**
- Non-response trials excluded
- IPIs < 10 ms treated as artifacts

### 4.9 Exploratory analyses (not preregistered)

1. Separate-equations model (lambda choice-only + omega vigor-only, no shared W)
2. Scaled single-parameter model (M3b: theta as omega, alpha*theta as kappa)
3. Posterior predictive checks
4. Affect index split-half reliability
5. Encounter spike individual differences (CV, split-half, parameter correlations)
6. Clinical regressions (pooled N ~ 580): all questionnaire scores on omega + kappa + affect
7. Trial-level anxiety-vigor coupling (LMM)
8. Frequentist robustness for H4/H5
9. Normative benchmark (model-derived optimal strategy vs participant behavior)

---

## 5. Data and Results

### 5.1 Data Files

| File | Description | How to Use |
|------|-------------|------------|
| `data/model_input_exploratory/` | Choice trials + vigor cell means (N=290) | Model fitting input |
| `data/model_input_confirmatory/` | Choice trials + vigor cell means (N=281) | Model fitting input |
| `results/stats/joint_optimal/exploratory/mcmc_model_comparison.csv` | WAIC/LOO for all models (exploratory) | H3 tables |
| `results/stats/joint_optimal/confirmatory/mcmc_model_comparison.csv` | WAIC/LOO for all models (confirmatory) | H3 tables |
| `results/stats/joint_optimal/exploratory/mcmc_m4_params.csv` | Per-subject omega, kappa (exploratory) | H4/H5 analyses |
| `results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv` | Per-subject omega, kappa (confirmatory) | H4/H5 analyses |
| `results/stats/confirmatory_hypothesis_results.csv` | All 24 test results with pass/fail | Summary table |
| `drafts/results.md` | Full results writeup | Results section |
| `drafts/preregistration_osf.md` | OSF preregistration | Methods + supplement |

### 5.2 Actual Results

**Result 1: H1 — Threat drives adaptive behavioral shifts (5/5 confirmed)**
- Finding: Threat reduces P(heavy) by beta = -0.91, increases vigor within cookie d = 0.45-0.76, increases anxiety beta = +0.53, decreases confidence beta = -0.67
- Significance: All P < 0.001, all replicate across samples
- Context: Establishes the basic phenomenon — threat shifts the full behavioral repertoire

**Result 2: H2 — Vigor follows the imminence continuum (3/3 confirmed)**
- Finding: Encounter spike d = 0.65, threat-independent (P = 0.16). GAM temporal signatures: encounter chi-squared = 1025, threat chi-squared = 115
- Significance: All P < 0.001
- Context: First motor vigor evidence for the predatory imminence continuum in humans

**Result 3: H3 — Joint model wins all comparisons (3/3 confirmed)**
- Finding: M4 outperforms M1 (DWAIC = +3,785), M2 (+1,621), M3 (+3,474). WAIC and LOO agree on all.
- Significance: Decisive (DWAIC in thousands)
- Context: The two parameters are necessary and separable — one function, two traits, both channels

**Result 4: H4 — Parameters predict survival, errors, and optimality (5/7 confirmed)**
- Finding: Omega to escape beta = +0.046. Kappa to vigor beta = -0.196. Angle to optimality beta = -0.054. 90% of errors overcautious.
- Significance: All HDIs exclude zero except H4e (consistency to earnings did not replicate)
- Context: The avoid-activate decomposition has real ecological consequences

**Result 5: H5 — Metacognition monitors the computation (7/7 confirmed)**
- Finding: Calibration improves LOO for 3/3 outcomes. Omega to confidence beta = -0.181 but omega to anxiety null (ROPE). Confidence to overcautious beta = -1.48, to reckless beta = +0.29.
- Significance: All HDIs exclude zero. Full replication.
- Context: The metacognitive layer adds information the computation doesn't contain

**Null / Negative results:**
- H4e: Model consistency (choice + intensity deviation) did not predict earnings in confirmatory sample. The indirect link between computational consistency and task performance is weaker than direct parameter-outcome relationships.
- Computational parameters (omega, kappa) do not directly predict clinical symptoms in either sample. The clinical bridge is through the affect layer (confidence, calibration), not the computation.

### 5.3 Figures and Tables

| File | Caption | Placement |
|------|---------|-----------|
| `results/figs/paper/fitness_function.png` | W(u) by threat, omega, and kappa | Methods / Model |
| `results/figs/paper/H1a_choice_surface.png` | P(heavy) by threat x distance | Results, H1 |
| `results/figs/paper/H1c_vigor_by_threat.png` | Normalized press rate by threat (within cookie) | Results, H1 |
| `results/figs/paper/H2a_vigor_by_threat.png` | Vigor by threat within cookie type | Results, H2 |
| `results/figs/paper/H3_model_comparison.png` | DWAIC bar chart for both samples | Results, H3 |
| To generate | Parameter space (omega vs kappa) with foraging profiles | Results, H4 |
| To generate | Affect-clinical dissociation heatmap | Results, H5/Clinical |
| To generate | Encounter-aligned vigor timecourse | Results, H2 |

**Figure formatting notes:**
Nature Communications standard: single column (88mm) or double column (180mm). 300 DPI minimum. Sans-serif fonts. Panels labelled a, b, c.

### 5.4 Expected vs. Actual

All core predictions confirmed (22/24 = 92%). The two failures (H4e) reflect the weakest, most indirect prediction — that computational consistency with one's own fitness function predicts total earnings. The direct parameter-outcome links (H4a-d) all replicated strongly.

The clinical findings exceeded expectations: the triple dissociation (anxiety level to clinical anxiety; confidence to depression/apathy; calibration to apathy) was not preregistered but emerged clearly in the pooled sample and is theoretically coherent.

**Surprise:** Computational parameters are clinically silent — omega and kappa do not predict depression, anxiety, or any clinical measure directly. This was the preregistered expectation but contradicts the intuition that "people who perceive more threat should be more anxious." The affect layer (confidence, calibration) is the bridge, not the computation.

---

## 6. Scope and Boundaries

**In scope:**
- Joint computational model of choice and vigor under threat
- Individual differences in avoidance sensitivity and activation intensity
- Metacognitive monitoring (anxiety, confidence) of the foraging computation
- Predatory imminence continuum in motor vigor
- Clinical correlates (exploratory, pooled sample)
- Preregistered two-sample confirmation

**Out of scope:**
- Neural correlates (no neuroimaging)
- Developmental or longitudinal changes
- Cross-species comparison (though the model is grounded in animal ecology)
- Treatment implications (we identify correlates, not causes)
- Individual item-level questionnaire analysis (factor analysis is supplementary)

**Target length:** 5,000-7,000 words (Nature Communications Article format)

**Target audience:** Computational neuroscientists, behavioral ecologists, clinical/computational psychiatrists, decision scientists

---

## 7. Paper Outline

```
Abstract (150 words)

1. Introduction
   1.1 The foraging-under-threat problem
   1.2 Optimal foraging theory: from animals to humans
   1.3 The predatory imminence continuum and motor vigor
   1.4 Metacognitive monitoring of threat computations
   1.5 Present study and preregistration

2. Results
   2.1 Threat drives adaptive shifts in choice, vigor, and affect (H1)
   2.2 Vigor dynamics follow the predatory imminence continuum (H2)
   2.3 A joint fitness model outperforms alternatives (H3)
   2.4 Model parameters define foraging profiles (H4)
   2.5 Metacognitive signals monitor the computation (H5)
   2.6 Task affect dissociates clinical dimensions (exploratory)

3. Discussion
   3.1 A common computational structure across decision, action, and emotion
   3.2 The avoid-activate decomposition
   3.3 Metacognition as the bridge to psychopathology
   3.4 Limitations
   3.5 Conclusion

4. Methods
   4.1 Participants and design
   4.2 Task
   4.3 Computational model
   4.4 Statistical analyses
   4.5 Preregistration

Supplementary Information
   S1 Model specification details
   S2 Parameter recovery
   S3 Posterior predictive checks
   S4 M3 non-convergence
   S5 Full clinical regression tables
   S6 Split-half reliability of affect indices
```

---

## 8. Key References

- Bednekoff, P. A. (2007). Foraging in the face of danger. In Foraging: Behavior and Ecology.
- Brown, J. S. (1999). Vigilance, patch use and habitat selection. Evolutionary Ecology Research.
- Lima, S. L., & Dill, L. M. (1990). Behavioral decisions made under the risk of predation. Canadian Journal of Zoology.
- Fanselow, M. S. (1994). Neural organization of the defensive behavior system. Psychonomic Bulletin & Review.
- Mobbs, D., et al. (2020). Space, time, and fear: survival computations along defensive circuits. Trends in Cognitive Sciences.
- Shadmehr, R., & Krakauer, J. W. (2008). A computational neuroanatomy for motor control. Experimental Brain Research.
- Yoon, T., et al. (2018). Control of movement vigor and decision making during foraging. PNAS.
- Fleming, S. M., & Daw, N. D. (2017). Self-evaluation of decision-making. Psychological Review.
- Lazarus, R. S. (1991). Emotion and Adaptation. Oxford University Press.
- Wells, A. (2009). Metacognitive Therapy for Anxiety and Depression. Guilford Press.
- Shenhav, A., et al. (2013). The expected value of control. Neuron.
- Capretto, T., et al. (2022). Bambi: A simple interface for fitting Bayesian linear models in Python.

---

## 9. Reviewer Guidance

**Preferred reviewers:** Simulate reviewers with expertise in:
1. Computational psychiatry / reinforcement learning modeling (e.g., someone from Daw, Dolan, or Huys labs)
2. Foraging / behavioral ecology in humans (e.g., Kolling, Hayden, or Mobbs-adjacent)
3. Metacognition and anxiety (e.g., Fleming, Paulus, or Robinson labs)

**Review emphasis:**
- Is the joint model genuinely novel or incremental over existing choice-only or vigor-only models?
- Is the preregistered confirmation convincing? Are 22/24 tests sufficient?
- Is the clinical story (exploratory) appropriate for the main text or should it be supplementary?
- Does the paper overreach in its theoretical claims?

---

## 10. Notes to the Writer Agent

- This is a preregistered study. Emphasize the two-sample design throughout. The exploratory sample was used to develop everything; the confirmatory sample tests it. This is the paper's main methodological strength.
- The tone should be confident but precise. We have strong results (22/24 replicate) and a clean model comparison (DWAIC in thousands). Don't hedge unnecessarily.
- The model is grounded in Bednekoff's life-history framework from animal ecology. This is not "just another computational model" — it has deep theoretical roots in a 30-year literature on foraging under predation.
- The clinical results are exploratory. Frame them as "extending the framework" not as primary claims. They belong in the main text but clearly marked as exploratory.
- H4e failed. Report it honestly — this strengthens credibility. The direct parameter-outcome links (H4a-d) replicate; the indirect consistency link doesn't. Frame as: direct effects robust, consistency metric needs refinement.
- Nature Communications format: Results before Methods. Introduction should be concise (4-5 paragraphs). Discussion should be focused. Methods in detail at the end.
- All code and data will be made available. The analysis notebooks are designed so reviewers can reproduce every result.
- Do NOT use the word "novel" — Nature Comms editors hate it. Use "demonstrate" or "show" instead.
- Refer to the preregistration throughout as evidence of confirmatory intent.
