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

## 2. Hypothesis

**Central hypothesis:**
A single fitness function W(u) = S(u)R - (1-S(u))omega(R+C) - kappa(u-req)^2 D, derived from optimal foraging theory (Bednekoff 2007; Brown 1999), simultaneously predicts patch selection and motor vigor through two separable individual-difference parameters: avoidance sensitivity (omega) and activation intensity (kappa). Metacognitive signals — anxiety and confidence — monitor this computation and bridge to clinical outcomes.

**Alternative hypotheses (tested as M1-M3):**
- M1 (Effort-only): Choice depends only on effort cost, threat is irrelevant, vigor has no systematic structure
- M2 (Threat-only): Only survival matters; individual effort differences are irrelevant
- M3 (Single-parameter): One trait governs both avoidance and activation — they are not separable

**Null hypothesis:**
The joint model (M4) does not outperform simpler alternatives; omega and kappa are not identifiable as separate traits; metacognitive signals do not add information beyond the computational parameters.

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

**Approach:**
Preregistered two-sample design: exploratory (N=290) for hypothesis development, independent confirmatory (N=281) for testing. All hypotheses, model specifications, and thresholds preregistered on OSF prior to confirmatory analysis.

**Task:**
Online foraging task built in Unity (WebGL). Participants choose between high-reward/high-effort and low-reward/low-effort patches in a circular arena under probabilistic predation risk (T in {0.1, 0.5, 0.9}), then press keys (S+D+F) to transport the chosen reward to safety. Predators may appear and capture the participant (costing 5 points + cookie reward). Probe trials collect prospective anxiety and confidence ratings. 81 trials per participant (45 choice + 36 probe). Participants recruited from Prolific (18-65 years, English-speaking). Performance bonus proportional to total score.

**Computational model:**
Joint fitness function W(u) with saturating survival function S(u,T,D) = exp(-h T^gamma D / speed(u)), where speed = sigmoid((u - 0.25 req) / sigma_sp). Two per-subject parameters (omega, kappa) estimated via hierarchical Bayesian inference (NumPyro HMC/NUTS, 4 chains, 2000 warmup + 4000 samples). Joint likelihood over choice (Bernoulli) and vigor (Normal on condition cell means).

**Statistical framework:**
- H1/H2: Frequentist (logistic regression, LMMs, paired t-tests, GAMs). Threshold: P < 0.01.
- H3: WAIC (primary) + PSIS-LOO (robustness). Both must agree.
- H4/H5: Bayesian linear models (bambi). Threshold: 95% HDI excludes zero.
- Clinical analyses: Exploratory, pooled (N=563), Bayesian.

**Questionnaires:** DASS-21, PHQ-9, OASIS, STAI (State + Trait), AMI, MFIS, STICSA.

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
