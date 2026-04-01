Affect Modeling Analysis Plan
Overview
For each subject, we estimate how anxiety and confidence respond to trial conditions on probe trials using a first-stage regression. The resulting per-subject parameter vectors are then used as predictors or outcomes in second-stage analyses targeting three questions: (1) are anxiety and confidence informationally dissociable, (2) does the computational model predict coping appraisal but not threat appraisal, and (3) does the anxiety-confidence gap predict maladaptive behavior and clinical risk.

Stage 1: Per-Subject Affect Models
Data: Probe trials only (N=36 per subject, forced-choice with ratings collected before pressing).
Models: Two parallel mixed models, one for anxiety, one for confidence:
rating_ij = b0_i + b1_i(T) + b2_i(D) + b3_i(is_heavy) + e_ij
where:

T ∈ {0.1, 0.5, 0.9} — threat probability, entered as continuous
D ∈ {1, 2, 3} — distance of heavy cookie, entered as continuous
is_heavy — whether the forced-choice probe involves the heavy cookie (check first whether this varies on probe trials — if not, drop b3)
i indexes subjects, j indexes trials

Fit via statsmodels MixedLM with random intercepts and random slopes on T and D. Extract posterior means of per-subject random effects as the Stage 1 parameters.
Output per subject, per rating:
ParameterSymbolInterpretationInterceptb0Baseline affect at T=0.1, D=1Threat slopeb1Sensitivity to stated threat probabilityDistance slopeb2Sensitivity to effort cost of heavy cookieCookie slopeb3Affect shift when heavy cookie is involved (if estimable)
You now have eight parameters per subject — four for anxiety, four for confidence — that fully characterize each person's affective response profile across conditions.

Stage 2: Core Analyses
Analysis A — Are anxiety and confidence informationally dissociable?
A1. Slope correlation structure
For each pair of corresponding parameters, compute the across-subject correlation:
r(b1_anx, b1_conf)  — threat sensitivity alignment
r(b2_anx, b2_conf)  — distance sensitivity alignment
r(b0_anx, b0_conf)  — baseline alignment
Prediction: b0 correlations will be moderate-to-strong negative (scared people feel less confident at baseline — these are not independent). b1 correlations will be weaker — threat sensitivity in anxiety and confidence are more dissociable than baseline levels. b2 correlations will be weakest or near zero — distance (effort cost) enters the two systems differently.
This gives you an empirical dissociation profile rather than a theoretical claim.
A2. Unique variance in behavior
Regress escape rate and optimality separately onto:

Model 1: b1_anx alone
Model 2: b1_conf alone
Model 3: b1_anx + b1_conf jointly

If both slopes contribute unique variance in Model 3, anxiety and confidence carry non-redundant behavioral information. If only b1_conf survives, confidence is the behaviorally relevant signal and anxiety is epiphenomenal. Either result is interesting and interpretable.

Analysis B — Does omega predict coping appraisal specifically?
B1. Omega predicts confidence threat slope, not anxiety threat slope
OLS: b1_conf ~ omega_i
OLS: b1_anx ~ omega_i
Prediction: omega positively predicts b1_conf (high capture cost → confidence rises more steeply with threat, because high-omega people know they'll press harder and escape more as stakes rise) but does not predict b1_anx (the computational danger estimate has no purchase on affective threat sensitivity).
This is a stronger and more mechanistic version of your current H5f. It says omega doesn't just predict confidence level, it predicts the dynamic structure of confidence across the threat gradient.
B2. Kappa predicts distance sensitivity of confidence
OLS: b2_conf ~ kappa_i
OLS: b2_anx ~ kappa_i
Prediction: kappa negatively predicts b2_conf — people for whom effort is costly feel less confident as distance increases, because distance amplifies their effort cost and they know it. Again no prediction for anxiety. This would be a clean double dissociation: omega maps onto threat sensitivity of confidence, kappa maps onto effort sensitivity of confidence, neither maps onto anxiety.

Analysis C — The anxiety-confidence gap
C1. Operationalize the gap
For each subject compute a gap parameter at each threat level:
gap_i(T) = b0_anx_i + b1_anx_i * T - (b0_conf_i + b1_conf_i * T)
This is the predicted anxiety minus predicted confidence at each threat level for each person. It captures how much their fear outpaces their sense of coping capacity, and how that relationship changes across threat levels.
Summarize per subject as:

gap_baseline: gap at T=0.1 (baseline mismatch)
gap_slope: how rapidly the gap widens as threat increases (b1_anx - b1_conf)

C2. Gap predicts overcaution
OLS: overcaution_ratio ~ gap_baseline + gap_slope + omega + kappa
Prediction: gap_baseline predicts overcaution beyond omega and kappa. People whose fear chronically outpaces their coping estimate avoid even when the computation says they shouldn't. This is a formal operationalization of the Lazarus primary/secondary appraisal mismatch and it explains overcaution in a way that neither omega nor kappa alone can.
C3. Gap predicts clinical symptoms
OLS: STAI_state ~ gap_baseline + gap_slope
OLS: DASS_anxiety ~ gap_baseline + gap_slope
OLS: OASIS ~ gap_baseline + gap_slope
Prediction: gap_baseline specifically predicts clinical anxiety symptoms. gap_slope may not — clinical anxiety is better characterized as a chronic mismatch than as threat-reactive amplification. This would tie the task measure to clinical constructs more precisely than mean anxiety level does, because it's not just that anxious people feel more fear — it's that their fear is uncoupled from their coping estimate.
C4. Gap versus calibration as predictors
Run head-to-head regressions putting your current calibration measure (within-subject r of anxiety with threat) against gap_baseline as predictors of the same outcomes. This lets you evaluate whether the gap construct adds something beyond what calibration already captures, and gives you a principled way to choose between the two framings.

Analysis D — Metacognitive accuracy
D1. Operationalize accuracy
For each subject compute:
conf_accuracy_i = r(confidence_ij, omega_predicted_escape_ij)
where omega_predicted_escape is the model-implied survival probability S(u*, T, D) for that probe trial. This is the correlation between what the person feels (confidence) and what the model predicts they should feel given their own omega and kappa. High accuracy means their confidence tracks their actual computational situation. Low accuracy means their confidence is decoupled from their objective coping capacity.
D2. Accuracy predicts error type
OLS: n_overcautious ~ conf_accuracy_i
OLS: n_reckless ~ conf_accuracy_i
Prediction: low accuracy in the pessimistic direction (feels less confident than omega predicts) → overcaution. Low accuracy in the optimistic direction → recklessness. This replaces your current H5g (confidence level predicts error type) with a more precise claim about miscalibration rather than level, which is theoretically cleaner and harder to dismiss.
D3. Does anxiety accuracy exist independently?
Compute the analogous measure for anxiety:
anx_accuracy_i = r(anxiety_ij, T_j)
This is just your existing calibration measure. But now you can directly compare conf_accuracy and anx_accuracy as independent predictors of behavior and clinical outcomes, making the dissociation empirical rather than assumed.

What to run on the exploratory sample now
In order:

Check probe trial design matrix — verify T, D, cookie variation on probe trials
Fit Stage 1 mixed models, extract per-subject slope parameters
Run Analysis A (dissociation structure) — this tells you whether the framework is viable
Run Analysis B (omega/kappa → slopes) — this is your strongest mechanistic claim
Run Analysis C (gap construct) — this is your clinical bridge
Run Analysis D (accuracy) — this is most dependent on model output so do it last

Then look at what held up and preregister the effects you want to confirm. Realistically you'll probably find that B1 and C2 are the strongest and those become your confirmatory tests. Everything else goes in as exploratory replication.