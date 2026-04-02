# Three separable cost signals govern foraging under threat: effort, threat aversion, and defensive vigor

Noah Okada^1^, Ketika Garg^1^, Toby Wise^2^, Dean Mobbs^1,3^

^1^ Division of the Humanities and Social Sciences, California Institute of Technology, Pasadena, CA, USA
^2^ Department of Neuroimaging, King's College London, London, UK
^3^ Computation and Neural Systems Program, California Institute of Technology, Pasadena, CA, USA

---

## Abstract

Foraging under predation risk requires integrating energetic cost and mortality risk into a single decision variable, yet whether these costs reflect a unitary sensitivity or separable computational signals remains unknown. Here we develop a three-parameter choice-vigor model and test it in a large online effort-foraging task under parametric threat (N = 293). Three subject-level parameters---effort cost (*k*), threat aversion ($\beta$), and capture aversion (*c*~d~)---govern distinct behavioral channels: *k* and $\beta$ enter the choice equation as separable linear costs of effort and threat, while *c*~d~ scales the penalty for failed escape and governs motor vigor. The model achieves per-subject choice *r*^2^ = 0.981 and trial-level vigor *r*^2^ = 0.424. Parameter recovery confirms all three are identifiable (*k*: *r* = 0.85; $\beta$: *r* = 0.84; *c*~d~: *r* = 0.93) and that *k* and $\beta$ are orthogonal (cross-recovery *r* = 0.03; empirical *r* = -0.006). The central result is a triple dissociation: *k* explains 77% of unique variance in overcautious foraging errors (partial *r* = 0.93); $\beta$ explains threat-driven choice avoidance (partial *r* = 0.78); and *c*~d~ explains the vigor gap relative to optimal pressing (*r* = 0.58). Together, the three parameters explain 89% of variance in overcaution (*R*^2^ = 0.887). At the condition level, *k* dominates low-threat choice (*r* = -0.94 with P(heavy) at *T* = 0.1) while $\beta$ dominates high-threat choice (*r* = -0.84 with P(heavy) at *T* = 0.9). Encounter dynamics reveal that *c*~d~ bridges tonic vigor and phasic defensive reflexes (*r* = 0.39), while $\beta$ does not predict reactivity (*r* = 0.08, *p* = .19), confirming that threat aversion in choice and defensive motor mobilization are dissociable. The model's survival signal tracks trial-level anxiety ($\beta$ = -0.56, *t* = -14.0) and confidence ($\beta$ = 0.58, *t* = 13.5). Affective calibration predicts policy alignment ($\Delta$*R*^2^ = 0.4%, *p* = .07), while affective discrepancy explains residual overcaution beyond the three parameters ($\Delta$*R*^2^ = 0.2%, *p* = .035) and predicts anxiety symptoms across 8 of 10 clinical instruments (*r* = 0.18--0.33). The only significant parameter-clinical link is $\beta$ to apathy (AMI: *r* = 0.14, *p* = .015); *k* and *c*~d~ show no clinical associations. A preregistered confirmatory replication (N = 350) is underway.

---

## Introduction

Animals foraging under predation risk face a fundamental optimization problem: the energy gained from distant, high-value resources must be weighed against increased exposure to capture during transport^1,2^. This trade-off shapes behavior across taxa, from birds adjusting provisioning rates near raptor territories^3^ to fish venturing from shelter to access food patches^4^. In humans, analogous computations arise whenever pursuing goals demands sustained effort under threat---persisting at hazardous work, commuting through dangerous environments, or investing cognitive effort when the costs of failure are severe. Theoretical ecology has long modeled this trade-off using reproductive value as the common currency^1,5^, but translating these models into a computational framework that specifies how cost-benefit calculation governs both what an organism chooses and how vigorously it acts has remained an open challenge.

Three research traditions have addressed the components of this problem in isolation. First, work on effort-based decision-making has established that humans discount reward value by the physical or cognitive effort required to obtain it^6--9^, with individual differences in effort cost linked to apathy, fatigue, and motivational disorders^10,11^. Second, research on defensive behavior has characterized how organisms modulate responses across the predatory imminence continuum---from strategic avoidance at a distance to reactive flight when danger is immediate^5,12--14^. Third, the vigor of motor execution reflects the marginal value of time under the current motivational state^15,16^, suggesting that threat should intensify physical effort even after the decision to forage has been made. Yet these three literatures have developed largely in parallel: effort discounting models ignore survival, defensive behavior models ignore energetic cost, and vigor theories have not been tested under ecological threat where both costs operate simultaneously.

A critical open question is whether effort sensitivity and threat sensitivity reflect a single latent variable or separable computational signals. If one person avoids a dangerous, effortful foraging opportunity, is that because they are generally cautious, because they are specifically effort-averse, or because they are specifically threat-averse? Existing paradigms cannot answer this question because they manipulate effort and threat in isolation. A joint manipulation is necessary, but even then, a model must be specified that can separate these influences. The Expected Value of Control (EVC) framework^17,18^ provides a candidate architecture: EVC proposes that the brain computes the expected payoff of allocating control effort, selecting the intensity that maximizes reward minus cost. Because EVC treats effort allocation as a continuous optimization, it naturally extends to predict both discrete choices and graded action vigor within a unified framework. However, EVC has been applied primarily to cognitive control in abstract tasks^19,20^ and has not been extended to physical effort under ecological threat.

Here we develop a three-parameter choice-vigor model and test it in a virtual effort-foraging task under parametric threat (N = 293). The model separates three cost signals: effort cost (*k*), threat aversion ($\beta$), and capture aversion (*c*~d~). Effort cost and threat aversion enter the choice equation as independent linear terms---there is no shared survival function in choice, so the two costs are structurally separable. Capture aversion governs vigor through a speed-dependent survival function that captures the physical mechanics of escape. We derive condition-specific optimal policies and show that each parameter drives a specific type of deviation from optimality, producing a triple dissociation. We then examine the model's survival signal as a computational substrate of moment-to-moment affect, and ask whether individual differences in the accuracy and bias of the affect-danger relationship carry additional functional consequences beyond what the model parameters explain. A preregistered confirmatory replication (N = 350) with identical task design, model specification, and analysis plan is underway.

---

## Results

### Threat and distance deter high-effort foraging and modulate vigor

We designed an effort-foraging task in which participants (N = 293, after five-stage quality screening of 350 recruits) chose between a high-reward cookie (5 points, requiring sustained keypressing at 60--100% of individually calibrated maximum capacity) and a low-reward cookie (1 point, requiring 40% of maximum) while facing predation risk (threat probability *T* $\in$ {0.1, 0.5, 0.9}) at varying distances (*D* $\in$ {1, 2, 3}; Fig. 1a). On attack trials, a predator appeared and pursued the participant; capture incurred a 5-point penalty and loss of the cookie's value. Participants completed 45 choice trials and 36 probe trials (forced-choice with identical options, paired with affect ratings).

Both threat and distance reduced high-effort choice (Fig. 1b). A logistic mixed-effects model confirmed threat as the dominant deterrent ($\beta$ = -1.28, *z* = -32.0, *p* < 10^-200^), with distance ($\beta$ = -0.65, *z* = -16.3, *p* < 10^-59^) and their interaction ($\beta$ = -0.18, *z* = -4.5, *p* < 10^-5^) also significant. At the extremes, P(heavy) dropped from 0.81 at *T* = 0.1, *D* = 1 to 0.08 at *T* = 0.9, *D* = 3. The mean proportion of heavy choices was 0.43 overall, declining from 0.69 at low threat to 0.21 at high threat (threat sensitivity: mean = 0.48, SD = 0.30).

Threat also increased motor vigor, but this effect was masked by a Simpson's paradox in unconditional analyses. High threat shifts choice toward light cookies, and light cookies have lower required press rates, so collapsing across cookie type makes average vigor appear flat or even decline. Conditioning on chosen cookie type revealed robust threat-driven vigor increases (heavy cookies: *t* = 6.6, *p* < 10^-10^, *d* = 0.42; light cookies: *t* = 7.5, *p* < 10^-13^, *d* = 0.49; Fig. 1c). Within each cookie type, participants pressed harder when threat was higher---consistent with the prediction that the marginal survival benefit of faster pressing increases with danger. This Simpson's paradox illustrates why a joint model of choice and vigor is necessary: analyzing either channel in isolation yields misleading conclusions about how threat shapes behavior.

### A three-parameter model separates effort cost, threat aversion, and defensive vigor

We formalized the foraging decision as a comparison of expected utilities:

$\Delta EU = 4 - k_i \times \text{effort}(D) - \beta_i \times T$

where *k*~i~ is the subject-specific effort cost, $\beta$~i~ is the subject-specific threat aversion, effort(*D*) captures the distance-dependent physical cost of the heavy cookie relative to the light cookie, and *T* is the stated threat probability. The choice equation is deliberately simple: effort and threat enter as separable linear costs, with no shared survival function. This means that *k* and $\beta$ are structurally identified from different dimensions of the choice surface---*k* from the distance gradient at fixed threat, $\beta$ from the threat gradient at fixed distance. Choices follow a softmax rule: P(heavy) = sigmoid($\Delta$EU / $\tau$), with population-level inverse temperature $\tau$ = 1.06.

Three features of this specification deserve emphasis. First, compared to our earlier model that incorporated a survival function *S* and probability weighting $\gamma$ in the choice equation, the present formulation drops both. This simplification was motivated by the finding that $\gamma$ and a separate effort-efficacy parameter $\varepsilon$ confounded the identification of $\beta$: when threat probability is transformed nonlinearly through *S*, the threat and effort contributions to choice become entangled. The linear specification achieves superior choice prediction (per-subject *r*^2^ = 0.981 vs. 0.951 in the earlier model) while enabling clean separation of the two cost signals.

Second, the absence of a survival function in choice does not mean participants ignore survival---it means that the *choice-relevant* threat signal is well-captured by a linear function of stated threat probability. The interaction between threat and distance that is observed in the behavioral data emerges naturally from the model because effort(*D*) increases with distance while $\beta \times T$ increases with threat; the sigmoid nonlinearity in the softmax link then generates the observed T $\times$ D interaction pattern.

Third, the model estimates three per-subject parameters: *k* (median = 1.48, IQR = [1.05, 2.04]), $\beta$ (median = 4.30, IQR = [3.21, 5.71]), and *c*~d~ (median = 30.9, IQR = [10.7, 49.6]). The key empirical result is that *k* and $\beta$ are essentially uncorrelated (*r* = -0.006), meaning that knowing how effort-sensitive a person is provides no information about how threat-sensitive they are. This orthogonality is not guaranteed by the model architecture---it is an empirical finding about the structure of individual differences in this sample.

For the vigor component, the model computes optimal press rate *u** by maximizing:

$EU(u) = S(u) \times R - (1 - S(u)) \times c_{d,i} \times (R + C) - c_{e,\text{vigor}} \times (u - \text{req})^2 \times D$

where *c*~d,i~ is the subject-specific capture aversion, *c*~e,vigor~ = 0.003 is the population-level motor deviation cost, and *S*(*u*) is the speed-dependent survival function:

$S(u) = (1 - T) + T \times p_{\text{esc}} \times \text{sigmoid}((u - \text{req}) / \sigma_{\text{motor}})$

with population-level escape probability *p*~esc~ = 0.002 and motor noise $\sigma$~motor~ = 0.82. The survival function in vigor captures the physical mechanics of escape: pressing faster than required improves the probability of outrunning a predator. Unlike the choice equation, the vigor equation necessarily includes *S*(*u*) because the speed-escape relationship is a physical constraint of the task, not a subjective weighting. Higher *c*~d~ drives steeper threat-vigor gradients because the marginal benefit of pressing faster grows with the stakes of failed escape.

Probe trials (forced conditions with identical options) anchor *c*~d~ estimation across all threat-by-distance cells without selection bias. The model predicted trial-level vigor with *r*^2^ = 0.424 (per-subject *r*^2^ = 0.669; Fig. 2c), with population-level vigor observation noise $\sigma$~v~ = 0.241.

Parameter recovery confirmed identifiability. Simulating synthetic subjects at the fitted population distribution and refitting yielded correlations of *r* = 0.85 for *k*, *r* = 0.84 for $\beta$, and *r* = 0.93 for *c*~d~ (all exceeding the *r* > 0.70 threshold; Fig. 2d). Critically, cross-parameter recovery was near zero: recovering *k* from data generated with known $\beta$ yielded *r* = 0.03, confirming that the two choice parameters are truly separable and not trading off against each other in the fitting procedure. The choice model achieved accuracy = 82.5%.

### A triple dissociation links each parameter to a distinct behavioral signature

The central result is that the three parameters show a triple dissociation with respect to foraging behavior (Fig. 3).

**Effort cost drives overcautious foraging.** We derived the EV-maximizing choice for each of the 9 conditions (3 threat $\times$ 3 distance) using empirical conditional survival rates (Table 1). Comparing each participant's choices against this optimal policy, participants achieved 69.8% optimality (SD = 12.0%), with 21.3% overcautious errors (choosing light when heavy was optimal; SD = 14.4%) and 8.9% overrisky errors (choosing heavy when light was optimal; SD = 8.4%).

Effort cost predicted overcautious errors with remarkable strength: the bivariate correlation between *k* and overcautious rate was *r* = 0.885 (*p* < 10^-98^), and the partial correlation controlling for $\beta$ and *c*~d~ was *r* = 0.933 (*p* < 10^-131^; Fig. 3a). In a multiple regression with all three parameters, *k* explained 76.8% of unique variance in overcaution, while $\beta$ contributed 10.2% and *c*~d~ was negligible (0.07%, *p* = .20). Together, the three parameters explained 88.7% of variance in overcautious error rate (*R*^2^ = 0.887). Effort cost also strongly predicted total earnings (*r* = -0.789, *p* < 10^-63^): participants with higher *k* earned substantially less because they systematically avoided high-value options.

**Threat aversion drives threat-specific choice avoidance.** The parameter $\beta$ captured sensitivity to threat in choice: bivariate *r* = 0.574 (*p* < 10^-27^), partial *r* = 0.779 (*p* < 10^-61^) controlling for *k* and *c*~d~ (Fig. 3b). At the condition level, the dissociation was sharp: *k* dominated low-threat choice (*r* = -0.937 with P(heavy) at *T* = 0.1) while $\beta$ dominated high-threat choice (*r* = -0.840 with P(heavy) at *T* = 0.9). This means that when threat is low, whether someone chooses the effortful option depends almost entirely on their effort cost; when threat is high, it depends primarily on their threat aversion. The T $\times$ D interaction in choice arises because *k* and $\beta$ have different "catchment areas" across the threat-distance surface.

**Capture aversion drives the vigor gap.** The parameter *c*~d~ predicted how far below optimal each participant pressed: bivariate *r* = 0.580 (*p* < 10^-27^), partial *r* = 0.587 (*p* < 10^-28^) controlling for *k* and $\beta$ (Fig. 3c). This differential prediction pattern confirms that the three parameters capture distinct dimensions of foraging behavior: *k* governs foregone opportunity (passing up profitable options), $\beta$ governs threat-driven avoidance (avoiding options when danger is high), and *c*~d~ governs vigor investment (pressing less hard than the stakes warrant).

**Table 1. Optimal foraging policy by condition.**

| Threat | Distance | Optimal choice | EV advantage |
|--------|----------|---------------|--------------|
| 0.1 | 1 | Heavy | +2.15 |
| 0.1 | 2 | Heavy | +1.55 |
| 0.1 | 3 | Heavy | +0.39 |
| 0.5 | 1 | Heavy | +0.80 |
| 0.5 | 2 | Heavy | +0.07 |
| 0.5 | 3 | Light | -0.84 |
| 0.9 | 1 | Light | -0.94 |
| 0.9 | 2 | Light | -0.75 |
| 0.9 | 3 | Light | -1.07 |

### Encounter dynamics: capture aversion bridges tonic and phasic defense

The static model predicts trial-level average vigor, but foraging under threat unfolds dynamically within each trial. We examined the 20 Hz vigor timeseries aligned to predator encounter events to test whether *c*~d~ extends to within-trial temporal dynamics.

Encounter reactivity---the difference between post-encounter and pre-encounter excess vigor---showed no reliable population mean effect (*M* = -0.019, *SD* = 0.28, *t* = -1.15, *p* = .25), but massive individual variation. This variation was highly stable across task blocks (mean cross-block *r* = 0.78), identifying encounter reactivity as a trait-like individual difference. Piecewise regression confirmed a qualitative shift at the encounter: pre-encounter vigor declined while post-encounter vigor increased (slope change = 0.050, *t* = 5.91, *p* < 10^-8^).

Encounter reactivity correlated with capture aversion (*r* = 0.39 with *c*~d~, *p* < 10^-12^) but not with effort cost (*r* = -0.11, *p* = .06) and, critically, not with threat aversion ($\beta$: *r* = 0.08, *p* = .19; Fig. 4a). This last null is important: $\beta$ captures how much threat deters choice, but it does not predict how vigorously someone mobilizes when the predator actually appears. Threat aversion in choice and defensive motor mobilization are governed by different systems. Threat probability did not modulate the encounter response (one-way ANOVA: *F* = 0.04, *p* = .96; Fig. 4b). This dissociation---threat modulates strategic choice ($\beta$ = -1.28 in the logistic model) but not the phasic encounter reflex---maps directly onto the distinction between strategic and reactive defensive modes^5,14^.

That *c*~d~ captures both tonic pressing level (trial-average vigor) and this phasic reflex (*r* = 0.39) supports interpreting it as a general defensive motor readiness parameter spanning the continuum from sustained vigilance to acute encounter response. This parameter bridges the strategic-reactive divide that the threat-imminence literature has described qualitatively^5,14^: the same individuals who press harder throughout a dangerous trial also mobilize more sharply when the predator appears, and this consistency is captured by a single computational quantity. Meanwhile, $\beta$ is purely strategic---it governs the decision about whether to enter a dangerous situation, not the motor response once danger materializes.

### The model's survival signal tracks moment-to-moment affect

On probe trials, participants rated either anxiety ("How anxious are you about being captured?") or confidence ("How confident are you about reaching safety?") on a 0--7 scale, after choosing but before pressing. The model's survival probability *S* strongly predicted both ratings via linear mixed models with random intercepts and slopes by subject:

- Anxiety: $\beta$ = -0.557, SE = 0.040, *t* = -14.04, *p* = 8.8 $\times$ 10^-45^ (N~obs~ = 5,274)
- Confidence: $\beta$ = +0.575, SE = 0.043, *t* = +13.48, *p* = 2.1 $\times$ 10^-41^ (N~obs~ = 5,272)

These effects translate to approximately 2 points on the 0--7 scale when moving from the safest (*T* = 0.1, *S* $\approx$ 0.85) to the most dangerous (*T* = 0.9, *S* $\approx$ 0.15) condition. Substantial random slope variance indicates meaningful individual differences in how tightly affect tracks survival. This individual variation forms the basis of the decomposition below.

Task-derived affect showed convergent validity with clinical instruments: mean task anxiety correlated with STAI (*r* = 0.31) and STICSA (*r* = 0.27), while mean task confidence was negatively associated with AMI (*r* = -0.25). Within-subject anxiety and confidence were only weakly correlated (*r* = -0.25), and between-subject means were essentially independent (*r* = -0.01), indicating partially separable affective channels.

### Affective calibration and discrepancy carry distinct functional consequences

We decomposed each participant's affect-danger relationship into two dimensions. **Calibration** is the within-subject Pearson correlation between anxiety ratings and model-derived danger (1 - *S*), computed across each participant's 18 anxiety probe trials; higher calibration indicates anxiety that more accurately tracks the computational danger signal. **Discrepancy** is the mean residual of a participant's anxiety ratings after removing the population-level anxiety-danger relationship; positive discrepancy indicates systematically elevated anxiety beyond what the danger signal warrants.

These dimensions were orthogonal (*r* = 0.032, *p* = .59; Fig. 5a), confirming that the accuracy of threat monitoring and the magnitude of affective bias are genuinely independent.

The central question is whether these affective dimensions carry functional consequences---costs or benefits visible in task performance---beyond what the three model parameters already explain. We tested this with hierarchical regressions, asking whether calibration and discrepancy add predictive power over and above *k*, $\beta$, and *c*~d~.

**Calibration and policy alignment.** Participants whose anxiety accurately tracked danger showed better alignment between their actual choices and the model's predictions: calibration correlated with policy alignment at *r* = 0.315 (*p* < 10^-8^). The partial correlation controlling for the three parameters was *r* = 0.107 (*p* = .07), and adding calibration to a regression of policy alignment on *k* + $\beta$ + *c*~d~ yielded $\Delta$*R*^2^ = 0.4% (*p* = .07; Fig. 5b). While this effect is marginal with three parameters in the base model (compared to $\Delta$*R*^2^ = 6.4% in the two-parameter model), calibration also predicted total earnings (*r* = 0.21, *p* < .001), suggesting that accurate anxiety still functions as useful information for foraging efficiency.

**Discrepancy predicts residual overcaution.** After regressing overcautious error rate on *k* + $\beta$ + *c*~d~ (base model *R*^2^ = 88.5%), discrepancy predicted the residuals: *r* = 0.123 (*p* = .036), yielding $\Delta$*R*^2^ = 0.2% (*p* = .035; Fig. 5c). This effect is small but statistically reliable: participants with excess anxiety beyond what the danger signal warrants show a pattern of additional caution that the three model parameters cannot explain. The practical consequence is modest: discrepancy was associated with lower total earnings (*r* = -0.24, *p* < 10^-4^).

We emphasize the effect-size asymmetry in a different light than in previous analyses. With three parameters capturing 88.7% of overcaution variance (compared to 85.4% with two parameters), there is less residual for affect to explain. The stronger finding is the triple dissociation itself; the additional finding that affective bias contributes beyond the parameters is real but small.

**Independence of pathways.** The routes analysis (Fig. 5d) confirmed that the predictors (*k*, $\beta$, *c*~d~, calibration, discrepancy, encounter reactivity) show a sparse, structured pattern of associations with foraging outcomes (overcautious rate, threat sensitivity, vigor gap, policy alignment, total earnings). Each parameter has a primary target: *k* $\rightarrow$ overcaution (*r* = 0.883); $\beta$ $\rightarrow$ threat sensitivity (*r* = 0.574); *c*~d~ $\rightarrow$ vigor gap (*r* = 0.579). Encounter reactivity was independent of choice-level outcomes (all *p* > .05). This structured specificity---rather than a general factor of "task performance"---supports the interpretation that the model identifies genuinely distinct computational dimensions of individual variation.

**Clinical convergence.** The three computational parameters showed almost no association with psychiatric symptom severity. The sole exception was $\beta$ predicting AMI total score (*r* = 0.143, *p* = .015): individuals with higher threat aversion in choice reported more apathy. All other parameter-clinical associations were null (*k*: all *p* > .20; *c*~d~: all *p* > .26; $\beta$: all other *p* > .13).

By contrast, affective discrepancy predicted symptom severity across 8 of 10 clinical instruments. STAI-State (*r* = 0.327, *p* < 10^-8^), DASS-Anxiety (*r* = 0.250, *p* < 10^-5^), STICSA (*r* = 0.264, *p* < 10^-5^), DASS-Stress (*r* = 0.232, *p* < 10^-4^), DASS-Depression (*r* = 0.213, *p* < 10^-3^), PHQ-9 (*r* = 0.206, *p* < 10^-3^), OASIS (*r* = 0.204, *p* < 10^-3^), and MFIS (*r* = 0.183, *p* = .002) all showed positive associations. Discrepancy did not predict STAI-Trait (*r* = -0.223, opposite sign, reflecting trait anxiety's inverse relationship with state reactivity) or AMI (*r* = 0.026, *p* = .66). These clinical associations, while consistent with the task-level findings, are from the discovery sample and await confirmatory replication. We note that all cross-validated *R*^2^ values for predicting individual symptom outcomes were negative, confirming group-level patterns rather than individually predictive biomarkers.

---

## Discussion

We developed a three-parameter choice-vigor model that separates how humans process effort cost, threat aversion, and defensive motor readiness during foraging under predation risk. The triple dissociation---*k* $\rightarrow$ overcaution, $\beta$ $\rightarrow$ threat-driven avoidance, *c*~d~ $\rightarrow$ vigor gap---demonstrates that what might appear as a unitary "cautiousness" in fact reflects three computationally and behaviorally distinct signals.

### Separable cost signals, not a unitary threat sensitivity

The central finding is that effort cost and threat aversion are orthogonal (*r* = -0.006). This is not a trivial consequence of model design: the two parameters enter the same choice equation, and in principle could have been correlated if individuals who are effort-sensitive also tend to be threat-sensitive. That they are not suggests that the neural and psychological mechanisms underlying effort discounting^6--9^ and threat processing^12--14^ are genuinely separable at the level of individual differences. The practical consequence is diagnostic: when a person avoids a dangerous, effortful option, we can now decompose that avoidance into an effort component (would they avoid it even without threat?) and a threat component (would they avoid it even without effort?). The condition-level dissociation makes this concrete: at low threat, *k* dominates choice (*r* = -0.937 with P(heavy) at *T* = 0.1); at high threat, $\beta$ dominates (*r* = -0.840 with P(heavy) at *T* = 0.9).

This separability has implications for clinical assessment. Motivational deficits in depression and apathy are typically attributed to elevated effort costs^10,11^, while avoidance in anxiety disorders is attributed to threat hypersensitivity^34^. Our results suggest these are independently varying dimensions: one could have high effort cost but normal threat sensitivity, or vice versa. Consistent with this, $\beta$ predicted apathy (*r* = 0.14) but not anxiety, while *k* predicted neither---a pattern that warrants investigation in clinical samples.

### Integration without fusion: architectural separability

The model achieves integration of effort and threat in the sense that both costs enter the same choice equation---a single $\Delta$EU drives the decision. But the integration is architectural, not computational: effort and threat remain separable terms rather than being fused through a shared nonlinear transformation. This contrasts with models that route both costs through a survival function^1,5^, which produces an inherent correlation between effort and threat sensitivity. The linear specification not only improves model fit (per-subject choice *r*^2^ = 0.981 vs. 0.951 with a survival-function-based model) but makes a stronger theoretical claim: the brain maintains distinct registers for "how hard is this?" and "how dangerous is this?" even when both inform the same choice.

The vigor channel provides a complementary form of integration. The survival function *S*(*u*) does appear in the vigor equation, because the speed-escape relationship is a physical constraint: pressing faster genuinely improves escape probability. Here, threat and effort are fused through *S*(*u*), but this fusion reflects the task's physical mechanics rather than a subjective weighting. The parameter *c*~d~ scales the penalty term (1 - *S*(*u*)) $\times$ (*R* + *C*), converting physical danger into a motivational signal that drives motor vigor above the required rate.

### Deviations from optimal: structured errors, not noise

By deriving condition-specific optimal policies, we showed that participants achieved 69.8% optimality on average, with the remaining errors structured by the model's three parameters. The strength of the *k*-overcaution association (*r* = 0.885, partial *r* = 0.933) is striking: individual variation in effort cost explains nearly all between-person variance in a specific error type. This is not a tautological relationship---*k* is estimated from the pattern of choices across conditions, while overcaution is computed relative to an independently derived optimal policy. The addition of $\beta$ captures an additional 10.2% of unique variance in overcaution, corresponding to threat-driven avoidance beyond what effort cost explains.

The *c*~d~-vigor gap association (*r* = 0.58) is more moderate but theoretically important: individuals with higher capture aversion press harder overall, but still fall short of what optimal investment would predict, producing a "vigor gap" that scales with the parameter.

### Strategic versus reactive defense: $\beta$ and *c*~d~ dissociate

The encounter dynamics results reveal a further dissociation within the threat-processing system. Threat aversion ($\beta$) governs the strategic decision of whether to enter a dangerous situation, but it does not predict how the organism responds once danger materializes (*r* = 0.08, *p* = .19). Capture aversion (*c*~d~) governs both tonic motor readiness and the phasic encounter reflex (*r* = 0.39). This maps onto the distinction between cognitive and reactive fear circuits^13^: $\beta$ reflects a deliberative assessment that enters the choice computation, while *c*~d~ reflects a motor-defensive system that operates both anticipatorily (tonic vigor) and reflexively (encounter reactivity).

The encounter reflex is threat-independent (*F* = 0.04, *p* = .96) and trait-stable (cross-block *r* = 0.78). This dissociation---strategic threat modulates choice but not the encounter response---suggests that the encounter reflex is triggered by predator detection per se, not by the probability of predator appearance. The brain distinguishes between "how likely is danger" (influencing choice via $\beta$) and "danger is here now" (mobilizing the motor system via *c*~d~).

### Affect as functional information

The survival signal's ability to predict trial-level anxiety (*t* = -14.0) and confidence (*t* = 13.5) establishes it as a computational substrate of moment-to-moment affect. The decomposition into calibration and discrepancy reveals that affect carries functional information, though with three parameters in the base model, the incremental contribution is smaller than in simpler models. Calibration's bivariate association with policy alignment (*r* = 0.315) suggests that accurate affective tracking helps foraging efficiency, even if the partial effect after controlling for three parameters is marginal ($\Delta$*R*^2^ = 0.4%, *p* = .07). Discrepancy's contribution to residual overcaution ($\Delta$*R*^2^ = 0.2%, *p* = .035) is statistically reliable but small, consistent with the three-parameter model capturing most of the variance that affect would otherwise explain.

The clinical convergence strengthens the interpretation that discrepancy captures something meaningful: it predicts anxiety symptoms across multiple instruments (*r* = 0.18--0.33), while the computational parameters largely do not. This suggests that the bridge from adaptive foraging computation to psychiatric vulnerability runs through affect---through how people feel about danger, not how they compute it^22^. The sole exception is the $\beta$-AMI link (*r* = 0.14), suggesting that threat aversion in choice may share variance with motivational withdrawal.

### Limitations

Several limitations warrant consideration. First, distance confounds effort duration and threat exposure by design---farther cookies require both more sustained pressing and more time in danger. This confound is shared with natural foraging^1^ and is partially addressed by the factorial crossing of distance with effort demand, but a fully orthogonal design would strengthen causal claims.

Second, the task uses explicit threat probabilities rather than learned threat. Our model assumes stationary threat and does not capture trial-by-trial belief updating that may occur even with stated probabilities.

Third, the model is static: it does not capture learning, strategic adjustment, or affect-behavior feedback within sessions.

Fourth, the population-level escape probability (*p*~esc~ = 0.002) is very low, meaning the vigor model's survival function is relatively flat over the observed press-rate range. This limits how strongly vigor can be predicted from survival mechanics alone, and the moderate vigor *r*^2^ = 0.424 likely reflects this constraint alongside irreducible motor noise.

Fifth, our online sample (Prolific) provides dimensional variation in self-reported psychiatric symptoms but not clinical diagnoses. The $\beta$-AMI association (*r* = 0.14) and the discrepancy-symptom associations (*r* = 0.18--0.33) are modest effect sizes that require replication in clinical populations.

---

## Methods

### Participants

We recruited 350 participants from Prolific (https://prolific.co). After a five-stage quality control pipeline---(1) task completion, (2) comprehension checks, (3) behavioral consistency screening, (4) effort calibration validation (minimum 10 presses in 10 seconds), and (5) outlier removal (escape rate < 35%)---the final sample comprised N = 293 participants (83.7% retention). No post-hoc exclusions were applied based on model fit quality or statistical extremity. The study was approved by the Caltech Institutional Review Board, and all participants provided informed consent. A preregistered confirmatory replication (N = 350) with identical task and analysis was conducted separately.

### Task design

Participants completed an effort-based foraging task implemented in Unity (WebGL) and presented in a desktop browser. The task featured a circular arena viewed from above, with a safe zone at the center and cookie targets at radial distances.

**Effort calibration.** Before the main task, participants completed three 10-second trials pressing the S, D, and F keys simultaneously as fast as possible. The maximum press count defined each participant's calibrated maximum (*f*~max~).

**Trial structure.** On choice trials, two cookies appeared: a heavy cookie (5 points, requiring pressing at 60--100% of *f*~max~) at distance *D* $\in$ {1, 2, 3} (5, 7, 9 game units), and a light cookie (1 point, requiring 40% of *f*~max~) always at *D* = 1. Participants clicked to select (irrevocable), then pressed S+D+F repeatedly to transport the cookie to safety. Movement speed followed a step function: full speed at $\geq$100% of required rate, half at $\geq$50%, quarter at $\geq$25%, zero below 25%.

**Threat manipulation.** Each trial had predation probability *T* $\in$ {0.1, 0.5, 0.9}. On attack trials, a predator spawned at the perimeter closest to the participant at a pre-determined encounter time, approached at 0.5 units/sec, then struck at a Gaussian-distributed time at 4$\times$ the participant's maximum speed. Capture cost 5 points plus the cookie value.

**Probe trials.** Twelve of 27 events per block were probe trials with identical options (both heavy or both light). After selection, the game paused for a rating of either anxiety about capture or confidence in reaching safety (0--7 scale). Probes were prospective: collected after commitment but before motor execution.

**Structure.** Three blocks of 27 events (81 total): 15 choice trials and 12 probe trials per block, yielding 45 choice trials and 36 probe trials. Conditions fully crossed: 3 threat $\times$ 3 distance $\times$ 5 repetitions (choice); 3 threat $\times$ 3 distance $\times$ 2 cookie types $\times$ 2 rating types (probe).

### Psychiatric assessment

Between blocks, participants completed the DASS-21, PHQ-9, OASIS, STAI-State, AMI, MFIS, and STICSA. All subscale scores were z-scored across participants before analysis.

### Three-parameter choice-vigor model

**Per-subject parameters** (log-normal prior, non-centered parameterization):
- *k* (effort cost, governs choice; median = 1.48, IQR = [1.05, 2.04])
- $\beta$ (threat aversion, governs choice; median = 4.30, IQR = [3.21, 5.71])
- *c*~d~ (capture aversion, governs vigor; median = 30.9, IQR = [10.7, 49.6])

**Population parameters:** $\tau$ = 1.06 (choice temperature); *p*~esc~ = 0.002 (escape probability); $\sigma$~motor~ = 0.82 (motor noise); *c*~e,vigor~ = 0.003 (deviation cost); $\sigma$~v~ = 0.241 (vigor observation noise).

**Choice model.** $\Delta$EU = 4 - *k*~i~ $\times$ effort(*D*) - $\beta$~i~ $\times$ *T*. Effort(*D*) captures the distance-dependent physical cost difference between heavy and light cookies. P(heavy) = sigmoid($\Delta$EU / $\tau$).

**Vigor model.** EU(*u*) = *S*(*u*) $\times$ *R* - (1 - *S*(*u*)) $\times$ *c*~d,i~ $\times$ (*R* + *C*) - *c*~e,vigor~ $\times$ (*u* - req)^2^ $\times$ *D*. Survival is speed-dependent: *S*(*u*) = (1 - *T*) + *T* $\times$ *p*~esc~ $\times$ sigmoid((*u* - req) / $\sigma$~motor~). Optimal *u** computed via softmax-weighted grid search.

**Joint likelihood.** Choice trials contribute Bernoulli likelihood; all 81 trials contribute Normal likelihood for vigor (observation noise $\sigma$~v~).

### Model fitting

The primary fit used NumPyro stochastic variational inference (SVI) with an AutoNormal guide (mean-field approximation), Adam optimizer (lr = 0.002), 40,000 steps. BIC computed as 2 $\times$ loss + *k* $\times$ log(*n*); we note that using ELBO loss for BIC is non-standard, as ELBO is a lower bound on the marginal likelihood.

### Parameter recovery

Synthetic datasets were generated from the fitted population distribution. Simulated data were refitted with the identical SVI procedure. Recovery assessed as Pearson *r* between true and recovered parameters. Cross-parameter recovery (e.g., true *k* vs. recovered $\beta$) assessed to confirm separability.

### Optimal policy derivation

For each of the 9 conditions (3 threat $\times$ 3 distance), we computed the EV-maximizing choice using empirical conditional survival rates from the full sample. Per-subject deviations were classified as optimal, overcautious (chose light when heavy was optimal), or overrisky (chose heavy when light was optimal).

### Within-trial vigor dynamics

The 20 Hz kernel-smoothed vigor timeseries was aligned to predator encounter time. Pre- and post-encounter excess vigor (cookie-type-centered) were computed in symmetric windows around the encounter. Encounter reactivity = post minus pre difference. Cross-block stability assessed as Pearson correlation. Threat modulation tested via one-way ANOVA on encounter-level reactivity. Associations with all three model parameters tested via Pearson correlation.

### Affect decomposition and functional consequences

**Calibration:** per-subject Pearson *r* between anxiety ratings and model-derived danger (1 - *S*). **Discrepancy:** per-subject mean residual from population-level regression of anxiety on *S*.

Functional consequences assessed via hierarchical regression: base model (*k* + $\beta$ + *c*~d~ $\rightarrow$ outcome), augmented model (+ calibration or + discrepancy). $\Delta$*R*^2^ and *F*-change tests assess incremental predictive power.

### Clinical analysis

Pearson correlations between each parameter (*k*, $\beta$, *c*~d~) and z-scored clinical measures (DASS-21 subscales, PHQ-9, OASIS, STAI-State, STAI-Trait, AMI, MFIS, STICSA). Discrepancy-clinical associations tested with the same approach. All tests two-tailed.

### Statistical analysis

All tests two-tailed. Effect sizes reported as Pearson *r*, partial *r*, or *R*^2^. Multiple comparisons corrected by Benjamini-Hochberg FDR where applicable. All analyses in Python 3.11 using NumPyro, JAX, statsmodels, scipy, and PyMC.

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
21. Bednekoff, P. A. Foraging in the face of danger. In *Foraging* (eds Stephens, D. W., Brown, J. S. & Ydenberg, R. C.) 305--329 (Univ. Chicago Press, 2007).
22. Paulus, M. P. & Stein, M. B. Interoception in anxiety and depression. *Brain Struct. Funct.* **214**, 451--463 (2010).
23. McNamara, J. M. & Houston, A. I. *The Currency of Nature: The Biology of Energy* (Cambridge Univ. Press, 2009).
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
35. Wells, A. *Metacognitive Therapy for Anxiety and Depression* (Guilford, 2009).
36. Cisek, P. & Kalaska, J. F. Neural mechanisms for interacting with a world full of action choices. *Annu. Rev. Neurosci.* **33**, 269--298 (2010).

---

## Supplementary Information

### Supplementary Note: Simpson's Paradox in Vigor

High threat shifts choice toward light cookies (which have lower required press rates), so unconditional analyses show flat or declining vigor with threat. Conditioning on chosen cookie type reveals robust threat-driven vigor increases (heavy: *d* = 0.42; light: *d* = 0.49). This paradox highlights the necessity of jointly modeling choice and vigor.
