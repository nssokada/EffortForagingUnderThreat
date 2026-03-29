# Integrating effort and threat in human foraging: a unified computation of choice and vigor under predation risk

Noah Okada^1^, Ketika Garg^1^, Toby Wise^2^, Dean Mobbs^1,3^

^1^ Division of the Humanities and Social Sciences, California Institute of Technology, Pasadena, CA, USA
^2^ Department of Neuroimaging, King's College London, London, UK
^3^ Computation and Neural Systems Program, California Institute of Technology, Pasadena, CA, USA

---

## Abstract

Foraging under predation risk demands integrating two incommensurable costs---energetic expenditure and mortality risk---into a single decision variable. How a unified computation governs both the choice of what to pursue and the vigor of pursuit, and how individual variation in that computation produces structured deviations from optimal foraging policy, remains unknown. Here we develop a joint choice-vigor model with a cost structure inspired by linear-quadratic optimal control and test it in a large online effort-foraging task under parametric threat (N = 293). Two subject-level parameters---effort cost (*c*~e~, identified from choice) and capture aversion (*c*~d~, identified from vigor)---plus population-level probability weighting ($\gamma$ = 0.21) jointly predict foraging decisions (per-subject choice *r*^2^ = 0.951) and press-rate vigor (*r*^2^ = 0.511). Both parameters are recoverable in simulation (*c*~e~: *r* = 0.92; *c*~d~: *r* = 0.94) and approximately independent (*r* = -0.14). Within-trial encounter dynamics reveal that *c*~d~ bridges tonic vigor and phasic defensive reflexes (*r* = 0.50), with the encounter response stable across blocks (mean *r* = 0.78) yet independent of stated threat probability (*F* = 0.04, *p* = .96). We derive condition-specific optimal foraging policies and show that individual parameters drive specific deviations: *c*~e~ explains 83% of variance in foregone-opportunity errors (*r* = 0.92), while *c*~d~ explains 30% of variance in the vigor gap relative to optimal (*r* = 0.55). Probability weighting ($\gamma$ = 0.21) shifts the optimal surface, reclassifying 4 of 9 conditions and accounting for 20% of apparent suboptimality. The model's survival signal tracks trial-level anxiety ($\beta$ = -0.56, *t* = -14.0) and confidence ($\beta$ = 0.58, *t* = 13.5). Decomposing the affect-danger relationship reveals that affective calibration predicts policy alignment ($\Delta$*R*^2^ = 6.4%), while affective discrepancy explains residual overcaution beyond what the parameters capture ($\Delta$*R*^2^ = 0.3%, *p* = .015). A preregistered confirmatory replication [confirmatory: N = XXX] is underway.

---

## Introduction

Animals foraging under predation risk face a fundamental optimization problem: the energy gained from distant, high-value resources must be weighed against increased exposure to capture during transport^1,2^. This trade-off shapes behavior across taxa, from birds adjusting provisioning rates near raptor territories^3^ to fish venturing from shelter to access food patches^4^. In humans, analogous computations arise whenever pursuing goals demands sustained effort under threat---persisting at hazardous work, commuting through dangerous environments, or investing cognitive effort when the costs of failure are severe. Theoretical ecology has long modeled this trade-off using reproductive value as the common currency^1,5^, but translating these models into a computational framework that specifies how the same cost-benefit calculation governs both what an organism chooses and how vigorously it acts has remained an open challenge.

Three research traditions have addressed the components of this problem in isolation. First, work on effort-based decision-making has established that humans discount reward value by the physical or cognitive effort required to obtain it^6--9^, with individual differences in effort cost linked to apathy, fatigue, and motivational disorders^10,11^. Second, research on defensive behavior has characterized how organisms modulate responses across the predatory imminence continuum---from strategic avoidance at a distance to reactive flight when danger is immediate^5,12--14^. Third, the vigor of motor execution reflects the marginal value of time under the current motivational state^15,16^, suggesting that threat should intensify physical effort even after the decision to forage has been made. Yet these three literatures have developed largely in parallel: effort discounting models ignore survival, defensive behavior models ignore energetic cost, and vigor theories have not been tested under ecological threat where both costs operate simultaneously.

The Expected Value of Control (EVC) framework^17,18^ provides a candidate architecture for this integration. EVC proposes that the brain computes the expected payoff of allocating control effort, selecting the intensity that maximizes reward minus cost. Because EVC treats effort allocation as a continuous optimization, it naturally extends to predict both discrete choices and graded action vigor within a unified framework. However, EVC has been applied primarily to cognitive control in abstract tasks^19,20^ and has not been extended to physical effort under ecological threat, where the cost function must jointly encode energetic expenditure and survival probability. A separate gap concerns optimal policy: knowing that a model fits behavior does not reveal whether that behavior is *good*. Without deriving the foraging policy that maximizes expected value under the task's actual contingencies, we cannot distinguish adaptive strategies from structured errors, nor identify which individual differences produce which kinds of suboptimality.

Here we develop a joint choice-vigor model with a cost structure inspired by linear-quadratic optimal control^27,28^ and test it in a virtual effort-foraging task under parametric threat (N = 293). We designed the task as a reductionist probe of the effort-threat integration problem: it captures the core foraging-under-threat trade-off but deliberately abstracts away patch dynamics, metabolic urgency, and learned threat probabilities. By fixing the information environment, we can identify how individuals differ in effort sensitivity and threat aversion without confounding these with learning or state-dependent variation. Two subject-level parameters---effort cost (*c*~e~) and capture aversion (*c*~d~)---jointly predict choice and vigor through distinct channels. We derive condition-specific optimal policies and show that each parameter drives a specific type of deviation from optimality. We then examine the model's survival signal as a computational substrate of moment-to-moment affect, and ask whether individual differences in the accuracy and bias of the affect-danger relationship carry additional functional consequences beyond what the model parameters explain. A preregistered confirmatory replication [confirmatory: N = XXX] with identical task design, model specification, and analysis plan is underway.

---

## Results

### Threat and distance deter high-effort foraging and modulate vigor

We designed an effort-foraging task in which participants (N = 293, after five-stage quality screening of 350 recruits) chose between a high-reward cookie (5 points, requiring sustained keypressing at 60--100% of individually calibrated maximum capacity) and a low-reward cookie (1 point, requiring 40% of maximum) while facing predation risk (threat probability *T* $\in$ {0.1, 0.5, 0.9}) at varying distances (*D* $\in$ {1, 2, 3}; Fig. 1a). On attack trials, a predator appeared and pursued the participant; capture incurred a 5-point penalty and loss of the cookie's value. Participants completed 45 choice trials and 36 probe trials (forced-choice with identical options, paired with affect ratings).

Both threat and distance reduced high-effort choice (Fig. 1b). A logistic mixed-effects model confirmed threat as the dominant deterrent ($\beta$ = -1.28, *z* = -32.0, *p* < 10^-200^), with distance ($\beta$ = -0.65, *z* = -16.3, *p* < 10^-59^) and their interaction ($\beta$ = -0.18, *z* = -4.5, *p* < 10^-5^) also significant. At the extremes, P(heavy) dropped from 0.81 at *T* = 0.1, *D* = 1 to 0.08 at *T* = 0.9, *D* = 3.

Threat also increased motor vigor, but this effect was masked by a Simpson's paradox in unconditional analyses. High threat shifts choice toward light cookies, and light cookies have lower required press rates, so collapsing across cookie type makes average vigor appear flat or even decline. Conditioning on chosen cookie type revealed robust threat-driven vigor increases (heavy cookies: *t* = 6.6, *p* < 10^-10^, *d* = 0.42; light cookies: *t* = 7.5, *p* < 10^-13^, *d* = 0.49; Fig. 1c). Within each cookie type, participants pressed harder when threat was higher---consistent with the prediction that the marginal survival benefit of faster pressing increases with danger. This Simpson's paradox illustrates why a joint model of choice and vigor is necessary: analyzing either channel in isolation yields misleading conclusions about how threat shapes behavior.

### A joint model with LQR-inspired cost structure captures choice and vigor

We formalized the foraging decision as a comparison of expected utilities:

$\Delta EU = S \times 4 - c_{e,i} \times (0.81D_H - 0.16)$

where *S* is the subjective survival probability, *c*~e,i~ is the subject-specific effort cost, and the effort term reflects the commitment cost inspired by linear-quadratic optimal control theory---the squared required press rate scaled by distance. The survival function incorporates probability weighting:

$S = (1 - T^{\gamma}) + \varepsilon \times T^{\gamma} \times p_{\text{esc}}$

where $\gamma$ = 0.209 is the population probability-weighting exponent, indicating substantial compression of stated threat probabilities (a nominal 50% threat is experienced as *T*^0.209^ = 0.86), and $\varepsilon$ = 0.098 reflects near-universal underweighting of effort's benefit to survival. Choices follow a softmax rule: P(heavy) = sigmoid($\Delta$EU / $\tau$), with inverse temperature $\tau$ = 0.476.

With per-subject *c*~e~ (log-normal prior, non-centered parameterization), the model reproduced the full threat-by-distance choice surface (Fig. 2a). The model achieved per-subject choice *r*^2^ = 0.951, choice accuracy = 79.3%, and AUC = 0.876 (Fig. 2b).

For the vigor component, the model computes optimal press rate *u** by maximizing:

$EU(u) = S(u) \times R - (1 - S(u)) \times c_{d,i} \times (R + C) - c_{e,\text{vigor}} \times (u - \text{req})^2 \times D$

where *c*~d,i~ is the subject-specific capture aversion and *c*~e,vigor~ = 0.003 is the population-level motor deviation cost. Pressing faster than required improves escape probability through a speed-dependent survival function. Three features of this architecture merit attention.

First, *c*~d~ is excluded from the choice equation due to collinearity: the residual *c*~d~ term in the choice difference scales with (*R*~H~ - *R*~L~)---the same factor as the reward term---rendering the two empirically inseparable. We therefore estimate *c*~e~ exclusively from choice and *c*~d~ exclusively from vigor.

Second, the distinction between commitment cost (req^2^ $\times$ *D*, governing choice) and deviation cost ((*u* - req)^2^ $\times$ *D*, governing vigor), inspired by the LQR separation of reference-trajectory and tracking costs^27^, resolves a scaling conflict that allows a single framework to span both decision stages. We note that this is an analogy to LQR optimal control, not a formal implementation: our model has no state dynamics, no feedback law, and no Riccati equation.

Third, probe trials (forced conditions with identical options) anchor *c*~d~ estimation across all threat-by-distance cells without selection bias.

The model predicted trial-level vigor with *r*^2^ = 0.511 (subject-level *r* = 0.829; Fig. 2c). Higher *c*~d~ drives steeper threat-vigor gradients because the marginal benefit of pressing faster grows with the stakes of failed escape.

Model comparison across six variants confirmed that every component was necessary (Table 1). Removing threat information entirely (M1: $\Delta$BIC = +18,659), individual effort cost (M2: $\Delta$BIC = +2,094, choice *r*^2^ = 0.006), joint estimation (M3: $\Delta$BIC = +10,393), or probability weighting (M5: $\Delta$BIC = +2,071) all substantially degraded fit. The LQR-inspired deviation cost performed comparably to a standard *u*^2^ cost (M6: $\Delta$BIC = -142), confirming empirical equivalence; we retain the LQR-inspired framing for theoretical motivation.

**Table 1. Model comparison.**

| Model | Description | BIC | $\Delta$BIC | Choice *r*^2^ | Vigor *r*^2^ |
|-------|------------|-----|------------|-------------|------------|
| **Full model** | **EVC 2+2** | **32,133** | **---** | **0.951** | **0.511** |
| M1 | Effort only (no threat) | 50,792 | +18,659 | 0.950 | 0.000 |
| M2 | Threat only (no individual *c*~e~) | 42,767 | +10,634 | 0.006 | 0.294 |
| M3 | Separate choice + vigor | 42,526 | +10,393 | 0.955 | 0.440 |
| M4 | Population *c*~e~ | 30,860 | -1,273 | 0.001 | 0.512 |
| M5 | No $\gamma$ ($\gamma$ = 1) | 34,204 | +2,071 | 0.955 | 0.425 |
| M6 | Standard *u*^2^ cost | 31,991 | -142 | 0.952 | 0.508 |

*Note.* M4 achieves lower BIC but fails to predict individual choice (choice *r*^2^ = 0.001), sacrificing the primary behavioral target for marginal vigor improvement. M6 is empirically equivalent to the full model.

Parameter recovery confirmed identifiability. Simulating 150 synthetic subjects across 3 datasets at the fitted population distribution and refitting yielded correlations (in log space) of *r* = 0.92 for *c*~e~ and *r* = 0.94 for *c*~d~ (Fig. 2d). The two parameters were approximately independent in the empirical sample (*r* = -0.14, *p* = .02). MCMC validation (NUTS, 4 chains $\times$ 200 warmup + 200 samples) produced zero divergent transitions and near-perfect agreement with the primary SVI estimates (log(*c*~e~): *r* = 0.999; log(*c*~d~): *r* = 0.999; Supplementary Note).

The probability weighting parameter ($\gamma$ = 0.21) indicates dramatic compression of threat probabilities---substantially stronger than estimates from monetary gambles ($\gamma$ ~ 0.65--0.70)^30^. This amplification is consistent with the embodied nature of virtual predation engaging defensive circuitry more powerfully than abstract monetary losses^5,13^. The functional consequence is that even low stated threats (10%) are experienced as substantial danger (*T*^0.21^ = 0.62), compressing the subjective threat range and shifting the optimal foraging policy (see below).

### Encounter dynamics: capture aversion bridges tonic and phasic defense

The static model predicts trial-level average vigor, but foraging under threat unfolds dynamically within each trial. We examined the 20 Hz vigor timeseries aligned to predator encounter events to test whether *c*~d~ extends to within-trial temporal dynamics.

Encounter reactivity---the difference between post-encounter and pre-encounter excess vigor---showed no reliable population mean effect (*M* = -0.019, *SD* = 0.28, *t* = -1.15, *p* = .25), but massive individual variation. This variation was highly stable across task blocks (mean cross-block *r* = 0.78), identifying encounter reactivity as a trait-like individual difference. Piecewise regression confirmed a qualitative shift at the encounter: pre-encounter vigor declined while post-encounter vigor increased (slope change = 0.050, *t* = 5.91, *p* < 10^-8^). The magnitude of this slope change correlated with *c*~d~ (*r* = 0.18, *p* = .002).

Encounter reactivity correlated strongly with capture aversion (Pearson *r* = 0.50 with log(*c*~d~), *p* < 10^-19^; Spearman $\rho$ = 0.41, *p* < 10^-12^) but not with effort cost (*r* = -0.09 with log(*c*~e~), *p* = .14). Critically, threat probability did not modulate the encounter response (one-way ANOVA: *F* = 0.04, *p* = .96; Fig. 3a). This dissociation---threat modulates strategic choice ($\beta$ = -1.28) but not the phasic encounter reflex---maps directly onto the distinction between strategic and reactive defensive modes^5,14^. Choice reflects probability-weighted evaluation; the encounter response is an automatic mobilization triggered by predator detection, independent of the stated probability of that predator appearing.

That *c*~d~ captures both tonic pressing level (trial-average vigor) and this phasic reflex (*r* = 0.50) supports interpreting it as a general defensive motor readiness trait spanning the continuum from sustained vigilance to acute encounter response. This parameter thus bridges the strategic-reactive divide that the threat-imminence literature has described qualitatively^5,14^: the same individuals who press harder throughout a dangerous trial also mobilize more sharply when the predator appears, and this consistency is captured by a single computational quantity.

### Individual parameters drive specific deviations from optimal foraging policy

Having established that the model fits behavior, we next asked whether that behavior is *good*---whether participants forage optimally given the task's contingencies, and if not, what specific errors they make.

We derived the EV-maximizing choice for each of the 9 conditions (3 threat $\times$ 3 distance) using empirical conditional survival rates (Table 2). Under objective probabilities, heavy cookies maximize EV in 6 of 9 conditions, including all low-threat and two medium-threat conditions. But under probability weighting ($\gamma$ = 0.21), the optimal surface shifts: 4 of 9 conditions change their optimal policy from heavy to light, because compressed threat probabilities make even moderate danger feel extreme, reducing the expected value of risky high-reward foraging.

**Table 2. Optimal foraging policy by condition.**

| Threat | Distance | Objective optimal | EV advantage | Subjective optimal |
|--------|----------|-------------------|--------------|-------------------|
| 0.1 | 1 | Heavy | +2.15 | Heavy |
| 0.1 | 2 | Heavy | +1.55 | Light |
| 0.1 | 3 | Heavy | +0.39 | Light |
| 0.5 | 1 | Heavy | +0.80 | Light |
| 0.5 | 2 | Heavy | +0.07 | Light |
| 0.5 | 3 | Light | -0.84 | Light |
| 0.9 | 1 | Light | -0.94 | Light |
| 0.9 | 2 | Light | -0.75 | Light |
| 0.9 | 3 | Light | -1.07 | Light |

Probability weighting accounts for approximately 20% of apparent suboptimality: choices that look irrational under objective probabilities become rational under the subjective threat surface. After accounting for $\gamma$, the remaining deviations from the subjective optimal policy were substantial---participants made optimal choices on 69.8% of trials (SD = 12.0%), with 21.3% overcautious errors (choosing light when heavy was optimal; SD = 14.4%) and 8.9% overrisky errors (choosing heavy when light was optimal; SD = 8.4%).

Each parameter drove a specific type of deviation (Fig. 4). Effort cost predicted overcautious errors with remarkable strength: the bivariate correlation between log(*c*~e~) and overcautious rate was *r* = 0.92 (*p* < 10^-121^), and a multiple regression confirmed that *c*~e~ explained 83.1% of unique variance in overcaution ($\beta$~ce~ = 0.196, *t* = 40.8, *p* < 10^-121^) while *c*~d~ contributed negligibly (unique *R*^2^ = 0.09%, *p* = .19). Conversely, *c*~d~ predicted the vigor gap---how far below optimal each participant pressed---with *r* = 0.55 (*p* < 10^-24^), partial *r* = 0.55 after controlling for *c*~e~. This differential prediction pattern confirms that the two parameters capture distinct dimensions of suboptimality: *c*~e~ governs foregone opportunity (passing up profitable options), while *c*~d~ governs vigor investment (pressing less hard than the stakes warrant).

Effort cost also predicted total earnings (*r* = -0.81, *p* < 10^-68^): participants with higher *c*~e~ earned substantially less because they systematically avoided high-value options. By contrast, *c*~d~ had a small positive association with earnings (*r* = 0.15, *p* = .01), suggesting that higher capture aversion, while producing vigor gaps, also improved survival in dangerous conditions.

A descriptive median-split of *c*~e~ $\times$ *c*~d~ revealed four behavioral profiles (Supplementary Table S1). The "Vigilant" profile (low *c*~e~, high *c*~d~; N = 82) earned the most (mean = 103.5 points), combining willingness to pursue high-value resources with vigorous motor execution during transport. The "Cautious" profile (high *c*~e~, low *c*~d~; N = 65) and "Disengaged" profile (high *c*~e~, high *c*~d~; N = 82) earned the least (67.6 and 66.4 points, respectively), while the "Bold" profile (low *c*~e~, low *c*~d~; N = 64) earned moderately (96.1 points) by pursuing high-value options but without the vigorous execution to survive at the highest rates.

### The model's survival signal tracks moment-to-moment affect

On probe trials, participants rated either anxiety ("How anxious are you about being captured?") or confidence ("How confident are you about reaching safety?") on a 0--7 scale, after choosing but before pressing. The model's survival probability *S* strongly predicted both ratings via linear mixed models with random intercepts and slopes by subject:

- Anxiety: $\beta$ = -0.557, SE = 0.040, *t* = -14.04, *p* = 8.8 $\times$ 10^-45^ (N~obs~ = 5,274)
- Confidence: $\beta$ = +0.575, SE = 0.043, *t* = +13.48, *p* = 2.1 $\times$ 10^-41^ (N~obs~ = 5,272)

These effects translate to approximately 2 points on the 0--7 scale when moving from the safest (*T* = 0.1, *S* $\approx$ 0.85) to the most dangerous (*T* = 0.9, *S* $\approx$ 0.15) condition. Substantial random slope variance indicates meaningful individual differences in how tightly affect tracks survival. This individual variation forms the basis of the decomposition below.

Task-derived affect showed convergent validity with clinical instruments: mean task anxiety correlated with STAI (*r* = 0.31) and STICSA (*r* = 0.27), while mean task confidence was negatively associated with AMI (*r* = -0.25). Within-subject anxiety and confidence were only weakly correlated (*r* = -0.25), and between-subject means were essentially independent (*r* = -0.01), indicating partially separable affective channels.

### Affective calibration and discrepancy carry distinct functional consequences

We decomposed each participant's affect-danger relationship into two dimensions. **Calibration** is the within-subject Pearson correlation between anxiety ratings and model-derived danger (1 - *S*), computed across each participant's 18 anxiety probe trials; higher calibration indicates anxiety that more accurately tracks the computational danger signal. **Discrepancy** is the mean residual of a participant's anxiety ratings after removing the population-level anxiety-danger relationship; positive discrepancy indicates systematically elevated anxiety beyond what the danger signal warrants.

These dimensions were orthogonal (*r* = 0.019, *p* = .75; Fig. 5a), confirming that the accuracy of threat monitoring and the magnitude of affective bias are genuinely independent.

The central question is whether these affective dimensions carry functional consequences---costs or benefits visible in task performance---beyond what the model parameters already explain. We tested this with hierarchical regressions, asking whether calibration and discrepancy add predictive power over and above log(*c*~e~) and log(*c*~d~).

**Calibration predicts policy alignment.** Participants whose anxiety accurately tracked danger showed better alignment between their actual choices and the model's predictions: calibration correlated with policy alignment at *r* = 0.30 (*p* < 10^-7^), and this association held after controlling for both parameters (partial *r* = 0.31, *p* < 10^-7^). Adding calibration to a regression of policy alignment on log(*c*~e~) + log(*c*~d~) yielded $\Delta$*R*^2^ = 6.4% (*p* < 10^-7^; Fig. 5b). Calibration also predicted survival (*r* = 0.18, *p* = .002) and total earnings (*r* = 0.20, *p* < .001). Accurate anxiety appears to function as useful information that improves foraging efficiency---participants who feel appropriately anxious when danger is high and appropriately calm when danger is low make choices more consistent with the value computation.

**Discrepancy predicts residual overcaution.** After regressing overcautious error rate on log(*c*~e~) + log(*c*~d~) (base model *R*^2^ = 85.4%), discrepancy predicted the residuals: *r* = 0.14 (*p* = .015), yielding $\Delta$*R*^2^ = 0.3% (*F*~change~ = 6.0, *p* = .015; Fig. 5c). This effect is small but statistically reliable: participants with excess anxiety beyond what the danger signal warrants show a pattern of additional caution that the model parameters cannot explain. Discrepancy also predicted excess vigor on low-threat trials (*r* = -0.12, *p* = .038), suggesting that overanxious individuals mobilize motor resources even when threat is minimal. The practical consequence is modest: discrepancy was associated with lower total earnings (*r* = -0.24, *p* < 10^-4^) and lower survival (*r* = -0.14, *p* = .017).

We emphasize the effect-size asymmetry: calibration's contribution to policy alignment ($\Delta$*R*^2^ = 6.4%) is substantially larger than discrepancy's contribution to residual overcaution ($\Delta$*R*^2^ = 0.3%). The stronger finding is that accurate affective tracking helps; the additional finding that affective bias hurts is real but small.

**Independence of pathways.** The four-routes analysis (Fig. 5d) confirmed that the five predictors (log(*c*~e~), log(*c*~d~), calibration, discrepancy, encounter reactivity) show a sparse, structured pattern of associations with the five outcomes (overcautious rate, survival, policy alignment, residual overcaution, total earnings). Each predictor has a primary target: *c*~e~ $\rightarrow$ overcaution (*r* = 0.92); *c*~d~ $\rightarrow$ survival (*r* = -0.02, ns) and vigor gap (*r* = 0.55); calibration $\rightarrow$ policy alignment (*r* = 0.30); discrepancy $\rightarrow$ residual overcaution (*r* = 0.14). Encounter reactivity showed a selective association with survival (*r* = -0.24) but was independent of choice-level outcomes and clinical anxiety measures (all *p* > .14). This structured specificity---rather than a general factor of "task performance"---supports the interpretation that the model identifies distinct computational and affective dimensions of individual variation.

**Clinical convergence (Supplementary Note).** Bayesian regressions controlling for log(*c*~e~) and log(*c*~d~) confirmed that discrepancy predicted anxiety symptoms across multiple instruments (STAI-State: $\beta$ = 0.338, 94% HDI [0.23, 0.45]; STICSA: $\beta$ = 0.285, HDI [0.17, 0.39]; DASS-Anxiety: $\beta$ = 0.275, HDI [0.16, 0.40]; DASS-Stress: $\beta$ = 0.255, HDI [0.13, 0.37]; all HDIs excluding zero). The computational parameters showed no reliable associations with symptom severity (65--93% of posterior mass within the ROPE of |$\beta$| < 0.10). Encounter reactivity was selectively associated with apathy (AMI: *r* = -0.17, *p* = .004) but not with any anxiety measure (all *p* > .20). These clinical associations, while consistent with the task-level findings, are from the discovery sample and await confirmatory replication. We note that all cross-validated *R*^2^ values for predicting individual symptom outcomes were negative, confirming group-level patterns rather than individually predictive biomarkers.

---

## Discussion

We developed a joint choice-vigor model that captures how humans integrate effort cost and survival probability into a unified foraging computation. Two parameters---effort cost (*c*~e~) and capture aversion (*c*~d~)---explain 95.1% of between-subject choice variance and 51.1% of trial-level vigor variance. Each parameter drives a specific type of deviation from optimal foraging policy: *c*~e~ governs foregone opportunity, *c*~d~ governs vigor investment. The model's survival signal tracks moment-to-moment affect, and individual differences in the accuracy of the affect-danger relationship carry functional consequences visible in task performance.

### A unified computation spanning choice, vigor, and encounter dynamics

The central modeling contribution is a joint framework that bridges three previously isolated literatures. Effort discounting models^6--9^ explain choice but ignore the vigor with which chosen actions are executed. Defensive behavior models^5,12--14^ characterize the threat-imminence continuum but do not specify how organisms trade effort against mortality risk. Vigor theories^15,16^ predict that motivational state should regulate motor intensity but have not been tested under ecological threat. Our model integrates these by embedding effort cost and survival probability in a single expected-utility computation, with the LQR-inspired^27^ separation of commitment and deviation costs providing a principled basis for distinct choice and vigor channels.

The probability weighting parameter ($\gamma$ = 0.21) deserves emphasis. This value indicates dramatically stronger distortion of threat probabilities than observed in monetary gambles ($\gamma$ ~ 0.65--0.70)^30^, consistent with the proposal that embodied threats engage defensive circuitry more powerfully than abstract losses^5,13^. The functional consequence is substantial: under $\gamma$ = 0.21, the optimal foraging policy shifts in 4 of 9 conditions, reclassifying choices that appear irrational under objective probabilities as rational under the subjective threat surface. This suggests that much of what appears as overcautious behavior in threat-foraging tasks may reflect probability weighting rather than irrationality---a finding relevant to interpreting "risk-averse" behavior in anxious populations.

### Deviations from optimal: structured errors, not noise

A model that fits behavior well does not, on its own, reveal whether behavior is adaptive. By deriving condition-specific optimal policies, we showed that participants achieved 69.8% optimality on average, with the remaining errors structured by the model's parameters. The strength of the *c*~e~-overcaution association (*r* = 0.92) is striking: individual variation in a single computational parameter explains nearly all between-person variance in a specific error type. This is not a tautological relationship---*c*~e~ is estimated from the pattern of choices across conditions, while overcaution is computed relative to an independently derived optimal policy. The finding means that effort cost sensitivity does not simply "shift choice thresholds" in some general sense; it specifically causes people to forego profitable opportunities when the effort-to-reward ratio crosses their subjective threshold.

The *c*~d~-vigor gap association (*r* = 0.55) is more moderate but theoretically important: individuals with higher capture aversion press harder overall, but still fall short of what optimal investment would predict, producing a "vigor gap" that scales with the parameter. That *c*~d~ also captures phasic encounter dynamics (*r* = 0.50) suggests that defensive motor readiness---whether measured tonically (trial-average vigor), phasically (encounter reactivity), or as deviation from optimal---coheres around a single individual-difference dimension.

### Strategic versus reactive defense: *c*~d~ bridges the imminence continuum

The encounter dynamics results place our computational framework in direct contact with the threat-imminence literature^5,14^. The encounter reflex is threat-independent (*F* = 0.04, *p* = .96), trait-stable (cross-block *r* = 0.78), and correlated with *c*~d~ (*r* = 0.50) but not *c*~e~ (*r* = -0.09). This dissociation maps onto the distinction between cognitive and reactive fear circuits^13^: choice reflects probability-weighted strategic evaluation, while the encounter response is an automatic motor mobilization triggered by predator detection. That a single parameter captures both strategic vigor investment and reactive encounter engagement suggests that the strategic-reactive distinction, while qualitatively real, may share a common generative source at the level of individual differences in threat-related motor readiness.

### Affect as functional information

The survival signal's ability to predict trial-level anxiety (*t* = -14.0) and confidence (*t* = 13.5) establishes it as a computational substrate of moment-to-moment affect. The decomposition into calibration and discrepancy reveals that affect carries functional information beyond what the model parameters encode. The stronger finding is calibration's contribution to policy alignment ($\Delta$*R*^2^ = 6.4%): participants whose anxiety accurately tracks danger make choices more consistent with the value computation, suggesting that well-calibrated affect functions as an adaptive alarm that improves foraging efficiency^22,32^. This is consistent with Wise and colleagues'^33^ finding that confidence tracks the quality of internal threat models, and with functional accounts of anxiety as a computationally useful signal rather than mere epiphenomenon.

Discrepancy's contribution to residual overcaution ($\Delta$*R*^2^ = 0.3%) is statistically reliable but small. We interpret this honestly: excess anxiety beyond computed danger has measurable but modest functional costs within the task. The convergent clinical associations---discrepancy predicts anxiety symptoms across multiple instruments while the computational parameters do not---strengthen the interpretation that this affective dimension captures something meaningful about threat processing that the model misses. But we do not claim that discrepancy is a strong predictor of real-world outcomes; its value lies in identifying a specific mechanistic pathway (affective bias $\rightarrow$ residual suboptimality) that the computational parameters alone cannot account for.

### Clinical implications

The clinical findings are placed in supplementary material because they require confirmatory replication. Nonetheless, the pattern merits brief discussion. Discrepancy---not the computational parameters---predicts symptom severity across anxiety, depression, and stress instruments ($\beta$ = 0.18--0.34). This suggests that the bridge from adaptive foraging computation to psychiatric vulnerability runs through affect---through how people feel about danger, not how they compute it^22^. Encounter reactivity independently predicts apathy (*r* = -0.17) but not anxiety, suggesting a separate motor-motivational pathway. We emphasize that these are modest effect sizes (3--11% of variance) in a non-clinical sample. All references to "anxiety symptoms" refer to dimensional variation on validated scales, not clinical diagnoses. Whether these associations strengthen in clinical populations is an important next step that the confirmatory study and planned combined-sample analyses (total N $\approx$ 580) will address.

### Limitations

Several limitations warrant consideration. First, the effort-efficacy parameter $\varepsilon$ is estimated at the population level because it is not individually recoverable. If some participants believe effort helps survival more than others, their apparent "excess caution" might be rational rather than biased. Individual differences in effort-efficacy beliefs remain an unmeasured source of variation.

Second, distance confounds effort duration and threat exposure by design---farther cookies require both more sustained pressing and more time in danger. This confound is shared with natural foraging^1^ and is partially addressed by the factorial crossing of distance with effort demand, but a fully orthogonal design would strengthen causal claims.

Third, the task uses explicit threat probabilities rather than learned threat. Our model assumes stationary threat and does not capture trial-by-trial belief updating that may occur even with stated probabilities.

Fourth, the model is static: it does not capture learning, strategic adjustment, or affect-behavior feedback within sessions. The moderate block-to-block stability of discrepancy (*r* = 0.48--0.68) suggests partial stability but also meaningful within-session change that dynamic models could characterize.

Fifth, our online sample (Prolific) provides dimensional variation in self-reported psychiatric symptoms but not clinical diagnoses. Effect sizes observed in non-clinical samples may not generalize to clinical populations.

---

## Methods

### Participants

We recruited 350 participants from Prolific (https://prolific.co). After a five-stage quality control pipeline---(1) task completion, (2) comprehension checks, (3) behavioral consistency screening, (4) effort calibration validation (minimum 10 presses in 10 seconds), and (5) outlier removal (escape rate < 35%)---the final sample comprised N = 293 participants (83.7% retention). No post-hoc exclusions were applied based on model fit quality or statistical extremity. The study was approved by the Caltech Institutional Review Board, and all participants provided informed consent. A preregistered confirmatory replication [confirmatory: N = XXX] with identical task and analysis was conducted separately.

### Task design

Participants completed an effort-based foraging task implemented in Unity (WebGL) and presented in a desktop browser. The task featured a circular arena viewed from above, with a safe zone at the center and cookie targets at radial distances.

**Effort calibration.** Before the main task, participants completed three 10-second trials pressing the S, D, and F keys simultaneously as fast as possible. The maximum press count defined each participant's calibrated maximum (*f*~max~).

**Trial structure.** On choice trials, two cookies appeared: a heavy cookie (5 points, requiring pressing at 60--100% of *f*~max~) at distance *D* $\in$ {1, 2, 3} (5, 7, 9 game units), and a light cookie (1 point, requiring 40% of *f*~max~) always at *D* = 1. Participants clicked to select (irrevocable), then pressed S+D+F repeatedly to transport the cookie to safety. Movement speed followed a step function: full speed at $\geq$100% of required rate, half at $\geq$50%, quarter at $\geq$25%, zero below 25%.

**Threat manipulation.** Each trial had predation probability *T* $\in$ {0.1, 0.5, 0.9}. On attack trials, a predator spawned at the perimeter closest to the participant at a pre-determined encounter time, approached at 0.5 units/sec, then struck at a Gaussian-distributed time at 4$\times$ the participant's maximum speed. Capture cost 5 points plus the cookie value.

**Probe trials.** Twelve of 27 events per block were probe trials with identical options (both heavy or both light). After selection, the game paused for a rating of either anxiety about capture or confidence in reaching safety (0--7 scale). Probes were prospective: collected after commitment but before motor execution.

**Structure.** Three blocks of 27 events (81 total): 15 choice trials and 12 probe trials per block, yielding 45 choice trials and 36 probe trials. Conditions fully crossed: 3 threat $\times$ 3 distance $\times$ 5 repetitions (choice); 3 threat $\times$ 3 distance $\times$ 2 cookie types $\times$ 2 rating types (probe).

### Psychiatric assessment

Between blocks, participants completed the DASS-21, PHQ-9, OASIS, STAI-State, AMI, MFIS, and STICSA. All subscale scores were z-scored across participants before analysis.

### Joint choice-vigor model

**Per-subject parameters** (log-normal prior, non-centered parameterization): *c*~e~ (effort cost, governs choice; median = 0.62, IQR = 0.41--0.78) and *c*~d~ (capture aversion, governs vigor; median = 31.3, IQR = 8.7--79.3).

**Population parameters:** $\gamma$ = 0.209 (probability weighting); $\varepsilon$ = 0.098 (effort efficacy); *c*~e,vigor~ = 0.003 (deviation cost); $\tau$ = 0.476 (choice temperature); *p*~esc~ = 0.018 (escape probability); $\sigma$~motor~ = 1.169 (motor noise); $\sigma$~v~ = 0.229 (vigor observation noise).

**Choice model.** $\Delta$EU = *S* $\times$ 4 - *c*~e,i~ $\times$ (0.81*D*~H~ - 0.16). The effort term is the difference in commitment costs between heavy and light cookies: req~H~^2^ $\times$ *D*~H~ - req~L~^2^ $\times$ *D*~L~ = 0.9^2^ $\times$ *D*~H~ - 0.4^2^ $\times$ 1 = 0.81*D*~H~ - 0.16. P(heavy) = sigmoid($\Delta$EU / $\tau$).

**Vigor model.** EU(*u*) = *S*(*u*) $\times$ *R* - (1 - *S*(*u*)) $\times$ *c*~d,i~ $\times$ (*R* + *C*) - *c*~e,vigor~ $\times$ (*u* - req)^2^ $\times$ *D*. Survival is speed-dependent: *S*(*u*) = (1 - *T*^$\gamma$^) + $\varepsilon$ $\times$ *T*^$\gamma$^ $\times$ *p*~esc~ $\times$ sigmoid((*u* - req) / $\sigma$~motor~). Optimal *u** computed via softmax-weighted grid search.

**Joint likelihood.** Choice trials contribute Bernoulli likelihood; all 81 trials contribute Normal likelihood for vigor (observation noise $\sigma$~v~).

### Model fitting

The primary fit used NumPyro stochastic variational inference (SVI) with an AutoNormal guide (mean-field approximation), Adam optimizer (lr = 0.002), 40,000 steps. MCMC validation (NUTS, 4 chains $\times$ 200 warmup + 200 samples) confirmed convergence with zero divergent transitions and near-perfect agreement with SVI estimates (Supplementary Note). BIC = 2 $\times$ loss + *k* $\times$ log(*n*); we note that using ELBO loss for BIC is non-standard, as ELBO is a lower bound on the marginal likelihood.

### Parameter recovery

Three synthetic datasets of 50 subjects each (150 total) were generated from the fitted population distribution. Simulated data were refitted with identical SVI procedure. Recovery assessed as Pearson *r* between true and recovered parameters in log space.

### Optimal policy derivation

For each of the 9 conditions (3 threat $\times$ 3 distance), we computed the EV-maximizing choice using empirical conditional survival rates from the full sample. Subjective optimal policies were computed under probability weighting ($\gamma$ = 0.21). Per-subject deviations were classified as optimal, overcautious (chose light when heavy was optimal), or overrisky (chose heavy when light was optimal) relative to the subjective optimal.

### Within-trial vigor dynamics

The 20 Hz kernel-smoothed vigor timeseries was aligned to predator encounter time. Pre- and post-encounter excess vigor (cookie-type-centered) were computed in symmetric windows around the encounter. Encounter reactivity = post minus pre difference. Cross-block stability assessed as Pearson correlation. Threat modulation tested via one-way ANOVA on encounter-level reactivity. Piecewise regression estimated pre- and post-encounter vigor slopes.

### Affect decomposition and functional consequences

**Calibration:** per-subject Pearson *r* between anxiety ratings and model-derived danger (1 - *S*). **Discrepancy:** per-subject mean residual from population-level regression of anxiety on *S*.

Functional consequences assessed via hierarchical regression: base model (log(*c*~e~) + log(*c*~d~) $\rightarrow$ outcome), augmented model (+ calibration or + discrepancy). $\Delta$*R*^2^ and *F*-change tests assess incremental predictive power.

### Clinical analysis (Supplementary)

Bayesian linear regression (bambi/PyMC, weakly informative priors, 2,000 draws $\times$ 4 chains) predicting each z-scored clinical measure from log(*c*~e~) + log(*c*~d~) + discrepancy + calibration. ROPE = |$\beta$| < 0.10. Cross-validated prediction via elastic net and ridge regression with nested 10-fold CV (5 repeats).

### Statistical analysis

All tests two-tailed. Effect sizes reported as Pearson *r*, standardized $\beta$, or *R*^2^. Multiple comparisons corrected by Benjamini-Hochberg FDR where applicable. All analyses in Python 3.11 using NumPyro, JAX, statsmodels, scipy, bambi, and PyMC.

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

### Supplementary Table S1. Behavioral profiles (median split on *c*~e~ $\times$ *c*~d~).

| Profile | N | Mean *c*~e~ | Mean *c*~d~ | Mean earnings | Overcautious rate | Survival rate |
|---------|---|-----------|-----------|--------------|-------------------|---------------|
| Vigilant (low *c*~e~, high *c*~d~) | 82 | Low | High | 103.5 | Low | High |
| Bold (low *c*~e~, low *c*~d~) | 64 | Low | Low | 96.1 | Low | Moderate |
| Cautious (high *c*~e~, low *c*~d~) | 65 | High | Low | 67.6 | High | Moderate |
| Disengaged (high *c*~e~, high *c*~d~) | 82 | High | High | 66.4 | High | Low |

### Supplementary Note: MCMC Validation

MCMC validation using NUTS (4 chains $\times$ 200 warmup + 200 samples) produced zero divergent transitions. Parameter estimates showed near-perfect agreement with the primary SVI fit: Pearson *r* between SVI and MCMC posterior means was 0.999 for both log(*c*~e~) and log(*c*~d~).

### Supplementary Note: Clinical Associations

**Table S2. Bayesian regression: discrepancy predicts anxiety symptoms.**

| Measure | $\beta$(discrepancy) | 94% HDI | P(positive) |
|---------|---------------------|---------|-------------|
| STAI-State | 0.338 | [0.23, 0.45] | 1.00 |
| STICSA | 0.285 | [0.17, 0.39] | 1.00 |
| DASS-Anxiety | 0.275 | [0.16, 0.40] | 1.00 |
| DASS-Stress | 0.255 | [0.13, 0.37] | 1.00 |
| DASS-Depression | 0.228 | [0.11, 0.35] | 1.00 |
| PHQ-9 | 0.225 | [0.11, 0.34] | 1.00 |
| OASIS | 0.228 | [0.12, 0.35] | 1.00 |
| AMI-Emotional | -0.222 | [-0.34, -0.11] | 0.00 |

*Note.* All models control for log(*c*~e~) and log(*c*~d~). ROPE = |$\beta$| < 0.10; computational parameters showed 65--93% posterior mass within ROPE. Encounter reactivity: AMI *r* = -0.17, *p* = .004; all anxiety measures *p* > .20.

### Supplementary Note: Simpson's Paradox in Vigor

High threat shifts choice toward light cookies (which have lower required press rates), so unconditional analyses show flat or declining vigor with threat. Conditioning on chosen cookie type reveals robust threat-driven vigor increases (heavy: *d* = 0.42; light: *d* = 0.49). This paradox highlights the necessity of jointly modeling choice and vigor.

### Supplementary Note: Distance Gradient and Population *c*~e~

Model M4 (population-level *c*~e~) achieves BIC = 30,860 ($\Delta$BIC = -1,273 vs. the full model), a nominal improvement. However, it fails completely at its primary purpose: per-subject choice *r*^2^ = 0.001. The apparent BIC advantage arises because fewer parameters are penalized, but the model cannot distinguish individuals who always choose heavy from those who always choose light. Individual *c*~e~ is essential for capturing the distance gradient that drives approximately 40% of between-subject choice variance.
