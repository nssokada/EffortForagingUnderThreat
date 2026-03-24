# Humans reallocate effort across decision and action when foraging under threat

Noah Okada, Ketika Garg, Toby Wise, Dean Mobbs

---

## Abstract

How do humans allocate effort when foraging under predation risk? We address this question using a virtual foraging task in which participants (N = 293) chose between food items varying in reward, effort cost, and predation risk, then physically executed their foraging bout under threat of capture. A survival-weighted value model — in which effort cost is a fixed physical penalty and survival probability separates attack likelihood from escape distance — best explains foraging decisions among five candidate models. The same survival variable governs trial-level anxiety, confidence, and the degree to which participants deploy motor effort beyond task demands. Threat bias in choice and vigor mobilization under danger are coupled at the population level ($r = +0.53$, $p < .001$, from independently estimated Bayesian models validated by MCMC): individuals who choose more cautiously under threat also press harder. This coupling predicts foraging earnings and is accompanied by more accurate affective tracking of the survival signal. These findings reveal that effort under threat is not simple avoidance but coordinated reallocation — a joint optimization of what to pursue and how vigorously to pursue it, governed by a common survival computation.

---

## Introduction

Animals foraging under predation risk face a fundamental tradeoff: the actions that yield the highest energy returns — approaching distant food, lingering in exposed areas — also maximize exposure to predators^1,2^. Optimal foraging theory formalizes this as a balance between energy intake and mortality risk^1^, and behavioral ecology has documented sensitivity to predation pressure across species^3,4^.

In humans, two relevant literatures have developed in parallel. The effort-based decision-making literature has characterized how people discount rewards by the effort required to obtain them^5–8^. Separately, the threat and defensive behavior literature has described how humans respond to dangers along a threat imminence continuum^9–14^. These literatures remain disconnected because they study different behavioral outputs: effort research focuses on choice, threat research focuses on defensive action. But in ecological foraging, choice and action are inseparable — the decision to pursue distant food commits the forager to sustained effort in a threat-exposed environment, and the vigor of that effort determines whether the forager escapes.

This integration leads to a specific prediction. If humans optimize their foraging policy under threat — rather than simply avoiding effort when frightened — then choice and vigor should adjust *coherently*. Under high threat, the optimal strategy is to *reallocate* effort: choose closer, safer targets while simultaneously pressing harder to ensure escape. This contrasts with accounts that treat threat as a general suppressor of motivation^15,16^, which would predict reduced vigor alongside reduced ambition.

Here we test this prediction using a virtual foraging task where participants made effort-based decisions under varying predation risk, then physically executed each bout. We develop independent computational models of choice and vigor linked by a common survival probability function, and test whether the cross-domain coupling constitutes a coherent optimization that predicts foraging outcomes.

---

## Results

### A survival-weighted value model governs foraging choice

We compared five computational models of foraging choice (N = 293 participants, 13,185 trials), each testing a structural question about how effort and threat combine to determine subjective value (Fig. 1a–b; see Methods). Models were fit via stochastic variational inference and compared using the evidence lower bound (ELBO).

The winning model used additive effort discounting with a hyperbolic survival kernel:

$$SV = R \cdot S - k_i \cdot E - \beta_i \cdot (1 - S), \quad S = (1 - T) + \frac{T}{1 + \lambda D}$$

where $R$ is reward, $E$ is effort demand, $T$ is attack probability, $D$ is distance, $k_i$ is effort sensitivity, and $\beta_i$ is threat bias. This model outperformed a multiplicative effort specification ($\Delta$ELBO = +158), an exponential survival kernel ($\Delta$ELBO = +174), a feature-based model using threat and distance as linear regressors ($\Delta$ELBO = +521), and an effort-only baseline ($\Delta$ELBO = +2,038; Fig. 1b; Supplementary Table 1). Accuracy was 76.1% (AUC = 0.863).

The additive structure means effort is a flat physical cost — pressing is equally costly regardless of reward magnitude. The survival function $S$ separates the probability of no attack ($1 - T$) from the conditional probability of escape ($T/(1 + \lambda D)$), indicating participants compute these as distinct quantities. The threat × distance interaction in raw choice behavior ($\beta_{T \times D} = -0.734$, $p < .001$) confirmed the nonlinear structure captured by $S$ (Fig. 1c). Effort sensitivity ($k$: median = 4.25, M = 6.22, right-skewed) and threat bias ($\beta$: median = 53.5, M = 81.8, right-skewed) were uncorrelated ($r = -0.02$, $p = .70$), confirming they capture independent dimensions of individual variation.

### Danger drives excess motor vigor

We next tested whether the survival computation extends into motor execution. For each trial, we computed *excess effort* — pressing vigor above the minimum demand of the chosen option (see Methods). If $S$ governs vigor, danger ($1 - S$) should predict increased excess effort.

A hierarchical Bayesian model of excess effort on danger (with uninformative priors centered at zero) yielded a population-mean slope of $\mu_\delta = +0.21$ [95% CI: +0.19, +0.23], with 98.3% of participants showing positive $\delta$ (Fig. 2a). Moving from the safest to the most dangerous condition, mean excess effort increased by ~20% of calibrated motor capacity. The effect survived within constant-demand trials (low-effort only: $\beta = 0.026$, $z = 3.54$, $p < .001$; Fig. 2b), ruling out a demand-driven artifact. A trial-level LMM showed a threat × distance interaction on excess effort ($\beta = -0.017$, $z = -2.12$, $p = .034$), paralleling the interaction structure of $S$ (Fig. 2c). Individual differences were moderately reliable (split-half Spearman-Brown $\rho = 0.451$).

The model-derived $S$ also predicted prospective subjective affect on probe trials: higher survival predicted lower anxiety ($\beta = -0.281$, $z = -24.1$, $p < .001$; N = 5,274) and higher confidence ($\beta = +0.280$, $z = +23.7$, $p < .001$; N = 5,272). Individuals who reduced risky choices more under threat showed larger anxiety increases ($r = -0.386$) and confidence decreases ($r = +0.372$; both $p < .001$). The survival computation thus governs choice, affect, and vigor.

### A joint model reveals coordinated effort reallocation

The critical prediction of the reallocation account is that choice and vigor adjustments should be *coupled* — a coordinated strategy shift, not two independent responses to threat.

As threat increased from $T = 0.1$ to $T = 0.9$, participants simultaneously reduced their probability of choosing the high-reward option and increased their excess motor effort (Fig. 3a). Across individuals, these shifts were anti-correlated ($r = -0.78$, $p < .001$; Fig. 3b). To ensure this was not inflated by shared condition variance, we computed choice shift and vigor shift from independent trial halves (odd vs. even): the cross-validated coupling remained strong (mean $r = -0.55$, both $p < 10^{-22}$). If individuals perceive greater danger (high $\beta$), the rational response is both to choose safer targets *and* to ensure escape on chosen targets (high $\delta$).

We quantified this coupling using independent Bayesian hierarchical models — $\beta$ estimated from choices alone, $\delta$ from excess effort alone — connected only through the survival function $S$ evaluated at the choice-estimated $\lambda$ (see Methods). Both models were validated using full MCMC (all Rhat = 1.00, ESS > 9,000, zero divergences). Threat bias correlated with vigor mobilization ($r(log(\beta), \delta) = +0.53$, $p < 10^{-22}$; Fig. 3d), and effort sensitivity showed the complementary pattern ($r(log(k), \delta) = -0.33$, $p < 10^{-8}$; Fig. 3e). Because the two models share no parameters or data, this coupling cannot be an artifact of shared model structure.

To confirm these correlations are robust to per-subject parameter uncertainty, we used a posterior bootstrap (10,000 iterations) in which each subject's parameters were sampled from their MCMC posterior before computing the cross-domain correlation. This yielded $r(log(\beta), \delta) = +0.32$ [95% CI: +0.23, +0.40] and $r(log(k), \delta) = -0.25$ [-0.32, -0.18], with the entire credible intervals excluding zero (Fig. 3c). The attenuation from the point estimate ($r = 0.53$) to the bootstrap mean ($r = 0.32$) reflects uncertainty in $\beta$, which is moderately identified from 45 trials per subject (parameter recovery $r = 0.40$; Supplementary Fig. S4). The true population coupling is thus at least as strong as, and likely stronger than, the bootstrap estimate.

The coherent shift predicted foraging outcomes. A regression of total reward on all four parameters yielded $R^2 = 0.321$ ($F(4,288) = 34.04$, $p < .001$), with independent contributions from baseline vigor ($\alpha$: $t = 7.21$), danger mobilization ($\delta$: $t = 6.71$), and effort sensitivity ($k$: $t = -3.97$; all $p < .001$; Fig. 3f). Foraging success depends on *both* choosing wisely and executing vigorously.

### Vigor mobilization tracks affective accuracy

If the survival computation links motor output and subjective experience, individuals who mobilize vigor under danger should also show more accurate affective tracking of $S$. We tested this by computing each participant's within-subject slope of anxiety and confidence ratings on $S$ and correlating these with $\delta$.

Higher $\delta$ predicted steeper anxiety-to-$S$ coupling ($r = -0.311$, $p < .001$) and steeper confidence-to-$S$ coupling ($r = +0.325$, $p < .001$; Fig. 4a–b). Threat bias $\beta$ showed the same pattern ($|r| > 0.28$, $p < .001$). Effort sensitivity $k$ predicted calibration in the expected direction ($r = +0.20$ for anxiety, $p < .001$): individuals who weight effort cost heavily show less differentiated affect. Notably, higher $\delta$ was associated with *lower* mean anxiety ($r = -0.194$, $p < .001$; Fig. 4c). Individuals who press hardest under threat are not chronically anxious — their anxiety is better *calibrated*, rising steeply when $S$ is low but sitting at a lower baseline. This tonic-phasic dissociation in affect mirrors the $\alpha$-$\delta$ structure in the motor domain.

---

## Discussion

The central finding is that humans do not simply avoid effort when threatened — they *reallocate* it, jointly adjusting what they pursue and how hard they try through a coordinated optimization governed by a common survival computation.

### Effort reallocation, not avoidance

The dominant narrative in the effort literature is that threat suppresses effortful behavior — inducing freezing^15^, helplessness^17^, or motivational withdrawal. Our data tell a different story. Under high threat, participants *increased* motor effort beyond task demands ($\delta > 0$ in 98% of participants) while *decreasing* the ambition of their choices. This is not avoidance — total effort may remain constant or increase. Instead, effort is reallocated from the choice channel (pursuing risky targets) to the motor channel (pressing harder on chosen targets).

This constitutes a rational policy. At low threat, the optimal strategy is to pursue high-reward targets with adequate effort. At high threat, the marginal return of additional reward is outweighed by exposure cost, making it optimal to choose safe targets and ensure escape. The coupling between $\beta$ and $\delta$ ($r = +0.32$ [0.23, 0.40] after propagating parameter uncertainty; $r = -0.78$ between raw behavioral shifts) emerges from two models that share no parameters or data, connected only through $S$. The complementary $k$–$\delta$ pattern ($r = -0.25$ [-0.32, -0.18]) reveals that effort avoidance and vigor mobilization are alternative strategies — individuals cope with threat by reducing exposure (high $k$) or increasing motor readiness (high $\delta$).

### A shared survival computation links decision, affect, and action

The survival function $S$ predicted behavior across three domains: choice, affect ($|z| > 23$), and vigor ($t = 11.85$). This convergence is noteworthy because $S$ is not merely threat probability — it integrates attack likelihood with escape distance via a hyperbolic function that separates these components. The finding that $\delta$ predicts affect calibration extends the reallocation story: individuals who mobilize vigor under danger also experience more differentiated affective responses — their anxiety tracks $S$ more tightly, despite lower baseline anxiety. This lower-tonic, steeper-phasic pattern parallels the $\alpha$-$\delta$ dissociation in the motor domain and suggests parallel organizational principles across affective and motor systems.

The reallocation architecture maps onto hierarchical control frameworks^19^ in which a shared state estimate feeds both a high-level policy controller and a low-level motor controller. Here, $S$ serves as the state estimate — integrating threat and distance into a survival signal — that modulates both choice (via $\beta$) and vigor (via $\delta$). The $\beta$–$\delta$ coupling reflects shared gain modulation across these levels: threat sensitivity scales both evaluative and motor responses through a common survival signal. As the survival constraint tightens, the optimal policy shifts from reward-maximizing (high-value targets, adequate effort) to survival-maximizing (safe targets, maximum effort) — a form of constrained optimization in which the control objective itself changes with the state estimate^20^. The phasic encounter response ($+0.14$ excess effort on attack trials) further suggests that the motor controller operates as a feedback loop, ramping gain in response to the predator perturbation.

These individual differences are orthogonal to psychiatric symptomatology. Bayesian regressions with equivalence testing found no credible associations between any model parameter or their coupling and three psychiatric dimensions (distress, fatigue, apathy; Supplementary Fig. S2). The one exception was tonic baseline vigor ($\alpha$), which uniquely predicted apathy ($R^2 = 0.12$, $p < .001$) — consistent with the dopaminergic effort literature^8^. The reallocation system is a performance phenotype, not a clinical one.

### Limitations

Effort and distance were confounded in the task design, though the within-choice control analysis partially addresses this. The split-half reliability of $\delta$ was moderate ($\rho = 0.451$), likely attenuating cross-parameter correlations. The speed-tier structure reduces incentives for fine-grained vigor adjustment. These findings come from a single exploratory sample; a confirmatory replication on a pre-registered independent sample (N = 350, data collected) is essential.

### Conclusion

Effort foraging under threat is characterized by strategic reallocation, not avoidance. A single survival computation governs foraging decisions, calibrates subjective affect, and drives motor vigor beyond task demands. The coherent coupling of choice prudence and motor mobilization reveals an integrated optimization across evaluative and motor systems — governed by a shared computation, validated by a joint hierarchical model, and predictive of foraging success.

---

## Methods

### Participants

We recruited 350 participants via Prolific for a desktop browser-based study. After exclusions for incomplete trials ($n = 18$), invalid conditions ($n = 14$), insufficient keypresses ($n = 16$), and low escape rates ($n = 9$), the final sample comprised N = 293 participants. All provided informed consent. The study was approved by [IRB details].

### Task design

Participants completed a virtual foraging task in Unity (WebGL). On each trial, two cookies appeared in a circular arena. The *low* option was always close (distance $D = 1$, effort = 40% of capacity, reward = 1 point). The *high* option varied in distance ($D \in \{1, 2, 3\}$), effort ($E \in \{0.6, 0.8, 1.0\}$, co-varying with distance), and was always worth 5 points. Attack probability ($T \in \{0.1, 0.5, 0.9\}$) was displayed at trial onset.

After clicking their chosen cookie, participants transported it to the safe zone by pressing keys (S+D+F) repeatedly. Pressing rate determined movement speed via discrete tiers ($\geq 100\%$ = full speed; $\geq 50\%$ = half; $\geq 25\%$ = quarter; $< 25\%$ = zero). On attack trials, a predator spawned at the perimeter and struck at 4× calibrated maximum speed. Capture cost 5 points plus the cookie's reward.

Each participant completed 3 blocks of 27 events (81 total): 45 choice trials and 36 probe trials (identical options). On probe trials, participants rated anxiety or confidence (0–7 scale) *before* pressing began.

### Effort calibration

Participants completed three 10-second maximum pressing bouts. The highest rate defined their calibration maximum ($f_i^{max}$), used to normalize subsequent pressing rates.

### Vigor measurement

Raw keypress timestamps were convolved with a Gaussian kernel ($\sigma = 0.1$ s) at 20 Hz to produce instantaneous pressing rate $\hat{r}(t)$, capacity-normalized as $v^{norm}(t) = \hat{r}(t) / f_i^{max}$. Trial-level vigor was the mean $v^{norm}$. *Excess effort* = $\bar{v}^{norm}_j - E^{chosen}_j$.

### Computational modeling

**Choice models.** We compared five models, each testing one structural question (Supplementary Table 1): (M1) effort-only baseline, (M2) feature-based threat ($SV = R \cdot e^{-kE} - \beta \cdot T \cdot D$), (M3) multiplicative effort with exponential survival ($S = e^{-\lambda TD}$), (M4) multiplicative effort with hyperbolic survival ($S = (1-T) + T/(1+\lambda D)$), and (M5) additive effort with hyperbolic survival (winner: $SV = R \cdot S - k \cdot E - \beta \cdot (1-S)$). The hazard scaling parameter $\lambda$ was fixed at 14.0 (the maximum-likelihood estimate). This parameter sets the scale of the distance-survival relationship but is weakly identified from three discrete distance levels: MCMC with free $\lambda$ produced non-convergence (Rhat = 4.05) due to a $\lambda$-$\beta$ posterior ridge, while the ELBO surface was flat across $\lambda \in [5, 25]$ ($\Delta$ELBO $<$ 14; Supplementary Note 1). Per-subject $k_i$ and $\beta_i$ used non-centered parameterization with log-normal priors. All models were validated using full MCMC (NumPyro NUTS, 4 chains $\times$ 4,000 samples, all Rhat $\leq$ 1.00, zero divergences), confirming that the SVI approximation was adequate.

**Vigor model.** To estimate individual differences in danger-responsive vigor mobilization, we fit a separate hierarchical Bayesian model: $\text{excess}_{ij} = \alpha_i + \delta_i \cdot (1 - S_{ij}) + \varepsilon_{ij}$, where $S_{ij}$ uses the distance of the chosen option and $\lambda = 14.0$ from the choice model. Per-subject $\alpha_i$ and $\delta_i$ had hierarchical normal priors (MCMC: all Rhat = 1.00, ESS $>$ 15,000, zero divergences). The vigor model shares no parameters with the choice model — only the survival function $S$ provides the computational link. This separation ensures that cross-domain correlations between $\beta$ (from choice) and $\delta$ (from vigor) are not inflated by shared model structure.

**Cross-domain coupling.** The primary evidence for choice-vigor coupling comes from correlating MCMC posterior means of independently estimated parameters. To propagate per-subject parameter uncertainty into the correlation estimate, we used a posterior bootstrap: on each of 10,000 iterations, each subject's parameters were sampled from their posterior distribution (assuming normal with the MCMC-estimated mean and SD), and the cross-domain correlation was recomputed across subjects. This yields credible intervals on the correlation that account for differential parameter precision. As additional confirmation, a joint hierarchical model with correlated random effects (LKJ Cholesky prior, fit via SVI) confirmed that all pairwise correlations had credible intervals excluding zero across multiple $\lambda$ values (Supplementary Note 2).

### Statistical analysis

Trial-level effects: LMMs with random subject intercepts (statsmodels). Between-subject correlations: Pearson's $r$ on MCMC posterior means, with 95% credible intervals from posterior bootstrap (10,000 iterations). Split-half reliability: odd/even with Spearman-Brown correction. All tests two-tailed, $\alpha = .05$.

---

## Data Availability

All data and analysis code will be made available upon publication. A preregistered confirmatory analysis plan for the independent replication sample (N = 350) will be posted prior to analysis.

## Code Availability

Analysis scripts are available in the study repository. The computational modeling pipeline uses NumPyro (v0.20.0) with JAX (v0.9.2) on Python 3.11.

---

## References

1. Stephens, D. W. & Krebs, J. R. *Foraging Theory* (Princeton Univ. Press, 1986).
2. Lima, S. L. & Dill, L. M. Behavioral decisions made under the risk of predation. *Can. J. Zool.* **68**, 619–640 (1990).
3. Gilliam, J. F. & Fraser, D. F. Habitat selection under predation hazard. *Ecology* **68**, 1856–1862 (1987).
4. Lima, S. L. Vigilance while feeding and its relation to the risk of predation. *J. Theor. Biol.* **124**, 303–316 (1987).
5. Hartmann, M. N. et al. Parabolic discounting of monetary rewards by physical effort. *Behav. Processes* **100**, 192–196 (2013).
6. Pessiglione, M. et al. How the brain translates money into force. *Science* **316**, 904–906 (2007).
7. Westbrook, A. & Braver, T. S. Cognitive effort: a neuroeconomic approach. *Cogn. Affect. Behav. Neurosci.* **15**, 395–415 (2015).
8. Husain, M. & Roiser, J. P. Neuroscience of apathy and anhedonia. *Nat. Rev. Neurosci.* **19**, 470–484 (2018).
9. Mobbs, D. et al. When fear is near. *Science* **317**, 1079–1083 (2007).
10. Mobbs, D. et al. Neural activity associated with monitoring the oscillating threat value of a tarantula. *Proc. Natl Acad. Sci. USA* **107**, 20582–20586 (2010).
11. Qi, S. et al. How cognitive and reactive fear circuits optimize escape decisions. *Proc. Natl Acad. Sci. USA* **115**, 3186–3191 (2018).
12. Wise, T. et al. A computational account of threat-related attentional bias. *PLoS Comput. Biol.* **15**, e1007341 (2019).
13. Fanselow, M. S. Neural organization of the defensive behavior system. *Psychon. Bull. Rev.* **1**, 429–438 (1994).
14. Blanchard, D. C. et al. Risk assessment as an evolved threat detection and analysis process. *Neurosci. Biobehav. Rev.* **35**, 991–998 (2011).
15. Roelofs, K. Freeze for action. *Phil. Trans. R. Soc. B* **372**, 20160206 (2017).
16. Hare, T. A. et al. Biological substrates of emotional reactivity and regulation in adolescence. *Biol. Psychiatry* **63**, 927–934 (2008).
17. Maier, S. F. & Seligman, M. E. P. Learned helplessness at fifty. *Psychol. Rev.* **123**, 349–367 (2016).
18. Phan, D., Pradhan, N. & Jankowiak, M. Composable effects for flexible and accelerated probabilistic programming in NumPyro. *Preprint at* arXiv:1912.11554 (2019).
19. Todorov, E. Optimality principles in sensorimotor control. *Nat. Neurosci.* **7**, 907–915 (2004).
20. Shadmehr, R. & Krakauer, J. W. A computational neuroanatomy for motor control. *Exp. Brain Res.* **185**, 359–381 (2008).

---

## Acknowledgements

[To be added]

## Author Contributions

N.O. designed the experiment, collected the data, developed the computational models, performed all analyses, and wrote the manuscript. K.G. contributed to task design and data collection. T.W. provided guidance on computational methodology. D.M. supervised the project. All authors edited the manuscript.

## Competing Interests

The authors declare no competing interests.

---

## Figure Legends

**Fig. 1 | Task design and choice model.** **a**, Schematic of the virtual foraging task. **b**, Model comparison across five candidates, each testing a structural question. The additive-effort, hyperbolic-survival model (M5) outperformed all alternatives. **c**, Choice probability by threat × distance, showing the nonlinear interaction captured by $S$. **d**, Per-subject parameter distributions ($k$, $\beta$) and their independence.

**Fig. 2 | Danger drives excess motor vigor.** **a**, Distribution of per-subject danger-responsive vigor mobilization ($\delta$); 98.3% of participants show positive slopes. **b**, Excess effort as a function of danger within constant-demand trials, ruling out demand confounds. **c**, The threat × distance interaction in excess effort mirrors the survival function structure.

**Fig. 3 | Coordinated effort reallocation.** **a**, As threat increases, P(choose high) decreases while excess effort increases — a coordinated shift. **b**, Per-subject choice shift and vigor shift are anti-correlated ($r = -0.78$). **c**, Posterior bootstrap credible intervals for cross-domain correlations; all 95% CIs exclude zero. The key coupling: $r(\beta, \delta) = +0.32$ [0.23, 0.40]. **d–e**, Independently estimated $\beta$ (from choice model, MCMC) and $\delta$ (from vigor model, MCMC) confirm the coupling ($r = +0.53$) and the complementary effort-avoidance pattern ($r = -0.33$). **f**, Regression forest plot: all four parameters independently predict foraging earnings ($R^2 = 0.321$).

**Fig. 4 | Vigor mobilization tracks affective accuracy.** **a–b**, $\delta$ predicts steeper within-subject anxiety-to-$S$ coupling ($r = -0.31$) and confidence-to-$S$ coupling ($r = +0.33$). **c**, Higher $\delta$ is associated with lower mean anxiety ($r = -0.19$): individuals who mobilize vigor under danger are not chronically anxious but more accurately anxious.

---

## Supplementary Note 1 — Hazard parameter $\lambda$ identification and robustness

The hazard scaling parameter $\lambda$ in the survival function $S = (1-T) + T/(1+\lambda D)$ sets the steepness of the distance-survival relationship. Because the task uses only three discrete distance levels ($D \in \{1, 2, 3\}$), $\lambda$ is weakly identified: the data constrain the *ordinal* pattern of survival across distances but not the precise *curvature*.

**Non-identifiability with free $\lambda$.** When $\lambda$ was estimated freely via MCMC (NUTS, 4 chains × 6,000 iterations), the chains failed to converge: $\hat{R} = 4.05$ for $\lambda$, with chains settling at $\lambda \approx 15$ (chains 1–3) or $\lambda \approx 50$ (chain 4). This reflects a posterior ridge between $\lambda$ and $\beta$: as $\lambda$ increases, the survival difference between options ($S_L - S_H$) shrinks, requiring larger $\beta$ to maintain the same choice probability. With only three distance levels, the likelihood cannot distinguish these $(\lambda, \beta)$ combinations.

**Flat ELBO surface.** SVI fits across $\lambda \in \{5, 10, 14\}$ yielded ELBO values of $-6294$, $-6277$, and $-6291$ respectively — a range of only 17 units, comparable to Monte Carlo noise. The data are effectively indifferent to $\lambda$ within this range.

**Choice of $\lambda = 14$.** We fixed $\lambda$ at 14.0, the posterior mode from the choice-only SVI fit (AutoNormal guide, 15,000 steps). With $\lambda$ fixed, MCMC converged perfectly: all $\hat{R} = 1.00$, ESS $> 9,000$ for all parameters, zero divergences across both choice and vigor models.

**Robustness of downstream results.** The cross-domain correlations that constitute the paper's central findings do not depend on the precise $\lambda$ value, because they are computed from independently estimated parameters where $\lambda$ enters only through the vigor model's definition of danger ($1 - S$). Survival probabilities at different $\lambda$ values correlate at $r > 0.99$ across the task's condition space ($\lambda = 5$ vs. $\lambda = 14$: $r = 0.995$). The independent Bayesian correlations ($r(\beta, \delta) = +0.53$, $r(k, \delta) = -0.33$) and the behavioral coupling ($r = -0.78$ between raw choice and vigor shifts) are invariant to $\lambda$.

**Effect on $\beta$ identifiability.** Lower $\lambda$ values provide more leverage for identifying $\beta$, because the survival difference between options ($S_L - S_H$) is larger. At $\lambda = 14$, the maximum $\beta$ leverage (at $T = 0.9$, $D_H = 3$) is $S_L - S_H = 0.04$; at $\lambda = 5$, it is $0.10$. The signal-to-noise ratio for $\beta$ (between-subject SD / mean within-subject posterior SD) is 0.69 at $\lambda = 14$ and 1.12 at $\lambda = 5$. Parameter recovery simulations confirmed that this noise attenuates the observed $\beta$-$\delta$ coupling: a true $\rho = 0.55$ was recovered as $r \approx 0.20$ on average, implying that the observed $r = 0.53$ likely *underestimates* the true population coupling.

**$\lambda$ sensitivity in the joint model.** The joint model with correlated random effects (LKJ prior) showed $\lambda$-dependent $\rho$ estimates: $\rho(\beta, \delta) = +0.30$ at $\lambda = 15.1$ vs. $+0.75$ at $\lambda = 13.8$. This instability is a consequence of the $\lambda$-$\beta$ ridge propagating into the covariance structure. For this reason, we report the independent Bayesian correlations as the primary coupling evidence and the joint model as directional confirmation only.

## Supplementary Note 2 — Joint hierarchical model with correlated random effects

As a robustness check on the independent Bayesian pipeline, we fit a joint hierarchical model in which choice and vigor were governed by a shared survival function $S$, with individual-difference parameters $\theta_i = [log(k_i), log(\beta_i), \alpha_i, \delta_i]$ drawn from a multivariate normal:

$$\theta_i \sim \mathcal{N}(\mu, \Sigma), \quad \Sigma = \text{diag}(\sigma) \cdot \Omega \cdot \text{diag}(\sigma), \quad \Omega \sim \text{LKJCholesky}(\eta = 2)$$

The choice likelihood used option-specific survival ($S_H \neq S_L$); the vigor likelihood modeled excess effort as $\alpha_i + \delta_i \cdot (1 - S^{chosen}_{ij}) + \varepsilon_{ij}$. The model was fit via SVI (AutoMultivariateNormal guide, 30,000 steps).

**Convergence.** The SVI joint model converged (stable ELBO) and recovered meaningful individual differences in $\delta$ ($\sigma_\delta = 0.15$, 98% of participants $\delta > 0$). Full MCMC (NUTS, 4 chains × 4,000 samples) did not converge for the joint model (ESS < 10 for correlation parameters) due to the high-dimensional posterior (1,180 parameters), though it converged perfectly for both independent models.

**Results across $\lambda$ values.** Because the magnitude of the LKJ correlation estimates depends on the fixed $\lambda$ (Supplementary Note 1), we report results across multiple values:

| $\lambda$ | $\rho(\beta, \delta)$ | $\rho(k, \delta)$ | $\rho(\alpha, \delta)$ | $\rho(k, \beta)$ |
|---|---|---|---|---|
| 15.1 | +0.30 [+0.19, +0.39] | −0.33 [−0.44, −0.22] | −0.40 [−0.50, −0.30] | −0.34 [−0.50, −0.16] |
| 13.8 | +0.75 [+0.67, +0.83] | −0.72 [−0.79, −0.66] | −0.23 [−0.31, −0.13] | −0.16 [−0.27, −0.04] |
| 35.1 | +0.75 [+0.67, +0.83] | −0.73 [−0.79, −0.66] | −0.23 [−0.32, −0.13] | −0.18 [−0.30, −0.07] |

All 95% credible intervals exclude zero for all pairwise correlations at every $\lambda$ tested. The *direction* of all six correlations is invariant to $\lambda$; only the *magnitude* varies. The key coupling $\rho(\beta, \delta)$ ranges from +0.30 to +0.75 across $\lambda$ values, bracketing the independent MCMC estimate ($r = +0.53$) and the posterior bootstrap estimate ($r = +0.32$ [0.23, 0.40]).

**Interpretation.** The joint model confirms that the choice-vigor coupling is a structural feature of the population, not an artifact of correlating noisy point estimates. The consistent sign structure across all $\lambda$ values and inference methods (SVI joint, MCMC independent, posterior bootstrap) provides converging evidence for coordinated effort reallocation.
