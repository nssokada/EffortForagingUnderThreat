# Movement Vigor as a Traitlike Attribute of Individuality

**Source:** Reppert, Rigas, Herzfeld, Sedaghat-Nejad, Komogortsev & Shadmehr (2018). *Journal of Neurophysiology*, 120, 741–757.

---

## Central Thesis

Movement vigor — the speed with which a person moves relative to the population average — is a **trait-like attribute of individuality** that is conserved across movement modalities (especially skeletal movements). This trait does not reflect a speed-accuracy tradeoff but instead likely reflects individual differences in the subjective evaluation of reward and effort, mediated by dopaminergic circuits in the basal ganglia.

---

## Defining Vigor

Vigor is quantified as a **scaling factor** of the population-average velocity–displacement relationship.

### The Canonical Velocity–Displacement Function

$$g(x) = \alpha \left(1 - \frac{1}{1 + \beta x}\right)$$

Where x is movement displacement (amplitude) and α, β are population-level parameters fit across all subjects. This hyperbolic function captures the saturating relationship between amplitude and peak velocity.

### Individual Vigor Parameter

For subject n, peak velocity on trial i is modeled as:

$$v_{n,i} = k_n \cdot g(x_i) + \varepsilon_n$$

Where:

- **k_n > 1:** Subject moves faster than the population average (high vigor).
- **k_n < 1:** Subject moves slower than the population average (low vigor).
- **ε_n:** Zero-mean Gaussian noise with subject-specific variance σ²_n.

The parameter k_n is estimated via maximum likelihood, providing a single scalar that characterizes each person's vigor for a given movement type.

---

## Two Competing Hypotheses for Vigor Differences

### Hypothesis 1: Speed-Accuracy Tradeoff

Classic motor control theory (signal-dependent noise) predicts that faster movements should be less accurate. Under this view, high-vigor individuals are simply more tolerant of inaccuracy.

### Hypothesis 2: Willingness to Expend Effort

Neuroeconomic models predict that vigor reflects the brain's evaluation of utility (reward − effort) / time. Under this view, high-vigor individuals assign greater subjective value to the reward or have a higher cost of time, making them willing to expend more effort.

---

## Key Results

### Experiment 1: Head-Fixed Saccades (n ≈ 289)

**Conservation across directions:**

- Individuals with high vigor in horizontal saccades also had high vigor in vertical saccades (R = 0.80, p < 10⁻⁶⁵).
- Some people move their eyes at nearly twice the velocity of others.

**Vigor–reaction time link:**

- Higher saccade vigor → shorter reaction times for both horizontal and vertical saccades.
- People with longer mean reaction times also had greater trial-to-trial reaction time variability (R = 0.59), consistent with a drift-diffusion model where utility drives the integration rate.

**No speed-accuracy tradeoff:**

- Higher vigor → greater variability in peak velocity (motor command noise), confirming signal-dependent noise.
- But this mid-movement variability did **not** translate into greater endpoint error. High-vigor individuals were equally accurate.
- The brain corrects mid-movement variability online (cerebellar feedback for saccades; sensory feedback for reaching).

**Sex differences:**

- Women had greater vigor in vertical (but not horizontal) saccades.
- Men had shorter reaction times for both horizontal and vertical saccades (~10 ms faster).
- Within each sex, higher vigor still predicted shorter reaction times.

### Experiment 2: Head-Free Reaching (n = 36)

**Cross-modal conservation:**

- Arm and head vigor were strongly correlated: R = 0.83 (p < 10⁻⁶). People who moved their arm fast also moved their head fast.
- Saccade vigor did **not** strongly predict arm or head vigor (positive trends but non-significant: R ≈ 0.27–0.31).
- This dissociation suggests that oculomotor vigor and skeletal motor vigor may be controlled by partially separate basal ganglia circuits.

**Arm vigor–reaction time link:**

- Higher arm vigor → shorter arm reaction times (R = −0.49, p = 0.003).
- Same pattern as saccades: those who move faster also initiate movement sooner.

**Arm vigor does not trade off with accuracy:**

- Higher arm vigor → greater peak velocity variability (R = 0.56), but no increase in endpoint error (R = 0.15, n.s.).

**Reaction time increases with target distance:**

- For reaching, reaction time increased with distance — consistent with the idea that greater anticipated effort reduces the utility signal and slows the evidence accumulation process.
- For saccades, reaction time showed a U-shaped pattern: decreasing for small amplitudes (release-from-fixation effect) then increasing for larger amplitudes (effort effect).

---

## The Trait Vigor Hypothesis (Bayesian Model)

### Generative Model

A single latent variable x_n (trait vigor) generates all measured variables for subject n:

$$\mathbf{y}_n = \mathbf{a} x_n + \mathbf{b} + \boldsymbol{\varepsilon}$$

Where y_n is a vector of measurements (peak velocities and reaction times across modalities), a and b are population-level parameters, and ε is Gaussian noise.

### Results

- **Experiment 1 (saccades only):** 77.6% posterior probability that a single trait vigor generated horizontal/vertical velocity and reaction time data. Confirmed by AIC comparison (p < 10⁻⁴⁹).
- **Experiment 2 (eye + head + arm):** 65% posterior probability that a single trait vigor generated all velocity and reaction time data across modalities. Confirmed by AIC (p < 10⁻⁶).
- The weaker cross-modal result is driven by the dissociation between saccade vigor and skeletal vigor. When restricted to arm and head only, posterior probability was 66% — essentially the same.

---

## Control Studies

### Reproducibility of Vigor

- **Head vigor:** Highly reproducible with and without concurrent arm movements (R = 0.97). The correlation isn't an artifact of co-movement.
- **Saccade vigor:** Highly reproducible across head-fixed and head-free conditions (R = 0.81).
- Prior work (Choi et al. 2014) showed within-subject saccade vigor varies ~3.5% across days, compared to ~50% range between subjects.

### Reaction Time Measurement Bias

- A velocity threshold was used to detect movement onset, which could introduce a bias (slower movements take longer to cross threshold).
- Simulation showed this bias accounts for ≈2 ms, while measured vigor-dependent reaction time differences were ≈10 ms.
- The vigor–reaction time relationship is not an artifact.

---

## Neural Basis of Vigor

### Basal Ganglia Circuit for Saccades

1. **Caudate nucleus** receives dopamine projections; fires more before rewarding saccades.
2. **Globus pallidus external (GPe)** inhibits SNr; fires more before vigorous saccades. Bilateral lesion eliminates reward-driven vigor modulation.
3. **Substantia nigra pars reticulata (SNr)** tonically inhibits superior colliculus. Deeper pause in SNr firing → more vigorous saccade.
4. Chronic dopamine depletion in caudate reduces saccade vigor by ~30%.

### Separate Circuits for Eye vs. Skeletal Vigor

- Saccade vigor: SNr → superior colliculus pathway.
- Skeletal vigor: GPi (internal globus pallidus) → thalamus pathway.
- These distinct pathways explain why saccade vigor and arm/head vigor are not strongly correlated, even though both are influenced by basal ganglia dopamine.

### Dopamine and Individual Differences

- Between-subject differences in dopamine D2/D3 binding in the striatum correlate with willingness to exert effort (Treadway et al. 2012).
- If vigor reflects willingness to exert effort, then individual differences in vigor may trace back to individual differences in striatal dopamine transmission.

---

## Theoretical Framework: Vigor as Utility Maximization

The theory connecting vigor to utility proceeds as follows:

1. The purpose of a voluntary movement is to acquire a rewarding state.
2. Motor commands = effort expenditure; movement duration = temporal discount on reward.
3. The brain's objective: maximize the **rate of reward** = (reward − effort) / time.
4. Variables that increase utility (higher reward, lower effort, higher cost of time) should increase vigor.
5. Individuals who move vigorously likely evaluate utility differently — higher subjective cost of time, greater willingness to expend effort, or higher valuation of reward.

This framework predicts:

- Reward increases vigor (confirmed empirically).
- Effort decreases vigor (confirmed empirically).
- Individuals with high temporal discounting (impatient) should have higher vigor (confirmed: Choi et al. 2014).
- Vigor differences should not reflect speed-accuracy tradeoff (confirmed here).

---

## Summary of Key Takeaways

1. Vigor is a stable, trait-like individual difference in movement speed relative to the population.
2. It is conserved across horizontal and vertical saccades, and across arm and head movements — but eye vigor and skeletal vigor are partially dissociated.
3. High vigor individuals react sooner and move faster, without sacrificing accuracy.
4. The brain compensates for greater mid-movement noise in high-vigor movements through online corrections.
5. Vigor likely reflects individual differences in the subjective evaluation of reward, effort, and time — not a willingness to accept inaccuracy.
6. The neural substrate involves dopaminergic modulation of basal ganglia circuits, with partially separate pathways for oculomotor and skeletal vigor.
