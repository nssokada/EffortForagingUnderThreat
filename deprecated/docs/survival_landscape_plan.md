# Survival Landscape Computation Plan (Updated)

## The Problem

The model estimates p_esc = 0.002, implying that pressing harder has essentially no survival benefit. This makes the vigor model's survival optimization story hollow — c_d can't drive meaningful vigor variation through a flat survival gradient.

Two likely causes:

1. **Step function masking:** The game uses a step function for movement speed, but the model fits a smooth sigmoid to the press-rate-survival relationship. The step function has flat plateaus — pressing at 100% and 150% of required gives identical speed. The sigmoid averages over the threshold discontinuity and washes out the actual survival mechanism.

2. **Wrong survival mechanism:** The real survival benefit of pressing harder isn't about moving faster at full speed — it's about **reducing exposure time**. The predator can strike starting ~1.5 seconds after encounter. After that, every additional second the participant is still in the field is a second they can be caught. The marginal exposure times are approximately 1 second at D=1, 2 seconds at D=2, and 3 seconds at D=3. Pressing harder reduces this exposure window by getting the participant home sooner.

The core mechanism is: **pressing harder → more time at full speed → less travel time → shorter exposure window → higher survival probability.** The survival function should reflect this exposure-time mechanism directly.

---

## Step 0: Empirical Ground Truth

Before modeling anything, compute the raw empirical relationships. These numbers determine whether the survival landscape has any gradient at all.

### 0a. Escape rates by speed tier

For all attack trials in the dataset:

```python
# Classify each trial by the dominant speed tier
# Using the 20 Hz press rate data, compute what fraction of the 
# transport phase each participant spent in each speed tier:
#   full_speed:    press_rate >= 1.0 * req
#   half_speed:    0.5 * req <= press_rate < 1.0 * req  
#   quarter_speed: 0.25 * req <= press_rate < 0.5 * req
#   stopped:       press_rate < 0.25 * req

# For each trial, compute:
frac_full_speed = time_at_full_speed / total_transport_time
frac_half_or_below = 1 - frac_full_speed

# Then compute:
# 1. Escape rate among trials where frac_full_speed > 0.95 (nearly perfect pressing)
# 2. Escape rate among trials where frac_full_speed is 0.80-0.95
# 3. Escape rate among trials where frac_full_speed < 0.80
# 
# Break this down by distance (D=1, D=2, D=3)
```

**What we're looking for:** A meaningful difference in escape rates between speed tiers. If trials with >95% full-speed time escape at 25% and trials with <80% full-speed time escape at 5%, we have a real survival gradient. If both are near 2%, the task genuinely doesn't reward pressing and we should reframe c_d as arousal.

### 0b. Escape rates at full speed by distance

```python
# Among attack trials where the participant maintained full speed 
# throughout (frac_full_speed > 0.95):
# 
# escape_rate_D1 = ?  (expect highest — shortest exposure window ~1 sec)
# escape_rate_D2 = ?  (intermediate — exposure window ~2 sec)
# escape_rate_D3 = ?  (expect lowest — longest exposure window ~3 sec)
```

**What we're looking for:** These values directly test the exposure-time mechanism. If escape rates decrease substantially with distance among full-speed trials, it confirms that the distance-dependent exposure window is the primary survival determinant. Expected pattern: escape_rate_D1 >> escape_rate_D3.

### 0c. Correlation between mean excess vigor and survival

```python
# Simple bivariate:
excess_vigor = mean_press_rate - required_rate  # per trial
survived = 1 or 0  # per trial

# Correlation within attack trials only, controlling for T and D
# Also compute separately within each distance level
```

**What we're looking for:** The correlation should be strongest at D=3 (where the exposure window is longest and there's the most time to shave off by pressing harder) and weakest at D=1 (where the exposure window is already short). This distance-dependent gradient would confirm the exposure mechanism.

### 0d. Press rate variability and survival

```python
# For each trial, compute:
press_rate_sd = std(20Hz_press_rate_during_transport)
press_rate_cv = press_rate_sd / mean_press_rate

# Among attack trials, does lower variability predict survival?
# Control for mean press rate, T, and D
```

**What we're looking for:** If press rate consistency predicts survival above and beyond mean press rate, it confirms that staying above the full-speed threshold (and thus maintaining minimum travel time) is the key mechanism. Variability causes momentary speed drops that extend travel time and therefore exposure.

### 0e. Exposure time and survival (direct test)

```python
# For each attack trial, compute the actual exposure time:
# exposure_time = time from first possible strike (~1.5 sec after encounter) 
#                 until participant reaches safe zone
# 
# If participant reached safety before first possible strike:
#     exposure_time = 0 (escaped with certainty)
#
# Correlate exposure_time with survival (binary) within attack trials
# Break down by distance
```

**What we're looking for:** A strong negative relationship between exposure time and survival. This is the most direct test of the exposure mechanism. If exposure time predicts survival well, the exposure-based survival function is the right specification.

---

## The Exposure-Time Survival Function

### Core Mechanism

After the predator encounters the participant (~halfway through the return trip), the predator can begin striking at approximately 1.5 seconds post-encounter. From that point on, the participant is vulnerable. The **exposure time** is the duration between when the predator can first strike and when the participant reaches safety:

```
exposure(u, D) = max(0, travel_time_remaining(u, D) - t_safe)
```

where:
- `travel_time_remaining(u, D)` = time from encounter to reaching the safe zone, depends on press rate and distance
- `t_safe` ≈ 1.5 seconds = the grace period after encounter before the predator can strike

If the participant reaches safety within t_safe seconds of the encounter, exposure = 0 and they escape with certainty. Otherwise, they are exposed for the remaining time.

### Approximate exposure times at full speed

From the known task mechanics, the marginal exposure at full speed is approximately:
- D=1: ~1 second of exposure
- D=2: ~2 seconds of exposure
- D=3: ~3 seconds of exposure

### Escape probability as a function of exposure

During the exposure window, the predator strikes according to a stochastic process. The simplest model: a constant hazard rate λ during the exposure window.

```
P_escape(u, D) = exp(-λ × exposure(u, D))
```

This means:
- Zero exposure → P_escape = 1 (certain escape)
- Exposure = 1 sec → P_escape = exp(-λ)
- Exposure = 3 sec → P_escape = exp(-3λ)

For λ = 0.3/sec (moderate strike rate):
- D=1 at full speed: P_escape ≈ 0.74
- D=2 at full speed: P_escape ≈ 0.55
- D=3 at full speed: P_escape ≈ 0.41

These are substantial, distance-separated values with real gradient. Compare to the previous p_esc = 0.002.

### How press rate affects exposure

Pressing harder → higher fraction of time at full speed → lower travel time → shorter exposure.

The travel time after encounter depends on the fraction of time spent at full vs. half speed. If fraction f of the remaining trip is at full speed and (1-f) at half speed:

```
travel_time_remaining(f, D) = remaining_distance(D) / effective_speed(f)
```

where:

```
effective_speed(f) = f × v_full + (1-f) × 0.5 × v_full = v_full × (0.5 + 0.5f)
```

So:

```
travel_time_remaining(f, D) = remaining_distance(D) / (v_full × (0.5 + 0.5f))
```

And exposure:

```
exposure(f, D) = max(0, remaining_distance(D) / (v_full × (0.5 + 0.5f)) - t_safe)
```

### Connecting press rate to full-speed fraction

The fraction of time at full speed depends on the buffer between mean press rate and required rate, mediated by motor noise:

```
f = Φ((u - req) / σ_motor)
```

where Φ is the normal CDF and σ_motor is the within-trial press rate variability.

### Complete survival function

Combining all components:

```
S(u, T, D) = (1 - T) + T × P_escape(u, D)
```

where:

```
f(u) = Φ((u - req) / σ_motor)
effective_speed(f) = v_full × (0.5 + 0.5 × f)
travel_remaining(f, D) = remaining_dist(D) / effective_speed(f)
exposure(f, D) = max(0, travel_remaining(f, D) - t_safe)
P_escape(u, D) = exp(-λ × exposure(f(u), D))
```

### The derivative (for the first-order condition)

```
dP_escape/du = λ × P_escape × |d(exposure)/du|

d(exposure)/du = d(exposure)/df × df/du

d(exposure)/df = -remaining_dist(D) × 0.5 × v_full / (v_full × (0.5 + 0.5f))²
               = -remaining_dist(D) / (2 × v_full × (0.5 + 0.5f)²)

df/du = φ((u - req) / σ_motor) / σ_motor
```

The key feature: this derivative is largest when:
1. f is near 0.5 (press rate near required — the normal PDF is maximized at the threshold)
2. D is large (more remaining distance, so speed changes have larger time effects)
3. Exposure is already long (the exponential makes reductions in exposure more valuable when you're more exposed)

This creates exactly the survival gradient we need: pressing harder helps most at far distances under high threat — the conditions where c_d should have maximum leverage.

---

## Option A: Step Function Fidelity (Direct 20 Hz Approach)

### Concept

Instead of modeling the press-rate-to-full-speed-fraction mapping through the buffer model, compute the full-speed fraction **directly from the 20 Hz timeseries** and use it as the vigor variable.

### Vigor variable

```python
# For each trial, from the 20 Hz timeseries during transport:
f_i,t = (number of samples where press_rate >= req) / (total samples during transport)
```

### Survival function

Same exposure-time formulation:

```
S(f, T, D) = (1 - T) + T × exp(-λ × exposure(f, D))
```

where:

```
effective_speed(f) = v_full × (0.5 + 0.5 × f)
travel_remaining(f, D) = remaining_dist(D) / effective_speed(f)
exposure(f, D) = max(0, travel_remaining(f, D) - t_safe)
```

### Vigor equation

```
EU(f) = S(f, T, D) × R - (1-S(f, T, D)) × c_d,i × (R+C) - κ_i × g(f) × D
```

The effort cost g(f) links back to press rate through motor noise:

```
# To achieve full-speed fraction f, the mean press rate must be:
u(f) = req + σ_motor × Φ⁻¹(f)

# The effort cost is the excess effort above required:
g(f) = α × σ_motor² × [Φ⁻¹(f)]²
```

This preserves the quadratic effort cost in press rate while expressing it in terms of the full-speed fraction.

### Advantages
- Uses the 20 Hz data directly — no assumption about Gaussian motor noise for the vigor measurement
- The measured f is the actual fraction, not a model-derived approximation
- The exposure-time survival function operates on f directly

### Disadvantages
- f is computed from data, not predicted from parameters — model predicts optimal f*, comparison to observed f requires defining f from the timeseries
- The effort cost function g(f) requires the Φ⁻¹ mapping, which reintroduces the Gaussian assumption
- Less compatible with the Shadmehr framework (which uses movement speed/duration, not a threshold fraction)

---

## Option B: Buffer Model with Exposure-Time Survival (Recommended)

### Concept

Keep mean press rate as the vigor variable. Model survival through the exposure-time mechanism, connecting press rate to exposure through the chain: buffer → full-speed fraction → effective speed → travel time → exposure.

### Survival function

```
S(u, T, D) = (1 - T) + T × exp(-λ × exposure(u, D))
```

(Full specification in the section above.)

### Vigor equation

```
EU(u) = S(u, T, D) × R - (1-S(u, T, D)) × c_d,i × (R+C) - κ_i × α × (u - req)² × D
```

### First-order condition

```
dS/du × [R + c_d × (R+C)] = 2κ × α × (u - req) × D
```

With the exposure-time survival function, dS/du is substantial in the range where participants actually press. Higher c_d shifts the left curve up (marginal survival benefit), increasing optimal u*. Higher κ shifts the right curve up (marginal effort cost), decreasing optimal u*.

### Choice equation

Evaluate EU at the required press rate for each cookie:

```
S_H = (1-T) + T × exp(-λ × exposure_at_req(D_H))
S_L = (1-T) + T × exp(-λ × exposure_at_req(1))
```

At u = req, the buffer is 0, f = Φ(0) = 0.5, and effective speed = 0.75 × v_full. Exposure times:

```
exposure_H = remaining_dist(D_H) / (0.75 × v_full) - t_safe
exposure_L = remaining_dist(1) / (0.75 × v_full) - t_safe
```

Because D_H > D_L = 1, exposure_H > exposure_L, and S_H < S_L. The heavy cookie is more dangerous. c_d amplifies this survival difference in the choice equation.

### Advantages
- Keeps mean press rate as the vigor variable (Shadmehr-compatible)
- The exposure-time survival function is grounded in actual game mechanics
- The exponential hazard model has a single interpretable parameter (λ)
- c_d has leverage in both choice and vigor
- Effort cost stays quadratic in (u - req)

### Disadvantages
- Buffer → f mapping assumes Gaussian motor noise
- σ_motor is a population parameter that might vary across subjects
- The effective speed formula is a linear approximation of the discrete step function

---

## Comparison

| Feature | Option A | Option B |
|---------|----------|----------|
| Vigor variable | Full-speed fraction f | Mean press rate u |
| Survival mechanism | Exposure time from measured f | Exposure time from modeled f(u) |
| Uses 20 Hz data | Yes, directly | No (uses trial-mean press rate) |
| Effort cost | Needs Φ⁻¹ mapping | Standard quadratic (u-req)² |
| Compatibility with Shadmehr | Moderate | High |
| Key survival parameter | λ (hazard rate) | λ (hazard rate) + σ_motor (buffer noise) |

**Recommendation: Start with Option B.** Use Option A as a robustness check — compute f from the 20 Hz data and verify that Option B's predicted f matches the observed f. If they diverge, Option A provides ground truth.

---

## Implementation Order

### Phase 1: Empirical ground truth (Step 0)

Run all five empirical analyses (0a-0e). Results determine everything:
- Escape rates decrease with distance among full-speed trials → confirms exposure mechanism
- Exposure time predicts survival strongly → confirms hazard model
- Vigor-survival correlation strongest at D=3 → confirms distance-dependent gradient
- Escape rates uniformly near zero → task doesn't reward pressing, reframe c_d as arousal

### Phase 2: Compute exposure-time parameters from game mechanics

```python
# Known or measurable quantities:
v_full = ?          # full speed in game units/sec (from calibration)
D_game = {1: 5, 2: 7, 3: 9}  # game units per distance level
t_safe = 1.5        # seconds after encounter before strike possible

# For each distance, compute exposure at full vs half speed:
for D in [1, 2, 3]:
    remaining_dist = D_game[D]  # half of round trip
    
    # At full speed (f=1):
    travel_full = remaining_dist / v_full
    exposure_full = max(0, travel_full - t_safe)
    
    # At half speed (f=0):
    travel_half = remaining_dist / (0.5 * v_full)
    exposure_half = max(0, travel_half - t_safe)
    
    print(f"D={D}: exposure_full={exposure_full:.1f}s, "
          f"exposure_half={exposure_half:.1f}s, "
          f"exposure_range={exposure_half - exposure_full:.1f}s")
```

**Estimate λ from the data:**

```python
# From attack trials where exposure time can be computed or approximated:
# Fit: P(escape) = exp(-λ × exposure)
# Via logistic regression or MLE

# Alternatively, compute from the predator strike time distribution:
# If strikes follow Gaussian(mean, sd) starting at t_safe after encounter,
# the hazard rate is approximately:
#   λ(t) = φ((t - mean)/sd) / (1 - Φ((t - mean)/sd))
# For a constant-hazard approximation, use the average hazard over the 
# typical exposure window.
```

### Phase 3: Estimate σ_motor from 20 Hz data

```python
# For each subject, compute within-trial press rate variability:
for subject in subjects:
    trial_sds = []
    for trial in subject.trials:
        press_rates = trial.get_20Hz_press_rates()  # transport phase only
        trial_sds.append(np.std(press_rates))
    sigma_motor[subject] = np.mean(trial_sds)

# Use population mean as fixed parameter (simplest):
sigma_motor_pop = np.mean(list(sigma_motor.values()))
```

### Phase 4: Implement Option B survival function

```python
from scipy.stats import norm
import numpy as np

def compute_survival(u, T, D, req, sigma_motor, v_full, t_safe, lam):
    """
    Exposure-time survival function.
    """
    # Buffer → full-speed fraction
    buffer = u - req
    f = norm.cdf(buffer / sigma_motor)
    
    # Full-speed fraction → effective speed
    eff_speed = v_full * (0.5 + 0.5 * f)
    
    # Effective speed → remaining travel time
    D_game = {1: 5, 2: 7, 3: 9}[D]
    remaining_dist = D_game  # half of round trip
    travel_time = remaining_dist / eff_speed
    
    # Travel time → exposure
    exposure = max(0, travel_time - t_safe)
    
    # Exposure → escape probability (exponential hazard)
    p_escape = np.exp(-lam * exposure)
    
    # Full survival
    return (1 - T) + T * p_escape
```

### Phase 5: Precompute optimal vigor grid

```python
def vigor_EU(u, T, D, R, C, req, kappa, cd, alpha, 
             sigma_motor, v_full, t_safe, lam):
    s = compute_survival(u, T, D, req, sigma_motor, v_full, t_safe, lam)
    effort_cost = kappa * alpha * (u - req)**2 * D
    return s * R - (1 - s) * cd * (R + C) - effort_cost

# Precompute for parameter grid
kappa_grid = np.linspace(0.1, 5.0, 50)
cd_grid = np.linspace(1, 80, 50)

optimal_vigor = {}
for kappa in kappa_grid:
    for cd in cd_grid:
        for T in [0.1, 0.5, 0.9]:
            for D in [1, 2, 3]:
                for cookie in ['heavy', 'light']:
                    R = 5 if cookie == 'heavy' else 1
                    req_val = req_heavy if cookie == 'heavy' else req_light
                    
                    u_grid = np.arange(req_val, 3 * req_val, 0.001)
                    eu_vals = [vigor_EU(u, T, D, R, 5, req_val, kappa, cd,
                               alpha, sigma_motor, v_full, t_safe, lam) 
                               for u in u_grid]
                    u_star = u_grid[np.argmax(eu_vals)]
                    
                    optimal_vigor[(kappa, cd, T, D, cookie)] = u_star
```

### Phase 6: Fit the full model

NumPyro SVI as specified in the modeling specification document, using the exposure-time survival function from Phase 4. Per-subject κ_i and c_d,i, population-level τ and σ_v.

Parameters that can be fixed from earlier phases:
- λ from Phase 2 (or estimate jointly)
- σ_motor from Phase 3 (or estimate jointly)
- α from preliminary grid search
- t_safe = 1.5 (from game mechanics)
- v_full from game calibration

### Phase 7: Compare survival specifications

Fit with three survival functions:

| Specification | Function | Parameters |
|--------------|----------|------------|
| Original (flat sigmoid) | p_esc × sigmoid((u-req)/σ) | p_esc, σ |
| Exposure-time (Option B) | exp(-λ × exposure(u,D)) | λ, σ_motor |
| Direct f (Option A) | exp(-λ × exposure(f_measured,D)) | λ |

Report for each: BIC/WAIC, choice r², vigor r², parameter recovery.

### Phase 8: Generate interaction plots

With the exposure-time survival function:
1. Heatmap of u* across κ × c_d space — should show meaningful variation
2. EU curves for four behavioral profiles — should show clearly separated peaks
3. Cross-sections: u* vs κ at fixed c_d, u* vs c_d at fixed κ
4. Survival landscape: P_escape as a function of press rate at each distance

---

## What Success Looks Like

1. **Phase 1 confirms the exposure mechanism:** Escape rates decline with distance among full-speed trials. Exposure time strongly predicts survival. Vigor-survival correlation strongest at D=3.

2. **The exposure-time survival function gives c_d real leverage:** With P_escape at full speed in the 40-75% range, pressing harder meaningfully improves survival, and c_d amplifies this benefit.

3. **The κ × c_d interaction is visible:** The heatmap shows clear separation between profiles. Vigilant (low κ, high c_d) has substantially higher optimal u* than Disengaged (high κ, low c_d).

4. **c_d has leverage on choice:** The exposure difference between heavy cookies at D=3 and light cookies at D=1 translates to a meaningful survival difference, and c_d amplifies this in the choice equation.

5. **Model fit improves:** Vigor r² improves because the survival function matches actual game mechanics.

## What Failure Looks Like

Phase 1 shows exposure time doesn't predict survival — escape rates are uniformly low (~2-5%) regardless of speed tier, distance, or exposure. In this case:

1. The predator mechanics are too lethal — once an attack happens, capture is nearly certain
2. c_d cannot be identified through survival optimization
3. Reframe c_d as defensive motor arousal — captures threat-driven motor activation, not rational escape computation
4. The encounter dynamics (threat-independent, phasic, trait-stable) support the arousal interpretation
5. The paper shifts from "unified survival optimization" to "separable strategic and arousal-driven channels"

This is still a good paper — it tells a different but honest story about what c_d means.
