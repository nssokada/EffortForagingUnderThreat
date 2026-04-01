# Theoretical Framework: Foraging Under Predation Risk

A reference document covering the key models from optimal foraging theory, predator-prey decision-making, motor control, and threat imminence that ground our approach.

---

## 1. The Fundamental Problem

An animal foraging under predation risk must balance energy acquisition against survival. More foraging effort yields more food but increases exposure to predators. The organism must decide:

1. **Where/what to forage** — which patches to visit, which prey to pursue
2. **How much effort to invest** — how fast to move, how vigilant to be, how hard to work
3. **When to flee** — at what point does the threat warrant abandoning the foraging bout

These decisions are interlinked: the choice of patch determines the effort required, the effort invested affects survival probability, and survival probability feeds back into patch valuation.

---

## 2. Bednekoff (2007) — Foraging in the Face of Danger

**Source:** Bednekoff, P. A. (2007). Foraging in the face of danger. In Stephens, Brown, & Ydenberg (Eds.), *Foraging: Behavior and Ecology* (pp. 305–329). University of Chicago Press.

### The Life-History Framework

Bednekoff formalizes the foraging-danger tradeoff using a fitness equation that unifies the costs (death) and benefits (food) of foraging:

**W(u) = S(u) · V(u)**

where:
- **u** = foraging effort (fraction of maximum, 0 to 1)
- **S(u)** = survival probability = exp[−k · u^z]
- **V(u)** = future reproductive value gained from foraging = κu
- **k** = mortality constant (how dangerous the environment is)
- **z** = mortality exponent (shape of the effort-danger tradeoff)

### Optimal Foraging Effort

Differentiating W(u) and solving:

**u* = 1 / (k·z)^(1/z)**

Key results:
- Optimal effort **decreases** as danger constant k increases
- The decline is **less steep** as mortality exponent z increases
- When the starvation minimum R exceeds u*, the animal is forced to **maximize survival** — life-history converges on survival-maximization
- The **shape** of the mortality function (z) is often more important than its absolute level (k)

### The Growth Scenario (No Time Limit)

For accumulating a fixed amount of resources K:

**u* = z·R / (z − 1)**

Critical insight: **optimal effort does not depend on k** (absolute danger). Only the shape z matters. This produces Gilliam's M/g rule — minimize mortality per unit gain.

### Danger Changes Over Time (Risk Allocation)

When danger fluctuates between high (α₁) and low (α₂) periods, the optimal feeding rates satisfy:

**α₁ · u₁ = α₂ · u₂**

The ratio of feeding rates is the **inverse** of the ratio of attack rates. Animals change behavior **more** in response to *variations* in danger than to *average* rates. This is Lima & Bednekoff's (1999) risk allocation hypothesis.

### Automatic State Dependence

As an animal accumulates resources, its future reproductive value (what it stands to lose from death) **increases**. This creates an inherent tendency toward greater caution with greater success — a natural state-dependent shift in the effort-danger tradeoff.

### Relevance to Our Task

Our foragers face exactly this problem. They choose between patches (cookies) that differ in reward and exposure, then execute with variable effort (pressing). The mortality function S(u) maps onto the survival probability given pressing rate and predator dynamics. Individual differences in k (perceived danger) and z (tradeoff shape) would produce the observed variation in both choice and vigor.

**The critical question Bednekoff's framework raises:** Is foraging effort u a single variable that governs both patch selection and motor vigor? Or does it decompose into separate dimensions?

---

## 3. Lima & Dill (1990) — Behavioral Decisions Under Predation Risk

**Source:** Lima, S. L. & Dill, L. M. (1990). Behavioral decisions made under the risk of predation: a review and prospectus. *Canadian Journal of Zoology*, 68, 619–640.

### Decomposing Predation Risk

P(death) = 1 − exp(−α · d · T)

where:
- **α** = encounter rate between predator and prey
- **d** = probability of death given an encounter
- **T** = time spent vulnerable

Each component is assessable and partially under behavioral control:
- Reduce **T** by spending less time exposed
- Reduce **α** by avoiding predator-dense areas
- Reduce **d** by feeding close to cover (improving escape probability)

### The Decision Hierarchy

Decisions organized large-to-small:
1. **When to feed** — temporal windows of higher/lower risk
2. **Where to feed** — habitat/patch selection trading profitability against safety
3. **What to eat** — diet selection considering handling time vs vigilance
4. **How to handle food** — handling time under behavioral control

### Key Principle

The foraging-danger tradeoff is NOT a constraint. The behavioral options lie on a continuum from pure energy maximization to pure risk minimization. The animal chooses optimally within this continuum.

### Relevance

Our task operationalizes decisions 2 (which cookie = where to forage) and the motor dimension (how hard to press = how to handle food / how vigilant to be). Lima & Dill's framework predicts that both are under behavioral control and trade off against each other.

---

## 4. Brown (1988, 1999) — Giving-Up Density and Vigilance

**Source:** Brown, J. S. (1988). Patch use as an indicator of habitat preference, predation risk, and competition. *Behavioral Ecology and Sociobiology*, 22, 37–47. Brown, J. S. (1999). Vigilance, patch use and habitat selection. *Evolutionary Ecology Research*, 1, 49–71.

### The GUD Equation

At the optimal quitting point in a depleting patch:

**H = C + P + MOC**

where:
- **H** = harvest rate at the giving-up density
- **C** = metabolic cost of foraging
- **P** = predation cost of foraging (= μ × α × F)
- **MOC** = missed opportunity cost

### The Predation Cost P

P = μ · α(u) · F

where:
- **μ** = predator encounter rate
- **α(u)** = lethality, decreasing in vigilance u
- **F** = fitness value of the animal (what it stands to lose)

### Joint Optimization of Vigilance and Patch Use (Brown 1999)

The animal simultaneously optimizes:
1. **Vigilance level u*** — how much time to scan vs forage
2. **Patch residency time** — when to leave (determines GUD)

under a single fitness objective. The first-order condition for optimal vigilance:

marginal value of vigilance for safety = marginal cost of vigilance in lost harvest

**This is the closest existing precedent to our two-variable optimization (patch choice × motor vigor).**

### Relevance

Our λ parameter maps onto P — the foraging cost of predation that drives patch disengagement. Our ω parameter maps onto α(u) — the lethality that vigilance (pressing hard) reduces. Brown's framework predicts joint optimization; our data show decomposition.

---

## 5. Houston, McNamara & Hutchinson (1993) — The Energy-Predation Tradeoff

**Source:** Houston, A. I., McNamara, J. M., & Hutchinson, J. M. C. (1993). General results concerning the trade-off between gaining energy and avoiding predation. *Phil. Trans. R. Soc. B*, 341, 375–397.

### The General Framework

An animal chooses behavior u, producing:
- **γ(u)** = energy gain rate
- **μ(u)** = mortality rate

The set of achievable (γ, μ) pairs forms a **tradeoff curve**. Options with higher γ tend to have higher μ.

### The Fitness Objective

Maximize: **V'(x) · γ(u) − V(x) · μ(u)**

where V'(x) = marginal value of energy, V(x) = current reproductive value.

### Special Cases

- **Growing to a threshold:** Minimize μ/γ (Gilliam's M/g rule)
- **Surviving to a fixed time:** Risk-spreading theorem — constant behavior

### Relevance

This provides the most general formulation. Our foragers are on a tradeoff curve in (reward, mortality) space. λ determines where they operate on this curve for patch selection; ω determines where they operate for vigor. The Houston framework predicts a single operating point; our data show two independent positions — one for choice, one for vigor.

---

## 6. Lima & Bednekoff (1999) — Risk Allocation Hypothesis

**Source:** Lima, S. L. & Bednekoff, P. A. (1999). Temporal variation in danger drives antipredatory behavior: the predation risk allocation hypothesis. *American Naturalist*, 153, 649–659.

### The Model

Time alternates between high-danger (d_H) and low-danger (d_L) periods. The animal allocates feeding effort f_H (during danger) and f_L (during safety) to:
- Maximize survival: S = exp(−(d_H · f_H · p_H + d_L · f_L · p_L) · T)
- Subject to meeting a foraging threshold: f_H · p_H + f_L · p_L ≥ F*/T

### Key Predictions

- Allocate MORE feeding to low-danger periods, MORE antipredator behavior to high-danger periods
- When high-danger periods are rare and brief → strongest antipredator response during them
- When high-danger periods are chronic → maintain feeding even during danger

### Type of Optimization

**Joint** — the animal simultaneously optimizes f_H and f_L under a single objective. This is a constrained optimization (Lagrangian).

### Relevance

Our threat levels (T=0.1, 0.5, 0.9) are analogous to their danger periods. The risk allocation prediction: people should shift effort away from high-T trials and toward low-T trials. This is exactly what we observe in choice (avoid at T=0.9, engage at T=0.1). The vigor increase at high T is the "maintained feeding during chronic danger" pattern — you can't completely disengage, so you compensate with motor mobilization.

---

## 7. Ydenberg & Dill (1986) — Economics of Fleeing

**Source:** Ydenberg, R. C. & Dill, L. M. (1986). The economics of fleeing from predators. *Advances in the Study of Behavior*, 16, 229–249.

### The Model

Flight initiation distance d* occurs where:

**C_flee(d) = C_stay(d)**

- **C_flee(d)** = cost of fleeing (lost foraging, energy of flight) — decreasing in distance from predator
- **C_stay(d)** = cost of not fleeing (risk of capture) — increasing as predator approaches

### Relevance

Our ω maps onto C_stay — the cost of not fleeing (being captured). Higher ω → flee sooner / press harder to escape. This framework treats escape as a single decision (when to flee), not a continuous effort variable. Our task extends it by making escape effort continuous (pressing rate) rather than binary (flee/stay).

---

## 8. Shadmehr & Krakauer (2008) / Yoon et al. (2018) — Vigor and Foraging

**Source:** Shadmehr, R. & Krakauer, J. W. (2008). A computational neuroanatomy for motor control. *Experimental Brain Research*, 185, 359–381. Yoon, T., et al. (2018). Control of movement vigor and decision making during foraging. *PNAS*, 115, E10476–E10485.

### The Vigor Framework

Movement vigor (speed) is chosen to maximize the global capture rate:

**J̄ = Σ[f(t_h) − u_m(d, t_m)] / Σ[t_h + t_m]**

Two control variables:
1. **t_h** — how long to stay at each patch (decision making)
2. **t_m** — how fast to move between patches (motor control)

### Yoon et al. (2018) Key Result

Both harvest duration and movement vigor are derived from a SINGLE normative principle (maximize global capture rate). The brain compares local utility with its history. This is the closest existing model to jointly optimizing choice and motor output in a foraging context.

### The Motor Cost

Movement effort follows a quadratic cost:

**cost(u) = c · u² · D**

where c = metabolic cost coefficient, u = press rate, D = distance. Standard in optimal feedback control (Todorov & Jordan 2002).

### Relevance

Our vigor equation directly implements this framework. The quadratic motor cost (u−req)²×D is the Shadmehr/Todorov cost. ω enters as the survival incentive that drives vigor up despite the motor cost. Yoon et al.'s joint optimization of harvest + vigor is the precedent for our attempt to unify choice and vigor (M6) — though our data show the decomposition (M5) fits better than the joint optimization.

---

## 9. Thura et al. (2025) — Integrated Control of Decision and Movement Vigor

**Source:** Thura, D., et al. (2025). The integrated control of decision and movement vigor. *Trends in Cognitive Sciences*, 29, 1146–1157.

### Co-Regulation vs Decoupling

Decision vigor (speed of deciding) and movement vigor (speed of executing) are:
- **Co-regulated by default** — both rise and fall together, even when not beneficial
- **Decoupled when needed** — at a cost to accuracy

### Two Mechanisms

1. **SNR modulation** → co-regulation (default): enhanced signal-to-noise in sensorimotor areas speeds both deciding and moving
2. **Inhibitory control** → decoupling (flexible override): frontal/basal ganglia circuits suppress one while releasing the other

### Relevance

Our λ-ω independence may reflect the **decoupled** mode. Choice (decision vigor → which cookie) and pressing (movement vigor → how hard) are not co-regulated in our data. The decoupling may be driven by the task structure — choice commitment is irrevocable, so the motor system cannot influence the decision system once committed. The weak positive correlation in raw threat shifts (r=0.16, which disappears after conditioning on threat) may be the residual co-regulation signal.

---

## 10. Fanselow (1994) / Mobbs et al. (2018, 2020) — Threat Imminence

**Source:** Fanselow, D. M. (1994). Neural organization of the defensive behavior system. *Psychonomic Bulletin & Review*, 1, 429–438. Mobbs, D., et al. (2018). Surviving threats. *Nature Reviews Neuroscience*, 19, 562. Mobbs, D., et al. (2020). Space, time, and fear. *Trends in Cognitive Sciences*, 24, 731.

### The Predatory Imminence Continuum

Three defensive modes, each with distinct behavior and neural substrates:

| Mode | Trigger | Behavior | Neural basis |
|------|---------|----------|-------------|
| **Pre-encounter** | Environment where predation is possible | Cautious exploration, meal reorganization, strategic avoidance | Prefrontal, hippocampus, amygdala |
| **Post-encounter** | Predator detected | Freezing, escape preparation | Amygdala → ventral PAG |
| **Circa-strike** | Imminent contact | Vigorous flight/fight | Dorsolateral PAG (forebrain inhibited) |

### Key Features

- **Not a single optimization** — different neural circuits dominate at different distances
- Model-based (distal) → model-free (proximal) transition
- Forebrain circuits are **inhibited** during circa-strike (the reactive phase actively suppresses deliberation)
- No formal mathematical model exists for the transitions

### Relevance

Our task captures the pre-encounter → post-encounter transition. Pre-encounter: λ-driven, probability-dependent, strategic avoidance (prefrontal). Post-encounter: ω-driven, probability-independent, motor activation (subcortical). The decomposition of λ and ω into separable channels maps onto the neural architecture of the defense cascade. The failure of the unified model (M6) is consistent with different circuits controlling pre-encounter and post-encounter behavior.

---

## Summary: How These Models Relate to Our Approach

| Model | What it predicts | Our test | Our finding |
|-------|-----------------|----------|-------------|
| Bednekoff (2007) | Single u* optimizes W = S·V | Does one parameter govern both channels? | No (M3 fails) |
| Brown (1999) | Joint optimization of vigilance + patch use | Does joint V computation improve choice? | No (M6 fails, ΔBIC +6,278) |
| Houston et al. (1993) | Single tradeoff curve in (γ, μ) space | Do choice and vigor land on the same curve? | No (independent after conditioning on T) |
| Lima & Bednekoff (1999) | Allocate effort across danger levels | Do people shift both channels with threat? | Yes, but independently (r→0 after residualization) |
| Yoon & Shadmehr (2018) | Joint optimization of harvest + vigor | Does the unified model beat the decomposed? | No (M5 >> M6) |
| Thura et al. (2025) | Co-regulation by default, decoupling possible | Are choice and vigor co-regulated? | Decoupled (independent λ, ω) |
| Fanselow/Mobbs | Different circuits for pre- vs post-encounter | Do the channels map onto the imminence gradient? | Yes (λ pre-encounter, ω post-encounter) |

**The synthesis:** Normative foraging theory predicts unified optimization over patch selection and motor vigor. Humans decompose this into parallel channels — avoidance (λ) and activation (ω) — that share an environmental threat input but are independently parameterized at the individual-difference level. This decomposition maps onto the threat imminence continuum, with λ governing strategic pre-encounter avoidance and ω governing reactive post-encounter motor mobilization.
