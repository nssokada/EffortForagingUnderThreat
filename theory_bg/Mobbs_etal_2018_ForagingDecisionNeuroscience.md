# Foraging for Foundations in Decision Neuroscience: Insights from Ethology

**Mobbs, D., Trimmer, P. C., Blumstein, D. T., & Dayan, P. (2018). Foraging for foundations in decision neuroscience: insights from ethology. *Nature Reviews Neuroscience*, 19, 419–427.**

---

## Central Thesis

Decision neuroscience has built powerful computational accounts of human choice using tightly controlled laboratory paradigms, but these paradigms often lack ecological grounding. Ethology and behavioral ecology offer the missing **functional context** — the survival problems brains evolved to solve. This paper advocates integrating ethological frameworks (particularly foraging and escape) with neuroscientific methods to produce biologically realistic accounts of decision-making.

---

## The Frameworks: Huxley, Tinbergen, Mayr, and Marr

The paper maps multiple explanatory frameworks onto one another:

### Tinbergen's Four Questions
1. **Mechanism (causation)**: What neural/behavioral mechanisms support decisions?
2. **Ontogeny (development)**: How do decision processes change over a lifetime?
3. **Function (adaptation)**: Why does the animal make the decisions it makes?
4. **Phylogeny (evolution)**: How have decision mechanisms changed over evolutionary time?

### Mayr's Two Classes
- **Proximate questions**: mechanism + ontogeny (how)
- **Ultimate questions**: function + phylogeny (why)

### Marr's Levels of Analysis
- **Computational level**: What problem is being solved? (parallels function/adaptation)
- **Algorithmic level**: What representations and processes solve it?
- **Implementational level**: How is it physically realized in neural hardware?

The argument: decision neuroscience has largely operated within Marr's framework (especially algorithmic and implementational levels) but has insufficiently addressed the computational level in its ethological sense — what problems did the brain actually evolve to solve?

---

## The Critique of Standard Decision Neuroscience

Standard paradigms include:
- Bandit problems (explore vs. exploit)
- Random dot kinematograms (perceptual decisions)
- Two-step tasks (model-based vs. model-free learning)
- Trust games (social decisions)
- Intertemporal choice (temporal discounting)

These have been productive but:
- They use a **limited set of task structures** that may not be ethologically representative
- They rarely consider the survival problems that shaped decision mechanisms
- They ignore **state dependence** (most subjects cannot run out of resources)
- Completely optimal behavior is uncommon — approximations (heuristics) must be tailored to common natural environments
- The **boundaries** between different decision regimes are poorly understood

---

## Foraging Theory Applied to Neuroscience

### Net Rate Maximization

The simplest foraging principle: maximize the difference between energy benefits and costs per unit time. Elements include energy content, opportunity costs, handling time, and travel costs.

**Neural correlate**: The **dorsal anterior cingulate cortex (dACC)** appears involved in cost–benefit analysis of effortful situations and in energizing behavior. Damage to dACC impairs motivation; electrical stimulation elicits perseverance and vigor.

### The Marginal Value Theorem (MVT)

In depleting environments, the optimal strategy balances current diminishing returns against the opportunity cost of traveling to a fresh patch.

**Neuroscience application**: Hayden, Pearson & Platt (2011) recorded from monkey dACC during a patch-foraging task. As patches depleted, dACC neurons gradually increased firing until a threshold triggered patch-leaving. The signal rose more slowly when travel time was longer — showing integration of multiple control signals.

Key finding: monkeys made **near-optimal decisions** in foraging contexts, with fewer temporal biases than in standard intertemporal choice tasks. The foraging context may reduce biases.

**Generalization**: The MVT framework has been extended to:
- Human attention allocation (Pirolli & Card 1999)
- Macaque social information foraging (Turrin et al. 2017)
- Human semantic memory recall (Hills, Jones & Todd 2012)

### Exploration vs. Exploitation

Kolling et al. (2012) designed a human task making the trade-off between searching for alternatives (exploring) and engaging with current options (exploiting) explicit. Results:
- **dACC** encoded both the average value of the foraging environment and the costs of foraging
- **vmPFC** (ventromedial prefrontal cortex) encoded well-defined economic values of options
- Proposed division: dACC for searching/switching, vmPFC for staying/exploiting

### State Dependence

Despite its ubiquity in natural foraging, state dependence is largely ignored in decision neuroscience. Most experiments don't allow subjects to face genuine resource constraints. When reserves are tight and homeostatic challenges are critical, the **variance** of outcomes becomes important, and animals become more risk-seeking. The brain may process money acquisition similarly to food acquisition (primary reinforcers influencing endogenous abilities).

---

## Competitive Foraging

### Game Theory and the Ideal Free Distribution (IFD)

When payoffs depend on others' actions, game theory applies. The IFD predicts that foragers distribute themselves in proportion to food availability relative to competition density.

**Harper's duck experiment**: Mallards distributed across two bread-dropping sites roughly matched the IFD prediction, but despotic (dominant) individuals received unequal shares.

**Neural implementation** (Mobbs et al. 2013): fMRI during competitive foraging showed:
- **dACC, supplementary motor area, insula**: activation associated with the drive to switch patches
- **Striatum, medial prefrontal cortex**: activation when staying in an advantageous habitat
- **Amygdala**: activity predicted individual differences in competition avoidance

---

## Foraging Under Predation Risk

### Common Currency

To trade off different benefits (foraging gain vs. predation avoidance), a **common currency** is needed. Behavioral ecology uses **reproductive value** — expected future reproductive success under different strategies. The strategy maximizing reproductive value is the normative expectation under natural selection.

Implication: if an individual is near starvation, reproductive value approaches zero without food, so decision processes should take relatively **less account of predation risk** when reserves are low.

### Escape Theory

Ydenberg & Dill (1986) modeled escape as an economic decision: the **flight initiation distance (FID)** is where the cost of not fleeing (rising with predator proximity) crosses the cost of fleeing (declining with proximity because less flight is needed).

Prey are remarkably adept at calibrating FID based on:
- Prior experience with the predator
- Predator's relative position, bearing, lethality, and velocity
- Value of the current location (food, mates)

### Neural Basis of Escape

**Threat imminence and brain regions**:

| Threat Distance | Dominant Brain Region | Proposed Function |
|---|---|---|
| Distant threat | vmPFC | Strategic assessment, cognitive fear |
| Approaching threat | Hippocampus, posterior cingulate | Contextual evaluation, planning |
| Imminent threat | Midbrain PAG, midcingulate cortex | Reactive fear, fast motor response |

Key finding (Mobbs et al. 2007): As a virtual predator approached, brain activity shifted from vmPFC to midbrain periaqueductal grey (PAG). This replicates across studies and connects to:
- Panic-related motor errors correlating with PAG/dorsal raphe activity
- Amygdala–ACC–vmPFC connectivity during predator evasion
- Hippocampal activation during threat engagement (when not under time pressure)

**Bayesian escape optimization** (Qi et al. 2018): Applied Bayesian decision theory to FID task. The midcingulate cortex tracked optimal escape from fast-attacking predators; hippocampus tracked optimal escape from slow-attacking threats.

---

## Future Directions

### For Decision Neuroscience
- Understanding **trade-offs across tasks** — how a mechanism adapted for multiple tasks coordinates function
- Virtual ecologies for studying adaptive and affective decision-making
- Applications to psychiatric disorders (gambling linked to suboptimal explore–exploit, depression linked to foraging deficits, effort impairments in mood disorders)
- Alignment with RDoC (Research Domain Criteria) framework

### Reciprocal Benefits for Ethology
- Understanding brain mechanisms helps identify what conditions mechanisms evolved for
- Decision-theoretic complexity and heuristics provide constraints for ethological models
- McNamara & Houston's "evo-mecho" approach: function and mechanism co-evolve and must be understood together

---

## Key Takeaways

1. Decision neuroscience risks isolating itself by losing touch with the natural problems brains evolved to solve
2. **Foraging theory** (MVT, IFD, optimal diet) provides ethologically grounded paradigms that can be adapted for neuroscience
3. The **dACC** emerges as a key region across foraging contexts — encoding effort costs, environment value, and switching signals
4. **State dependence** is ubiquitous in nature but largely absent from neuroscience paradigms
5. **Escape theory** connects behavioral ecology directly to neural survival circuits, revealing a gradient from cognitive (vmPFC, hippocampus) to reactive (PAG) processing as threat proximity increases
6. **Reproductive value** provides the common currency for trading off foraging gains against predation costs
7. The integration is bidirectional: neuroscience can ground ethological theory in mechanism, while ethology can ground neuroscience in function
