# Confidence, Metacognition, and the Control of Behavior: A Theoretical Overview

**Context:** Kumaran, Daw, Osindero, Veličković & Patraucean (2026). *Causal Evidence that Language Models use Confidence to Drive Behavior.* arXiv:2603.22161v1.

---

## The Central Problem

Across species and now artificial systems, agents face a recurring problem: having made a decision, should they commit to it or withhold? This is a **meta-decision** — a decision about a decision — and it requires an internal signal about the quality of the primary decision. That signal is **confidence**: an estimate of the probability that a choice is correct. The theoretical question is not just whether such signals exist, but whether they are actually *used* to control behavior.

---

## The Two-Stage Architecture of Metacognitive Control

The dominant framework, formalized by Fleming & Daw (2017) and Kepecs & Mainen (2012), decomposes metacognitive control into two separable stages.

### Stage 1: Confidence Formation

The system generates a primary decision and, as part of or immediately following that process, produces a confidence signal. There is a core theoretical debate about where this signal originates:

**First-order accounts** (Kepecs & Mainen, 2012; Kiani & Shadlen, 2009) hold that confidence is a direct byproduct of the evidence accumulation process that drives the primary decision. If the evidence strongly favors one option, confidence is high; if the evidence is ambiguous, confidence is low. No separate monitoring mechanism is required — confidence is simply the strength or quality of the decision variable at the moment of commitment.

**Second-order (higher-order) accounts** (Fleming & Daw, 2017) propose that a distinct metacognitive system reads out and evaluates the quality of the first-order process. This allows dissociations between decision accuracy and confidence: an agent can be correct but uncertain, or incorrect but confident. The second-order monitor may have access to noisy or degraded copies of first-order signals, introducing systematic biases (overconfidence, underconfidence) even when the primary decision is optimal.

Both accounts predict that confidence should correlate with accuracy. They diverge on whether confidence can be manipulated independently of the primary decision, and on whether lesions or interventions can selectively impair metacognitive monitoring while leaving primary performance intact.

### Stage 2: Threshold-Based Action Selection

The confidence signal is compared against a criterion (threshold T) to produce a binary meta-decision — act or withhold, answer or abstain, wager high or low. The mapping from continuous confidence to binary action follows a sigmoid decision function:

$$P(\text{abstain}) = \sigma\left(\frac{T - C}{\tau}\right)$$

Where C is the confidence signal, T is the threshold (the indifference point at which the agent is equally likely to act or withhold), and τ is the policy temperature (governing the sharpness of the transition). A low τ produces a near-deterministic step function; a high τ produces a gradual, probabilistic transition.

This is structurally identical to signal detection theory's framework for perceptual decisions, but operating one level up — applied to the agent's own internal signals about decision quality rather than to external stimuli.

---

## The Evidence Accumulation Foundation

The deeper grounding comes from sequential sampling models of decision-making (Gold & Shadlen, 2007). In these models, decisions arise from the noisy accumulation of evidence toward a bound. Confidence emerges naturally as a function of the state of the accumulator at the time of commitment:

- **Strong, rapid evidence** → high confidence
- **Weak, slow, or conflicting evidence** → low confidence
- **Evidence arriving after the commitment** can update confidence post-decisionally

Pouget, Drugowitsch & Kepecs (2016) drew an important distinction between two quantities that are often conflated:

- **Confidence:** The probability that a *specific decision* is correct. This is action-oriented — it guides meta-decisions about whether to commit, wager, or seek more information.
- **Certainty:** The precision of a *probabilistic belief* about the state of the world. This is belief-oriented — it guides learning and belief updating.

These serve different computational purposes and may be implemented by partially distinct neural mechanisms. The abstention paradigm specifically engages confidence in Pouget et al.'s sense: a judgment about a particular choice that determines whether to act on it.

---

## Key Empirical Precedents from Neuroscience

### Neural Correlates of Confidence

**Kepecs et al. (2008)** — Neurons in rat orbitofrontal cortex (OFC) encoded decision confidence during an odor categorization task. After the rat committed to a choice but before feedback, OFC firing rates predicted whether the choice was correct or incorrect. This provided the first clear neural correlate of a confidence signal that tracked decision accuracy in real time.

**Kiani & Shadlen (2009)** — Neurons in monkey lateral intraparietal area (LIP) encoded confidence during perceptual decisions. The state of neural activity at the moment of commitment predicted the animal's subsequent confidence wagers. The signal was well-described by a drift-diffusion model, consistent with first-order accounts where confidence is a readout of evidence accumulation.

### Behavioral Use of Confidence

**Foote & Crystal (2007)** — Rats in a "decline-to-test" paradigm selectively declined difficult perceptual discrimination trials where they were likely to err, choosing a smaller but guaranteed reward instead. This demonstrated that rats monitored their own uncertainty and used it to regulate behavior — the hallmark of metacognitive control.

**Lak et al. (2014)** — Orbitofrontal cortex was *causally required* for optimal waiting based on decision confidence. Rats with OFC lesions could still make primary decisions normally, but they could no longer use confidence to decide whether to wait for a reward-contingent outcome or abort and start fresh. This double dissociation — intact primary decisions, impaired confidence-based meta-decisions — provided causal evidence for the two-stage separation between confidence formation and confidence use.

### The Post-Decision Wagering Paradigm

The experimental paradigm used by Kumaran et al. is a direct descendant of post-decision wagering paradigms in psychophysics and neuroscience. The common structure is:

1. **Primary decision:** Categorize a stimulus, answer a question, choose between options.
2. **Meta-decision:** Wager on the correctness of the primary decision, choose to wait for a performance-contingent reward vs. initiating a new trial, or answer vs. abstain.

The critical feature is that the meta-decision involves a tradeoff between the potential reward for a correct primary decision and the cost of committing to an incorrect one. This tradeoff can only be navigated adaptively if the agent has access to a graded internal signal about primary decision quality — i.e., confidence.

---

## Confidence in Language Models: From Extraction to Use

### The Prior State of the Field

A substantial body of work has demonstrated that confidence-like signals can be extracted from LLM outputs:

- **Logit-based confidence:** The probability assigned to the chosen token (or answer option) in the next-token distribution provides a natural confidence measure (Guo et al., 2017).
- **Verbalized confidence:** Models can be prompted to state how confident they are, though verbal reports are often markedly overconfident and poorly calibrated (Steyvers et al., 2025; Griot et al., 2025).
- **Calibration:** Post-hoc methods like temperature scaling can align logit-based confidence with empirical accuracy, such that a model reporting 70% confidence is correct approximately 70% of the time (Guo et al., 2017; Xiong et al., 2023).
- **Self-knowledge:** Kadavath et al. (2022) showed that language models "mostly know what they know" — their confidence estimates discriminate between questions they can and cannot answer.

However, extraction and calibration are observer-side operations. They demonstrate that confidence information *exists* in the model's outputs, but not that the model *uses* it to guide its own behavior. The distinction is analogous to showing that a brain region encodes confidence (a correlational finding) versus showing that lesioning or stimulating that region changes confidence-guided behavior (a causal finding).

### Engineered vs. Native Abstention

Prior approaches to LLM abstention have been primarily engineered:

- **Post-hoc thresholding:** Extract confidence, then apply an externally determined threshold to decide when the model should abstain (Yadkori et al., 2024; Plaut et al., 2024; Tomani et al., 2024).
- **Supervised fine-tuning:** Train models with explicit abstention labels derived from external evaluations of answer quality (Tjandra et al., 2024; Chuang et al., 2024; Zhang et al., 2024).
- **Refusal direction:** Arditi et al. (2024) identified a single direction in activation space that mediates refusal behavior, but this concerned safety-related refusal rather than uncertainty-based abstention.

These approaches do not examine whether models natively deploy confidence to guide behavior. They leave open whether models are engaging in genuine metacognitive control or simply pattern-matching to surface features correlated with training supervision.

---

## The Kumaran et al. Contribution: Causal Evidence

### Experimental Design Mapped to Theory

The four-phase paradigm maps precisely onto the two-stage framework:

| Phase | Theoretical Target | What It Tests |
|-------|-------------------|---------------|
| Phase 1 | Stage 1 isolation | Elicits confidence without any meta-decision context — a "pure" measure uncontaminated by abstention |
| Phase 2 | Stage 2 revealed | Models freely decide whether to answer or abstain, revealing their implicit threshold and policy temperature |
| Phase 3 | Stage 1 intervention | Activation steering directly manipulates confidence representations, testing whether confidence *causes* abstention changes |
| Phase 4 | Stage 2 intervention | Instructed thresholds manipulate the decision policy, testing whether the policy can be controlled while leaving confidence intact |

### The Key Causal Findings

**Phase 3 (activation steering)** provides the strongest causal evidence. By injecting high-confidence or low-confidence activation patterns into the model's residual stream, the authors showed that artificially boosting confidence decreased abstention rates, while suppressing confidence increased them. Mediation analysis revealed that 67% of the steering effect operated through **confidence redistribution** (reallocating probability mass from the abstention option toward answer options) and 26% through **policy shifts** (altering the mapping from confidence to action). This confirms that steering operates primarily at Stage 1 while partially affecting Stage 2.

**Phase 4 (instructed thresholds)** provides complementary evidence at Stage 2. When models were told to abstain below a specified confidence level, abstention rates increased systematically with threshold — but Phase 1 confidence retained its predictive power across all threshold conditions. This demonstrates that threshold instructions alter the decision policy without fundamentally distorting the underlying confidence representations, confirming the operational independence of the two stages.

### The Dominance of Confidence Over Alternatives

A particularly important finding is that confidence was the **dominant predictor** of abstention, with standardized effect sizes approximately 10× larger than alternative mechanisms:

- **Question difficulty** (aggregate accuracy across model runs) — captures how objectively hard a question is, independently of any single trial's confidence
- **RAG scores** (retrieval-augmented generation similarity) — captures how accessible relevant knowledge is from the training corpus
- **Sentence embeddings** (distributional semantic features) — captures surface-level patterns in question structure, topic, and domain

This rules out the possibility that abstention is driven by pattern-matching on question characteristics rather than genuine metacognitive assessment.

---

## Cross-Model Variation and Invariant Architecture

Despite substantial variation in abstention behavior across models (baseline abstention rates ranged from 27% to 82%; confidence-to-threshold weighting ranged from 0.66 to 1.80), the **two-stage architecture was preserved in every model tested**. Confidence robustly predicted abstention in GPT-4o, Gemma 3 27B, DeepSeek 671B, and Qwen 80B.

The variation occurred in the *parameters* of the architecture:

- **Shift (baseline bias):** How cautious the model is by default, independently of confidence or threshold. Ranged from near-zero (DeepSeek) to strongly conservative (GPT-4o at −97.6%).
- **Scale (confidence weighting):** How heavily the model weights its own confidence relative to instructed thresholds. Values above 1.0 mean the model trusts itself more than the instruction; below 1.0 means it defers to the instruction.
- **Policy temperature:** How sharply the model transitions between answering and abstaining around the indifference point. Lower values produce crisper boundaries.

This dissociation — variation in parameters, invariance in structure — suggests that while training procedures shape the specifics of abstention policy, the use of confidence to guide meta-decisions is a convergent computational solution.

---

## Broader Theoretical Implications

### Convergent Solutions Across Substrates

The strongest theoretical claim is that the computational demands of metacognitive control impose similar architectures across biological and artificial systems. If any system must decide when to commit to an uncertain answer versus withhold it, and if it has access to graded internal signals about decision quality, a threshold-based policy over those signals is a natural — perhaps inevitable — solution. The two-stage structure may not be an accident of biology replicated in transformers; it may be the efficient solution to the problem.

### Metacognitive Control vs. Introspective Access

The paper draws an important distinction between two capacities:

- **Metacognitive control:** Using internal evaluations of decision quality to regulate behavior. This is what the paper demonstrates — models deploy confidence to decide when to abstain.
- **Introspective access:** The ability to accurately report on one's own internal states. Anthropic's recent work on introspection (2025) has explored this capacity in frontier models.

These may be dissociable. Verbal confidence reports from LLMs are often poorly calibrated and overconfident (Steyvers et al., 2025), yet the same models adaptively deploy internal (logit-based) confidence signals to guide behavior. This parallels findings in human metacognition where behavioral signatures of confidence-based control can be intact even when verbal confidence judgments are systematically biased.

### Implications for AI Safety and Autonomy

The practical significance is direct: as models transition from passive assistants to autonomous agents, the capacity to recognize their own uncertainty becomes critical. A model that can reliably abstain when uncertain — rather than confidently hallucinating — is safer in high-stakes domains (medicine, law, engineering). The finding that this capacity operates natively, through intrinsic confidence dynamics rather than externally imposed guardrails, suggests it may be more robust and generalizable than engineered approaches, but also that it may have failure modes (e.g., the conservatism or overconfidence biases observed in different models) that need to be understood and managed.

---

## Key References from the Theoretical Landscape

| Reference | Contribution |
|-----------|-------------|
| Fleming & Daw (2017) | Bayesian framework for metacognitive computation; formalizes two-stage architecture with separable confidence formation and action selection |
| Kepecs & Mainen (2012) | Computational framework for confidence in humans and animals; links confidence to evidence accumulation |
| Kepecs et al. (2008) | Neural correlates of decision confidence in rat OFC; firing rates predict correctness before feedback |
| Kiani & Shadlen (2009) | Confidence representation in monkey LIP; decision-variable readout predicts post-decision wagers |
| Pouget, Drugowitsch & Kepecs (2016) | Distinguishes confidence (action-oriented, specific to a decision) from certainty (belief-oriented, about world states) |
| Gold & Shadlen (2007) | Evidence accumulation framework for decision-making; foundation for first-order confidence accounts |
| Lak et al. (2014) | Causal evidence that OFC is required for confidence-based waiting; dissociates primary decision from meta-decision |
| Foote & Crystal (2007) | Metacognitive uncertainty monitoring in rats via decline-to-test paradigm |
| Guo et al. (2017) | Temperature scaling for calibrating neural network confidence; standard method used across the LLM calibration literature |
| Kadavath et al. (2022) | LLMs "mostly know what they know" — logit-based confidence discriminates answerable from unanswerable questions |
| Steyvers et al. (2025) | Comparison of human and LLM uncertainty communication; documents overconfidence in verbal LLM reports |
| Arditi et al. (2024) | Refusal in LLMs mediated by a single activation direction; related but distinct from uncertainty-based abstention |
| Kumaran et al. (2025) | Companion paper on confidence-driven change of mind in LLMs; shows overconfidence in initial choices and underconfidence under criticism |
