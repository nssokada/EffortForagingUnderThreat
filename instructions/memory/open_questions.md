# Open Questions

Unresolved theoretical, methodological, and empirical questions that need decisions or further analysis.

---

## Theoretical

### What does β (threat bias) capture computationally?
- β is the residual threat sensitivity beyond what the survival-weighted EV accounts for
- Right-skewed distribution (β̄=1.44, SD=1.89, range [0.20, 13.58]) — is this a genuine individual difference or noise?
- Correlates with anticipatory mobilization (vigor) — consistent with a "preparatory arousal" interpretation
- Open: does β relate to psychiatric measures (anxiety, STAI-T) more than z does?

### Does the paper's "common computational structure" claim hold for affect?
- Choice ✅, Vigor ✅, Affect ✅ — all show S_probe/model-param links (NB12 complete)
- z → chronic confidence deficit (β=−0.199, p_fdr=0.013); p_threat → phasic anxiety/confidence (β=±0.575/0.586)
- **NEW NUANCE:** κ also predicts trait affect (anxiety + confidence). z×threat moderation NULL.
- **Cross-domain null:** vigor × affect r all <0.13, none FDR-significant → parallel but independent reactive systems
- **Resolution:** Two-system framing (deliberative vs. reactive) is the right story. NOT a simple "common substrate."

### Tonic-phasic tradeoff interpretation
- r = -0.36 to -0.48 between tonic vigor and reactive spike on attack trials
- Reactive surge is also larger in low-threat patches (ceiling effect confirmed in NB11)
- Could be ceiling/floor effect, resource allocation, or compensatory strategy
- Open: does this tradeoff vary by threat level or model parameters?

---

## Methodological

### Factor analysis of psychiatric battery
- DASS-21, PHQ-9, OASIS, STAI, AMI, MFIS, STICSA collected but not yet factor-analyzed
- NB07 (clinical prediction) is blocked on this
- Key question: how many factors? Expected: negative affect, apathy/fatigue, anxiety specificity
- Should we do EFA first, then CFA on confirmatory sample?

### Should we run full 7-model MCMC comparison on N=293?
- WAIC table in paper currently from N=270 GPU fit
- For the paper: need WAIC comparison on the same dataset as the winning model
- Computational cost: ~6 hrs per model × 7 models = ~42 hrs total (or parallelized)

### Confirmatory sample (N=350)
- Not yet preprocessed or analyzed
- Will need: full preprocessing pipeline, then refit FETExponentialBias, rerun all vigor analyses
- Should confirmatory analysis be pre-registered?

---

## Pipeline

### Parameter recovery — does FETExponentialBias recover well?
- `02_parameter_recovery.ipynb` has not been run against N=293 fit
- Important for paper: shows model identifiability and parameter independence

### What's the right time window for "terminal vigor"?
- Currently: last 2s before trial resolution (escape or capture)
- Alternative: fixed window before encounter; or dynamic window from encounter to resolution
- Open: sensitivity analysis on window size

---

## Statistical

### Multiple comparison correction strategy for vigor → psych regressions (NB04 psych)
- Currently using FDR (Benjamini-Hochberg) within each DV set
- Alternative: Bonferroni, or family-wise correction across all tests
- Decision needed before writing up NB04 results

### Should z-scored parameters be used throughout or raw?
- NB08/09 currently use z-scored (z_z, kappa_z, beta_z) for mixed models
- Raw params used in NB06 scatter plots
- Standardization helps interpretability of β coefficients across params with different scales
