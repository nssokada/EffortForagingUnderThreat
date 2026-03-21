# Vigor × Model Parameters: Analysis Design

**Question:** How do computational model parameters (z, κ, β) relate to the
structure of defensive action vigor across trial phases?

---

## The Problem in Concrete Terms

We have two multivariate datasets at the **subject level** (N = 293):

**X — Model parameters (p = 3)**
| Param | Meaning | z–κ correlation |
|-------|---------|-----------------|
| z | Hazard sensitivity (distance → danger) | r = 0.18 |
| κ | Effort discounting | |
| β | Residual threat bias | z–β: r = 0.06, κ–β: r = 0.14 |

Params are largely orthogonal. β is right-skewed (log-transformed for regression).

**Y — Vigor features (q = 7, residualized stream)**
Computed as subject means over trials from `phase_vigor_metrics.parquet`:

| Feature | Phase window | Captures |
|---------|-------------|----------|
| `onset_slope_resid` | 0–2 s, slope | Rate of pre-encounter ramp-up |
| `onset_mean_resid` | 0–2 s, mean | Overall anticipatory drive level |
| `enc_pre_mean_resid` | encounter−1 s, mean | Baseline just before encounter |
| `enc_post_mean_resid` | encounter+1 s, mean | Post-encounter vigor level |
| `enc_spike_resid` | post − pre | Encounter-triggered reactive increment |
| `term_mean_resid` | resolution−2 s, mean | Terminal persistence magnitude |
| `term_slope_resid` | resolution−2 s, slope | Terminal persistence acceleration |

"Resid" = after subtracting demand curve (effort-level × time mean), so these
capture *vigor above and beyond what the task mechanically requires*.

**Observed correlation structure (current 4-dim summary vs. params):**

|  | z | κ | β |
|--|--|--|--|
| tonic_vigor | +0.07 | **−0.21** | +0.03 |
| anticipatory_mobilization | +0.10 | −0.11 | **+0.14** |
| reactive_spike | +0.02 | −0.01 | +0.05 |
| terminal_persistence | −0.02 | +0.00 | −0.03 |

Key observations:
- **κ** drives the strongest effects and affects *both* tonic and anticipatory (global suppression)
- **β** selectively affects anticipatory mobilization
- **Reactive spike and terminal persistence have near-zero correlations with all params**
- Correlations are individually weak (r ≈ 0.10–0.21) — any approach needs to handle this

**The core problem with the current approach:**
The 4 summary dimensions were defined a priori by aggregating trial-level
features with ad hoc rules. Then each was independently regressed on all 3
params (4 × 3 = 12 tests). This:
1. Ignores the correlation between vigor features within phases
2. Has no principled criterion for which features to aggregate
3. Creates an implicit multiple-testing problem even after FDR correction
4. Doesn't characterize the *shared* multivariate structure between the two sets

---

## Candidate Approaches

---

### A. PCA on Vigor Features → Correlate PCs with Params (two-step)

**Procedure:**
1. PCA on the 7 subject-level vigor feature means → extract 2–3 PCs
2. Regress each PC on z, κ, β simultaneously

**What it gives you:**
Uncorrelated latent axes of vigor variation. PC1 will likely reflect overall
vigor level (positive loadings across all phases), PC2 will likely contrast
anticipatory vs. reactive, etc. You then ask: which param drives which PC?

**Pros:**
- Simple and interpretable; easily visualized as biplots
- PCA is unsupervised — decomposition doesn't "know" the params, so
  finding that PCs align with params afterward is genuine evidence of structure
- Well-understood in the field; reviewers won't object to the method
- The PC loadings can be reported as a supplementary figure

**Cons:**
- **Fundamentally two-step and inefficient.** PCA finds axes of maximum
  variance in vigor space, not maximum covariance with params. The
  param-relevant signal could be entirely in a low-variance PC
- If PC1 is "everyone presses harder" (global level), it will dominate the
  decomposition regardless of whether it's what z/κ/β map onto
- No built-in significance test for the structure
- Rotation problem: oblique vs. orthogonal, and which rotation?

**Fit to our data:**
The vigor feature inter-correlations are moderate (tonic × anticipatory r ≈ 0.40;
reactive spike is orthogonal to everything). PCA would likely find:
- PC1: general vigor level (tonic + anticipatory + terminal, r ≈ 0.40+)
- PC2: reactive spike (orthogonal to everything)
- PC3: terminal slope vs. onset slope contrast
PC1 would correlate with κ (r ≈ −0.20), which is the main finding. But this
just repackages what we already know.

**Verdict:** Adequate but not optimal. Best used as a descriptive/visualization
tool in figures, not as the primary statistical test.

---

### B. Canonical Correlation Analysis (CCA)

**Procedure:**
Find linear combinations of vigor features (U = Ya) and model params (V = Xb)
that maximize Corr(U, V). Extract multiple canonical variate pairs.

**What it gives you:**
The *shared* latent structure between the two sets, simultaneously reducing
both. The first canonical pair explains the most covariance between X and Y.

**Pros:**
- Directly answers: "what is the shared structure between vigor and params?"
  — this is precisely the scientific question
- Symmetric: treats vigor features and params as co-equal sets, not
  predictor/outcome
- The canonical loadings tell you which vigor features and which params
  load on each shared dimension
- With permutation testing, canonical correlations have valid significance tests

**Cons:**
- **Severely underpowered / prone to overfitting** with our data: N = 293,
  q = 7 vigor features, p = 3 params. Classical CCA needs N >> p + q.
  With only p = 3 params, we have at most 3 canonical variates — and the
  first will already inflate correlation by exploiting chance structure
- Standard CCA has no regularization. With weak true effects (r ≈ 0.1–0.2),
  the canonical correlations will be optimistically biased
- Difficult to report compactly in a manuscript: loadings, weights, and
  structure coefficients need careful interpretation
- Significance of individual canonical variates (after the first) is hard to test

**Regularized variant:** RCCA (regularized CCA) with cross-validated
shrinkage parameters (e.g., the `rcca` package or scikit-learn's CCA with
ridge penalty) addresses overfitting. Still complex to report.

**Fit to our data:**
Given p = 3 params and q = 7 vigor features, classical CCA will find at most
3 canonical variate pairs. With effect sizes r ≈ 0.10–0.21, canonical
correlations will be substantially inflated without regularization. Permutation
testing of the full CCA is feasible but computationally expensive.

**Verdict:** Conceptually correct but implementation risk is high. If using CCA,
must use regularized CCA with cross-validation, which complicates reporting.
Good for exploratory analysis; hard to present cleanly in a top-tier paper.

---

### C. LASSO / Elastic Net Regression

**Procedure:**
For each model parameter (z, κ, β) separately, regress it on all 7 vigor
features with L1 (LASSO) or L1 + L2 (elastic net) penalty. Cross-validate
the penalty hyperparameter. Identify which vigor features survive shrinkage.

**What it gives you:**
A sparse subset of vigor features that predict each param. Addresses
multiple-testing by automatically zeroing out irrelevant predictors.

**Pros:**
- Principled solution to the variable selection problem
- Sparse solutions are easy to interpret and report ("vigor features X and Y
  predict κ; none reliably predict β")
- Well-understood by reviewers in clinical/systems neuroscience
- Can run separately for each param without correction burden (the regularization
  handles it)
- Cross-validated R² gives a natural out-of-sample fit metric

**Cons:**
- **Directional**: treats vigor as predictor and params as outcomes (or vice
  versa). Neither direction is obviously correct — params and vigor were
  measured independently
- Running 3 separate regressions (one per param) doesn't characterize the
  *joint* structure between the two sets
- LASSO is unstable with correlated predictors (tonic × anticipatory r ≈ 0.40).
  Elastic net is better but introduces a second hyperparameter
- Effect sizes are small (r ≈ 0.10–0.21). With N = 293, LASSO may zero out
  everything or select near-arbitrarily between correlated features
- No natural uncertainty quantification on which features are selected

**Fit to our data:**
The directional ambiguity is real: it makes more causal sense to say params
→ vigor (params are computed first from choice data; vigor is an outcome). But
the regression direction doesn't change the correlations. The sparsity is
probably too aggressive given effect sizes: with r ≈ 0.1, LASSO on N = 293
may find nothing. Elastic net would be more sensitive.

**Verdict:** Useful as a robustness/sensitivity check. Not the primary analysis
because it's directional, single-DV, and may not recover anything with these
effect sizes.

---

### D. Partial Least Squares (PLS) Regression

**Procedure:**
Decompose X (params) and Y (vigor features) simultaneously into latent
components that maximize *covariance* (not correlation) between the two sets.
Use cross-validation to select number of components.

**What it gives you:**
Ordered latent dimensions of the X–Y relationship. PLS component 1 captures
the largest shared covariance; component 2 the next largest orthogonal component.
Each component has loadings on both the param set and the vigor feature set.

**Pros:**
- Directly designed for the case where N is modest relative to p + q — more
  stable than CCA
- Maximizes covariance (not just correlation), so it's sensitive to the
  magnitude of the shared signal, not just its direction
- Unlike PCA, it finds the param-relevant axes of vigor variation — the
  first PLS component will capture what κ drives in vigor space, even if
  that's not the largest axis of raw vigor variance
- Cross-validated prediction (Q² = LOO or k-fold) provides honest estimate
  of generalizability; permutation test of components is standard
- Natural to report: loading plots (X scores vs. Y scores) are visually
  intuitive for a paper figure
- Well-established in neuroimaging (similar problem structure: neural
  features × behavioral variables)

**Cons:**
- Less familiar than PCA or LASSO in cognitive/computational psych literature;
  reviewers from a modeling background may be less comfortable with it
- PLS maximizes covariance, not correlation — the first component can be
  dominated by a high-variance parameter or feature regardless of
  theoretical relevance
- With p = 3 params, at most 3 PLS components are possible, and with
  small effect sizes the permutation tests for component 2+ may be underpowered

**Fit to our data:**
PLS is arguably the most natural tool here. With p = 3 params and q = 7 vigor
features, we expect 1–2 meaningful components:
- **Component 1**: likely dominated by κ → global vigor suppression
- **Component 2**: likely z/β → anticipatory mobilization specifically
- The reactive spike will load near zero on both components, cleanly
  showing it's not part of the shared structure

**Verdict:** The strongest candidate for the primary analysis. Handles the
feature selection problem, characterizes the joint structure, is appropriate
for our N, and produces interpretable figures. Permutation testing is
straightforward.

---

### E. Bayesian Multivariate Regression with Shrinkage Priors

**Procedure:**
Specify a multivariate regression: Y (7 vigor features) ~ X (3 params) with
a shared shrinkage prior on the 7 × 3 coefficient matrix. Options:
- **Horseshoe prior**: aggressive shrinkage toward zero with heavy tails to
  preserve large effects; automatic relevance determination over the
  coefficient matrix
- **Low-rank factorized prior**: coefficient matrix Γ = Λ₁ Λ₂ᵀ (rank-r
  decomposition), directly testing whether the relationship operates through
  a low-dimensional latent structure (equivalent to RRR in a Bayesian setting)
- **Factor regression**: model vigor features as arising from shared latent
  factors, some of which are predicted by params

**What it gives you:**
Full posterior over the coefficient matrix, posterior inclusion probabilities
for each (param, vigor feature) pair, uncertainty-quantified loadings, and
natural model comparison (Bayes factors, LOO-CV) between low-rank and
full-rank models.

**Pros:**
- Principled uncertainty quantification — especially important when effect
  sizes are small; posterior credible intervals will honestly reflect the
  weakness of evidence
- The horseshoe prior's adaptive shrinkage is better calibrated than LASSO
  for recovering weak signals: it neither shrinks everything to zero (as
  LASSO might) nor inflates weak effects (as OLS does)
- Low-rank Bayesian regression directly tests the hypothesis that the
  vigor–param relationship is low-dimensional
- Bayes factors for the shared component structure are interpretable in terms
  of evidence, not just p-values — more defensible for a paper claiming a
  cross-domain result

**Cons:**
- **Substantially harder to implement** than the other options. NumPyro/Stan
  required; non-trivial to specify and validate the model
- Harder to present to reviewers: "we fit a Bayesian low-rank regression
  with horseshoe prior" requires more explanation
- Posterior summaries (95% HDI, posterior inclusion probabilities) may not
  communicate the finding as intuitively as a PLS biplot
- With N = 293 and effect sizes r ≈ 0.1–0.2, even a well-specified Bayesian
  model will return wide credible intervals — the posterior might just say
  "we're uncertain" rather than giving a clean story

**Fit to our data:**
Most principled approach, but the effect sizes in our data are genuinely small.
The posterior on the coefficient matrix will be broad. This is honest, but
may produce a message like "κ is probably negatively associated with tonic
vigor, but we cannot distinguish its effect on anticipatory from reactive
phases" — which doesn't support the selective-effects narrative cleanly.

**Verdict:** Best for an honest characterization of uncertainty, and worth
doing as a robustness check. Not the best primary vehicle for the narrative
result, because broad posteriors don't communicate selective effects cleanly.
Would be a strong supplementary analysis ("Bayesian analysis confirms PLS
result with correct uncertainty quantification").

---

## Recommendation

### Primary analysis: PLS Regression with permutation test
- Decomposes the shared vigor–param structure into 1–3 interpretable components
- Appropriate for N = 293, p = 3, q = 7
- Cross-validated Q² gives honest generalizability estimate
- Loading biplot is a clean figure showing which vigor features and which
  params co-load on each component
- Permutation test (shuffle param labels, recompute PLS) gives valid inference
- Component 1 likely recovers the κ → global suppression finding cleanly;
  Component 2 likely recovers z/β → anticipatory mobilization

### Secondary analysis: Bayesian multivariate regression (horseshoe or low-rank)
- Confirms PLS result with proper uncertainty quantification
- Reports posterior inclusion probabilities per (param, feature) cell
- Particularly important for reactive spike / terminal persistence: the
  posterior will honestly show near-zero effects with appropriate uncertainty
- Can be reported as supplement or in the methods as robustness check

### Descriptive: PCA biplots for figures
- Use PCA for visualization — intuitive for readers
- Run it alongside PLS; if PC1 aligns with PLS component 1, this confirms
  the result is robust to method

### Not recommended as primary: CCA (overfitting risk), LASSO (too directional
and aggressive for effect sizes this small)

---

## What the Result Should Show

Given the observed correlations, the expected PLS result is:
- **Component 1** (explains most shared covariance): κ loads negatively;
  tonic and anticipatory vigor features load negatively. Interpretation:
  *effort discounting globally suppresses pre-encounter motor drive.*
- **Component 2** (orthogonal to component 1): z and β load positively;
  onset_slope and enc_pre_mean_resid load positively; enc_spike and terminal
  features near zero. Interpretation: *hazard sensitivity and threat bias
  selectively mobilize anticipatory vigor but not reactive escape.*
- **Reactive spike and terminal persistence**: near-zero loadings on both
  components. This is the key double dissociation — reactive and terminal
  vigor are not part of the deliberative system captured by choice params.

This result is interpretable as a functional architecture of threat response:
deliberative parameters (computed offline from foraging choices) govern the
*anticipatory* phase of motor behavior; the *reactive* phase is governed by
something else (automatic, encounter-triggered).

---

## Resolved Decisions

1. **Unit of analysis → subject-level (N = 293)** using subject-mean vigor
   features. Avoids pseudo-replication. Within-subject dynamics are a
   separate analysis (see Analysis 2 below).

2. **Vigor stream → residualized (`_resid`)** throughout. Removes demand-curve
   variance so features capture individual motor drive above task requirements.

3. **Component retention → permutation test** (≥ 5,000 shuffles of param
   labels). Stop retaining components when permuted p > 0.05.

4. **Reactive / terminal loadings**: explicitly test loadings vs. zero using
   bootstrap 95% CI. Report as an interpretable null (deliberative params do
   not predict automatic reactive/terminal phases) rather than an absence.

5. **Trial-level survival → vigor** (Analysis 2, separate notebook section):
   compute trial-specific S_i = exp(−T_i · D_i^{z_i}) using fitted z_i per
   subject, merge with phase_vigor_metrics, run LMM. The two analyses answer
   different questions and both are needed for the paper:
   - **PLS** (Analysis 1): trait-level — do people with higher κ chronically
     produce less vigor? Characterizes stable individual differences.
   - **LMM** (Analysis 2): state-level — when *this trial* has lower computed
     survival, does that person mobilize more vigor? Characterizes within-subject
     dynamic modulation and the z × S interaction the draft claims.

---

## Analysis Plan

### Analysis 1 — PLS at Subject Level
**Notebook:** `notebooks/03_vigor_analysis/10_pls_vigor_params.ipynb`

Inputs:
- `phase_vigor_metrics.parquet` → aggregate to subject-level means (7 resid features)
- `results/stats/FET_Exp_Bias_{z,k,beta}_params.csv` → posterior means per subject

Steps:
1. Compute subject-mean vigor features (7 × `_resid` columns)
2. Z-score both X (params, p=3) and Y (vigor features, q=7)
3. Fit PLS with 1–3 components; cross-validate with LOO Q²
4. Permutation test per component (5,000 shuffles of param rows)
5. Report: X loadings (params), Y loadings (vigor features), component scores biplot
6. Bootstrap 95% CI on all loadings; flag reactive/terminal loadings vs. zero
7. Sensitivity: re-run with norm stream, compare component structure

### Analysis 2 — Trial-Level Survival → Vigor LMM
**Notebook:** same notebook, separate section

Inputs:
- `phase_vigor_metrics.parquet` (trial-level vigor)
- Subject params merged in (z_i, κ_i, β_i)
- Trial features: T (threat), D (distance from selected option)

Steps:
1. Compute trial-level S_i = exp(−T_i · D_i^{z_i}) for each trial
2. Z-score S_i within the dataset
3. For each vigor phase (onset_slope, enc_spike, term_mean) run:
   `vigor_phase ~ S_i + threat_c + choice + startDistance + (1|subj)`
4. Test z_i × S_i interaction (allows for subject-specific sensitivity of
   vigor to within-trial survival variation)
5. Compare β estimates to draft's claimed β = −0.23 for model-derived survival
