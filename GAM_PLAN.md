# GAM Analysis Plan for Vigor Dynamics

## The Approach: Spline Basis + Mixed Model in Python

A GAM is just a linear model where the predictor enters through a set of basis functions. We can implement this in Python by:

1. Creating spline basis columns using `patsy.dmatrix('cr(t_enc, df=K)')` (natural cubic regression splines with K degrees of freedom)
2. Adding these columns to the dataframe as regular predictors
3. Fitting with `statsmodels.mixedlm` which gives us random intercepts by subject

This is mathematically equivalent to `mgcv::gamm(rate ~ s(t_enc) + cookie + (1|subj))` in R.

## Data Structure

One row per trial per 200ms time bin:
- `rate`: normalized press rate (median IPI⁻¹ / calibMax)
- `t_enc`: time from encounter in seconds
- `cookie`: 0/1 (heavy/light)
- `threat`: 0.1/0.5/0.9
- `is_attack`: 0/1
- `subj`: participant ID

~40K rows after per-subject per-bin aggregation.

## Three GAM Models

### Model 1: Encounter Effect
```
rate ~ s(t_enc, K=15) + s(t_enc, K=15):is_attack + cookie + threat + (1|subj)
```

Implementation:
```python
from patsy import dmatrix

# Create spline basis for the main smooth
spline_base = dmatrix('cr(t_enc, df=15) - 1', df)  # 15 basis functions

# Create interaction spline basis: spline × is_attack
# These capture HOW the timecourse shape DIFFERS for attack vs non-attack
spline_interact = spline_base * df['is_attack'].values[:, None]

# Add all columns to dataframe
for j in range(15):
    df[f'spl_{j}'] = spline_base[:, j]           # main smooth
    df[f'spl_atk_{j}'] = spline_interact[:, j]   # interaction smooth

# Fit mixed model
formula = 'rate ~ ' + ' + '.join([f'spl_{j}' for j in range(15)]) + \
          ' + ' + ' + '.join([f'spl_atk_{j}' for j in range(15)]) + \
          ' + cookie + threat'
model = smf.mixedlm(formula, df, groups=df['subj']).fit()
```

The interaction spline coefficients tell you WHEN attack trials differ from non-attack trials. To test significance: compare full model (with interaction splines) to reduced model (without) via likelihood ratio test.

For pointwise inference: predict the smooth at a grid of t_enc values for attack=1 and attack=0, compute the difference, and get confidence bands from the model's covariance matrix.

### Model 2: Threat Effect
```
rate ~ s(t_enc, K=15) + s(t_enc, K=15):threat_high + cookie + is_attack + (1|subj)
```

Same approach but the interaction smooth captures how the timecourse shape differs between T=0.9 and T=0.1.

### Model 3: Parameter Moderation
```
rate ~ s(t_enc, K=15) + s(t_enc, K=15):cd_z + cookie + threat + is_attack + (1|subj)
```

The interaction smooth captures how the timecourse shape varies with cd. High cd subjects should show a different shape (larger encounter response, higher post-encounter level).

## Inference

### Global test: Does the interaction smooth matter?
- Likelihood ratio test: full model vs model without the interaction spline terms
- Chi-squared test with df = number of interaction basis functions

### Pointwise test: WHERE does it matter?
- Predict the smooth at each t_enc value for both levels of the factor
- Compute the difference ± 95% CI from the model covariance matrix
- Mark time regions where the CI excludes zero

### Smooth significance: Is the smooth itself non-linear?
- Compare model with s(t_enc) to model with just a linear t_enc term
- LRT tells you whether the smooth curvature is needed

## Practical Considerations

### Number of basis functions (K)
- K=15 for a 7-second window (200ms bins × 35 = 35 bins, K should be less than half)
- The smooth penalty isn't automatic in this approach (we're fitting unpenalized)
- Use K=10 as a robustness check — if results change substantially, K matters
- Could add a ridge penalty manually by penalizing large spline coefficients

### Penalty (smoothing parameter)
- statsmodels mixedlm doesn't have built-in smooth penalties
- Two options:
  a. Use unpenalized splines with modest K (K=10-15) — the finite basis constrains wiggliness
  b. Treat the spline coefficients as random effects: `s(t_enc)` becomes a random slope on the spline basis, which IS a penalized smooth (Wand 2003, "Smoothing and mixed models")
- Option (a) is simpler and sufficient for publication if K is well-chosen

### Random effects structure
- Minimum: `(1|subj)` — random intercept by subject
- Better: `(1 + is_attack|subj)` — random intercept and random encounter effect
- Best: random slopes on each spline basis function (= subject-specific smooth shapes). But this is 15+ random effects per subject and won't converge with mixedlm.
- Practical: use `(1|subj)` and accept that the model captures population-average smooth shapes with subject-level intercept adjustment.

### Computational feasibility
- 40K rows × 15 spline columns + 15 interaction columns = 30 fixed effects + 1 random effect
- statsmodels mixedlm can handle this in ~30 seconds
- No GPU needed, no JAX, just numpy/scipy linear algebra

## Output

### Tables
1. Full model fixed effects: all spline coefficients, cookie, threat, LRT for interaction smooth
2. Model comparison: LRT chi-squared and p-value for each interaction smooth

### Figures
1. Fitted smooth for attack vs non-attack with 95% CI (controlling for cookie)
2. Difference smooth (encounter effect) with pointwise 95% CI
3. Fitted smooth for T=0.1 vs T=0.9 (controlling for cookie and attack)
4. Difference smooth (threat effect) with pointwise CI
5. cd moderation: fitted smooth at high cd (75th percentile) vs low cd (25th percentile)

## Why This Is Better Than What We Had

- Proper random effects (not ignored)
- Smooth function estimated from data (not arbitrary epoch boundaries)
- Cookie controlled as a covariate (not separate analyses)
- Pointwise inference with proper CIs (not mass univariate with permutation)
- Standard methodology reviewers will recognize
- Implementable entirely in Python (no R dependency)

## R Fallback

If the Python implementation has convergence issues, the R equivalent is:

```r
library(mgcv)
m <- gamm(rate ~ s(t_enc, k=15) + s(t_enc, by=is_attack, k=15) + cookie + threat,
          random = list(subj = ~1), data = df)
```

This has automatic penalty estimation (REML), proper smooth-by-factor interactions, and is the gold standard. But the Python version should produce equivalent results for the simple models we need.
