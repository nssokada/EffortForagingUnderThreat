# Statistics Verification Report — draft009/paper.md

Generated: 2026-03-28

Legend: Match = the paper value agrees with the source file; Mismatch = the paper value disagrees.

---

## 1. BIC = 32,133

- **Paper:** 32,133
- **Source (evc_model_comparison_final.csv, FINAL row):** 32,133.4
- **Verdict:** MATCH (rounded from 32,133.433)

## 2. Choice r^2 = 0.951

- **Paper:** 0.951
- **Source (evc_model_comparison_final.csv):** 0.951; **(evc_final_ppc.csv overall):** 0.951 (implicit from subj_choice_r = 0.9757, but the model comparison file directly gives 0.951)
- **Verdict:** MATCH

## 3. Vigor r^2 = 0.511

- **Paper:** 0.511
- **Source (evc_model_comparison_final.csv):** 0.511; **(evc_final_ppc.csv overall):** vigor_r2 = 0.5209
- **Note:** The model comparison file says 0.511; the PPC file says 0.521. Paper uses 0.511, consistent with the model comparison file. The PPC file's vigor_r2 = 0.521 is slightly different (possibly computed differently). Paper is consistent with model_comparison source.
- **Verdict:** MATCH (against model_comparison); note PPC file says 0.521

### Per-subject vigor r

- **Paper (abstract):** "press-rate vigor (r^2 = 0.511)"
- **Paper (Results, line 69):** "subj vigor r = 0.829"
- **Source (evc_final_ppc.csv):** subj_vigor_r = 0.836
- **Verdict:** MISMATCH -- Paper says 0.829; source says 0.836

## 4. Population parameters: gamma = 0.209, epsilon = 0.098, tau = 0.476

- **Paper:** gamma = 0.209, epsilon = 0.098, tau = 0.476
- **Source (oc_evc_final_81_population.csv):** gamma = 0.2093, epsilon = 0.0976, tau = 0.4755
- **Verdict:** MATCH (all round correctly to stated values)

## 5. Choice accuracy = 79.3%, AUC = 0.876

- **Paper:** accuracy = 79.3%, AUC = 0.876
- **Source (evc_final_ppc.csv):** choice_accuracy = 0.7933, choice_auc = 0.8764
- **Verdict:** MATCH

## 6. LMM anxiety beta = -0.557, t = -14.04

- **Paper (abstract):** beta = -0.557, t = -14.0
- **Paper (Results, line 103):** beta = -0.557, SE = 0.040, t = -14.04, p = 8.8 x 10^-45, N_obs = 5,274
- **Source (evc_final_affect.csv):** beta = -0.5567, SE = 0.0396, t = -14.041, p = 8.77e-45, n_obs = 5274
- **Verdict:** MATCH (all values agree to stated precision)

## 7. LMM confidence beta = 0.575, t = 13.48

- **Paper (abstract):** beta = 0.575, t = 13.5
- **Paper (Results, line 104):** beta = +0.575, SE = 0.043, t = +13.48, p = 2.1 x 10^-41, N_obs = 5,272
- **Source (evc_final_affect.csv):** beta = 0.5749, SE = 0.0426, t = 13.480, p = 2.06e-41, n_obs = 5272
- **Verdict:** MATCH

## 8. Calibration x choice quality r = 0.230

- **Paper (line 120):** r = 0.230, p < .001
- **Source (evc_final_dissociation.csv):** calibration -> choice_quality: r = 0.2297, p = 0.0001
- **Verdict:** MATCH (rounds to 0.230)

## 9. Calibration x survival r = 0.185

- **Paper (line 120):** r = 0.185, p = .002
- **Source (evc_final_dissociation.csv):** calibration -> survival_rate: r = 0.1787, p = 0.0023
- **Verdict:** MISMATCH -- Paper says 0.185; source says 0.179 (rounds to 0.179, not 0.185)

## 10. Table 2: Bayesian regression (discrepancy predicts symptoms)

Source: evc_bayesian_clinical.csv, model = full_metacog, predictor = discrepancy

| Measure | Paper beta | Source beta | Paper HDI | Source HDI | Match? |
|---------|-----------|-------------|-----------|------------|--------|
| STAI-State | 0.338 | 0.338 | [0.22, 0.45] | [0.23, 0.45] | MISMATCH (HDI low: paper 0.22, source 0.23) |
| STICSA | 0.285 | 0.285 | [0.17, 0.40] | [0.17, 0.39] | MISMATCH (HDI high: paper 0.40, source 0.39) |
| DASS-Anxiety | 0.275 | 0.275 | [0.16, 0.39] | [0.16, 0.40] | MISMATCH (HDI high: paper 0.39, source 0.40) |
| DASS-Stress | 0.217 | 0.255 | [0.10, 0.33] | [0.13, 0.37] | MISMATCH (beta: paper 0.217, source 0.255; HDI also differs) |
| DASS-Depression | 0.206 | 0.228 | [0.09, 0.32] | [0.11, 0.35] | MISMATCH (beta: paper 0.206, source 0.228; HDI also differs) |
| PHQ-9 | 0.212 | 0.225 | [0.10, 0.33] | [0.11, 0.34] | MISMATCH (beta: paper 0.212, source 0.225; HDI also differs) |
| OASIS | 0.180 | 0.228 | [0.07, 0.30] | [0.12, 0.35] | MISMATCH (beta: paper 0.180, source 0.228; HDI also differs) |
| AMI-Emotional | -0.222 | -0.222 | [-0.34, -0.11] | [-0.34, -0.11] | MATCH |

### Table 2: log(ce) % in ROPE

| Measure | Paper | Source | Match? |
|---------|-------|--------|--------|
| STAI-State | 77% | 91.4% | MISMATCH |
| STICSA | 82% | 79.4% | MISMATCH |
| DASS-Anxiety | 75% | 60.2% | MISMATCH |
| DASS-Stress | 80% | 62.0% | MISMATCH |
| DASS-Depression | 78% | 78.4% | MATCH (rounds to 78%) |
| PHQ-9 | 76% | 71.5% | MISMATCH |
| OASIS | 78% | 55.6% | MISMATCH |
| AMI-Emotional | 73% | 53.1% | MISMATCH |

### Table 2: log(cd) % in ROPE

| Measure | Paper | Source | Match? |
|---------|-------|--------|--------|
| STAI-State | 93% | 92.8% | MATCH |
| STICSA | 78% | 86.8% | MISMATCH |
| DASS-Anxiety | 90% | 90.4% | MATCH |
| DASS-Stress | 91% | 91.1% | MATCH |
| DASS-Depression | 82% | 80.8% | MATCH (rounds to 81%) |
| PHQ-9 | 80% | 88.8% | MISMATCH |
| OASIS | 75% | 77.2% | MISMATCH |
| AMI-Emotional | 65% | 54.2% | MISMATCH |

**Summary for Table 2:** Multiple mismatches. The STAI-State and AMI-Emotional discrepancy betas match. The DASS-Stress, DASS-Depression, PHQ-9, and OASIS discrepancy betas are wrong. Most ROPE percentages for log(ce) are wrong. Several log(cd) ROPE values are also wrong.

## 11. ROPE containment 65--93% (general claim in text)

- **Paper (line 139, 174):** "ROPE containment 65--93%"
- **Source:** log(ce) ROPE ranges from 53.1% to 91.4%; log(cd) ROPE ranges from 54.2% to 92.8%
- **Verdict:** MISMATCH -- The lower bound is ~53%, not 65%. The claimed range of 65--93% is too narrow; the actual range is approximately 53--93%.

## 12. Orthogonality r = 0.019

- **Paper:** r = 0.019, p = .75
- **Source (evc_final_dissociation.csv):** calibration -> discrepancy: r = 0.0190, p = 0.7476
- **Verdict:** MATCH

## 13. Parameter recovery: ce r = 0.92, cd r = 0.94

- **Paper:** ce r = 0.92, cd r = 0.94
- **Source (computed from evc_final_recovery.csv):** ce r = 0.916, cd r = 0.943
- **Verdict:** MATCH (rounds correctly)

## 14. Parameter independence: r = -0.14

- **Paper:** r = -0.14
- **Source (computed from oc_evc_final_params.csv):** r = -0.135
- **Verdict:** MATCH (rounds to -0.14)

## 15. Model comparison table (all 7 BIC values)

| Model | Paper BIC | Source BIC | Paper delta | Source delta | Match? |
|-------|----------|------------|-------------|-------------|--------|
| EVC 2+2 | 32,133 | 32,133.4 | --- | 0.0 | MATCH |
| M1 (Effort only) | 50,792 | 50,792.0 | +18,659 | +18,658.6 | MATCH |
| M2 (Threat only) | 34,227 | 42,767.0 | +2,094 | +10,633.6 | MISMATCH (paper BIC 34,227; source 42,767; delta paper +2,094; source +10,634) |
| M3 (Separate) | 42,563 | 42,526.0 | +10,430 | +10,392.6 | MISMATCH (paper BIC 42,563; source 42,526; delta paper +10,430; source +10,393) |
| M4 (Pop ce) | 30,860 | 30,860.0 | -1,274 | -1,273.4 | MATCH |
| M5 (No gamma) | 34,204 | 34,204.0 | +2,071 | +2,070.6 | MATCH |
| M6 (Standard u^2) | 31,991 | 31,991.0 | -142 | -142.4 | MATCH |

### M2 values mismatch details:
- **Paper:** M2 BIC = 34,227, delta = +2,094, choice_r2 = 0.006, vigor_r2 = 0.513
- **Source:** M2 BIC = 42,767, delta = +10,634, choice_r2 = 0.006, vigor_r2 = 0.294
- Vigor r2 also mismatches: paper says 0.513, source says 0.294
- choice_r2 matches (0.006)

### M3 values mismatch details:
- **Paper:** M3 BIC = 42,563, delta = +10,430, choice_r2 = 0.955, vigor_r2 = 0.441
- **Source:** M3 BIC = 42,526, delta = +10,393, choice_r2 = 0.955, vigor_r2 = 0.440
- Minor discrepancies in BIC (42,563 vs 42,526) and vigor_r2 (0.441 vs 0.440)

### Supplementary Table S1:
Same issues as main Table 1 -- M2 and M3 values differ.

## 16. Encounter dynamics

### Reactivity x cd r = 0.47

- **Paper (line 95, 97):** r = 0.47
- **Source (evc_vigor_dynamics.csv, reactivity_x_param log(c_death)):** r = 0.501 (Pearson); no-outlier r = 0.465; Spearman r = 0.407
- **Verdict:** MISMATCH -- Paper says 0.47; Pearson says 0.50, no-outlier says 0.47, Spearman says 0.41. The 0.47 matches the no-outlier version but not the standard Pearson. Paper does not indicate outlier removal was used.

### AMI r = -0.19

- **Paper (line 143):** AMI-Behavioural r = -0.191, p = .001; AMI-Total r = -0.188, p = .001
- **Source (evc_vigor_dynamics.csv, AMI_Total):** r = -0.167, p = 0.004
- **Note:** The source file only has AMI_Total = -0.167. Paper reports -0.188 for AMI_Total and -0.191 for AMI_Behavioural. These do not match the source file's AMI_Total = -0.167.
- **Verdict:** MISMATCH -- Paper says AMI_Total r = -0.188; source says r = -0.167

## 17. Cross-block stability r = 0.78

- **Paper:** r = 0.78
- **Source (evc_vigor_dynamics.csv, block_stability):** mean_cross_block_r = 0.782
- **Verdict:** MATCH

## 18. Threat ANOVA F = 0.04, p = .96

- **Paper:** F = 0.04, p = .96
- **Source (evc_vigor_dynamics.csv, encounter_change_anova):** F = 0.040, p = 0.960
- **Verdict:** MATCH

## 19. MCMC validation: ce r = 0.9994, cd r = 0.9986

- **Paper (line 238):** log(ce) r = 0.999, log(cd) r = 0.999
- **Paper (Supplementary Note, line 375):** log(ce) r = 0.999, log(cd) r = 0.999
- **Source (computed from oc_evc_mcmc_params.csv vs oc_evc_final_params.csv):** ce r = 0.9994, cd r = 0.9986
- **Verdict:** MATCH (both round to 0.999)

## 20. Other numbers in the paper

### Overall reactivity population mean
- **Paper (line 92):** M = -0.019, t = -1.15, p = .25, SD = 0.28
- **Source:** M = -0.0186, t = -1.151, p = 0.251
- **Note:** SD not directly in the vigor_dynamics file. M and t match.
- **Verdict:** MATCH

### Piecewise slope change
- **Paper (line 93):** t = 5.91, p < 10^-8
- **Source:** t = 5.909, p = 9.59e-9
- **Verdict:** MATCH

### Reactivity x ce
- **Paper (line 95):** r = -0.04, p = .49
- **Source:** r = -0.086, p = 0.141
- **Verdict:** MISMATCH -- Paper says r = -0.04; source says r = -0.09. Paper says p = .49; source says p = .14.

### Calibration x STAI
- **Paper (line 120):** r = 0.138, p = .02
- **Source (evc_final_dissociation.csv):** calibration -> STAI_State: r = 0.121, p = 0.040
- **Note:** Paper says r = 0.138. Source STAI_State = 0.121. There is also STAI_Trait = -0.061, p = 0.303. Neither matches 0.138 exactly. The evc_lqr_dissociation.csv shows calibration -> STAI_State r = 0.138, p = 0.019 -- so the paper is using values from the lqr version, not the final version.
- **Verdict:** MISMATCH -- Paper cites r = 0.138 (from evc_lqr_dissociation.csv); evc_final_dissociation.csv shows r = 0.121

### ce_vigor = 0.003
- **Paper (line 61):** c_e,vigor = 0.003
- **Source:** ce_vigor = 0.00284
- **Verdict:** MATCH (rounds to 0.003)

### Confidence-performance correlations
- **Paper (line 108):** confidence vs survival r = -0.05, p = .41; confidence vs choice quality r = -0.08, p = .16
- **Source (evc_final_metacognition.csv):** r_conf_survival = -0.048, p = 0.412; r_conf_choice_quality = -0.081, p = 0.165
- **Verdict:** MATCH

### Discrepancy x survival
- **Paper (Discussion, line 178):** r = -0.15, p = .009
- **Source (evc_final_dissociation.csv):** discrepancy -> survival_rate: r = -0.140, p = 0.017
- **Verdict:** MISMATCH -- Paper says r = -0.15, p = .009; source says r = -0.14, p = .017

### M2 description
- **Paper (line 71):** "Removing threat information (M1)... individual effort cost (M2: delta BIC = +2,094, choice r2 = 0.006)"
- **Source:** M2 delta BIC = +10,634
- **Verdict:** MISMATCH -- see item 15 above. The delta BIC for M2 is severely wrong in the paper.

### Paper (line 71): "joint estimation (M3: delta BIC = +10,430)"
- **Source:** M3 delta BIC = +10,393
- **Verdict:** Minor MISMATCH (10,430 vs 10,393)

### Paper (line 71): "probability weighting (M5: delta BIC = +2,071)"
- **Source:** M5 delta BIC = +2,071 (2070.6 rounds to 2,071)
- **Verdict:** MATCH

### Paper (line 71): M6 delta BIC = -142
- **Source:** -142.4
- **Verdict:** MATCH

### Calibration mean
- **Paper (line 118):** M = 0.47, SD = 0.32, 85% positive
- **Source:** Not directly in source files -- cannot verify
- **Verdict:** CANNOT VERIFY from available source files

### Discrepancy x reactivity independence
- **Paper (line 143, 150):** r = 0.096, p = .10
- **Source:** Not directly in provided source files -- cannot verify
- **Verdict:** CANNOT VERIFY from available source files

---

## Summary of Mismatches

### Critical mismatches (wrong numbers):

1. **M2 (Threat Only) in Table 1:** Paper BIC = 34,227, source BIC = 42,767. Paper delta = +2,094, source delta = +10,634. Paper vigor_r2 = 0.513, source = 0.294. **The entire M2 row appears to be from a different model fit.**

2. **Table 2 -- DASS-Stress discrepancy beta:** Paper = 0.217, source = 0.255
3. **Table 2 -- DASS-Depression discrepancy beta:** Paper = 0.206, source = 0.228
4. **Table 2 -- PHQ-9 discrepancy beta:** Paper = 0.212, source = 0.225
5. **Table 2 -- OASIS discrepancy beta:** Paper = 0.180, source = 0.228
6. **Table 2 -- Most log(ce) ROPE percentages are wrong** (e.g., OASIS: paper 78%, source 56%; DASS-Anxiety: paper 75%, source 60%)
7. **Table 2 -- Several log(cd) ROPE percentages are wrong** (e.g., STICSA: paper 78%, source 87%; PHQ-9: paper 80%, source 89%)

8. **Reactivity x log(cd):** Paper says r = 0.47; source Pearson r = 0.50 (no-outlier = 0.47)
9. **Reactivity x log(ce):** Paper says r = -0.04, p = .49; source says r = -0.09, p = .14
10. **Reactivity x AMI_Total:** Paper says r = -0.188; source says r = -0.167
11. **Sub-level vigor r:** Paper says 0.829; source says 0.836
12. **Calibration x survival:** Paper says r = 0.185; source says r = 0.179
13. **Discrepancy x survival:** Paper says r = -0.15, p = .009; source says r = -0.14, p = .017
14. **Calibration x STAI:** Paper says r = 0.138 (appears to be from evc_lqr_dissociation.csv, not evc_final_dissociation.csv which gives 0.121)
15. **ROPE range claim "65--93%":** Actual range is ~53--93%

### Minor mismatches (rounding or very small differences):

16. **M3 BIC:** Paper = 42,563, source = 42,526 (37-point difference)
17. **M3 vigor_r2:** Paper = 0.441, source = 0.440

### Values that match:

- BIC = 32,133
- Choice r2 = 0.951
- Vigor r2 = 0.511
- gamma = 0.209, epsilon = 0.098, tau = 0.476
- Choice accuracy = 79.3%, AUC = 0.876
- Anxiety beta = -0.557, t = -14.04
- Confidence beta = 0.575, t = 13.48
- Calibration x choice quality r = 0.230
- Orthogonality r = 0.019
- Parameter recovery: ce r = 0.92, cd r = 0.94
- Parameter independence r = -0.14
- Cross-block stability r = 0.78
- Threat ANOVA F = 0.04, p = .96
- MCMC validation: ce r = 0.999, cd r = 0.999
- M1, M4, M5, M6 BIC values
- Piecewise slope change t = 5.91
- ce_vigor = 0.003
- Confidence-performance correlations
- Population-level parameters
