# Statistics Verification Checklist — Draft 010

Every statistic in the paper traced to its source CSV and verified value.

## Population Parameters (oc_evc_final_81_population.csv)

| Stat | Paper value | CSV value | Match |
|------|-------------|-----------|-------|
| gamma | 0.209 / 0.21 | 0.2093 | YES |
| epsilon | 0.098 | 0.0976 | YES |
| ce_vigor | 0.003 | 0.00284 | YES (rounded) |
| tau | 0.476 | 0.4755 | YES |
| p_esc | 0.018 | 0.01788 | YES |
| sigma_motor | 1.169 | 1.1687 | YES |
| sigma_v | 0.229 | 0.2288 | YES |

## Model Comparison (evc_model_comparison_final.csv)

| Model | Paper BIC | CSV BIC | Paper deltaBIC | CSV deltaBIC | Match |
|-------|-----------|---------|----------------|--------------|-------|
| Full (FINAL) | 32,133 | 32133.433 | --- | 0 | YES |
| M1 (Effort only) | 50,792 | 50792 | +18,659 | 18658.567 | YES |
| M2 (Threat only) | 42,767 | 42767 | +10,634 | 10633.567 | YES |
| M3 (Separate) | 42,526 | 42526 | +10,393 | 10392.567 | YES |
| M4 (Pop ce) | 30,860 | 30860 | -1,273 | -1273.433 | YES |
| M5 (No gamma) | 34,204 | 34204 | +2,071 | 2070.567 | YES |
| M6 (u^2 cost) | 31,991 | 31991 | -142 | -142.433 | YES |

| Model | Paper choice r2 | CSV choice_r2 | Paper vigor r2 | CSV vigor_r2 | Match |
|-------|----------------|---------------|----------------|--------------|-------|
| Full | 0.951 | 0.951 | 0.511 | 0.511 | YES |
| M1 | 0.950 | 0.950 | 0.000 | 0.000 | YES |
| M2 | 0.006 | 0.006 | 0.294 | 0.294 | YES |
| M3 | 0.955 | 0.955 | 0.440 | 0.440 | YES |
| M4 | 0.001 | 0.001 | 0.512 | 0.512 | YES |
| M5 | 0.955 | 0.955 | 0.425 | 0.425 | YES |
| M6 | 0.952 | 0.952 | 0.508 | 0.508 | YES |

## Parameter Recovery (evc_final_recovery.csv, computed)

| Stat | Paper value | Computed value | Match |
|------|-------------|---------------|-------|
| log(ce) recovery r | 0.92 | 0.9159 | YES |
| log(cd) recovery r | 0.94 | 0.9428 | YES |
| N datasets | 3 | 3 | YES |
| Subjects per dataset | 50 | 50 | YES |

## Parameter Summary (oc_evc_final_params.csv, computed)

| Stat | Paper value | Computed value | Match |
|------|-------------|---------------|-------|
| ce median | 0.62 | 0.624 | YES |
| cd median | 31.3 | 31.263 | YES |
| ce-cd correlation r | -0.14 | -0.135 | YES |
| ce-cd correlation p | .02 | 0.0208 | YES |
| N subjects | 293 | 293 | YES |

## Deviation Analysis (per_subject_deviations.csv, computed)

| Stat | Paper value | Computed value | Match |
|------|-------------|---------------|-------|
| Optimality rate mean | 69.8% | 0.698 | YES |
| Optimality rate SD | 12.0% | 0.120 | YES |
| Overcautious rate mean | 21.3% | 0.213 | YES |
| Overcautious rate SD | 14.4% | 0.144 | YES |
| Overrisky rate mean | 8.9% | 0.089 | YES |
| Overrisky rate SD | 8.4% | 0.084 | YES |
| Mean total earnings | 83.5 | 83.5 | YES |

## Deviation-Parameter Associations (deviation_param_associations.csv)

| Stat | Paper value | CSV value | Match |
|------|-------------|-----------|-------|
| ce-overcautious bivariate r | 0.92 | 0.9242 | YES |
| ce-overcautious partial r | 0.92 | 0.9228 | YES |
| cd-vigor gap bivariate r | 0.55 | 0.5540 | YES |
| cd-vigor gap partial r | 0.55 | 0.5495 | YES |
| Overcautious multiple R2 | (0.855) | 0.8550 | YES |
| ce unique R2 in overcaution | 83.1% | 0.8312 | YES |
| cd unique R2 in overcaution | 0.09% | 0.000876 | YES |
| cd unique R2 p-value | .19 | 0.1865 | YES |
| beta_ce in overcaution regression | 0.196 | 0.19617 | YES |
| t_ce in overcaution regression | 40.8 | 40.774 | YES |
| ce-earnings r | -0.81 | -0.810 (four_routes) | YES |
| cd-earnings r | 0.15 | 0.148 (four_routes) | YES |
| ce-overrisky r | -0.78 | -0.7751 | YES |
| Gamma shift: conditions changed | 4/9 | 4/9 (optimal_policy.csv) | YES |
| Gamma distortion pct | ~20% | -0.200 (relative) | YES |

## Profiles (deviation_param_associations.csv)

| Profile | Paper earnings | CSV earnings | Paper N | CSV N | Match |
|---------|---------------|-------------|---------|-------|-------|
| Vigilant | 103.5 | 103.46 | 82 | 82 | YES |
| Cautious | 67.6 | 67.62 | 65 | 65 | YES |
| Bold | 96.1 | 96.06 | 64 | 64 | YES |
| Disengaged | 66.4 | 66.44 | 82 | 82 | YES |

## Residual Suboptimality (residual_suboptimality.csv)

| Stat | Paper value | CSV value | Match |
|------|-------------|-----------|-------|
| Calibration-discrepancy r | 0.019 | 0.01899 | YES |
| Calibration-discrepancy p | .75 | 0.7478 | YES |
| Base model overcaution R2 | 85.4% | 0.8535 | YES |
| Discrepancy-residual overcaution r | 0.14 | 0.14228 | YES |
| Discrepancy-residual overcaution p | .015 | 0.01550 | YES |
| Discrepancy delta R2 | 0.3% | 0.00303 | YES |
| Discrepancy F-change | 6.0 | 6.025 | YES |
| Discrepancy delta R2 p | .015 | 0.01470 | YES |
| Calibration-policy alignment bivariate r | 0.30 | 0.30423 | YES |
| Calibration-policy alignment partial r | 0.31 | 0.30518 | YES |
| Calibration delta R2 | 6.4% | 0.06390 | YES |
| Calibration delta R2 p | <10^-7 | 1.34e-07 | YES |
| Discrepancy-excess vigor low T | -0.12 | -0.12181 | YES |
| Discrepancy-excess vigor low T p | .038 | 0.03850 | YES |

## Four Routes (four_routes.csv)

| Predictor → Outcome | Paper r | CSV r | Match |
|---------------------|---------|-------|-------|
| log_ce → overcautious_rate | 0.92 | 0.9234 | YES |
| log_ce → policy_alignment | -0.56 | -0.5593 | YES |
| log_ce → total_earnings | -0.81 | -0.8100 | YES |
| log_cd → survival_rate | -0.02 (ns) | -0.0237 | YES |
| calibration → policy_alignment | 0.30 | 0.3042 | YES |
| calibration → survival_rate | 0.18 | 0.1787 | YES |
| calibration → total_earnings | 0.20 | 0.2006 | YES |
| discrepancy → residual_overcaution | 0.14 | 0.1423 | YES |
| discrepancy → total_earnings | -0.24 | -0.2363 | YES |
| discrepancy → survival_rate | -0.14 | -0.1404 | YES |
| encounter_reactivity → survival_rate | -0.24 | -0.2420 | YES |
| encounter_reactivity → overcautious_rate | ns | -0.086 (p=.14) | YES |

## Encounter Dynamics (evc_vigor_dynamics.csv)

| Stat | Paper value | CSV value | Match |
|------|-------------|-----------|-------|
| Mean encounter reactivity | -0.019 | -0.01858 | YES |
| SD encounter reactivity | 0.28 | 0.27644 | YES |
| t-stat overall | -1.15 | -1.1507 | YES |
| p-value overall | .25 | 0.2508 | YES |
| Cross-block r | 0.78 | 0.78204 | YES |
| Reactivity × log(cd) r | 0.50 | 0.50066 | YES |
| Reactivity × log(cd) p | <10^-19 | 5.39e-20 | YES |
| Reactivity × log(ce) r | -0.09 | -0.08615 | YES |
| Reactivity × log(ce) p | .14 | 0.14128 | YES |
| Encounter ANOVA F | 0.04 | 0.04045 | YES |
| Encounter ANOVA p | .96 | 0.96036 | YES |
| Slope change | 0.050 | 0.04955 | YES |
| Slope change t | 5.91 | 5.9086 | YES |
| Slope change p | <10^-8 | 9.59e-09 | YES |
| Slope change × log(cd) r | 0.18 | 0.17632 | YES |
| Slope change × log(cd) p | .002 | 0.00245 | YES |
| Spearman reactivity × cd | 0.41 | 0.40667 | YES |
| Reactivity × AMI r | -0.17 | -0.16672 | YES |
| Reactivity × AMI p | .004 | 0.00421 | YES |

## Clinical Bayesian (evc_bayesian_clinical.csv)

| Stat | Paper value | CSV value | Match |
|------|-------------|-----------|-------|
| STAI disc beta | 0.338 | 0.33755 | YES |
| STAI disc HDI | [0.23, 0.45] | [0.226, 0.448] | YES |
| STICSA disc beta | 0.285 | 0.28539 | YES |
| STICSA disc HDI | [0.17, 0.39] | [0.169, 0.394] | YES |
| DASS-Anx disc beta | 0.275 | 0.27530 | YES |
| DASS-Anx disc HDI | [0.16, 0.40] | [0.162, 0.396] | YES |
| DASS-Stress disc beta | 0.255 | 0.25522 | YES |
| DASS-Stress disc HDI | [0.13, 0.37] | [0.132, 0.368] | YES |
| DASS-Dep disc beta | 0.228 | 0.22819 | YES |
| PHQ9 disc beta | 0.225 | 0.22519 | YES |
| OASIS disc beta | 0.228 | 0.22834 | YES |
| AMI-Emo disc beta | -0.222 | -0.22237 | YES |

## Optimal Policy (optimal_policy.csv)

| Condition | Paper obj optimal | CSV optimal_choice | Paper EV advantage | CSV ev_advantage | Match |
|-----------|------------------|-------------------|-------------------|-----------------|-------|
| T=0.1,D=1 | Heavy | heavy | +2.15 | 2.148 | YES |
| T=0.1,D=2 | Heavy | heavy | +1.55 | 1.549 | YES |
| T=0.1,D=3 | Heavy | heavy | +0.39 | 0.389 | YES |
| T=0.5,D=1 | Heavy | heavy | +0.80 | 0.802 | YES |
| T=0.5,D=2 | Heavy | heavy | +0.07 | 0.070 | YES |
| T=0.5,D=3 | Light | light | -0.84 | -0.836 | YES |
| T=0.9,D=1 | Light | light | -0.94 | -0.944 | YES |
| T=0.9,D=2 | Light | light | -0.75 | -0.747 | YES |
| T=0.9,D=3 | Light | light | -1.07 | -1.067 | YES |

## Summary

- **Total statistics verified:** 107
- **Mismatches:** 0
- **All values match source CSVs** (with standard rounding to reported precision)
