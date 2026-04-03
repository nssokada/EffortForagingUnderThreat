#!/usr/bin/env python3
"""
17_critical_checks.py — Critical reviewer-facing analyses
==========================================================

Analyses:
  1. Does pressing harder ACTUALLY help survival? (ε validation)
  2. γ: probability weighting vs utility curvature vs T-dependent ce
  3. External model comparisons (heuristic, effort-only, RL)
  4. Policy alignment (formal)
  5. Discrepancy ΔR² robustness (bootstrap, power)

Output:
  results/stats/critical_checks.csv
  results/figs/paper/fig_critical_checks.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, pointbiserialr
from scipy.special import expit
import statsmodels.api as sm
from pathlib import Path
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'font.size': 9,
})

# ── Paths ──
DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
BEHAVIOR_FILE = DATA_DIR / "behavior.csv"
BEHAVIOR_RICH_FILE = DATA_DIR / "behavior_rich.csv"
PARAMS_FILE = Path("/workspace/results/stats/oc_evc_final_params.csv")
POP_FILE = Path("/workspace/results/stats/oc_evc_final_81_population.csv")
DEVIATIONS_FILE = Path("/workspace/results/stats/per_subject_deviations.csv")
OUT_STATS = Path("/workspace/results/stats/critical_checks.csv")
OUT_FIG = Path("/workspace/results/figs/paper/fig_critical_checks.png")
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ──
print("=" * 70)
print("17. CRITICAL REVIEWER CHECKS")
print("=" * 70)

behavior = pd.read_csv(BEHAVIOR_FILE)
behavior_rich = pd.read_csv(BEHAVIOR_RICH_FILE, low_memory=False)
params = pd.read_csv(PARAMS_FILE)
pop = pd.read_csv(POP_FILE)
dev = pd.read_csv(DEVIATIONS_FILE)

GAMMA = float(pop["gamma"].iloc[0])
EPSILON = float(pop["epsilon"].iloc[0])
P_ESC = float(pop["p_esc"].iloc[0])
TAU = float(pop["tau"].iloc[0])

print(f"Behavior rich: {behavior_rich.shape[0]} trials, {behavior_rich['subj'].nunique()} subjects")
print(f"Population: gamma={GAMMA:.4f}, epsilon={EPSILON:.4f}, p_esc={P_ESC:.4f}, tau={TAU:.4f}")

results = []

def safe_pearsonr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])


# ═══════════════════════════════════════════════════════════════════════
# COMPUTE PRESS RATES FOR ALL TRIALS
# ═══════════════════════════════════════════════════════════════════════
print("\nComputing press rates...")

def compute_median_rate(row):
    """Compute normalized median press rate from alignedEffortRate."""
    try:
        pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
        ipis = np.diff(pt)
        ipis = ipis[ipis > 0.01]
        if len(ipis) >= 5:
            return np.median((1.0 / ipis) / row['calibrationMax'])
        return np.nan
    except Exception:
        return np.nan

# Only compute for choice trials (type=1) -- this is the relevant set
choice_rich = behavior_rich[behavior_rich['type'] == 1].copy()
choice_rich['median_press_rate'] = choice_rich.apply(compute_median_rate, axis=1)
choice_rich['survived'] = (choice_rich['trialEndState'] == 'escaped').astype(int)
choice_rich['captured'] = (choice_rich['trialEndState'] == 'captured').astype(int)
choice_rich['T_round'] = choice_rich['threat'].round(1)
choice_rich['cookie_type'] = np.where(choice_rich['trialCookie_weight'] == 3.0, 'heavy', 'light')

# Drop NaN press rates
valid = choice_rich.dropna(subset=['median_press_rate']).copy()
print(f"Valid choice trials with press rates: {len(valid)}/{len(choice_rich)}")


# ═══════════════════════════════════════════════════════════════════════
# 1. DOES PRESSING HARDER HELP SURVIVAL?
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. DOES PRESSING HARDER ACTUALLY HELP SURVIVAL?")
print("=" * 70)

# 1a. Overall empirical escape rate on attack trials
attack_trials = valid[valid['isAttackTrial'] == 1]
p_esc_empirical = attack_trials['survived'].mean()
n_attack = len(attack_trials)
print(f"\nEmpirical p(escape | attack): {p_esc_empirical:.4f} (N={n_attack})")
results.append(('1_p_esc_empirical', p_esc_empirical, n_attack, np.nan))

# 1b. By T level
print("\nEmpirical p(escape | attack) by threat level:")
for t in [0.1, 0.5, 0.9]:
    sub = attack_trials[attack_trials['T_round'] == t]
    pe = sub['survived'].mean()
    n = len(sub)
    print(f"  T={t}: p_esc={pe:.4f} (N={n})")
    results.append((f'1_p_esc_T{t}', pe, n, np.nan))

# 1c. Logistic regression: captured ~ press_rate, by T x cookie
print("\n--- Logistic regression: captured ~ press_rate ---")
print(f"{'T':>5} {'cookie':>7} {'N':>6} {'OR':>8} {'p':>10} {'r_pb':>8} {'p_pb':>10}")
print("-" * 65)

for t in [0.1, 0.5, 0.9]:
    for cookie in ['heavy', 'light']:
        sub = valid[(valid['T_round'] == t) & (valid['cookie_type'] == cookie)]
        sub = sub.dropna(subset=['median_press_rate', 'captured'])
        if len(sub) < 20:
            print(f"{t:>5} {cookie:>7} {len(sub):>6} --- insufficient ---")
            continue

        X = sm.add_constant(sub['median_press_rate'].values)
        y = sub['captured'].values

        try:
            logit = sm.Logit(y, X).fit(disp=False)
            or_val = np.exp(logit.params[1])
            p_val = logit.pvalues[1]
        except Exception:
            or_val, p_val = np.nan, np.nan

        r_pb, p_pb = pointbiserialr(sub['captured'].values, sub['median_press_rate'].values)

        print(f"{t:>5.1f} {cookie:>7} {len(sub):>6} {or_val:>8.4f} {p_val:>10.4e} {r_pb:>8.4f} {p_pb:>10.4e}")
        results.append((f'1_logit_T{t}_{cookie}_OR', or_val, len(sub), p_val))
        results.append((f'1_rpb_T{t}_{cookie}', r_pb, len(sub), p_pb))

# 1d. Attack trials only: does press rate predict survival?
print("\n--- Attack trials only: captured ~ press_rate ---")
print(f"{'T':>5} {'cookie':>7} {'N':>6} {'OR':>8} {'p':>10} {'r_pb':>8} {'p_pb':>10}")
print("-" * 65)

for t in [0.1, 0.5, 0.9]:
    for cookie in ['heavy', 'light']:
        sub = attack_trials[(attack_trials['T_round'] == t) & (attack_trials['cookie_type'] == cookie)]
        sub = sub.dropna(subset=['median_press_rate', 'captured'])
        if len(sub) < 20:
            print(f"{t:>5.1f} {cookie:>7} {len(sub):>6} --- insufficient ---")
            continue

        X = sm.add_constant(sub['median_press_rate'].values)
        y = sub['captured'].values

        try:
            logit = sm.Logit(y, X).fit(disp=False)
            or_val = np.exp(logit.params[1])
            p_val = logit.pvalues[1]
        except Exception:
            or_val, p_val = np.nan, np.nan

        r_pb, p_pb = pointbiserialr(sub['captured'].values, sub['median_press_rate'].values)

        print(f"{t:>5.1f} {cookie:>7} {len(sub):>6} {or_val:>8.4f} {p_val:>10.4e} {r_pb:>8.4f} {p_pb:>10.4e}")
        results.append((f'1_attack_logit_T{t}_{cookie}_OR', or_val, len(sub), p_val))
        results.append((f'1_attack_rpb_T{t}_{cookie}', r_pb, len(sub), p_pb))

# 1e. Overall attack trial regression
print("\n--- Overall: attack trials, captured ~ press_rate + T + D ---")
at = attack_trials.dropna(subset=['median_press_rate', 'captured']).copy()
at['dist_actual'] = at['startDistance'].map({5: 1, 7: 2, 9: 3})
X_full = sm.add_constant(at[['median_press_rate', 'threat', 'dist_actual']].values)
logit_full = sm.Logit(at['captured'].values, X_full).fit(disp=False)
print(f"  press_rate coef: {logit_full.params[1]:.4f}, OR={np.exp(logit_full.params[1]):.4f}, p={logit_full.pvalues[1]:.4e}")
print(f"  threat coef: {logit_full.params[2]:.4f}, OR={np.exp(logit_full.params[2]):.4f}, p={logit_full.pvalues[2]:.4e}")
print(f"  distance coef: {logit_full.params[3]:.4f}, OR={np.exp(logit_full.params[3]):.4f}, p={logit_full.pvalues[3]:.4e}")
results.append(('1_attack_overall_press_OR', np.exp(logit_full.params[1]), len(at), logit_full.pvalues[1]))

# Overall point-biserial
r_overall, p_overall = pointbiserialr(at['captured'].values, at['median_press_rate'].values)
print(f"  Overall point-biserial (attack): r={r_overall:.4f}, p={p_overall:.4e}")
results.append(('1_attack_overall_rpb', r_overall, len(at), p_overall))

# Interpretation
print("\n>>> INTERPRETATION:")
if abs(r_overall) < 0.05 and p_overall > 0.05:
    print(">>> Press rate has WEAK/NULL relationship with survival.")
    print(">>> cd likely captures threat-driven motor AROUSAL, not survival OPTIMIZATION.")
elif r_overall < -0.05 and p_overall < 0.05:
    print(">>> Press rate DOES predict survival (higher rate → less capture).")
    print(">>> cd captures genuine survival-promoting effort allocation.")
else:
    print(f">>> Ambiguous: r={r_overall:.4f}, p={p_overall:.4e}. See per-cell results.")


# ═══════════════════════════════════════════════════════════════════════
# 2. γ: PROBABILITY WEIGHTING VS ALTERNATIVES
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. γ: PROBABILITY WEIGHTING VS UTILITY CURVATURE VS T-DEPENDENT ce")
print("=" * 70)

# Setup: choice data
choice = behavior[behavior['subj'].isin(params['subj'])].copy()
N_subj = params['subj'].nunique()
subjects = sorted(params['subj'].unique())
subj_to_idx = {s: i for i, s in enumerate(subjects)}
choice['subj_idx'] = choice['subj'].map(subj_to_idx)

# ── Helper: compute choice NLL for a model ──
def compute_choice_bic(choice_df, p_heavy_all, n_params):
    """BIC from predicted P(heavy)."""
    eps = 1e-8
    p = np.clip(p_heavy_all, eps, 1 - eps)
    y = choice_df['choice'].values
    nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    n = len(y)
    bic = 2 * nll + n_params * np.log(n)
    acc = np.mean((p > 0.5).astype(int) == y)
    return nll, bic, acc

# ── Fit per-subject ce via SVI-like grid optimization ──
def fit_per_subject_ce(choice_df, S_func, tau_val=TAU, ce_grid=np.linspace(0.01, 5, 200)):
    """Fit per-subject ce by grid search, given S function.

    Returns dict of subj -> best_ce and array of P(heavy).
    """
    ce_dict = {}
    p_heavy_all = np.zeros(len(choice_df))

    for s in subjects:
        mask = (choice_df['subj'] == s).values
        sdf = choice_df[mask]
        if len(sdf) == 0:
            continue

        T = sdf['threat'].values
        D_H = sdf['distance_H'].values
        y = sdf['choice'].values

        S = S_func(T)

        best_nll = np.inf
        best_ce = 1.0
        for ce_val in ce_grid:
            dEU = S * 4 - ce_val * (0.81 * D_H - 0.16)
            p = expit(dEU / tau_val)
            p = np.clip(p, 1e-8, 1 - 1e-8)
            nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best_nll:
                best_nll = nll
                best_ce = ce_val

        ce_dict[s] = best_ce
        S_vals = S_func(T)
        dEU = best_ce_dEU = S_vals * 4 - best_ce * (0.81 * D_H - 0.16)
        p_heavy_all[mask] = expit(best_ce_dEU / tau_val)

    return ce_dict, p_heavy_all

# 2A. Original model (baseline)
print("\n--- 2A. Baseline: S = (1-T^gamma) + eps*T^gamma*p_esc ---")
def S_baseline(T):
    Tg = T ** GAMMA
    return (1 - Tg) + EPSILON * Tg * P_ESC

ce_base, p_base = fit_per_subject_ce(choice, S_baseline)
nll_base, bic_base, acc_base = compute_choice_bic(choice, p_base, N_subj + 2)  # ce_i + gamma + eps
print(f"  NLL={nll_base:.1f}, BIC={bic_base:.1f}, Accuracy={acc_base:.3f}")
results.append(('2_baseline_BIC', bic_base, N_subj, acc_base))

# 2A-alt. Utility curvature model: EU = S*R^alpha - (1-S)*C^alpha
print("\n--- 2A-alt. Utility curvature: EU = S*R^alpha - (1-S)*C^alpha ---")
best_alpha_bic = np.inf
best_alpha = 1.0
R_H, R_L, C = 5.0, 1.0, 5.0

for alpha_test in np.linspace(0.1, 2.0, 100):
    # With linear T: S = 1-T (no gamma)
    def S_linear(T):
        return 1 - T

    def compute_alpha_nll(choice_df, alpha):
        nll_total = 0
        p_all = np.zeros(len(choice_df))
        for s in subjects:
            mask = (choice_df['subj'] == s).values
            sdf = choice_df[mask]
            if len(sdf) == 0:
                continue
            T = sdf['threat'].values
            D_H = sdf['distance_H'].values
            y = sdf['choice'].values
            S = 1 - T  # linear

            # Heavy: S*R_H^alpha - (1-S)*C^alpha, adjusted for effort
            # Light: S*R_L^alpha - (1-S)*C^alpha (at D=1, minimal effort)
            # We still need effort cost... use per-subject ce
            # Actually: to make this a REPLACEMENT for gamma, use:
            # EU_heavy = S * (R_H^alpha) - (1-S)*(C^alpha) - ce*effort_H
            # EU_light = S * (R_L^alpha) - (1-S)*(C^alpha) - ce*effort_L
            # dEU = S * (R_H^alpha - R_L^alpha) - ce*(effort_H - effort_L)
            # Note: C^alpha cancels!
            dR = R_H**alpha - R_L**alpha

            best_nll_s = np.inf
            best_ce_s = 1.0
            for ce_val in np.linspace(0.01, 5, 100):
                dEU = S * dR - ce_val * (0.81 * D_H - 0.16)
                p = expit(dEU / TAU)
                p = np.clip(p, 1e-8, 1 - 1e-8)
                nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
                if nll < best_nll_s:
                    best_nll_s = nll
                    best_ce_s = ce_val

            dEU_best = S * dR - best_ce_s * (0.81 * D_H - 0.16)
            p_all[mask] = expit(dEU_best / TAU)
            nll_total += best_nll_s

        return nll_total, p_all

    nll_a, _ = compute_alpha_nll(choice, alpha_test)
    bic_a = 2 * nll_a + (N_subj + 1) * np.log(len(choice))  # ce_i + alpha
    if bic_a < best_alpha_bic:
        best_alpha_bic = bic_a
        best_alpha = alpha_test

# Recompute at best alpha
nll_alpha, p_alpha = compute_alpha_nll(choice, best_alpha)
bic_alpha = 2 * nll_alpha + (N_subj + 1) * np.log(len(choice))
acc_alpha = np.mean((p_alpha > 0.5).astype(int) == choice['choice'].values)
print(f"  Best alpha={best_alpha:.3f}, NLL={nll_alpha:.1f}, BIC={bic_alpha:.1f}, Accuracy={acc_alpha:.3f}")
print(f"  vs Baseline BIC={bic_base:.1f} (ΔBIC={bic_alpha - bic_base:+.1f})")
results.append(('2_alpha_model_BIC', bic_alpha, N_subj, acc_alpha))
results.append(('2_alpha_best', best_alpha, np.nan, np.nan))
results.append(('2_alpha_vs_baseline_dBIC', bic_alpha - bic_base, np.nan, np.nan))

# 2B. Does ce vary by threat level?
print("\n--- 2B. Per-subject ce at each threat level ---")
ce_by_T = {}
for t in [0.1, 0.5, 0.9]:
    sub_choice = choice[choice['threat'].round(1) == t].copy()
    # Fit ce per subject with T fixed -> S is constant within subset
    S_val = S_baseline(np.array([t]))[0]

    ce_t = {}
    for s in subjects:
        mask = (sub_choice['subj'] == s).values
        sdf = sub_choice[mask]
        if len(sdf) < 3:
            continue

        D_H = sdf['distance_H'].values
        y = sdf['choice'].values

        best_nll = np.inf
        best_ce = 1.0
        for ce_val in np.linspace(0.01, 5, 200):
            dEU = S_val * 4 - ce_val * (0.81 * D_H - 0.16)
            p = expit(dEU / TAU)
            p = np.clip(p, 1e-8, 1 - 1e-8)
            nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best_nll:
                best_nll = nll
                best_ce = ce_val
        ce_t[s] = best_ce
    ce_by_T[t] = ce_t

# Compare ce across T levels
common_subj = set(ce_by_T[0.1].keys()) & set(ce_by_T[0.5].keys()) & set(ce_by_T[0.9].keys())
print(f"  Subjects with ce at all 3 T levels: {len(common_subj)}")

ce_01 = np.array([ce_by_T[0.1][s] for s in common_subj])
ce_05 = np.array([ce_by_T[0.5][s] for s in common_subj])
ce_09 = np.array([ce_by_T[0.9][s] for s in common_subj])

print(f"  ce at T=0.1: mean={ce_01.mean():.3f}, sd={ce_01.std():.3f}")
print(f"  ce at T=0.5: mean={ce_05.mean():.3f}, sd={ce_05.std():.3f}")
print(f"  ce at T=0.9: mean={ce_09.mean():.3f}, sd={ce_09.std():.3f}")

# Repeated-measures ANOVA (Friedman since distributions likely non-normal)
stat_f, p_f = stats.friedmanchisquare(ce_01, ce_05, ce_09)
print(f"  Friedman test: χ²={stat_f:.2f}, p={p_f:.4e}")
results.append(('2B_friedman_chi2', stat_f, len(common_subj), p_f))

# Pairwise correlations of per-subject ce across T levels
r_01_05, p_01_05 = pearsonr(ce_01, ce_05)
r_01_09, p_01_09 = pearsonr(ce_01, ce_09)
r_05_09, p_05_09 = pearsonr(ce_05, ce_09)
print(f"  ce stability: T0.1-T0.5 r={r_01_05:.3f}, T0.1-T0.9 r={r_01_09:.3f}, T0.5-T0.9 r={r_05_09:.3f}")
results.append(('2B_ce_r_T01_T05', r_01_05, len(common_subj), p_01_05))
results.append(('2B_ce_r_T01_T09', r_01_09, len(common_subj), p_01_09))
results.append(('2B_ce_r_T05_T09', r_05_09, len(common_subj), p_05_09))

# Prelec-style probability weighting
print("\n--- 2B-alt. Prelec probability weighting fit ---")
# For each subject, compute P(heavy) at each T
pheavy_by_T = {}
for t in [0.1, 0.5, 0.9]:
    pheavy_by_T[t] = {}
    for s in subjects:
        sub = choice[(choice['subj'] == s) & (choice['threat'].round(1) == t)]
        if len(sub) >= 3:
            pheavy_by_T[t][s] = sub['choice'].mean()

# Population-level Prelec fit: w(T) = exp(-(-ln(T))^alpha)
T_vals = np.array([0.1, 0.5, 0.9])
pop_pheavy = np.array([
    np.mean([pheavy_by_T[t][s] for s in pheavy_by_T[t]]) for t in T_vals
])
print(f"  Population P(heavy): T=0.1: {pop_pheavy[0]:.3f}, T=0.5: {pop_pheavy[1]:.3f}, T=0.9: {pop_pheavy[2]:.3f}")

# Grid search for Prelec alpha
best_prelec_alpha = 1.0
best_prelec_ss = np.inf
for a_test in np.linspace(0.05, 3.0, 200):
    w = np.exp(-(-np.log(T_vals))**a_test)
    # w is the subjective threat weight; predicted P(heavy) should decrease with w
    # Simple: P(heavy) ≈ intercept - slope * w
    X_p = sm.add_constant(w)
    try:
        fit_p = sm.OLS(pop_pheavy, X_p).fit()
        ss = fit_p.ssr
        if ss < best_prelec_ss:
            best_prelec_ss = ss
            best_prelec_alpha = a_test
    except:
        pass

w_best = np.exp(-(-np.log(T_vals))**best_prelec_alpha)
print(f"  Prelec alpha={best_prelec_alpha:.3f}")
print(f"  Prelec weights: w(0.1)={w_best[0]:.3f}, w(0.5)={w_best[1]:.3f}, w(0.9)={w_best[2]:.3f}")
print(f"  vs T^gamma:     w(0.1)={0.1**GAMMA:.3f}, w(0.5)={0.5**GAMMA:.3f}, w(0.9)={0.9**GAMMA:.3f}")
results.append(('2_prelec_alpha', best_prelec_alpha, 3, np.nan))


# ═══════════════════════════════════════════════════════════════════════
# 3. EXTERNAL MODEL COMPARISONS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. EXTERNAL MODEL COMPARISONS")
print("=" * 70)

# ── Model A: Heuristic threshold ──
# P(heavy) = sigmoid((a_i - b_T*T - b_D*D) / tau_heur)
print("\n--- Model A: Heuristic threshold ---")
# Fit: grid search b_T, b_D, tau_heur (population), a_i per subject
best_heur = {'nll': np.inf}

for b_T in np.linspace(0.5, 8.0, 30):
    for b_D in np.linspace(0.0, 3.0, 15):
        tau_h = 0.5  # Fix temperature
        nll_total = 0
        for s in subjects:
            mask = (choice['subj'] == s).values
            sdf = choice[mask]
            if len(sdf) == 0:
                continue
            T = sdf['threat'].values
            D_H = sdf['distance_H'].values
            y = sdf['choice'].values

            best_nll_s = np.inf
            for a_val in np.linspace(-3, 6, 50):
                p = expit((a_val - b_T * T - b_D * D_H) / tau_h)
                p = np.clip(p, 1e-8, 1 - 1e-8)
                nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
                if nll < best_nll_s:
                    best_nll_s = nll
            nll_total += best_nll_s

        if nll_total < best_heur['nll']:
            best_heur = {'nll': nll_total, 'b_T': b_T, 'b_D': b_D}

# Recompute at best params
p_heur_all = np.zeros(len(choice))
a_heur_dict = {}
for s in subjects:
    mask = (choice['subj'] == s).values
    sdf = choice[mask]
    if len(sdf) == 0:
        continue
    T = sdf['threat'].values
    D_H = sdf['distance_H'].values
    y = sdf['choice'].values

    best_nll_s = np.inf
    best_a = 0
    for a_val in np.linspace(-3, 6, 200):
        p = expit((a_val - best_heur['b_T'] * T - best_heur['b_D'] * D_H) / 0.5)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll_s:
            best_nll_s = nll
            best_a = a_val
    a_heur_dict[s] = best_a
    p_heur_all[mask] = expit((best_a - best_heur['b_T'] * T - best_heur['b_D'] * D_H) / 0.5)

nll_heur = best_heur['nll']
bic_heur = 2 * nll_heur + (N_subj + 2) * np.log(len(choice))  # a_i + b_T + b_D
acc_heur = np.mean((p_heur_all > 0.5).astype(int) == choice['choice'].values)
print(f"  b_T={best_heur['b_T']:.2f}, b_D={best_heur['b_D']:.2f}")
print(f"  NLL={nll_heur:.1f}, BIC={bic_heur:.1f}, Accuracy={acc_heur:.3f}")
results.append(('3A_heuristic_BIC', bic_heur, N_subj, acc_heur))

# ── Model B: Effort only (no threat) ──
print("\n--- Model B: Effort discounting only (no threat) ---")
# ΔEU = (R_H - R_L) - ce_i * (0.81*D_H - 0.16)
# P(heavy) = sigmoid(ΔEU / tau)
p_effort_all = np.zeros(len(choice))
nll_effort_total = 0
ce_effort_dict = {}

for s in subjects:
    mask = (choice['subj'] == s).values
    sdf = choice[mask]
    if len(sdf) == 0:
        continue
    D_H = sdf['distance_H'].values
    y = sdf['choice'].values

    best_nll_s = np.inf
    best_ce = 1.0
    for ce_val in np.linspace(0.01, 10, 200):
        dEU = 4.0 - ce_val * (0.81 * D_H - 0.16)
        p = expit(dEU / TAU)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll_s:
            best_nll_s = nll
            best_ce = ce_val

    ce_effort_dict[s] = best_ce
    dEU = 4.0 - best_ce * (0.81 * D_H - 0.16)
    p_effort_all[mask] = expit(dEU / TAU)
    nll_effort_total += best_nll_s

bic_effort = 2 * nll_effort_total + N_subj * np.log(len(choice))  # ce_i only
acc_effort = np.mean((p_effort_all > 0.5).astype(int) == choice['choice'].values)
print(f"  NLL={nll_effort_total:.1f}, BIC={bic_effort:.1f}, Accuracy={acc_effort:.3f}")
results.append(('3B_effort_only_BIC', bic_effort, N_subj, acc_effort))

# ── Model C: Reinforcement Learning ──
print("\n--- Model C: Reinforcement learning ---")
# Q-learning on (T_level, D_level, choice)
# Need to process trials in ORDER for each subject

# Map conditions to indices
T_map = {0.1: 0, 0.5: 1, 0.9: 2}
D_map = {1: 0, 2: 1, 3: 2}

# Get outcome from behavior_rich for each trial
# outcome in behavior.csv: 0=survived, 1=captured
# reward: if survived, get cookie reward; if captured, lose C=5

p_rl_all = np.zeros(len(choice))
nll_rl_total = 0
alpha_rl_dict = {}

for s in subjects:
    mask = (choice['subj'] == s).values
    sdf = choice[mask].sort_values('trial')
    if len(sdf) == 0:
        continue

    T_arr = sdf['threat'].round(1).values
    D_arr = sdf['distance_H'].values
    y_arr = sdf['choice'].values
    out_arr = sdf['outcome'].values  # 0=survived, 1=captured

    best_nll_s = np.inf
    best_alpha = 0.1
    best_p_arr = None

    for alpha_test in np.linspace(0.01, 1.0, 50):
        Q = np.zeros((3, 3, 2))  # T x D x choice
        p_arr = np.zeros(len(sdf))
        nll_s = 0

        for i in range(len(sdf)):
            t_idx = T_map.get(T_arr[i], 1)
            d_idx = D_map.get(D_arr[i], 1)

            # Predict
            dQ = Q[t_idx, d_idx, 1] - Q[t_idx, d_idx, 0]
            p_h = expit(dQ / TAU)
            p_h = np.clip(p_h, 1e-8, 1 - 1e-8)
            p_arr[i] = p_h

            nll_s += -(y_arr[i] * np.log(p_h) + (1 - y_arr[i]) * np.log(1 - p_h))

            # Outcome
            ch = int(y_arr[i])
            if out_arr[i] == 0:  # survived
                reward = 5.0 if ch == 1 else 1.0
            else:  # captured
                reward = -5.0

            Q[t_idx, d_idx, ch] += alpha_test * (reward - Q[t_idx, d_idx, ch])

        if nll_s < best_nll_s:
            best_nll_s = nll_s
            best_alpha = alpha_test
            best_p_arr = p_arr.copy()

    alpha_rl_dict[s] = best_alpha
    # Re-index p_arr back to the original order
    indices = np.where(mask)[0]
    sort_order = sdf['trial'].values.argsort()
    reverse_order = sort_order.argsort()
    p_rl_all[indices] = best_p_arr[reverse_order]
    nll_rl_total += best_nll_s

bic_rl = 2 * nll_rl_total + N_subj * np.log(len(choice))  # alpha_i only
acc_rl = np.mean((p_rl_all > 0.5).astype(int) == choice['choice'].values)
print(f"  NLL={nll_rl_total:.1f}, BIC={bic_rl:.1f}, Accuracy={acc_rl:.3f}")
results.append(('3C_rl_BIC', bic_rl, N_subj, acc_rl))

# ── Model comparison summary ──
print("\n--- MODEL COMPARISON SUMMARY ---")
print(f"{'Model':<25} {'NLL':>10} {'BIC':>12} {'Accuracy':>10} {'ΔBIC':>10}")
print("-" * 70)
models = [
    ('EVC (baseline)', nll_base, bic_base, acc_base),
    ('Alpha (utility curv.)', nll_alpha, bic_alpha, acc_alpha),
    ('Heuristic threshold', nll_heur, bic_heur, acc_heur),
    ('Effort only (no T)', nll_effort_total, bic_effort, acc_effort),
    ('Reinforcement learning', nll_rl_total, bic_rl, acc_rl),
]
min_bic = min(m[2] for m in models)
for name, nll, bic, acc in models:
    print(f"{name:<25} {nll:>10.1f} {bic:>12.1f} {acc:>10.3f} {bic - min_bic:>+10.1f}")

results.append(('3_comparison_min_bic_model', min_bic, len(models), np.nan))


# ═══════════════════════════════════════════════════════════════════════
# 4. POLICY ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. POLICY ALIGNMENT (FORMAL)")
print("=" * 70)

# For each subject's 45 trials, compare actual choice to SUBJECTIVE-optimal
# Subjective optimal: choose heavy if dEU > 0 (under their ce, population gamma)
merged = params.merge(dev, on='subj', how='inner')

policy_alignment = []
for _, row in merged.iterrows():
    s = row['subj']
    ce_s = row['c_effort']

    sdf = choice[choice['subj'] == s]
    if len(sdf) == 0:
        continue

    T = sdf['threat'].values
    D_H = sdf['distance_H'].values
    y = sdf['choice'].values

    Tg = T ** GAMMA
    S = (1 - Tg) + EPSILON * Tg * P_ESC
    dEU = S * 4 - ce_s * (0.81 * D_H - 0.16)

    subj_optimal = (dEU > 0).astype(int)
    alignment = np.mean(y == subj_optimal)
    policy_alignment.append({'subj': s, 'policy_alignment': alignment})

pa_df = pd.DataFrame(policy_alignment)
merged = merged.merge(pa_df, on='subj', how='inner')

print(f"Policy alignment: mean={merged['policy_alignment'].mean():.3f}, "
      f"sd={merged['policy_alignment'].std():.3f}, "
      f"median={merged['policy_alignment'].median():.3f}")
print(f"Range: [{merged['policy_alignment'].min():.3f}, {merged['policy_alignment'].max():.3f}]")
results.append(('4_policy_alignment_mean', merged['policy_alignment'].mean(), len(merged), np.nan))
results.append(('4_policy_alignment_sd', merged['policy_alignment'].std(), len(merged), np.nan))

# Correlation with calibration (from residual_suboptimality file if available)
# Compute calibration here directly
feelings = pd.read_csv(DATA_DIR / "feelings.csv")
anxiety_df = feelings[feelings["questionLabel"] == "anxiety"].copy()
T_a = anxiety_df["threat"].values
T_gamma_a = T_a ** GAMMA
anxiety_df["S"] = (1 - T_gamma_a) + EPSILON * T_gamma_a * P_ESC

pop_slope, pop_intercept = np.polyfit(anxiety_df["S"].values, anxiety_df["response"].values, 1)

calib_list = []
for s, sdf in anxiety_df.groupby("subj"):
    S_s = sdf["S"].values
    danger = 1 - S_s
    anxiety = sdf["response"].values

    if len(S_s) < 3:
        continue

    # Calibration: personal slope of anxiety on danger
    slope_s, _, r_s, _, _ = stats.linregress(danger, anxiety)
    calibration = slope_s

    # Discrepancy: mean residual from population model
    predicted = pop_intercept + pop_slope * S_s
    discrepancy = np.mean(anxiety - predicted)

    calib_list.append({'subj': s, 'calibration': calibration, 'discrepancy': discrepancy})

calib_df = pd.DataFrame(calib_list)
merged = merged.merge(calib_df, on='subj', how='inner')

# Policy alignment ~ calibration
r_cal, p_cal = safe_pearsonr(merged['calibration'].values, merged['policy_alignment'].values)
print(f"\nCalibration → Policy alignment: r={r_cal:.3f}, p={p_cal:.4e}")
results.append(('4_calibration_alignment_r', r_cal, len(merged), p_cal))

# Controlling for ce + cd
Z = np.column_stack([np.log(merged['c_effort'].values), np.log(merged['c_death'].values)])
r_partial, p_partial = safe_pearsonr(
    sm.OLS(merged['calibration'].values, sm.add_constant(Z)).fit().resid,
    sm.OLS(merged['policy_alignment'].values, sm.add_constant(Z)).fit().resid
)
print(f"  Partial (controlling ce, cd): r={r_partial:.3f}, p={p_partial:.4e}")
results.append(('4_calibration_alignment_partial_r', r_partial, len(merged), p_partial))

# Policy alignment ~ discrepancy
r_disc, p_disc = safe_pearsonr(merged['discrepancy'].values, merged['policy_alignment'].values)
print(f"Discrepancy → Policy alignment: r={r_disc:.3f}, p={p_disc:.4e}")
results.append(('4_discrepancy_alignment_r', r_disc, len(merged), p_disc))


# ═══════════════════════════════════════════════════════════════════════
# 5. DISCREPANCY ΔR² ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. DISCREPANCY → RESIDUAL OVERCAUTION ΔR² ROBUSTNESS")
print("=" * 70)

# Compute residual overcaution
log_ce = np.log(merged['c_effort'].values)
log_cd = np.log(merged['c_death'].values)
oc_rate = merged['subj_overcautious_rate'].values
disc = merged['discrepancy'].values

mask_valid = np.isfinite(log_ce) & np.isfinite(log_cd) & np.isfinite(oc_rate) & np.isfinite(disc)
log_ce_v = log_ce[mask_valid]
log_cd_v = log_cd[mask_valid]
oc_v = oc_rate[mask_valid]
disc_v = disc[mask_valid]
N_valid = mask_valid.sum()
print(f"Valid subjects: {N_valid}")

# Base model: overcaution ~ log(ce) + log(cd)
X_base = sm.add_constant(np.column_stack([log_ce_v, log_cd_v]))
base_fit = sm.OLS(oc_v, X_base).fit()
r2_base = base_fit.rsquared
print(f"Base model R²: {r2_base:.4f}")

# Full model: overcaution ~ log(ce) + log(cd) + discrepancy
X_full = sm.add_constant(np.column_stack([log_ce_v, log_cd_v, disc_v]))
full_fit = sm.OLS(oc_v, X_full).fit()
r2_full = full_fit.rsquared
dR2 = r2_full - r2_base
print(f"Full model R²: {r2_full:.4f}")
print(f"ΔR²: {dR2:.4f}")
print(f"Discrepancy coef: {full_fit.params[3]:.4f}, p={full_fit.pvalues[3]:.4e}")
results.append(('5_dR2_point', dR2, N_valid, full_fit.pvalues[3]))

# 5a. Bootstrap 95% CI for ΔR²
print("\n--- Bootstrap 95% CI for ΔR² ---")
np.random.seed(42)
n_boot = 1000
dR2_boot = np.zeros(n_boot)

for b in range(n_boot):
    idx = np.random.choice(N_valid, size=N_valid, replace=True)
    oc_b = oc_v[idx]
    X_base_b = X_base[idx]
    X_full_b = X_full[idx]

    try:
        r2_base_b = sm.OLS(oc_b, X_base_b).fit().rsquared
        r2_full_b = sm.OLS(oc_b, X_full_b).fit().rsquared
        dR2_boot[b] = r2_full_b - r2_base_b
    except:
        dR2_boot[b] = np.nan

dR2_boot = dR2_boot[np.isfinite(dR2_boot)]
ci_lo, ci_hi = np.percentile(dR2_boot, [2.5, 97.5])
print(f"  ΔR² = {dR2:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Bootstrap mean: {np.mean(dR2_boot):.4f}, sd: {np.std(dR2_boot):.4f}")
ci_excludes_zero = ci_lo > 0
print(f"  CI excludes zero: {ci_excludes_zero}")
results.append(('5_dR2_CI_lo', ci_lo, n_boot, np.nan))
results.append(('5_dR2_CI_hi', ci_hi, n_boot, np.nan))

# 5b. Bonferroni correction (if testing 5 outcomes)
p_disc_oc = full_fit.pvalues[3]
p_bonferroni = min(p_disc_oc * 5, 1.0)
print(f"\n--- Bonferroni correction (5 tests) ---")
print(f"  Raw p: {p_disc_oc:.4e}")
print(f"  Bonferroni-corrected p: {p_bonferroni:.4e}")
print(f"  Survives Bonferroni at α=0.05: {p_bonferroni < 0.05}")
results.append(('5_bonferroni_p', p_bonferroni, 5, np.nan))

# 5c. Minimum N for 80% power
# Use f² = ΔR² / (1 - R²_full) for the incremental effect
f2 = dR2 / (1 - r2_full) if r2_full < 1 else np.nan
print(f"\n--- Power analysis ---")
print(f"  Effect size f² = {f2:.4f}")

# Power formula for hierarchical regression:
# N_needed ≈ (z_alpha + z_beta)² / f² + k + 1
# For f-test with 1 numerator df, 3 total predictors
from scipy.stats import f as f_dist, norm
alpha_power = 0.05
beta = 0.20  # 80% power
z_alpha = norm.ppf(1 - alpha_power / 2)
z_beta = norm.ppf(1 - beta)

# More accurate: use noncentral F
# But simpler approximation: N ≈ (L / f²) + k + 1
# where L depends on alpha, power, df1=1
# L ≈ 7.85 for alpha=0.05, power=0.80, df1=1 (from Cohen's table)
L_val = 7.85
N_needed = int(np.ceil(L_val / f2)) + 3 + 1 if f2 > 0 else np.inf
print(f"  Approximate N needed for 80% power: {N_needed}")
results.append(('5_f_squared', f2, np.nan, np.nan))
results.append(('5_N_needed_80pct', N_needed, np.nan, np.nan))


# ═══════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, hspace=0.4, wspace=0.35)

# Panel A: Press rate → survival (attack trials)
ax1 = fig.add_subplot(gs[0, 0])
for i, t in enumerate([0.1, 0.5, 0.9]):
    sub = attack_trials[attack_trials['T_round'] == t].dropna(subset=['median_press_rate'])
    # Bin press rates
    bins = pd.qcut(sub['median_press_rate'], 5, duplicates='drop')
    grouped = sub.groupby(bins, observed=True).agg(
        rate=('median_press_rate', 'mean'),
        surv=('survived', 'mean'),
        n=('survived', 'count')
    ).dropna()
    ax1.plot(grouped['rate'], grouped['surv'], 'o-', label=f'T={t}', markersize=5)

ax1.set_xlabel('Median press rate (normalized)')
ax1.set_ylabel('P(survival)')
ax1.set_title('A. Press rate → survival\n(attack trials)')
ax1.legend(fontsize=8)

# Panel B: Model comparison BIC
ax2 = fig.add_subplot(gs[0, 1])
model_names = ['EVC\n(baseline)', 'Utility\ncurvature', 'Heuristic\nthreshold', 'Effort\nonly', 'RL']
bics = [bic_base, bic_alpha, bic_heur, bic_effort, bic_rl]
colors = ['#2ca02c' if b == min(bics) else '#1f77b4' for b in bics]
bars = ax2.bar(range(len(bics)), [b - min(bics) for b in bics], color=colors)
ax2.set_xticks(range(len(bics)))
ax2.set_xticklabels(model_names, fontsize=7)
ax2.set_ylabel('ΔBIC (from best)')
ax2.set_title('B. Model comparison')
ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.text(4.5, 12, 'ΔBIC=10', fontsize=7, color='red', alpha=0.7)

# Panel C: ce by threat level
ax3 = fig.add_subplot(gs[0, 2])
positions = [1, 2, 3]
bp = ax3.boxplot([ce_01, ce_05, ce_09], positions=positions, widths=0.5,
                  patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], ['#2ca02c', '#ff7f0e', '#d62728']):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)
ax3.set_xticks(positions)
ax3.set_xticklabels(['T=0.1', 'T=0.5', 'T=0.9'])
ax3.set_ylabel('Fitted ce')
ax3.set_title(f'C. ce by threat level\nFriedman p={p_f:.3e}')

# Panel D: Policy alignment distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(merged['policy_alignment'], bins=20, edgecolor='white', alpha=0.7, color='#1f77b4')
ax4.axvline(merged['policy_alignment'].mean(), color='red', linestyle='--', linewidth=1.5)
ax4.set_xlabel('Policy alignment')
ax4.set_ylabel('Count')
ax4.set_title(f'D. Policy alignment\nmean={merged["policy_alignment"].mean():.3f}')

# Panel E: Calibration → policy alignment
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(merged['calibration'], merged['policy_alignment'], alpha=0.3, s=15, c='#1f77b4')
# Regression line
m_cal = np.isfinite(merged['calibration']) & np.isfinite(merged['policy_alignment'])
if m_cal.sum() > 10:
    z = np.polyfit(merged.loc[m_cal, 'calibration'], merged.loc[m_cal, 'policy_alignment'], 1)
    x_line = np.linspace(merged['calibration'].min(), merged['calibration'].max(), 100)
    ax5.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2)
ax5.set_xlabel('Calibration')
ax5.set_ylabel('Policy alignment')
ax5.set_title(f'E. Calibration → alignment\nr={r_cal:.3f}, p={p_cal:.3e}')

# Panel F: Bootstrap ΔR² distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(dR2_boot, bins=40, edgecolor='white', alpha=0.7, color='#ff7f0e')
ax6.axvline(dR2, color='red', linestyle='--', linewidth=1.5, label=f'Observed: {dR2:.4f}')
ax6.axvline(ci_lo, color='blue', linestyle=':', linewidth=1, label=f'95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]')
ax6.axvline(ci_hi, color='blue', linestyle=':', linewidth=1)
ax6.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax6.set_xlabel('ΔR² (discrepancy increment)')
ax6.set_ylabel('Count')
ax6.set_title(f'F. Bootstrap ΔR² distribution\nN={n_boot} samples')
ax6.legend(fontsize=7)

# Panel G: Empirical escape rates by T×D
ax7 = fig.add_subplot(gs[2, 0])
for d in [1, 2, 3]:
    esc_rates = []
    for t in [0.1, 0.5, 0.9]:
        sub = attack_trials[(attack_trials['T_round'] == t)]
        sub_d = sub[sub['startDistance'].map({5: 1, 7: 2, 9: 3}) == d]
        if len(sub_d) > 0:
            esc_rates.append(sub_d['survived'].mean())
        else:
            esc_rates.append(np.nan)
    ax7.plot([0.1, 0.5, 0.9], esc_rates, 'o-', label=f'D={d}', markersize=5)
ax7.set_xlabel('Threat level')
ax7.set_ylabel('P(escape | attack)')
ax7.set_title('G. Empirical escape rates')
ax7.legend(fontsize=8)

# Panel H: RL learning rate distribution
ax8 = fig.add_subplot(gs[2, 1])
alpha_vals = [alpha_rl_dict[s] for s in alpha_rl_dict]
ax8.hist(alpha_vals, bins=20, edgecolor='white', alpha=0.7, color='#2ca02c')
ax8.set_xlabel('Learning rate (α)')
ax8.set_ylabel('Count')
ax8.set_title(f'H. RL learning rates\nmean={np.mean(alpha_vals):.3f}')

# Panel I: Prelec weighting vs power weighting
ax9 = fig.add_subplot(gs[2, 2])
T_fine = np.linspace(0.01, 0.99, 100)
w_power = T_fine ** GAMMA
w_prelec = np.exp(-(-np.log(T_fine))**best_prelec_alpha)
ax9.plot(T_fine, T_fine, 'k--', alpha=0.3, label='Identity')
ax9.plot(T_fine, w_power, 'b-', linewidth=2, label=f'T^γ (γ={GAMMA:.3f})')
ax9.plot(T_fine, w_prelec, 'r-', linewidth=2, label=f'Prelec (α={best_prelec_alpha:.3f})')
ax9.set_xlabel('Objective threat (T)')
ax9.set_ylabel('Subjective weight')
ax9.set_title('I. Probability weighting functions')
ax9.legend(fontsize=8)

fig.suptitle('Critical Reviewer Checks', fontsize=14, fontweight='bold', y=0.98)
plt.savefig(OUT_FIG, bbox_inches='tight', dpi=150)
plt.close()
print(f"Figure saved: {OUT_FIG}")

# ═══════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results, columns=['test', 'value', 'N', 'p_value'])
results_df.to_csv(OUT_STATS, index=False)
print(f"\nResults saved: {OUT_STATS}")

# ── Final summary ──
print("\n" + "=" * 70)
print("SUMMARY OF CRITICAL FINDINGS")
print("=" * 70)

print(f"""
1. PRESS RATE → SURVIVAL:
   Overall point-biserial (attack trials): r={r_overall:.4f}, p={p_overall:.4e}
   Empirical p(escape|attack) = {p_esc_empirical:.4f}
   {'Pressing harder DOES help survival' if (r_overall < -0.05 and p_overall < 0.05) else 'Pressing harder has WEAK/NULL effect on survival — cd captures arousal, not optimization'}

2. γ vs ALTERNATIVES:
   Baseline (γ) BIC = {bic_base:.1f}
   Utility curvature BIC = {bic_alpha:.1f} (ΔBIC = {bic_alpha - bic_base:+.1f}, best alpha={best_alpha:.3f})
   ce varies by T: Friedman p={p_f:.4e}
   {'ce is STABLE across T (γ is not absorbing T-dependent ce)' if p_f > 0.05 else 'ce VARIES by T — γ may partly absorb T-dependent effort sensitivity'}
   Prelec α={best_prelec_alpha:.3f}

3. MODEL COMPARISON:
   Best model: {min(models, key=lambda x: x[2])[0]} (BIC={min(m[2] for m in models):.1f})
   EVC wins over effort-only by ΔBIC={bic_effort - bic_base:+.1f}
   EVC wins over heuristic by ΔBIC={bic_heur - bic_base:+.1f}
   EVC wins over RL by ΔBIC={bic_rl - bic_base:+.1f}

4. POLICY ALIGNMENT:
   Mean = {merged['policy_alignment'].mean():.3f}, sd = {merged['policy_alignment'].std():.3f}
   Calibration → alignment: r={r_cal:.3f}, p={p_cal:.4e}
   Partial (controlling ce,cd): r={r_partial:.3f}, p={p_partial:.4e}

5. ΔR² ROBUSTNESS:
   ΔR² = {dR2:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]
   Bootstrap CI excludes zero: {ci_excludes_zero}
   Bonferroni p = {p_bonferroni:.4e}, survives: {p_bonferroni < 0.05}
   N needed for 80% power: {N_needed}
   f² = {f2:.4f}
""")

print("DONE.")
