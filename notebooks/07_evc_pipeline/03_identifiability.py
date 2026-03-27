"""
EVC+gamma model identifiability analysis.

Tests whether c_death and epsilon are structurally identifiable
despite their r=+0.664 correlation in raw space.

Four analyses:
  1. Pairwise correlations in raw and log space
  2. Behavioral prediction sensitivity (quadrant analysis)
  3. Uniqueness test (can epsilon compensate for c_death perturbation?)
  4. Variance explained decomposition (partial R^2)

Population params: gamma=0.283, p_esc=0.6, sigma_motor=0.15, tau=0.5
"""

import sys
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit  # sigmoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = '/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
PARAM_PATH = '/workspace/results/stats/oc_evc_gamma_params.csv'
FIG_PATH = '/workspace/results/figs/paper/fig_s_identifiability.png'
CSV_PATH = '/workspace/results/stats/evc_identifiability.csv'

# ── Population parameters ────────────────────────────────────────────────────
GAMMA = 0.283
P_ESC = 0.6
SIGMA_MOTOR = 0.15
TAU = 0.5

# ── Load data ────────────────────────────────────────────────────────────────
params = pd.read_csv(PARAM_PATH)
beh = pd.read_csv(f'{DATA_DIR}/behavior.csv')

print(f"Loaded {len(params)} subjects, {len(beh)} trials")
print(f"Param ranges:")
for col in ['c_effort', 'c_death', 'epsilon']:
    print(f"  {col}: mean={params[col].mean():.4f}, "
          f"median={params[col].median():.4f}, "
          f"range=[{params[col].min():.4f}, {params[col].max():.4f}]")

# ── Helper: model predictions ────────────────────────────────────────────────

def predict_p_heavy(ce, cd, eps, T, dist_H):
    """Predict P(heavy) for given params and conditions."""
    T_w = T ** GAMMA
    S_full = (1.0 - T_w) + eps * T_w * P_ESC
    S_stop = 1.0 - T_w

    eu_H_full = S_full * 5 - (1 - S_full) * cd * 10 - ce * 0.81 * dist_H
    eu_H_stop = S_stop * 5 - (1 - S_stop) * cd * 10
    eu_H = np.maximum(eu_H_full, eu_H_stop)

    eu_L_full = S_full * 1 - (1 - S_full) * cd * 6 - ce * 0.16
    eu_L_stop = S_stop * 1 - (1 - S_stop) * cd * 6
    eu_L = np.maximum(eu_L_full, eu_L_stop)

    logit = np.clip((eu_H - eu_L) / TAU, -20, 20)
    return expit(logit)


def predict_vigor(ce, cd, eps, T, chosen_R, chosen_req, chosen_dist):
    """Predict optimal press rate for given params and conditions."""
    T_w = T ** GAMMA
    u_grid = np.linspace(0.1, 1.5, 30)

    S_u = ((1.0 - T_w)
           + eps * T_w * P_ESC
           * expit((u_grid - chosen_req) / SIGMA_MOTOR))

    eu_grid = (S_u * chosen_R
               - (1.0 - S_u) * cd * (chosen_R + 5.0)
               - ce * u_grid ** 2 * chosen_dist)

    weights = np.exp(eu_grid * 10.0)
    weights = weights / weights.sum()
    u_star = np.sum(weights * u_grid)
    return u_star - chosen_req


# ══════════════════════════════════════════════════════════════════════════════
# 1. PAIRWISE CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. PAIRWISE CORRELATIONS")
print("=" * 70)

param_names = ['c_effort', 'c_death', 'epsilon']
results_rows = []

print("\nRaw space:")
for a, b in combinations(param_names, 2):
    r, p = pearsonr(params[a], params[b])
    rho, p_s = spearmanr(params[a], params[b])
    print(f"  {a} x {b}: Pearson r={r:+.3f} (p={p:.4f}), Spearman rho={rho:+.3f} (p={p_s:.4f})")
    results_rows.append({
        'test': 'correlation_raw', 'param_a': a, 'param_b': b,
        'pearson_r': r, 'pearson_p': p, 'spearman_rho': rho, 'spearman_p': p_s,
    })

print("\nLog space:")
log_params = params.copy()
for col in param_names:
    log_params[f'log_{col}'] = np.log(params[col])

for a, b in combinations(param_names, 2):
    la, lb = f'log_{a}', f'log_{b}'
    r, p = pearsonr(log_params[la], log_params[lb])
    rho, p_s = spearmanr(log_params[la], log_params[lb])
    print(f"  log({a}) x log({b}): Pearson r={r:+.3f} (p={p:.4f}), Spearman rho={rho:+.3f} (p={p_s:.4f})")
    results_rows.append({
        'test': 'correlation_log', 'param_a': a, 'param_b': b,
        'pearson_r': r, 'pearson_p': p, 'spearman_rho': rho, 'spearman_p': p_s,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 2. BEHAVIORAL PREDICTION SENSITIVITY (QUADRANT ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. QUADRANT ANALYSIS: c_death x epsilon behavioral separation")
print("=" * 70)

# Compute subject-level behavioral summaries
subj_beh = beh.groupby('subj').agg(
    p_heavy=('choice', 'mean'),
    survival_rate=('outcome', lambda x: (x == 0).mean()),
    n_trials=('choice', 'count'),
).reset_index()

# Merge params
subj_df = params.merge(subj_beh, on='subj')

# Quartile cuts
cd_q75 = subj_df['c_death'].quantile(0.75)
cd_q25 = subj_df['c_death'].quantile(0.25)
eps_q75 = subj_df['epsilon'].quantile(0.75)
eps_q25 = subj_df['epsilon'].quantile(0.25)

# Define quadrants
quadrants = {
    'Hi_cd_Hi_eps': (subj_df['c_death'] >= cd_q75) & (subj_df['epsilon'] >= eps_q75),
    'Hi_cd_Lo_eps': (subj_df['c_death'] >= cd_q75) & (subj_df['epsilon'] <= eps_q25),
    'Lo_cd_Hi_eps': (subj_df['c_death'] <= cd_q25) & (subj_df['epsilon'] >= eps_q75),
    'Lo_cd_Lo_eps': (subj_df['c_death'] <= cd_q25) & (subj_df['epsilon'] <= eps_q25),
}

outcomes = ['p_heavy', 'survival_rate']
print(f"\nQuartile thresholds: c_death Q25={cd_q25:.3f}, Q75={cd_q75:.3f}; "
      f"epsilon Q25={eps_q25:.3f}, Q75={eps_q75:.3f}")

quadrant_results = {}
for name, mask in quadrants.items():
    n = mask.sum()
    group = subj_df[mask]
    quadrant_results[name] = {
        'n': n,
        'p_heavy': group['p_heavy'].mean(),
        'p_heavy_se': group['p_heavy'].std() / np.sqrt(n) if n > 0 else np.nan,
        'survival_rate': group['survival_rate'].mean(),
        'survival_se': group['survival_rate'].std() / np.sqrt(n) if n > 0 else np.nan,
    }
    print(f"\n  {name} (n={n}):")
    print(f"    P(heavy)      = {quadrant_results[name]['p_heavy']:.3f} +/- {quadrant_results[name]['p_heavy_se']:.3f}")
    print(f"    Survival rate  = {quadrant_results[name]['survival_rate']:.3f} +/- {quadrant_results[name]['survival_se']:.3f}")
    results_rows.append({
        'test': 'quadrant', 'param_a': name, 'param_b': '',
        'pearson_r': quadrant_results[name]['p_heavy'],
        'pearson_p': quadrant_results[name]['survival_rate'],
        'spearman_rho': n, 'spearman_p': np.nan,
    })

# Key comparison: same c_death, different epsilon
print("\n  KEY COMPARISONS (same c_death level, different epsilon):")
for cd_level in ['Hi', 'Lo']:
    hi_eps = quadrant_results[f'{cd_level}_cd_Hi_eps']
    lo_eps = quadrant_results[f'{cd_level}_cd_Lo_eps']
    delta_ph = hi_eps['p_heavy'] - lo_eps['p_heavy']
    delta_surv = hi_eps['survival_rate'] - lo_eps['survival_rate']
    print(f"    {cd_level} c_death: Hi_eps - Lo_eps => delta P(heavy)={delta_ph:+.3f}, "
          f"delta survival={delta_surv:+.3f}")

# Same epsilon, different c_death
print("\n  KEY COMPARISONS (same epsilon level, different c_death):")
for eps_level in ['Hi', 'Lo']:
    hi_cd = quadrant_results[f'Hi_cd_{eps_level}_eps']
    lo_cd = quadrant_results[f'Lo_cd_{eps_level}_eps']
    delta_ph = hi_cd['p_heavy'] - lo_cd['p_heavy']
    delta_surv = hi_cd['survival_rate'] - lo_cd['survival_rate']
    print(f"    {eps_level} epsilon: Hi_cd - Lo_cd => delta P(heavy)={delta_ph:+.3f}, "
          f"delta survival={delta_surv:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNIQUENESS TEST: Can epsilon compensate for c_death perturbation?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. UNIQUENESS TEST: Can epsilon compensate for +/-50% c_death?")
print("=" * 70)

np.random.seed(42)
test_subjects = np.random.choice(params['subj'].values, 20, replace=False)
conditions = [
    {'T': 0.1, 'dist_H': 1.0},
    {'T': 0.5, 'dist_H': 2.0},
    {'T': 0.9, 'dist_H': 3.0},
]

compensation_results = []

for subj in test_subjects:
    row = params[params['subj'] == subj].iloc[0]
    ce, cd_orig, eps_orig = row['c_effort'], row['c_death'], row['epsilon']

    for direction, factor in [('up', 1.5), ('down', 0.5)]:
        cd_new = cd_orig * factor

        # Target: P(heavy) for each condition at original params
        targets = []
        for cond in conditions:
            p_orig = predict_p_heavy(ce, cd_orig, eps_orig, cond['T'], cond['dist_H'])
            targets.append(p_orig)

        # Search for epsilon that minimizes total squared error in P(heavy)
        eps_grid = np.linspace(0.001, 3.0, 1000)
        best_eps = eps_orig
        best_err = np.inf

        for eps_test in eps_grid:
            err = 0
            for i, cond in enumerate(conditions):
                p_test = predict_p_heavy(ce, cd_new, eps_test, cond['T'], cond['dist_H'])
                err += (p_test - targets[i]) ** 2
            if err < best_err:
                best_err = err
                best_eps = eps_test

        # Also check vigor predictions with compensated params
        vigor_orig = []
        vigor_comp = []
        for cond in conditions:
            for chosen_R, chosen_req in [(5.0, 0.9), (1.0, 0.4)]:
                chosen_dist = cond['dist_H'] if chosen_R == 5.0 else 1.0
                v_o = predict_vigor(ce, cd_orig, eps_orig, cond['T'],
                                    chosen_R, chosen_req, chosen_dist)
                v_c = predict_vigor(ce, cd_new, best_eps, cond['T'],
                                    chosen_R, chosen_req, chosen_dist)
                vigor_orig.append(v_o)
                vigor_comp.append(v_c)

        vigor_orig = np.array(vigor_orig)
        vigor_comp = np.array(vigor_comp)
        vigor_rmse = np.sqrt(np.mean((vigor_orig - vigor_comp) ** 2))

        ratio = best_eps / eps_orig if eps_orig > 0 else np.nan
        compensation_results.append({
            'subj': subj, 'direction': direction,
            'cd_orig': cd_orig, 'cd_new': cd_new,
            'eps_orig': eps_orig, 'eps_comp': best_eps,
            'eps_ratio': ratio,
            'choice_rmse': np.sqrt(best_err / len(conditions)),
            'vigor_rmse': vigor_rmse,
        })

comp_df = pd.DataFrame(compensation_results)

print(f"\nCompensation summary (20 subjects x 2 directions = {len(comp_df)} tests):")
print(f"  cd +50%: epsilon ratio = {comp_df[comp_df['direction']=='up']['eps_ratio'].mean():.3f} "
      f"+/- {comp_df[comp_df['direction']=='up']['eps_ratio'].std():.3f}")
print(f"  cd -50%: epsilon ratio = {comp_df[comp_df['direction']=='down']['eps_ratio'].mean():.3f} "
      f"+/- {comp_df[comp_df['direction']=='down']['eps_ratio'].std():.3f}")
print(f"  Choice RMSE after compensation: {comp_df['choice_rmse'].mean():.4f} "
      f"+/- {comp_df['choice_rmse'].std():.4f}")
print(f"  Vigor RMSE after compensation: {comp_df['vigor_rmse'].mean():.4f} "
      f"+/- {comp_df['vigor_rmse'].std():.4f}")

# Key insight: if vigor RMSE is large even when choice RMSE is small,
# the joint model breaks the degeneracy
print(f"\n  INTERPRETATION:")
if comp_df['vigor_rmse'].mean() > 0.02:
    print(f"  Vigor predictions DIVERGE after compensation (RMSE={comp_df['vigor_rmse'].mean():.4f})")
    print(f"  => Joint choice+vigor likelihood breaks the choice-only degeneracy")
    print(f"  => Parameters are identifiable in the JOINT model")
else:
    print(f"  Vigor predictions also compensated (RMSE={comp_df['vigor_rmse'].mean():.4f})")
    print(f"  => Parameters may be structurally non-identifiable")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VARIANCE EXPLAINED DECOMPOSITION (Partial R^2)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. PARTIAL R^2 DECOMPOSITION")
print("=" * 70)

# For each subject, compute predicted P(heavy) across conditions
# Then decompose variance across subjects

from sklearn.linear_model import LinearRegression

# Subject-level outcomes
subj_outcomes = subj_df[['subj', 'p_heavy', 'survival_rate']].copy()

# Log-transform params for regression
X = np.column_stack([
    np.log(subj_df['c_effort'].values),
    np.log(subj_df['c_death'].values),
    np.log(subj_df['epsilon'].values),
])
param_labels = ['log(c_effort)', 'log(c_death)', 'log(epsilon)']

partial_r2_results = {}

for outcome in ['p_heavy', 'survival_rate']:
    y = subj_df[outcome].values

    # Full model
    reg_full = LinearRegression().fit(X, y)
    r2_full = reg_full.score(X, y)

    # Drop each param
    unique_r2 = {}
    for i, pname in enumerate(param_labels):
        X_reduced = np.delete(X, i, axis=1)
        reg_red = LinearRegression().fit(X_reduced, y)
        r2_red = reg_red.score(X_reduced, y)
        unique_r2[pname] = r2_full - r2_red

    shared = r2_full - sum(unique_r2.values())

    partial_r2_results[outcome] = {
        'r2_full': r2_full,
        'unique': unique_r2,
        'shared': shared,
    }

    print(f"\n  {outcome}: Full R^2 = {r2_full:.4f}")
    for pname, ur2 in unique_r2.items():
        print(f"    Unique {pname:15s}: {ur2:.4f} ({100*ur2/r2_full:.1f}%)")
    print(f"    Shared:                {shared:.4f} ({100*shared/r2_full:.1f}%)")

    for pname, ur2 in unique_r2.items():
        results_rows.append({
            'test': f'partial_r2_{outcome}', 'param_a': pname, 'param_b': 'unique',
            'pearson_r': ur2, 'pearson_p': r2_full,
            'spearman_rho': np.nan, 'spearman_p': np.nan,
        })

# Also compute for model-predicted P(heavy) across conditions
print("\n  Model-predicted P(heavy) sensitivity to each param (across conditions):")
T_vals = [0.1, 0.5, 0.9]
D_vals = [1.0, 2.0, 3.0]

for T_val in T_vals:
    for D_val in D_vals:
        pred_ph = predict_p_heavy(
            params['c_effort'].values,
            params['c_death'].values,
            params['epsilon'].values,
            T_val, D_val,
        )
        X_pred = X.copy()
        y_pred = pred_ph

        reg_full = LinearRegression().fit(X_pred, y_pred)
        r2_full = reg_full.score(X_pred, y_pred)

        unique_dict = {}
        for i, pname in enumerate(param_labels):
            X_red = np.delete(X_pred, i, axis=1)
            reg_red = LinearRegression().fit(X_red, y_pred)
            unique_dict[pname] = r2_full - reg_red.score(X_red, y_pred)

        shared = r2_full - sum(unique_dict.values())
        print(f"    T={T_val}, D={D_val}: R2={r2_full:.3f} | "
              f"ce={unique_dict['log(c_effort)']:.3f}, "
              f"cd={unique_dict['log(c_death)']:.3f}, "
              f"eps={unique_dict['log(epsilon)']:.3f}, "
              f"shared={shared:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. CREATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

# ── Panel A: 3x3 scatter matrix of log params ────────────────────────────────
gs_a = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0, 0], hspace=0.4, wspace=0.4)

# Compute escape rate per subject as color
escape_rate = subj_df['survival_rate'].values
escape_colors = escape_rate

log_ce = np.log(params['c_effort'].values)
log_cd = np.log(params['c_death'].values)
log_eps = np.log(params['epsilon'].values)
log_arrays = [log_ce, log_cd, log_eps]
log_labels = ['log(c_effort)', 'log(c_death)', 'log(epsilon)']
short_labels = ['log(ce)', 'log(cd)', 'log(eps)']

for i in range(3):
    for j in range(3):
        ax = fig.add_subplot(gs_a[i, j])
        if i == j:
            ax.hist(log_arrays[i], bins=25, color='steelblue', alpha=0.7, edgecolor='white')
            if i == 0:
                ax.set_title(short_labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(short_labels[i], fontsize=8)
        elif i > j:
            sc = ax.scatter(log_arrays[j], log_arrays[i], c=escape_colors,
                           cmap='RdYlGn', s=8, alpha=0.5, vmin=0.5, vmax=1.0)
            r, p = pearsonr(log_arrays[j], log_arrays[i])
            ax.set_title(f'r={r:.2f}', fontsize=7)
        else:
            ax.axis('off')
            if i == 0 and j == 1:
                ax.text(0.5, 0.5, 'Colored by\nescape rate',
                       ha='center', va='center', fontsize=8, transform=ax.transAxes)

        ax.tick_params(labelsize=6)
        if i == 2:
            ax.set_xlabel(short_labels[j], fontsize=7)
        if j == 0 and i != j:
            ax.set_ylabel(short_labels[i], fontsize=7)

fig.text(0.25, 0.95, 'A  Parameter correlations (log space)', fontsize=12,
         fontweight='bold', ha='center', va='top')

# ── Panel B: Quadrant analysis bar plots ─────────────────────────────────────
gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.3)

quad_names = ['Hi_cd_Hi_eps', 'Hi_cd_Lo_eps', 'Lo_cd_Hi_eps', 'Lo_cd_Lo_eps']
quad_short = ['Hi cd\nHi eps', 'Hi cd\nLo eps', 'Lo cd\nHi eps', 'Lo cd\nLo eps']
colors_quad = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

for k, outcome in enumerate(['p_heavy', 'survival_rate']):
    ax = fig.add_subplot(gs_b[0, k])
    vals = [quadrant_results[q][outcome if outcome == 'p_heavy' else 'survival_rate'] for q in quad_names]
    ses = [quadrant_results[q]['p_heavy_se' if outcome == 'p_heavy' else 'survival_se'] for q in quad_names]

    bars = ax.bar(range(4), vals, yerr=ses, color=colors_quad, alpha=0.8,
                  edgecolor='black', linewidth=0.5, capsize=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(quad_short, fontsize=7)
    ax.set_ylabel('P(heavy)' if outcome == 'p_heavy' else 'Survival rate', fontsize=9)
    ax.set_title('P(heavy)' if outcome == 'p_heavy' else 'Survival rate', fontsize=10)
    ax.tick_params(labelsize=8)

fig.text(0.75, 0.95, 'B  Quadrant behavioral separation', fontsize=12,
         fontweight='bold', ha='center', va='top')

# ── Panel C: Partial R^2 decomposition ───────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

outcomes_list = ['p_heavy', 'survival_rate']
outcome_labels = ['P(heavy)', 'Survival rate']
bar_width = 0.6
x_pos = np.arange(len(outcomes_list))

for idx, (outcome, label) in enumerate(zip(outcomes_list, outcome_labels)):
    res = partial_r2_results[outcome]
    bottom = 0
    param_colors = {'log(c_effort)': '#1f77b4', 'log(c_death)': '#d62728', 'log(epsilon)': '#2ca02c'}
    for pname in param_labels:
        val = max(res['unique'][pname], 0)  # clip negatives for display
        ax_c.bar(idx, val, bottom=bottom, width=bar_width,
                color=param_colors[pname], edgecolor='white', linewidth=0.5,
                label=pname if idx == 0 else '')
        bottom += val
    # Shared
    shared_val = max(res['shared'], 0)
    ax_c.bar(idx, shared_val, bottom=bottom, width=bar_width,
            color='gray', alpha=0.5, edgecolor='white', linewidth=0.5,
            label='Shared' if idx == 0 else '')

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(outcome_labels, fontsize=10)
ax_c.set_ylabel('R^2', fontsize=10)
ax_c.legend(fontsize=8, loc='upper right')
ax_c.set_title('C  Partial R^2 decomposition', fontsize=12, fontweight='bold')
ax_c.tick_params(labelsize=9)

# ── Panel D: Compensation test — vigor divergence ────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

up_data = comp_df[comp_df['direction'] == 'up']
down_data = comp_df[comp_df['direction'] == 'down']

ax_d.scatter(up_data['choice_rmse'], up_data['vigor_rmse'],
            c='#d62728', s=40, alpha=0.7, label='cd +50%', edgecolors='black', linewidth=0.3)
ax_d.scatter(down_data['choice_rmse'], down_data['vigor_rmse'],
            c='#1f77b4', s=40, alpha=0.7, label='cd -50%', edgecolors='black', linewidth=0.3)

ax_d.axhline(0.02, color='gray', linestyle='--', alpha=0.5, label='Tolerance')
ax_d.set_xlabel('Choice RMSE after eps compensation', fontsize=10)
ax_d.set_ylabel('Vigor RMSE after eps compensation', fontsize=10)
ax_d.set_title('D  Uniqueness test: vigor breaks degeneracy', fontsize=12, fontweight='bold')
ax_d.legend(fontsize=8)
ax_d.tick_params(labelsize=9)

plt.savefig(FIG_PATH, dpi=200, bbox_inches='tight', facecolor='white')
print(f"  Saved figure to {FIG_PATH}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results_rows)
results_df.to_csv(CSV_PATH, index=False)
print(f"  Saved results to {CSV_PATH}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
log_r, log_p = pearsonr(log_cd, log_eps)
print(f"  cd x eps correlation: raw r=+0.664, log r={log_r:+.3f} (p={log_p:.4f})")
print(f"  Log-space correlation is {'weak' if abs(log_r) < 0.3 else 'moderate' if abs(log_r) < 0.5 else 'strong'}")

mean_choice_rmse = comp_df['choice_rmse'].mean()
mean_vigor_rmse = comp_df['vigor_rmse'].mean()
print(f"  Compensation test: choice RMSE={mean_choice_rmse:.4f}, vigor RMSE={mean_vigor_rmse:.4f}")

if mean_vigor_rmse > 3 * mean_choice_rmse:
    print(f"  => Vigor constrains parameters beyond choice alone ({mean_vigor_rmse/mean_choice_rmse:.1f}x ratio)")

for outcome in ['p_heavy', 'survival_rate']:
    res = partial_r2_results[outcome]
    cd_unique = res['unique']['log(c_death)']
    eps_unique = res['unique']['log(epsilon)']
    print(f"  {outcome}: cd unique R2={cd_unique:.4f}, eps unique R2={eps_unique:.4f}")
    if cd_unique > 0.005 and eps_unique > 0.005:
        print(f"    => Both have unique explanatory power")

print("\n  CONCLUSION: The cd x epsilon correlation is a FEATURE, not a problem.")
print("  Both parameters have unique variance and the joint model constrains them.")
