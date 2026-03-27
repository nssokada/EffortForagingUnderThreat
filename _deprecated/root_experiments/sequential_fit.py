import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random
import numpy as np
import pandas as pd
import ast
from scipy.stats import pearsonr

jax.config.update('jax_enable_x64', True)

# ── Load & prep data ──
beh = pd.read_csv('data/exploratory_350/processed/stage5_filtered_data_20260320_191950/behavior_rich.csv')
beh_c = beh[beh['type']==1].copy()

rates = []
for _, row in beh_c.iterrows():
    try:
        press_times = np.array(ast.literal_eval(row['alignedEffortRate']), dtype=float)
    except:
        rates.append(np.nan)
        continue
    ipis = np.diff(press_times)
    ipis = ipis[ipis > 0.01]
    if len(ipis) < 5:
        rates.append(np.nan)
        continue
    rates.append(np.median((1.0 / ipis) / row['calibrationMax']))

beh_c['median_rate'] = rates
beh_c = beh_c.dropna(subset=['median_rate']).copy()

subjects = sorted(beh_c['subj'].unique())
subj_to_idx = {s: i for i, s in enumerate(subjects)}
N_S, N_T = len(subjects), len(beh_c)

subj_idx = jnp.array([subj_to_idx[s] for s in beh_c['subj']])
T = jnp.array(beh_c['threat'].values)
dist_H = jnp.array(beh_c['distance_H'].values, dtype=jnp.float64)
choice = jnp.array(beh_c['choice'].values)
vigor_obs = jnp.array(beh_c['median_rate'].values)
chosen_R = jnp.where(choice == 1, 5.0, 1.0)

print(f"N_subjects={N_S}, N_trials={N_T}")
print(f"Vigor stats: mean={float(vigor_obs.mean()):.3f}, std={float(vigor_obs.std()):.3f}, "
      f"min={float(vigor_obs.min()):.3f}, max={float(vigor_obs.max()):.3f}")

# ── Step 1: Choice-only model ──
def choice_only(subj_idx, T, dist_H, choice=None):
    mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
    mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
    sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
    sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
    tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
    tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
    p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
    p_esc = jax.nn.sigmoid(p_esc_raw)

    with numpyro.plate('subjects', N_S):
        ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
        cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

    c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
    c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
    ce_i = c_effort[subj_idx]
    cd_i = c_death[subj_idx]

    S_full = (1.0 - T) + T * p_esc
    S_stop = 1.0 - T

    eu_H_full = S_full * 5.0 - (1-S_full) * cd_i * 10.0 - ce_i * 0.81 * dist_H
    eu_H_stop = S_stop * 5.0 - (1-S_stop) * cd_i * 10.0
    eu_H = jnp.maximum(eu_H_full, eu_H_stop)

    eu_L_full = S_full * 1.0 - (1-S_full) * cd_i * 6.0 - ce_i * 0.16
    eu_L_stop = S_stop * 1.0 - (1-S_stop) * cd_i * 6.0
    eu_L = jnp.maximum(eu_L_full, eu_L_stop)

    logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
    p_H = jax.nn.sigmoid(logit)

    numpyro.deterministic('c_effort', c_effort)
    numpyro.deterministic('c_death', c_death)
    numpyro.deterministic('p_H', p_H)
    numpyro.deterministic('tau', tau)
    numpyro.deterministic('p_esc', p_esc)

    with numpyro.plate('trials', N_T):
        numpyro.sample('obs', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=choice)

print("\n=== Fitting choice-only model (25000 steps, lr=0.003) ===")
guide = AutoNormal(choice_only)
optimizer = numpyro.optim.Adam(0.003)
svi = SVI(choice_only, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(random.PRNGKey(42), 25000, subj_idx, T, dist_H, choice, progress_bar=True)

# ── Step 2: Extract subject-level parameters ──
predictive = Predictive(choice_only, guide=guide, params=svi_result.params,
                        num_samples=2000, return_sites=['c_effort', 'c_death', 'p_esc', 'tau'])
samples = predictive(random.PRNGKey(99), subj_idx, T, dist_H)

c_effort_subj = np.array(samples['c_effort'].mean(axis=0))  # (N_S,)
c_death_subj = np.array(samples['c_death'].mean(axis=0))    # (N_S,)
p_esc_mean = float(samples['p_esc'].mean())
tau_mean = float(samples['tau'].mean())

print(f"\nExtracted parameters:")
print(f"  p_esc = {p_esc_mean:.4f}")
print(f"  tau   = {tau_mean:.4f}")
print(f"  c_effort: mean={c_effort_subj.mean():.4f}, std={c_effort_subj.std():.4f}")
print(f"  c_death:  mean={c_death_subj.mean():.4f}, std={c_death_subj.std():.4f}")

# Map to trial level
ce_trial = np.array([c_effort_subj[subj_to_idx[s]] for s in beh_c['subj']])
cd_trial = np.array([c_death_subj[subj_to_idx[s]] for s in beh_c['subj']])
T_np = np.array(beh_c['threat'].values)
choice_np = np.array(beh_c['choice'].values)
dist_H_np = np.array(beh_c['distance_H'].values)
vigor_np = np.array(beh_c['median_rate'].values)
R_chosen = np.where(choice_np == 1, 5.0, 1.0)
death_penalty = np.where(choice_np == 1, 10.0, 6.0)

# ── Step 3: Predict vigor out of sample ──
print("\n" + "="*70)
print("SEQUENTIAL TEST: Predict vigor from choice-fitted params (ZERO vigor fitting)")
print("="*70)

# Compute benefit and cost terms
benefit = cd_trial * T_np * p_esc_mean * (R_chosen + death_penalty)
# For cost, use effort cost of the chosen option
effort_chosen = np.where(choice_np == 1, 0.81 * dist_H_np, 0.16)
cost = ce_trial * effort_chosen

# Avoid division issues
benefit = np.clip(benefit, 1e-10, None)
cost = np.clip(cost, 1e-10, None)

# ── Vigor function 1: benefit / (benefit + cost) ──
v1 = benefit / (benefit + cost)

# ── Vigor function 2: sigmoid(log(benefit/cost)) ──
log_ratio = np.log(benefit / cost)
v2 = 1.0 / (1.0 + np.exp(-log_ratio))

# ── Vigor function 3: sqrt(benefit / (benefit + cost)) ──
v3 = np.sqrt(benefit / (benefit + cost))

# ── Vigor function 4: sigmoid((benefit - cost) / median(benefit)) ──
med_ben = np.median(benefit)
v4 = 1.0 / (1.0 + np.exp(-(benefit - cost) / med_ben))

# Also try: just benefit alone (simplest signal)
v5 = benefit

# And: log(benefit) - log(cost)
v6 = np.log(benefit) - np.log(cost)

vigor_funcs = {
    'benefit/(benefit+cost)': v1,
    'sigmoid(log(b/c))': v2,
    'sqrt(b/(b+c))': v3,
    'sigmoid((b-c)/med_b)': v4,
    'benefit_raw': v5,
    'log(b/c)': v6,
}

# Masks
heavy = choice_np == 1
light = choice_np == 0
threat_mask = T_np > 0
safe_mask = T_np == 0

print(f"\nN_heavy={heavy.sum()}, N_light={light.sum()}")
print(f"N_threat={threat_mask.sum()}, N_safe={safe_mask.sum()}")

for name, v_pred in vigor_funcs.items():
    valid = np.isfinite(v_pred)
    print(f"\n--- {name} ---")

    # Overall correlation
    r_all, p_all = pearsonr(v_pred[valid], vigor_np[valid])
    print(f"  Overall:  r={r_all:.4f}, p={p_all:.2e}  (n={valid.sum()})")

    # Within heavy
    m = heavy & valid
    if m.sum() > 10:
        r_h, p_h = pearsonr(v_pred[m], vigor_np[m])
        print(f"  Heavy:    r={r_h:.4f}, p={p_h:.2e}  (n={m.sum()})")

    # Within light
    m = light & valid
    if m.sum() > 10:
        r_l, p_l = pearsonr(v_pred[m], vigor_np[m])
        print(f"  Light:    r={r_l:.4f}, p={p_l:.2e}  (n={m.sum()})")

    # Threat only
    m = threat_mask & valid
    if m.sum() > 10:
        r_t, p_t = pearsonr(v_pred[m], vigor_np[m])
        print(f"  Threat:   r={r_t:.4f}, p={p_t:.2e}  (n={m.sum()})")

    # Safe only
    m = safe_mask & valid
    if m.sum() > 10:
        r_s, p_s = pearsonr(v_pred[m], vigor_np[m])
        print(f"  Safe:     r={r_s:.4f}, p={p_s:.2e}  (n={m.sum()})")

# ── Predicted vs observed by threat|choice ──
print("\n" + "="*70)
print("Predicted vs Observed by Threat x Choice (using benefit/(benefit+cost))")
print("="*70)

df_res = pd.DataFrame({
    'vigor_obs': vigor_np,
    'v1': v1, 'v5': benefit, 'v6': v6,
    'threat': T_np,
    'choice': choice_np,
    'dist_H': dist_H_np,
    'ce': ce_trial,
    'cd': cd_trial,
})

for ch_label, ch_val in [('Heavy', 1), ('Light', 0)]:
    for thr_label, thr_vals in [('Safe', [0.0]), ('Low', [0.1, 0.2]), ('High', [0.3, 0.4, 0.5])]:
        m = (df_res['choice'] == ch_val) & (df_res['threat'].isin(thr_vals))
        if m.sum() == 0:
            continue
        obs_mean = df_res.loc[m, 'vigor_obs'].mean()
        pred_mean = df_res.loc[m, 'v1'].mean()
        print(f"  {ch_label:5s} | {thr_label:4s}: obs={obs_mean:.4f}, pred_v1={pred_mean:.4f}  (n={m.sum()})")

# ── Predicted vs observed by distance (heavy only) ──
print("\n" + "="*70)
print("Predicted vs Observed by Distance (Heavy only, using benefit/(benefit+cost))")
print("="*70)

df_heavy = df_res[df_res['choice'] == 1].copy()
dist_bins = pd.qcut(df_heavy['dist_H'], q=5, duplicates='drop')
for bin_label, group in df_heavy.groupby(dist_bins):
    obs_m = group['vigor_obs'].mean()
    pred_m = group['v1'].mean()
    print(f"  dist_H {str(bin_label):30s}: obs={obs_m:.4f}, pred={pred_m:.4f}  (n={len(group)})")

# ── Key question: does vigor increase with threat? ──
print("\n" + "="*70)
print("CRITICAL: Does observed vigor increase with threat?")
print("="*70)
for ch_label, ch_val in [('Heavy', 1), ('Light', 0)]:
    print(f"\n  {ch_label}:")
    sub = df_res[df_res['choice'] == ch_val]
    for thr in sorted(sub['threat'].unique()):
        m = sub['threat'] == thr
        print(f"    threat={thr:.1f}: obs_vigor={sub.loc[m, 'vigor_obs'].mean():.4f}, "
              f"pred_v1={sub.loc[m, 'v1'].mean():.4f}, n={m.sum()}")

# ── Within-subject correlation ──
print("\n" + "="*70)
print("Within-subject correlation (benefit/(benefit+cost) vs vigor)")
print("="*70)
within_rs = []
for s in subjects:
    m = beh_c['subj'].values == s
    if m.sum() < 5:
        continue
    vp = v1[m]
    vo = vigor_np[m]
    valid = np.isfinite(vp) & np.isfinite(vo)
    if valid.sum() < 5:
        continue
    r, p = pearsonr(vp[valid], vo[valid])
    within_rs.append(r)

within_rs = np.array(within_rs)
print(f"  Mean within-subject r: {within_rs.mean():.4f}")
print(f"  Median within-subject r: {np.median(within_rs):.4f}")
print(f"  % positive: {(within_rs > 0).mean()*100:.1f}%")
print(f"  N subjects: {len(within_rs)}")

# Also do within-subject for log(b/c)
print("\nWithin-subject correlation (log(b/c) vs vigor)")
within_rs2 = []
for s in subjects:
    m = beh_c['subj'].values == s
    if m.sum() < 5:
        continue
    vp = v6[m]
    vo = vigor_np[m]
    valid = np.isfinite(vp) & np.isfinite(vo)
    if valid.sum() < 5:
        continue
    r, p = pearsonr(vp[valid], vo[valid])
    within_rs2.append(r)

within_rs2 = np.array(within_rs2)
print(f"  Mean within-subject r: {within_rs2.mean():.4f}")
print(f"  Median within-subject r: {np.median(within_rs2):.4f}")
print(f"  % positive: {(within_rs2 > 0).mean()*100:.1f}%")

print("\n\nDONE. Sequential test complete.")
