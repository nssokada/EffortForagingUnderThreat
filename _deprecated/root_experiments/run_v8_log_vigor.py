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
N_S = len(subjects)
N_T = len(beh_c)

subj_idx = jnp.array([subj_to_idx[s] for s in beh_c['subj']])
T = jnp.array(beh_c['threat'].values)
dist_H = jnp.array(beh_c['distance_H'].values, dtype=jnp.float64)
choice = jnp.array(beh_c['choice'].values)
vigor_obs = jnp.array(beh_c['median_rate'].values)
chosen_dist = jnp.where(choice == 1, dist_H, 1.0)
chosen_R = jnp.where(choice == 1, 5.0, 1.0)

print(f"N_subjects={N_S}, N_trials={N_T}")
print(f"Vigor obs: mean={float(vigor_obs.mean()):.3f}, std={float(vigor_obs.std()):.3f}, "
      f"min={float(vigor_obs.min()):.3f}, max={float(vigor_obs.max()):.3f}")

# ── Model ──
def model(subj_idx, T, dist_H, chosen_dist, chosen_R, choice, vigor_obs=None, N_S=None):
    if N_S is None:
        N_S = int(subj_idx.max()) + 1

    # Population priors
    mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
    sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(1.0))
    mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
    sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(1.0))

    tau = numpyro.sample('tau', dist.HalfNormal(5.0))
    p_esc = numpyro.sample('p_esc', dist.Beta(2.0, 2.0))
    sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))

    # Subject-level params (softplus to keep positive)
    ce_raw = numpyro.sample('ce_raw', dist.Normal(jnp.zeros(N_S), 1.0))
    cd_raw = numpyro.sample('cd_raw', dist.Normal(jnp.zeros(N_S), 1.0))
    ce_all = jax.nn.softplus(mu_ce + sigma_ce * ce_raw)
    cd_all = jax.nn.softplus(mu_cd + sigma_cd * cd_raw)

    ce_i = ce_all[subj_idx]
    cd_i = cd_all[subj_idx]

    # ── CHOICE MODEL (same as v7) ──
    C_total_H = 6.0 + 5.0   # reward_H + baseline
    C_total_L = 6.0 + 1.0   # reward_L + baseline

    # Heavy
    S_full_H = (1.0 - T) + T * p_esc
    S_stop_H = (1.0 - T)
    eu_H_full = S_full_H * 5.0 - (1.0 - S_full_H) * cd_i * C_total_H - ce_i * 0.81 * dist_H
    eu_H_stop = S_stop_H * 5.0 - (1.0 - S_stop_H) * cd_i * C_total_H
    eu_H = jnp.maximum(eu_H_full, eu_H_stop)

    # Light
    S_full_L = (1.0 - T) + T * p_esc
    S_stop_L = (1.0 - T)
    eu_L_full = S_full_L * 1.0 - (1.0 - S_full_L) * cd_i * C_total_L - ce_i * 0.16 * 1.0
    eu_L_stop = S_stop_L * 1.0 - (1.0 - S_stop_L) * cd_i * C_total_L
    eu_L = jnp.maximum(eu_L_full, eu_L_stop)

    logit_H = (eu_H - eu_L) / tau
    numpyro.sample('choice_obs', dist.Bernoulli(logits=logit_H), obs=choice)

    # ── VIGOR MODEL: log-scaled benefit/cost ──
    C_val = jnp.where(choice == 1, C_total_H, C_total_L)
    benefit = cd_i * T * p_esc * (chosen_R + C_val)
    cost = ce_i
    log_ratio = jnp.log(benefit / (cost + 1e-6) + 1e-6)
    v_pred = jax.nn.sigmoid(log_ratio)

    numpyro.sample('vigor_obs', dist.Normal(v_pred, sigma_v), obs=vigor_obs)


# ── Fit ──
guide = AutoNormal(model)
optim = numpyro.optim.Adam(0.003)
svi = SVI(model, guide, optim, loss=Trace_ELBO())
svi_result = svi.run(random.PRNGKey(0), 25000,
                     subj_idx, T, dist_H, chosen_dist, chosen_R, choice, vigor_obs, N_S,
                     progress_bar=True)

# ── Posterior samples ──
pred = Predictive(guide, params=svi_result.params, num_samples=500)
post = pred(random.PRNGKey(1), subj_idx, T, dist_H, chosen_dist, chosen_R, choice, None, N_S)

# Extract medians
tau_m = float(np.median(post['tau']))
p_esc_m = float(np.median(post['p_esc']))
sigma_v_m = float(np.median(post['sigma_v']))
mu_ce_m = float(np.median(post['mu_ce']))
sigma_ce_m = float(np.median(post['sigma_ce']))
mu_cd_m = float(np.median(post['mu_cd']))
sigma_cd_m = float(np.median(post['sigma_cd']))
ce_raw_m = np.median(post['ce_raw'], axis=0)
cd_raw_m = np.median(post['cd_raw'], axis=0)

from jax.nn import softplus
ce_subj = np.array(softplus(mu_ce_m + sigma_ce_m * ce_raw_m))
cd_subj = np.array(softplus(mu_cd_m + sigma_cd_m * cd_raw_m))

print("\n=== PARAMETERS ===")
print(f"tau = {tau_m:.4f}")
print(f"p_esc = {p_esc_m:.4f}")
print(f"sigma_v = {sigma_v_m:.4f}")
print(f"mu_ce = {mu_ce_m:.4f}, sigma_ce = {sigma_ce_m:.4f}")
print(f"mu_cd = {mu_cd_m:.4f}, sigma_cd = {sigma_cd_m:.4f}")
print(f"c_effort: min={ce_subj.min():.4f}, max={ce_subj.max():.4f}, mean={ce_subj.mean():.4f}, std={ce_subj.std():.4f}")
print(f"c_death:  min={cd_subj.min():.4f}, max={cd_subj.max():.4f}, mean={cd_subj.mean():.4f}, std={cd_subj.std():.4f}")
print(f"c_effort x c_death r = {pearsonr(ce_subj, cd_subj)[0]:.3f}")

# ── Choice predictions ──
ce_i_m = ce_subj[np.array(subj_idx)]
cd_i_m = cd_subj[np.array(subj_idx)]
T_np = np.array(T)
dist_H_np = np.array(dist_H)
choice_np = np.array(choice)
vigor_np = np.array(vigor_obs)

C_total_H = 11.0
C_total_L = 7.0

S_full = (1.0 - T_np) + T_np * p_esc_m
S_stop = (1.0 - T_np)

eu_H_full = S_full * 5.0 - (1.0 - S_full) * cd_i_m * C_total_H - ce_i_m * 0.81 * dist_H_np
eu_H_stop = S_stop * 5.0 - (1.0 - S_stop) * cd_i_m * C_total_H
eu_H = np.maximum(eu_H_full, eu_H_stop)

eu_L_full = S_full * 1.0 - (1.0 - S_full) * cd_i_m * C_total_L - ce_i_m * 0.16
eu_L_stop = S_stop * 1.0 - (1.0 - S_stop) * cd_i_m * C_total_L
eu_L = np.maximum(eu_L_full, eu_L_stop)

from scipy.special import expit
p_H = expit((eu_H - eu_L) / tau_m)
pred_choice = (p_H > 0.5).astype(int)

acc = np.mean(pred_choice == choice_np)
print(f"\n=== CHOICE ACCURACY ===")
print(f"Overall: {acc:.3f} ({acc*100:.1f}%)")

threats = sorted(beh_c['threat'].unique())
for t_val in threats:
    mask = T_np == t_val
    a = np.mean(pred_choice[mask] == choice_np[mask])
    print(f"  Threat={t_val:.1f}: {a:.3f} (n={mask.sum()})")

# ── Vigor predictions ──
chosen_R_np = np.where(choice_np == 1, 5.0, 1.0)
C_val_np = np.where(choice_np == 1, C_total_H, C_total_L)

benefit = cd_i_m * T_np * p_esc_m * (chosen_R_np + C_val_np)
cost = ce_i_m
log_ratio = np.log(benefit / (cost + 1e-6) + 1e-6)
v_pred = expit(log_ratio)

r_all, p_all = pearsonr(v_pred, vigor_np)
print(f"\n=== VIGOR CORRELATION ===")
print(f"Overall: r={r_all:.3f}, p={p_all:.2e}")
print(f"  v_pred: mean={v_pred.mean():.3f}, std={v_pred.std():.3f}, min={v_pred.min():.3f}, max={v_pred.max():.3f}")
print(f"  v_obs:  mean={vigor_np.mean():.3f}, std={vigor_np.std():.3f}, min={vigor_np.min():.3f}, max={vigor_np.max():.3f}")

# Within heavy
h_mask = choice_np == 1
r_h, p_h = pearsonr(v_pred[h_mask], vigor_np[h_mask])
print(f"Within Heavy: r={r_h:.3f}, p={p_h:.2e} (n={h_mask.sum()})")

# Within light
l_mask = choice_np == 0
r_l, p_l = pearsonr(v_pred[l_mask], vigor_np[l_mask])
print(f"Within Light: r={r_l:.3f}, p={p_l:.2e} (n={l_mask.sum()})")

# ── Vigor by threat|choice ──
print(f"\n=== VIGOR BY THREAT|CHOICE ===")
print(f"{'Threat':>6} {'Choice':>6} {'Pred':>8} {'Obs':>8} {'n':>5}")
for t_val in threats:
    for c_val in [1, 0]:
        mask = (T_np == t_val) & (choice_np == c_val)
        if mask.sum() == 0:
            continue
        label = 'Heavy' if c_val == 1 else 'Light'
        print(f"{t_val:6.1f} {label:>6} {v_pred[mask].mean():8.3f} {vigor_np[mask].mean():8.3f} {mask.sum():5d}")

# ── Heavy by distance ──
print(f"\n=== HEAVY BY DISTANCE ===")
print(f"{'Dist':>5} {'Pred':>8} {'Obs':>8} {'n':>5}")
for d in sorted(beh_c.loc[beh_c['choice']==1, 'distance_H'].unique()):
    mask = (choice_np == 1) & (dist_H_np == d)
    if mask.sum() < 5:
        continue
    print(f"{d:5.1f} {v_pred[mask].mean():8.3f} {vigor_np[mask].mean():8.3f} {mask.sum():5d}")
