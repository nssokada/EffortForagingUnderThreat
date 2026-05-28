"""
MCMC-based parameter recovery for V8c (M4) joint optimal control model.

Matches the paper's actual M4 inference protocol (NUTS, 4 chains, SVI warm start,
target_accept=0.95). Designed to be run on GPU — auto-detects and uses
`chain_method='vectorized'` to run all chains in parallel on one device.

Improvements over the SVI version (`param_recovery_v8c_full.py`):
  - Uses NUTS (not AutoNormal SVI), matching the actual paper inference.
  - Properly calibrated posterior credible intervals (SVI's AutoNormal is
    known to produce over-confident posteriors; the SVI run shows ~20-30%
    coverage at 80% nominal — MCMC should give ~80% coverage at 80% nominal).
  - Reports R-hat and bulk-ESS convergence diagnostics.
  - Saves raw MCMC samples so downstream analyses can use full posteriors.

CLI:
  python scripts/modeling/joint_optimal/param_recovery_v8c_mcmc.py \\
      [--n_subj 500] [--num_warmup 2000] [--num_samples 4000] [--num_chains 4] \\
      [--seed 42] [--no-svi-init] [--sample empirical|confirmatory]

Outputs:
    results/stats/joint_optimal/param_recovery_v8c_mcmc.csv            — per-subject
    results/stats/joint_optimal/param_recovery_v8c_mcmc_summary.csv    — metrics
    results/stats/joint_optimal/param_recovery_v8c_mcmc_population.csv — pop params
    results/stats/joint_optimal/param_recovery_v8c_mcmc_diagnostics.csv — R-hat, ESS
    results/stats/joint_optimal/param_recovery_v8c_mcmc_samples.npz    — raw samples
"""

import argparse, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value
from jax import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
from pathlib import Path

OUT_DIR = Path("results/stats/joint_optimal")
C_PENALTY = 5.0

# Population parameters from V8c fit (will be overridden by actual fitted values)
POP_PARAMS = {
    'gamma': 0.86,
    'hazard': 0.832,
    'tau': 2.01,
    'sigma_v': 0.12,
    'b_cookie': -0.03,
}

# Trial design (matches real subjects' free-choice trial count)
THREATS = [0.1, 0.5, 0.9]
DIST_H = [1, 2, 3]
N_TRIALS_PER_CELL = 5  # 9 cells × 5 = 45 choice trials/subject
N_VIGOR_PER_SUBJ = 81  # matches actual event-count per subject


# ============================================================
# Generative model (identical to SVI version)
# ============================================================

def exp_survival(u, T, D, gamma, hazard):
    T_w = T ** gamma
    return np.exp(-hazard * T_w * D / np.clip(u, 0.1, None))


def compute_W(u, omega, kappa, T, D, R, req, gamma, hazard):
    S = exp_survival(u, T, D, gamma, hazard)
    return S * R - (1 - S) * omega * (R + C_PENALTY) - kappa * (u - req)**2 * D


def find_ustar(omega, kappa, T, D, R, req, gamma, hazard):
    u_grid = np.linspace(0.1, 1.5, 100)
    W_vals = np.array([compute_W(u, omega, kappa, T, D, R, req, gamma, hazard)
                       for u in u_grid])
    best_idx = np.argmax(W_vals)
    return u_grid[best_idx], W_vals[best_idx]


def load_empirical_distribution(sample='exploratory'):
    """Read fitted M4 per-subject params; return empirical (μ, σ, r) in log-space."""
    p = Path(f'results/stats/joint_optimal/{sample}/mcmc_m4_params.csv')
    df = pd.read_csv(p)
    log_om = np.log(df['omega'].dropna().clip(lower=1e-9))
    log_kp = np.log(df['kappa'].dropna().clip(lower=1e-9))
    valid = df[['omega', 'kappa']].dropna()
    log_pair = np.log(valid.clip(lower=1e-9))
    r_logok = log_pair.corr().iloc[0, 1]
    print(f"[empirical M4 posterior — {sample}]")
    print(f"  N = {len(df)}")
    print(f"  ω: μ_log = {log_om.mean():+.4f},  σ_log = {log_om.std():.4f}")
    print(f"  κ: μ_log = {log_kp.mean():+.4f},  σ_log = {log_kp.std():.4f}")
    print(f"  r(log ω, log κ) = {r_logok:.4f}")
    return {
        'mu_log_om':    float(log_om.mean()),
        'sig_log_om':   float(log_om.std()),
        'mu_log_kap':   float(log_kp.mean()),
        'sig_log_kap':  float(log_kp.std()),
        'r_logok':      float(r_logok),
    }


def sample_correlated_params(N, emp, rng):
    """Sample (log ω, log κ) from bivariate normal matching empirical (μ, σ, r)."""
    mu = np.array([emp['mu_log_om'], emp['mu_log_kap']])
    sig_o, sig_k = emp['sig_log_om'], emp['sig_log_kap']
    r = emp['r_logok']
    cov = np.array([
        [sig_o**2,             r * sig_o * sig_k],
        [r * sig_o * sig_k,    sig_k**2],
    ])
    log_pair = rng.multivariate_normal(mu, cov, size=N)
    return np.exp(log_pair[:, 0]), np.exp(log_pair[:, 1])


def simulate_data(true_omega, true_kappa, true_baseline, rng):
    N_S = len(true_omega)
    gamma = POP_PARAMS['gamma']; hazard = POP_PARAMS['hazard']
    tau = POP_PARAMS['tau']; sigma_v = POP_PARAMS['sigma_v']
    b_cookie = POP_PARAMS['b_cookie']

    choice_records, vigor_records = [], []
    for s in range(N_S):
        omega = true_omega[s]; kappa = true_kappa[s]; base = true_baseline[s]
        for T in THREATS:
            for D_H in DIST_H:
                for _ in range(N_TRIALS_PER_CELL):
                    u_H, V_H = find_ustar(omega, kappa, T, D_H, 5.0, 0.9, gamma, hazard)
                    u_L, V_L = find_ustar(omega, kappa, T, 1.0, 1.0, 0.4, gamma, hazard)
                    chose_heavy = rng.random() < expit((V_H - V_L) / tau)
                    choice_records.append({
                        'subj': s, 'threat': T, 'distance_H': D_H,
                        'choice': int(chose_heavy),
                    })
                    if chose_heavy:
                        u_star, R, req, D, is_h = u_H, 5.0, 0.9, D_H, 1
                    else:
                        u_star, R, req, D, is_h = u_L, 1.0, 0.4, 1.0, 0
                    rate = u_star + base + b_cookie * is_h + rng.standard_normal() * sigma_v
                    vigor_records.append({
                        'subj': s, 'threat': T, 'actual_dist': D,
                        'actual_R': R, 'actual_req': req,
                        'median_rate': rate, 'is_heavy': is_h,
                    })

        n_extra = N_VIGOR_PER_SUBJ - len(THREATS) * len(DIST_H) * N_TRIALS_PER_CELL
        for _ in range(max(0, n_extra)):
            T = THREATS[rng.integers(3)]
            is_h = int(rng.random() < 0.5)
            R, req, D = (5.0, 0.9, float(DIST_H[rng.integers(3)])) if is_h else (1.0, 0.4, 1.0)
            u_star, _ = find_ustar(omega, kappa, T, D, R, req, gamma, hazard)
            rate = u_star + base + b_cookie * is_h + rng.standard_normal() * sigma_v
            vigor_records.append({
                'subj': s, 'threat': T, 'actual_dist': D,
                'actual_R': R, 'actual_req': req,
                'median_rate': rate, 'is_heavy': is_h,
            })

    return pd.DataFrame(choice_records), pd.DataFrame(vigor_records)


def prepare_sim_data(choice_df, vigor_df):
    N_S = choice_df['subj'].nunique()
    return {
        'ch_subj': jnp.array(choice_df['subj'].values),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_D_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_D_L': jnp.ones(len(choice_df)),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array(vigor_df['subj'].values),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['actual_R'].values),
        'vig_req': jnp.array(vigor_df['actual_req'].values),
        'vig_dist': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vig_rate': jnp.array(vigor_df['median_rate'].values),
        'vig_cookie': jnp.array(vigor_df['is_heavy'].values, dtype=jnp.float64),
        'N_S': N_S, 'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }


# ============================================================
# V8c model (identical to SVI version) — same priors as the paper M4 fit
# ============================================================

def exp_survival_jax(u, T, D, gamma, hazard):
    T_w = jnp.power(T, gamma)
    return jnp.exp(-hazard * T_w * D / jnp.clip(u, 0.1, None))


def vigor_eu_exp(omega, kappa, T, D, R, req, gamma, hazard, u_grid):
    u_g = u_grid[None, :]
    S = exp_survival_jax(u_g, T[:, None], D[:, None], gamma, hazard)
    W = (S * R[:, None]
         - (1.0 - S) * omega[:, None] * (R[:, None] + C_PENALTY)
         - kappa[:, None] * (u_g - req[:, None]) ** 2 * D[:, None])
    weights = jax.nn.softmax(W * 20.0, axis=1)
    u_star = jnp.sum(weights * u_g, axis=1)
    V_star = jnp.sum(weights * W, axis=1)
    return u_star, V_star


def make_v8c(N_S, N_ch, N_vig):
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist, vig_rate, vig_cookie):
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard = numpyro.deterministic('hazard', jnp.exp(hz_raw))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(1.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(1.0))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-2.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(1.5))  # widened to allow σ ≈ 1.4
        mu_base = numpyro.sample('mu_base', dist.Normal(0.0, 0.3))
        sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(0.2))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
            base_raw = numpyro.sample('base_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        baseline = mu_base + sigma_base * base_raw
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)
        numpyro.deterministic('baseline', baseline)
        numpyro.deterministic('log_omega', jnp.log(omega))
        numpyro.deterministic('log_kappa', jnp.log(kappa))

        u_grid = jnp.linspace(0.1, 1.5, 30)

        _, V_H = vigor_eu_exp(omega[ch_subj], kappa[ch_subj], ch_T, ch_D_H,
                              jnp.full(N_ch, 5.0), jnp.full(N_ch, 0.9),
                              gamma, hazard, u_grid)
        _, V_L = vigor_eu_exp(omega[ch_subj], kappa[ch_subj], ch_T, ch_D_L,
                              jnp.full(N_ch, 1.0), jnp.full(N_ch, 0.4),
                              gamma, hazard, u_grid)
        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice', N_ch):
            numpyro.sample('obs_ch', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + baseline[vig_subj] + b_cookie * vig_cookie
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


KWARGS_KEYS = ['ch_subj', 'ch_T', 'ch_D_H', 'ch_D_L', 'ch_choice',
               'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
               'vig_rate', 'vig_cookie']


# ============================================================
# MCMC inference with SVI warm-start
# ============================================================

def svi_warm_start(model_fn, kw, seed, n_steps=10000):
    """Short SVI fit to provide good init values for NUTS (M4 needs this)."""
    print(f"  [SVI warm-start: {n_steps} steps] ...")
    guide = AutoNormal(model_fn)
    optimizer = numpyro.optim.ClippedAdam(step_size=0.001, clip_norm=10.0)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kw)
    update_fn = jax.jit(svi.update)
    best_loss, best_params = float('inf'), None
    for i in range(n_steps):
        state, loss = update_fn(state, **kw)
        l = float(loss)
        if l < best_loss and not np.isnan(l):
            best_loss = l; best_params = svi.get_params(state)
    print(f"  [SVI warm-start done: best loss {best_loss:.1f}]")

    samples = Predictive(model_fn, guide=guide, params=best_params,
                         num_samples=1)(random.PRNGKey(seed + 1), **kw)
    # Keep only latent (sampled) sites, drop deterministics and observed
    DROPS = {'gamma', 'hazard', 'omega', 'kappa', 'log_omega', 'log_kappa',
             'baseline', 'obs_ch', 'obs_vig'}
    return {k: v[0] for k, v in samples.items() if k not in DROPS}


def fit_mcmc(model_fn, kw, num_warmup, num_samples, num_chains,
             target_accept_prob, max_tree_depth, seed, svi_init=True):
    init_params = svi_warm_start(model_fn, kw, seed) if svi_init else None

    has_gpu = any('cuda' in str(d).lower() or 'gpu' in str(d).lower()
                  for d in jax.devices())
    if init_params is not None:
        kernel = NUTS(model_fn, target_accept_prob=target_accept_prob,
                      max_tree_depth=max_tree_depth,
                      init_strategy=init_to_value(values=init_params))
    else:
        kernel = NUTS(model_fn, target_accept_prob=target_accept_prob,
                      max_tree_depth=max_tree_depth)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=True,
                chain_method='vectorized' if has_gpu else 'sequential')

    print(f"\n  [MCMC: NUTS, {num_chains} chains × {num_warmup} warmup + {num_samples} samples,"
          f" target_accept={target_accept_prob}, max_tree_depth={max_tree_depth}]")
    print(f"  [chain_method = {'vectorized (GPU)' if has_gpu else 'sequential (CPU)'}]")
    t0 = time.time()
    mcmc.run(random.PRNGKey(seed + 1000), **kw)
    elapsed = time.time() - t0
    print(f"  [MCMC done in {elapsed / 60:.1f} min]")
    return mcmc, elapsed


# ============================================================
# Convergence diagnostics (R-hat / ESS)
# ============================================================

def _split_rhat(chains):
    chains = np.array(chains, dtype=np.float64)
    if chains.ndim != 2 or chains.shape[1] < 4:
        return np.nan
    n_chains, n_samples = chains.shape
    mid = n_samples // 2
    split = np.concatenate([chains[:, :mid], chains[:, mid:2 * mid]], axis=0)
    n = split.shape[1]
    B = n * np.var(split.mean(axis=1), ddof=1)
    W = np.mean(split.var(axis=1, ddof=1))
    if W < 1e-10:
        return 1.0
    var_hat = ((n - 1) / n) * W + (1.0 / n) * B
    return float(np.sqrt(var_hat / W))


def _bulk_ess(chains):
    """Crude bulk ESS via autocorrelation sum (Geyer's initial-positive-sequence)."""
    chains = np.array(chains, dtype=np.float64)
    if chains.ndim != 2:
        return np.nan
    n_chains, n_samples = chains.shape
    flat = chains - chains.mean(axis=1, keepdims=True)
    var = flat.var(ddof=1)
    if var < 1e-10:
        return float(n_chains * n_samples)
    # autocorr per chain, average across chains
    max_lag = min(200, n_samples // 4)
    rho_sum = 0.0
    for lag in range(1, max_lag):
        rho_lag_per_chain = np.array([
            np.mean(flat[c, :-lag] * flat[c, lag:]) / var
            for c in range(n_chains)
        ])
        rho_avg = float(np.mean(rho_lag_per_chain))
        if rho_avg < 0.05:
            break
        rho_sum += rho_avg
    return float(n_chains * n_samples / (1 + 2 * rho_sum))


def check_convergence(mcmc):
    samples = mcmc.get_samples(group_by_chain=True)
    rows = []
    max_rhat, min_ess = 0.0, float('inf')
    for name, vals in samples.items():
        if vals.ndim == 2:  # scalar (chains, samples)
            rhat = _split_rhat(np.array(vals))
            ess  = _bulk_ess(np.array(vals))
            rows.append({'parameter': name, 'rhat': rhat, 'ess': ess,
                         'mean': float(np.mean(vals)), 'std': float(np.std(vals))})
            max_rhat = max(max_rhat, rhat); min_ess = min(min_ess, ess)
        elif vals.ndim == 3:  # per-subject (chains, samples, N_S)
            rhats, esses = [], []
            for j in range(vals.shape[2]):
                rhats.append(_split_rhat(np.array(vals[:, :, j])))
                esses.append(_bulk_ess(np.array(vals[:, :, j])))
            rows.append({'parameter': f'{name} (max across {vals.shape[2]})',
                         'rhat': float(np.max(rhats)),
                         'ess':  float(np.min(esses)),
                         'mean': float(np.mean(vals)),
                         'std':  float(np.std(vals))})
            max_rhat = max(max_rhat, float(np.max(rhats)))
            min_ess  = min(min_ess,  float(np.min(esses)))
    passed = (max_rhat < 1.01) and (min_ess > 400)
    print(f"\n  Convergence: max R-hat = {max_rhat:.4f}, min ESS = {min_ess:.0f}  "
          f"{'PASS' if passed else 'WARN'}")
    return passed, pd.DataFrame(rows)


# ============================================================
# Metrics (identical to SVI version)
# ============================================================

def compute_metrics(df):
    out = {}
    for p in ['omega', 'kappa']:
        t = df[f'log_{p}_true'].values
        r = df[f'log_{p}_rec'].values
        lo80, hi80 = df[f'log_{p}_lo80'].values, df[f'log_{p}_hi80'].values
        lo95, hi95 = df[f'log_{p}_lo95'].values, df[f'log_{p}_hi95'].values

        out[f'r_{p}'], _ = pearsonr(t, r)
        out[f'rho_{p}'], _ = spearmanr(t, r)
        out[f'rmse_{p}'] = float(np.sqrt(np.mean((t - r) ** 2)))
        out[f'mae_{p}']  = float(np.mean(np.abs(t - r)))
        out[f'cov80_{p}'] = float(np.mean((lo80 <= t) & (t <= hi80)))
        out[f'cov95_{p}'] = float(np.mean((lo95 <= t) & (t <= hi95)))

        n = len(t)
        k = max(1, n // 10)
        order_true = np.argsort(t)
        bot = order_true[:k]; top = order_true[-k:]
        idx = np.concatenate([bot, top])
        rho_ext, _ = spearmanr(t[idx], r[idx])
        out[f'rho_extremes_{p}'] = float(rho_ext)
        rec_top = set(np.argsort(r)[-k:].tolist())
        out[f'top_decile_overlap_{p}'] = float(len(set(top.tolist()) & rec_top) / k)

    out['cross_talk_om_to_kp'], _ = pearsonr(df['log_omega_true'], df['log_kappa_rec'])
    out['cross_talk_kp_to_om'], _ = pearsonr(df['log_kappa_true'], df['log_omega_rec'])
    return out


def print_metrics(m, label=''):
    print(f"\n--- Metrics ({label}) ---")
    for p in ['omega', 'kappa']:
        print(f"  {p}:")
        print(f"    r = {m[f'r_{p}']:+.3f},  ρ = {m[f'rho_{p}']:+.3f}")
        print(f"    RMSE = {m[f'rmse_{p}']:.3f},  MAE = {m[f'mae_{p}']:.3f}  (log scale)")
        print(f"    80% coverage = {m[f'cov80_{p}']:.2%},  95% coverage = {m[f'cov95_{p}']:.2%}")
        print(f"    rho_extremes (top/bot 10%) = {m[f'rho_extremes_{p}']:+.3f}")
        print(f"    top-decile overlap = {m[f'top_decile_overlap_{p}']:.2%}")
    print(f"  Cross-talk:")
    print(f"    r(true ω, rec κ) = {m['cross_talk_om_to_kp']:+.3f}")
    print(f"    r(true κ, rec ω) = {m['cross_talk_kp_to_om']:+.3f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_subj', type=int, default=500,
                        help='Number of simulated subjects (default 500)')
    parser.add_argument('--num_warmup', type=int, default=2000,
                        help='NUTS warmup samples per chain (default 2000)')
    parser.add_argument('--num_samples', type=int, default=4000,
                        help='NUTS post-warmup samples per chain (default 4000)')
    parser.add_argument('--num_chains', type=int, default=4,
                        help='Number of NUTS chains (default 4)')
    parser.add_argument('--target_accept', type=float, default=0.95,
                        help='NUTS target accept rate (default 0.95)')
    parser.add_argument('--max_tree_depth', type=int, default=10,
                        help='NUTS max tree depth (default 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default 42)')
    parser.add_argument('--no_svi_init', action='store_true',
                        help='Skip SVI warm-start (M4 may struggle without it)')
    parser.add_argument('--sample', default='exploratory',
                        choices=['exploratory', 'confirmatory'],
                        help='Empirical M4 posterior to draw true (μ,σ,r) from')
    parser.add_argument('--out_prefix', default='param_recovery_v8c_mcmc',
                        help='Output filename prefix (without extension)')
    args = parser.parse_args()

    if args.num_chains > 1:
        numpyro.set_host_device_count(args.num_chains)

    print("=" * 78)
    print(f"PARAMETER RECOVERY (MCMC): V8c / M4 Joint Optimal Control Model")
    print(f"  N = {args.n_subj} subjects, single dataset")
    print(f"  Empirical (μ, σ, r) from '{args.sample}/mcmc_m4_params.csv'")
    print(f"  jax devices: {jax.devices()}")
    print("=" * 78)
    t_total = time.time()

    emp = load_empirical_distribution(sample=args.sample)

    # Generate true parameters
    rng = np.random.default_rng(args.seed)
    true_om, true_kp = sample_correlated_params(args.n_subj, emp, rng)
    true_base = rng.standard_normal(args.n_subj) * 0.1

    print(f"\n[Simulating data for N = {args.n_subj}]")
    t0 = time.time()
    choice_df, vigor_df = simulate_data(true_om, true_kp, true_base, rng)
    data = prepare_sim_data(choice_df, vigor_df)
    print(f"  {data['N_choice']} choice trials, {data['N_vigor']} vigor trials  "
          f"(sim time: {time.time() - t0:.1f}s)")
    print(f"  Choice base rate (heavy): {choice_df['choice'].mean():.3f}")

    # Fit via MCMC
    kw = {k: data[k] for k in KWARGS_KEYS}
    model_fn = make_v8c(args.n_subj, data['N_choice'], data['N_vigor'])
    mcmc, mcmc_elapsed = fit_mcmc(
        model_fn, kw,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains, target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth, seed=args.seed,
        svi_init=not args.no_svi_init,
    )

    # Convergence
    passed_conv, diag_df = check_convergence(mcmc)

    # Per-subject posterior summaries from MCMC samples
    samples = mcmc.get_samples()       # flat: (chains * samples, ...)
    log_om_samp = np.array(samples['log_omega'])   # (n_draws, N_S)
    log_kp_samp = np.array(samples['log_kappa'])

    rec_log_om_mean = log_om_samp.mean(0)
    rec_log_kp_mean = log_kp_samp.mean(0)
    rec_log_om_lo80, rec_log_om_hi80 = np.percentile(log_om_samp, [10, 90], axis=0)
    rec_log_om_lo95, rec_log_om_hi95 = np.percentile(log_om_samp, [2.5, 97.5], axis=0)
    rec_log_kp_lo80, rec_log_kp_hi80 = np.percentile(log_kp_samp, [10, 90], axis=0)
    rec_log_kp_lo95, rec_log_kp_hi95 = np.percentile(log_kp_samp, [2.5, 97.5], axis=0)

    per_subj = pd.DataFrame({
        'subj': np.arange(args.n_subj),
        'log_omega_true': np.log(true_om),
        'log_omega_rec':  rec_log_om_mean,
        'log_omega_lo80': rec_log_om_lo80, 'log_omega_hi80': rec_log_om_hi80,
        'log_omega_lo95': rec_log_om_lo95, 'log_omega_hi95': rec_log_om_hi95,
        'log_kappa_true': np.log(true_kp),
        'log_kappa_rec':  rec_log_kp_mean,
        'log_kappa_lo80': rec_log_kp_lo80, 'log_kappa_hi80': rec_log_kp_hi80,
        'log_kappa_lo95': rec_log_kp_lo95, 'log_kappa_hi95': rec_log_kp_hi95,
    })

    metrics = compute_metrics(per_subj)
    print_metrics(metrics, label=f'MCMC, N = {args.n_subj}')

    # Population-level recovery
    mu_om_rec     = float(samples['mu_om'].mean())
    sigma_om_rec  = float(samples['sigma_om'].mean())
    mu_kp_rec     = float(samples['mu_kap'].mean())
    sigma_kp_rec  = float(samples['sigma_kap'].mean())
    gamma_rec     = float(samples['gamma'].mean())
    hazard_rec    = float(samples['hazard'].mean())

    print("\n--- Population-level recovery ---")
    print(f"  μ_log(ω)  : true = {emp['mu_log_om']:+.3f},  recovered = {mu_om_rec:+.3f}")
    print(f"  σ_log(ω)  : true = {emp['sig_log_om']:.3f},  recovered = {sigma_om_rec:.3f}")
    print(f"  μ_log(κ)  : true = {emp['mu_log_kap']:+.3f},  recovered = {mu_kp_rec:+.3f}")
    print(f"  σ_log(κ)  : true = {emp['sig_log_kap']:.3f},  recovered = {sigma_kp_rec:.3f}")
    print(f"  γ          : true = {POP_PARAMS['gamma']:.3f},  recovered = {gamma_rec:.3f}")
    print(f"  h (hazard) : true = {POP_PARAMS['hazard']:.3f},  recovered = {hazard_rec:.3f}")

    # ── Save outputs ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pfx = args.out_prefix
    per_subj.to_csv(OUT_DIR / f'{pfx}.csv', index=False)
    summary = pd.DataFrame([{
        **metrics, 'n_subj': args.n_subj,
        'num_warmup': args.num_warmup, 'num_samples': args.num_samples,
        'num_chains': args.num_chains, 'svi_init': not args.no_svi_init,
        'mcmc_minutes': mcmc_elapsed / 60.0,
        'convergence_passed': passed_conv,
    }])
    summary.to_csv(OUT_DIR / f'{pfx}_summary.csv', index=False)
    pd.DataFrame([{
        'param': 'mu_log_om',     'true': emp['mu_log_om'],     'recovered': mu_om_rec},
        {'param': 'sigma_log_om', 'true': emp['sig_log_om'],    'recovered': sigma_om_rec},
        {'param': 'mu_log_kap',   'true': emp['mu_log_kap'],    'recovered': mu_kp_rec},
        {'param': 'sigma_log_kap','true': emp['sig_log_kap'],   'recovered': sigma_kp_rec},
        {'param': 'gamma',        'true': POP_PARAMS['gamma'],  'recovered': gamma_rec},
        {'param': 'hazard',       'true': POP_PARAMS['hazard'], 'recovered': hazard_rec},
    ]).to_csv(OUT_DIR / f'{pfx}_population.csv', index=False)
    diag_df.to_csv(OUT_DIR / f'{pfx}_diagnostics.csv', index=False)
    # Raw samples (compressed) — for downstream coverage / posterior-predictive checks
    np.savez_compressed(OUT_DIR / f'{pfx}_samples.npz',
                        log_omega=log_om_samp, log_kappa=log_kp_samp,
                        true_log_omega=np.log(true_om),
                        true_log_kappa=np.log(true_kp))

    print(f"\nSaved:")
    print(f"  per-subject:   {OUT_DIR / f'{pfx}.csv'}")
    print(f"  summary:       {OUT_DIR / f'{pfx}_summary.csv'}")
    print(f"  population:    {OUT_DIR / f'{pfx}_population.csv'}")
    print(f"  diagnostics:   {OUT_DIR / f'{pfx}_diagnostics.csv'}")
    print(f"  raw samples:   {OUT_DIR / f'{pfx}_samples.npz'}")
    print(f"\nTotal wall time: {(time.time() - t_total) / 60:.1f} min")


if __name__ == '__main__':
    main()
