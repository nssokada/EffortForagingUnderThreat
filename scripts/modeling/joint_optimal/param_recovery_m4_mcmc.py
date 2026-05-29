"""
MCMC parameter recovery for the PRODUCTION M4 joint optimal-control model.

This supersedes `param_recovery_v8c_mcmc.py`, whose bespoke `make_v8c` model
DIVERGED from the production `make_m4` that actually converged on real data
(real exploratory M4: max R-hat = 1.0016). The v8c recovery (a) added a
per-subject `baseline` vigor intercept that production M4 does not have
(R-hat 6.9 there), (b) dropped the `- kappa*req*D` total-demand choice term
that gives kappa its identifiability, (c) widened the kappa-SD prior to
HalfNormal(1.5) vs production HalfNormal(0.5), and (d) used a 30-pt grid and a
raw-u survival instead of the production 40-pt grid + saturating-speed survival.

This script instead generates data from, and fits with, the EXACT production
`make_m4` / `eu_sat` (imported from model_comparison_cm.py), using the production
NUTS config. True per-subject (omega, kappa) are drawn from the empirical M4
posterior (mu, sigma, r) in log space; population vigor params are fixed to the
M4 fitted values. Cell/trial design is copied from the real subjects so the
information geometry matches the data that converged.

CLI:
  python scripts/modeling/joint_optimal/param_recovery_m4_mcmc.py \\
      [--n_subj 500] [--num_warmup 2000] [--num_samples 4000] [--num_chains 4] \\
      [--target_accept 0.95] [--max_tree_depth 10] [--seed 42] \\
      [--sample exploratory|confirmatory] [--out_prefix param_recovery_m4_mcmc]

Outputs (prefix `param_recovery_m4_mcmc`): same family as the v8c script
  *.csv (per-subject), *_summary.csv, *_population.csv, *_diagnostics.csv, *_samples.npz
"""

import argparse, sys, os, time, importlib.util
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

# ── Import the PRODUCTION model (make_m4, eu_sat) ──
_cm_path = os.path.join(os.path.dirname(__file__), 'model_comparison_cm.py')
_spec = importlib.util.spec_from_file_location('model_comparison_cm', os.path.abspath(_cm_path))
_cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cm)
make_m4 = _cm.make_m4
eu_sat = _cm.eu_sat
C_PENALTY = _cm.C  # 5.0

OUT_DIR = Path("results/stats/joint_optimal")
MODEL_INPUT_DIR = Path("data/model_input")

# Latent (sampled) site names in make_m4 — everything else is deterministic/observed.
M4_LATENT_SITES = {'gr', 'hr', 'tr', 'sv', 'bc', 'spr', 'mo', 'so', 'mk', 'sk', 'or', 'kr'}


# ============================================================
# Empirical truth: per-subject (omega, kappa) distribution + population vigor params
# ============================================================

def load_empirical_distribution(sample='exploratory'):
    """Read fitted M4 per-subject params; return empirical (mu, sigma, r) in log space."""
    p = Path(f'results/stats/joint_optimal/{sample}/mcmc_m4_params.csv')
    df = pd.read_csv(p)
    log_om = np.log(df['omega'].dropna().clip(lower=1e-9))
    log_kp = np.log(df['kappa'].dropna().clip(lower=1e-9))
    valid = df[['omega', 'kappa']].dropna()
    r_logok = np.log(valid.clip(lower=1e-9)).corr().iloc[0, 1]
    print(f"[empirical M4 posterior — {sample}]  N={len(df)}")
    print(f"  omega: mu_log={log_om.mean():+.4f}, sig_log={log_om.std():.4f}")
    print(f"  kappa: mu_log={log_kp.mean():+.4f}, sig_log={log_kp.std():.4f}")
    print(f"  r(log omega, log kappa) = {r_logok:.4f}")
    return {'mu_log_om': float(log_om.mean()), 'sig_log_om': float(log_om.std()),
            'mu_log_kap': float(log_kp.mean()), 'sig_log_kap': float(log_kp.std()),
            'r_logok': float(r_logok)}


def load_pop_vigor_truth(sample='exploratory'):
    """Read fitted M4 population vigor params (gamma, hazard, tau, sv, bc, sp) from diagnostics."""
    p = Path(f'results/stats/joint_optimal/{sample}/mcmc_convergence_diagnostics.csv')
    d = pd.read_csv(p)
    m4 = d[d['model'] == 'M4'].set_index('parameter')['mean']
    pop = {
        'gamma':  float(np.clip(np.exp(np.log(m4['gamma'])), .1, 3.)) if 'gamma' in m4 else float(m4['gamma']),
        'hazard': float(m4['hazard']),
        'tau':    float(np.clip(np.exp(m4['tr']), .01, 50.)),
        'sv':     float(m4['sv']),
        'bc':     float(m4['bc']),
        'sp':     float(np.clip(np.exp(m4['spr']), .01, 1.)),
    }
    # gamma is stored as a deterministic mean already; use it directly
    pop['gamma'] = float(m4['gamma'])
    print(f"[M4 population vigor truth — {sample}]")
    for k, v in pop.items():
        print(f"  {k} = {v:+.4f}")
    return pop


def sample_correlated_params(N, emp, rng):
    """Sample (log omega, log kappa) from bivariate normal matching empirical (mu, sigma, r)."""
    mu = np.array([emp['mu_log_om'], emp['mu_log_kap']])
    so, sk = emp['sig_log_om'], emp['sig_log_kap']
    r = emp['r_logok']
    cov = np.array([[so**2, r*so*sk], [r*so*sk, sk**2]])
    lp = rng.multivariate_normal(mu, cov, size=N)
    return np.exp(lp[:, 0]), np.exp(lp[:, 1])


# ============================================================
# Generative process — IDENTICAL structure to make_m4
# ============================================================

def _eu_sat_np(om, kap, T, D, R, req, pop, ug):
    """Call the production eu_sat with scalar per-subj (om,kap) broadcast over obs arrays.

    eu_sat indexes om[:, None] / kap[:, None], so they must be per-obs arrays.
    """
    n = len(T)
    om = jnp.full(n, float(om)); kap = jnp.full(n, float(kap))
    return eu_sat(om, kap, jnp.asarray(T), jnp.asarray(D), jnp.asarray(R),
                  jnp.asarray(req), pop['gamma'], pop['hazard'], pop['sp'], ug)


def build_templates():
    """Per-real-subject design templates (choice rows + vigor cells) to reuse for synthetic subjects."""
    ch = pd.read_csv(MODEL_INPUT_DIR / "choice_trials.csv")
    vg = pd.read_csv(MODEL_INPUT_DIR / "vigor_cell_means.csv")
    ch_templ = {s: g[['threat', 'distance_H']].to_numpy() for s, g in ch.groupby('subj_idx')}
    vg_templ = {s: g[['T_round', 'actual_dist', 'actual_R', 'actual_req',
                      'is_heavy', 'n_trials']].to_numpy() for s, g in vg.groupby('subj_idx')}
    real_ids = sorted(ch_templ.keys())
    return ch_templ, vg_templ, real_ids


def simulate_data(true_om, true_kp, pop, rng):
    """Generate choice + cell-mean vigor data from the production M4 generative model."""
    ch_templ, vg_templ, real_ids = build_templates()
    n_real = len(real_ids)
    ug = jnp.linspace(.1, 1.5, 40)

    choice_rows, vigor_rows = [], []
    for s in range(len(true_om)):
        om = float(true_om[s]); kap = float(true_kp[s])
        rid = real_ids[s % n_real]

        # ── Choice ── VH = V(W) - kap*req_H*D_H ; VL = V(W) - kap*req_L*D_L
        ct = ch_templ[rid]
        T = ct[:, 0].astype(float); DH = ct[:, 1].astype(float)
        nC = len(ct)
        _, VH_base = _eu_sat_np(om, kap, T, DH, np.full(nC, 5.), np.full(nC, .9), pop, ug)
        _, VL_base = _eu_sat_np(om, kap, T, np.ones(nC), np.full(nC, 1.), np.full(nC, .4), pop, ug)
        VH = np.asarray(VH_base) - kap * 0.9 * DH
        VL = np.asarray(VL_base) - kap * 0.4 * 1.0
        pH = expit(np.clip((VH - VL) / pop['tau'], -20, 20))
        ch = (rng.random(nC) < pH).astype(int)
        for i in range(nC):
            choice_rows.append({'subj': s, 'threat': T[i], 'distance_H': DH[i], 'choice': int(ch[i])})

        # ── Vigor cell means ── rp = u*(W) + bc*cookie ; mean ~ Normal(rp, sv/sqrt(n))
        vt = vg_templ[rid]
        vT = vt[:, 0].astype(float); vD = vt[:, 1].astype(float)
        vR = vt[:, 2].astype(float); vq = vt[:, 3].astype(float)
        vc = vt[:, 4].astype(float); vn = vt[:, 5].astype(float)
        us, _ = _eu_sat_np(om, kap, vT, vD, vR, vq, pop, ug)
        rp = np.asarray(us) + pop['bc'] * vc
        mean_rate = rp + rng.standard_normal(len(vt)) * (pop['sv'] / np.sqrt(np.clip(vn, 1, None)))
        for i in range(len(vt)):
            vigor_rows.append({'subj': s, 'T_round': vT[i], 'actual_dist': vD[i],
                               'actual_R': vR[i], 'actual_req': vq[i], 'is_heavy': vc[i],
                               'mean_rate': mean_rate[i], 'n_trials': vn[i]})

    return pd.DataFrame(choice_rows), pd.DataFrame(vigor_rows)


def prepare_sim_data(choice_df, vigor_df, N_S):
    """Build the KK-keyed data dict that make_m4 expects."""
    return {
        'cs': jnp.array(choice_df['subj'].values),
        'cT': jnp.array(choice_df['threat'].values),
        'cDH': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'cDL': jnp.ones(len(choice_df)),
        'cc': jnp.array(choice_df['choice'].values),
        'vs': jnp.array(vigor_df['subj'].values, dtype=jnp.int32),
        'vT': jnp.array(vigor_df['T_round'].values),
        'vR': jnp.array(vigor_df['actual_R'].values),
        'vq': jnp.array(vigor_df['actual_req'].values),
        'vD': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vr': jnp.array(vigor_df['mean_rate'].values),
        'vc': jnp.array(vigor_df['is_heavy'].values, dtype=jnp.float64),
        'vn': jnp.array(vigor_df['n_trials'].values, dtype=jnp.float64),
        'N_S': N_S, 'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }


# ============================================================
# MCMC with SVI warm-start (same protocol as production run_model_comparison_mcmc)
# ============================================================

def svi_warm_start(model_fn, kw, seed, n_steps=10000):
    print(f"  [SVI warm-start: {n_steps} steps] ...")
    guide = AutoNormal(model_fn)
    svi = SVI(model_fn, guide, numpyro.optim.ClippedAdam(step_size=0.001, clip_norm=10.0), Trace_ELBO())
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
    return {k: v[0] for k, v in samples.items() if k in M4_LATENT_SITES}


def fit_mcmc(model_fn, kw, num_warmup, num_samples, num_chains,
             target_accept_prob, max_tree_depth, seed, svi_init=True):
    init_params = svi_warm_start(model_fn, kw, seed) if svi_init else None
    has_gpu = any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in jax.devices())
    kernel = NUTS(model_fn, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth,
                  **({'init_strategy': init_to_value(values=init_params)} if init_params else {}))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                progress_bar=True, chain_method='vectorized' if has_gpu else 'sequential')
    print(f"\n  [MCMC: NUTS, {num_chains} chains x {num_warmup} warmup + {num_samples} samples,"
          f" target_accept={target_accept_prob}, max_tree_depth={max_tree_depth}]")
    print(f"  [chain_method = {'vectorized (GPU)' if has_gpu else 'sequential (CPU)'}]")
    t0 = time.time()
    mcmc.run(random.PRNGKey(seed + 1000), **kw)
    elapsed = time.time() - t0
    print(f"  [MCMC done in {elapsed / 60:.1f} min]")
    return mcmc, elapsed


# ============================================================
# Convergence (R-hat / ESS) — identical to v8c script
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
    return float(np.sqrt((((n - 1) / n) * W + (1.0 / n) * B) / W))


def _bulk_ess(chains):
    chains = np.array(chains, dtype=np.float64)
    if chains.ndim != 2:
        return np.nan
    n_chains, n_samples = chains.shape
    flat = chains - chains.mean(axis=1, keepdims=True)
    var = flat.var(ddof=1)
    if var < 1e-10:
        return float(n_chains * n_samples)
    max_lag = min(200, n_samples // 4)
    rho_sum = 0.0
    for lag in range(1, max_lag):
        rho = float(np.mean([np.mean(flat[c, :-lag] * flat[c, lag:]) / var for c in range(n_chains)]))
        if rho < 0.05:
            break
        rho_sum += rho
    return float(n_chains * n_samples / (1 + 2 * rho_sum))


def check_convergence(mcmc):
    samples = mcmc.get_samples(group_by_chain=True)
    rows = []
    max_rhat, min_ess = 0.0, float('inf')
    for name, vals in samples.items():
        if vals.ndim == 2:
            rhat, ess = _split_rhat(np.array(vals)), _bulk_ess(np.array(vals))
            rows.append({'parameter': name, 'rhat': rhat, 'ess': ess,
                         'mean': float(np.mean(vals)), 'std': float(np.std(vals))})
            max_rhat = max(max_rhat, rhat); min_ess = min(min_ess, ess)
        elif vals.ndim == 3:
            rhats = [_split_rhat(np.array(vals[:, :, j])) for j in range(vals.shape[2])]
            esses = [_bulk_ess(np.array(vals[:, :, j])) for j in range(vals.shape[2])]
            rows.append({'parameter': f'{name} (max across {vals.shape[2]})',
                         'rhat': float(np.max(rhats)), 'ess': float(np.min(esses)),
                         'mean': float(np.mean(vals)), 'std': float(np.std(vals))})
            max_rhat = max(max_rhat, float(np.max(rhats))); min_ess = min(min_ess, float(np.min(esses)))
    passed = (max_rhat < 1.01) and (min_ess > 400)
    print(f"\n  Convergence: max R-hat = {max_rhat:.4f}, min ESS = {min_ess:.0f}  "
          f"{'PASS' if passed else 'WARN'}")
    return passed, pd.DataFrame(rows)


# ============================================================
# Metrics — identical to v8c script
# ============================================================

def compute_metrics(df):
    out = {}
    for p in ['omega', 'kappa']:
        t, r = df[f'log_{p}_true'].values, df[f'log_{p}_rec'].values
        lo80, hi80 = df[f'log_{p}_lo80'].values, df[f'log_{p}_hi80'].values
        lo95, hi95 = df[f'log_{p}_lo95'].values, df[f'log_{p}_hi95'].values
        out[f'r_{p}'], _ = pearsonr(t, r)
        out[f'rho_{p}'], _ = spearmanr(t, r)
        out[f'rmse_{p}'] = float(np.sqrt(np.mean((t - r) ** 2)))
        out[f'mae_{p}'] = float(np.mean(np.abs(t - r)))
        out[f'cov80_{p}'] = float(np.mean((lo80 <= t) & (t <= hi80)))
        out[f'cov95_{p}'] = float(np.mean((lo95 <= t) & (t <= hi95)))
        n = len(t); k = max(1, n // 10)
        order = np.argsort(t); idx = np.concatenate([order[:k], order[-k:]])
        out[f'rho_extremes_{p}'], _ = spearmanr(t[idx], r[idx])
        rec_top = set(np.argsort(r)[-k:].tolist())
        out[f'top_decile_overlap_{p}'] = float(len(set(order[-k:].tolist()) & rec_top) / k)
    out['cross_talk_om_to_kp'], _ = pearsonr(df['log_omega_true'], df['log_kappa_rec'])
    out['cross_talk_kp_to_om'], _ = pearsonr(df['log_kappa_true'], df['log_omega_rec'])
    return out


def print_metrics(m, label=''):
    print(f"\n--- Metrics ({label}) ---")
    for p in ['omega', 'kappa']:
        print(f"  {p}:  r={m[f'r_{p}']:+.3f}  rho={m[f'rho_{p}']:+.3f}  "
              f"RMSE={m[f'rmse_{p}']:.3f}  cov80={m[f'cov80_{p}']:.2%}  cov95={m[f'cov95_{p}']:.2%}  "
              f"rho_ext={m[f'rho_extremes_{p}']:+.3f}")
    print(f"  cross-talk: r(true om, rec kp)={m['cross_talk_om_to_kp']:+.3f}  "
          f"r(true kp, rec om)={m['cross_talk_kp_to_om']:+.3f}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n_subj', type=int, default=500)
    ap.add_argument('--num_warmup', type=int, default=2000)
    ap.add_argument('--num_samples', type=int, default=4000)
    ap.add_argument('--num_chains', type=int, default=4)
    ap.add_argument('--target_accept', type=float, default=0.95)
    ap.add_argument('--max_tree_depth', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_svi_init', action='store_true')
    ap.add_argument('--sample', default='exploratory', choices=['exploratory', 'confirmatory'])
    ap.add_argument('--out_prefix', default='param_recovery_m4_mcmc')
    args = ap.parse_args()

    if args.num_chains > 1:
        numpyro.set_host_device_count(args.num_chains)

    print("=" * 78)
    print("PARAMETER RECOVERY (MCMC): PRODUCTION M4 Joint Optimal Control Model")
    print(f"  N = {args.n_subj} subjects | empirical (mu,sigma,r) from '{args.sample}/mcmc_m4_params.csv'")
    print(f"  jax devices: {jax.devices()}")
    print("=" * 78)
    t_total = time.time()

    emp = load_empirical_distribution(sample=args.sample)
    pop = load_pop_vigor_truth(sample=args.sample)

    rng = np.random.default_rng(args.seed)
    true_om, true_kp = sample_correlated_params(args.n_subj, emp, rng)

    print(f"\n[Simulating data for N = {args.n_subj} from production make_m4]")
    t0 = time.time()
    choice_df, vigor_df = simulate_data(true_om, true_kp, pop, rng)
    data = prepare_sim_data(choice_df, vigor_df, args.n_subj)
    print(f"  {data['N_choice']} choice trials, {data['N_vigor']} vigor cells  "
          f"(sim {time.time() - t0:.1f}s) | choice heavy-rate {choice_df['choice'].mean():.3f}")

    kw = {k: data[k] for k in _cm.KK}
    model_fn = make_m4(args.n_subj, data['N_choice'], data['N_vigor'])
    mcmc, mcmc_elapsed = fit_mcmc(model_fn, kw, args.num_warmup, args.num_samples, args.num_chains,
                                  args.target_accept, args.max_tree_depth, args.seed,
                                  svi_init=not args.no_svi_init)

    passed_conv, diag_df = check_convergence(mcmc)

    samples = mcmc.get_samples()
    log_om_samp = np.log(np.clip(np.array(samples['omega']), 1e-12, None))   # (n_draws, N_S)
    log_kp_samp = np.log(np.clip(np.array(samples['kappa']), 1e-12, None))

    def pctl(a, q):
        return np.percentile(a, q, axis=0)

    per_subj = pd.DataFrame({
        'subj': np.arange(args.n_subj),
        'log_omega_true': np.log(true_om), 'log_omega_rec': log_om_samp.mean(0),
        'log_omega_lo80': pctl(log_om_samp, 10), 'log_omega_hi80': pctl(log_om_samp, 90),
        'log_omega_lo95': pctl(log_om_samp, 2.5), 'log_omega_hi95': pctl(log_om_samp, 97.5),
        'log_kappa_true': np.log(true_kp), 'log_kappa_rec': log_kp_samp.mean(0),
        'log_kappa_lo80': pctl(log_kp_samp, 10), 'log_kappa_hi80': pctl(log_kp_samp, 90),
        'log_kappa_lo95': pctl(log_kp_samp, 2.5), 'log_kappa_hi95': pctl(log_kp_samp, 97.5),
    })

    metrics = compute_metrics(per_subj)
    print_metrics(metrics, label=f'production M4, N={args.n_subj}')

    mu_om_rec = float(samples['mo'].mean()); sigma_om_rec = float(samples['so'].mean())
    mu_kp_rec = float(samples['mk'].mean()); sigma_kp_rec = float(samples['sk'].mean())
    gamma_rec = float(samples['gamma'].mean()); hazard_rec = float(samples['hazard'].mean())
    print("\n--- Population-level recovery ---")
    print(f"  mu_log(om) : true={emp['mu_log_om']:+.3f}  rec={mu_om_rec:+.3f}")
    print(f"  sig_log(om): true={emp['sig_log_om']:.3f}  rec={sigma_om_rec:.3f}")
    print(f"  mu_log(kp) : true={emp['mu_log_kap']:+.3f}  rec={mu_kp_rec:+.3f}")
    print(f"  sig_log(kp): true={emp['sig_log_kap']:.3f}  rec={sigma_kp_rec:.3f}")
    print(f"  gamma      : true={pop['gamma']:.3f}  rec={gamma_rec:.3f}")
    print(f"  hazard     : true={pop['hazard']:.3f}  rec={hazard_rec:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pfx = args.out_prefix
    per_subj.to_csv(OUT_DIR / f'{pfx}.csv', index=False)
    pd.DataFrame([{**metrics, 'n_subj': args.n_subj, 'num_warmup': args.num_warmup,
                   'num_samples': args.num_samples, 'num_chains': args.num_chains,
                   'svi_init': not args.no_svi_init, 'mcmc_minutes': mcmc_elapsed / 60.0,
                   'convergence_passed': passed_conv}]).to_csv(OUT_DIR / f'{pfx}_summary.csv', index=False)
    pd.DataFrame([
        {'param': 'mu_log_om', 'true': emp['mu_log_om'], 'recovered': mu_om_rec},
        {'param': 'sigma_log_om', 'true': emp['sig_log_om'], 'recovered': sigma_om_rec},
        {'param': 'mu_log_kap', 'true': emp['mu_log_kap'], 'recovered': mu_kp_rec},
        {'param': 'sigma_log_kap', 'true': emp['sig_log_kap'], 'recovered': sigma_kp_rec},
        {'param': 'gamma', 'true': pop['gamma'], 'recovered': gamma_rec},
        {'param': 'hazard', 'true': pop['hazard'], 'recovered': hazard_rec},
    ]).to_csv(OUT_DIR / f'{pfx}_population.csv', index=False)
    diag_df.to_csv(OUT_DIR / f'{pfx}_diagnostics.csv', index=False)
    np.savez_compressed(OUT_DIR / f'{pfx}_samples.npz', log_omega=log_om_samp, log_kappa=log_kp_samp,
                        true_log_omega=np.log(true_om), true_log_kappa=np.log(true_kp))

    print(f"\nSaved {pfx}.csv / _summary / _population / _diagnostics / _samples.npz in {OUT_DIR}")
    print(f"Total wall time: {(time.time() - t_total) / 60:.1f} min")


if __name__ == '__main__':
    main()
