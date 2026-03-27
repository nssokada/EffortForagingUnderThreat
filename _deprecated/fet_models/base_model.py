"""
Base class for Factorized Effort-Threat (FET) choice models.
"""

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

jax.config.update("jax_enable_x64", True)


def _compute_hdi(samples: np.ndarray, hdi_prob: float = 0.94) -> tuple:
    """
    Compute Highest Density Interval (HDI) for posterior samples.
    
    The HDI is the narrowest interval containing the specified probability mass.
    Unlike equal-tailed intervals, HDI ensures all points inside have higher
    density than points outside.
    
    Parameters
    ----------
    samples : np.ndarray
        1D array of posterior samples
    hdi_prob : float
        Probability mass for the interval (default 0.94)
        
    Returns
    -------
    tuple
        (lower, upper) bounds of HDI
    """
    samples = np.asarray(samples).flatten()
    samples = samples[~np.isnan(samples)]
    n = len(samples)
    
    if n == 0:
        return (np.nan, np.nan)
    
    # Sort samples
    sorted_samples = np.sort(samples)
    
    # Number of samples to include in HDI
    n_included = int(np.ceil(hdi_prob * n))
    
    # Find narrowest interval
    n_intervals = n - n_included
    if n_intervals <= 0:
        return (sorted_samples[0], sorted_samples[-1])
    
    # Width of each candidate interval
    interval_widths = sorted_samples[n_included:] - sorted_samples[:n_intervals]
    
    # Find the narrowest one
    min_idx = np.argmin(interval_widths)
    hdi_lower = sorted_samples[min_idx]
    hdi_upper = sorted_samples[min_idx + n_included]
    
    return (float(hdi_lower), float(hdi_upper))


def configure_device(use_gpu: bool = True, gpu_memory_fraction: float = 0.9):
    """
    Configure JAX to use GPU or CPU.
    """
    if use_gpu:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(gpu_memory_fraction)
        try:
            devices = jax.devices('gpu')
            print(f"GPU configured: {devices[0]}")
        except RuntimeError:
            print("No GPU found, using CPU")
            jax.config.update('jax_platform_name', 'cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        jax.config.update('jax_platform_name', 'cpu')
        print("CPU configured")
    
    return jax.devices()[0]


class BaseEffortThreatModel(ABC):
    """
    Abstract base class for hierarchical Bayesian choice models.
    
    Key features:
    - Group-level tau (NOT hierarchical)
    - Subject-level k and z
    - S_u_H and S_u_L exposed for PPC
    """
    
    def __init__(self):
        self.mcmc = None
        self.posterior_samples = None
        self.subjects = None
        self.subj_to_idx = None
        self.n_subjects = None
        self.fit_params = None
        
    @abstractmethod
    def discount_function(self, effort: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @property
    @abstractmethod
    def use_effort_component(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def use_threat_component(self) -> bool:
        pass
    
    @staticmethod
    def survival_function(distance: jnp.ndarray, threat: jnp.ndarray, 
                          z: jnp.ndarray) -> jnp.ndarray:
        """S = exp(-threat * distance^z)"""
        exponent = -(threat * jnp.power(distance, z))
        return jnp.exp(exponent)
    
    def model(self, subj_idx, threat, distance_H, distance_L, effort_H, effort_L,
              choice=None, R_H=5.0, R_L=1.0, C=5.0, n_subjects=None, include_ppc_sites=False):
        """
        Hierarchical Bayesian choice model.
        
        Parameters
        ----------
        include_ppc_sites : bool
            If True, include extra deterministic sites for PPC (S_u_H, S_u_L, etc.)
            Set to False during fitting for faster sampling.
        """
        if n_subjects is None:
            raise ValueError("n_subjects must be provided")
        
        # Population-level hyperpriors
        if self.use_threat_component:
            mu_z_raw = numpyro.sample('mu_z_raw', dist.Normal(0, 0.5))
            sigma_z = numpyro.sample('sigma_z', dist.HalfNormal(0.3))
            mu_z = numpyro.deterministic('mu_z', jnp.exp(mu_z_raw))
        
        if self.use_effort_component:
            mu_k_raw = numpyro.sample('mu_k_raw', dist.Normal(0, 0.5))
            sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.3))
            mu_k = numpyro.deterministic('mu_k', jnp.exp(mu_k_raw))
        
        # tau - GROUP-LEVEL ONLY
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0, 0.5))
        tau = numpyro.deterministic('tau', jnp.clip(jnp.exp(tau_raw), 0.1, 10.0))
        
        # Subject-level parameters
        with numpyro.plate('subjects', n_subjects):
            if self.use_threat_component:
                z_raw = numpyro.sample('z_raw', dist.Normal(0, 1))
                z_log = mu_z_raw + sigma_z * z_raw
                z = numpyro.deterministic('z', jnp.clip(jnp.exp(z_log), 0.1, 3.0))
            else:
                z = None
            
            if self.use_effort_component:
                k_raw = numpyro.sample('k_raw', dist.Normal(0, 1))
                k_log = mu_k_raw + sigma_k * k_raw
                k = numpyro.deterministic('k', jnp.clip(jnp.exp(k_log), 0.01, 5.0))
            else:
                k = None
        
        # Trial-level computations
        z_i = z[subj_idx] if z is not None else None
        k_i = k[subj_idx] if k is not None else None
        
        # Value computation
        if self.use_effort_component and not self.use_threat_component:
            discount_H = self.discount_function(effort_H, k_i)
            discount_L = self.discount_function(effort_L, k_i)
            SV_H = R_H * discount_H
            SV_L = R_L * discount_L
            S_u_H = jnp.ones_like(threat)
            S_u_L = jnp.ones_like(threat)
            
        elif self.use_threat_component and not self.use_effort_component:
            S_u_H = self.survival_function(distance_H, threat, z_i)
            S_u_L = self.survival_function(distance_L, threat, z_i)
            SV_H = R_H * S_u_H - (1 - S_u_H) * C
            SV_L = R_L * S_u_L - (1 - S_u_L) * C
            discount_H = jnp.ones_like(threat)
            discount_L = jnp.ones_like(threat)
            
        else:
            discount_H = self.discount_function(effort_H, k_i)
            discount_L = self.discount_function(effort_L, k_i)
            effective_R_H = R_H * discount_H
            effective_R_L = R_L * discount_L
            S_u_H = self.survival_function(distance_H, threat, z_i)
            S_u_L = self.survival_function(distance_L, threat, z_i)
            SV_H = effective_R_H * S_u_H - (1 - S_u_H) * C
            SV_L = effective_R_L * S_u_L - (1 - S_u_L) * C
        
        # Only add PPC sites when needed (not during fitting)
        if include_ppc_sites:
            numpyro.deterministic('S_u_H', S_u_H)
            numpyro.deterministic('S_u_L', S_u_L)
            numpyro.deterministic('SV_H', SV_H)
            numpyro.deterministic('SV_L', SV_L)
            numpyro.deterministic('discount_H', discount_H)
            numpyro.deterministic('discount_L', discount_L)
            p_high = jax.nn.sigmoid(jnp.clip((SV_H - SV_L) / tau, -20, 20))
            numpyro.deterministic('p_high', p_high)
        
        # Choice probability
        SV_diff = jnp.clip(SV_H - SV_L, -20, 20)
        logit_p = jnp.clip(SV_diff / tau, -20, 20)
        
        n_trials = len(threat) if choice is None else len(choice)
        with numpyro.plate('trials', n_trials):
            numpyro.sample('obs', dist.Bernoulli(logits=logit_p), obs=choice)
            
    def fit(self, data: pd.DataFrame, 
            R_H: float = 5.0, R_L: float = 1.0, C: float = 5.0,
            num_warmup: int = 1000, num_samples: int = 1000,
            num_chains: int = 4, target_accept_prob: float = 0.85,
            max_tree_depth: int = 10, seed: int = 42,
            progress_bar: bool = True):
        """Fit the model using NUTS sampling."""
        
        self.subjects = data['subj'].unique()
        self.n_subjects = len(self.subjects)
        self.subj_to_idx = {s: i for i, s in enumerate(self.subjects)}
        subj_idx = np.array([self.subj_to_idx[s] for s in data['subj']], dtype=int)
        
        self.fit_params = {
            'R_H': R_H, 'R_L': R_L, 'C': C,
            'num_warmup': num_warmup, 'num_samples': num_samples,
            'num_chains': num_chains, 'seed': seed
        }
        
        subj_idx_jax = jnp.array(subj_idx)
        threat = jnp.array(data['threat'].values)
        distance_H = jnp.array(data['distance_H'].values)
        distance_L = jnp.array(data['distance_L'].values)
        effort_H = jnp.array(data['effort_H'].values)
        effort_L = jnp.array(data['effort_L'].values)
        choice = jnp.array(data['choice'].values)
        
        print(f"Fitting {self.__class__.__name__}:")
        print(f"  {self.n_subjects} subjects, {len(choice)} trials")
        print(f"  Device: {jax.devices()[0]}")
        
        kernel = NUTS(self.model, target_accept_prob=target_accept_prob,
                      max_tree_depth=max_tree_depth)
        
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                         num_chains=num_chains, chain_method="vectorized",
                         progress_bar=progress_bar)
        
        rng_key = jax.random.PRNGKey(seed)
        
        # NOTE: include_ppc_sites=False for faster fitting
        self.mcmc.run(rng_key, subj_idx=subj_idx_jax, threat=threat,
                      distance_H=distance_H, distance_L=distance_L,
                      effort_H=effort_H, effort_L=effort_L, choice=choice,
                      R_H=R_H, R_L=R_L, C=C, n_subjects=self.n_subjects,
                      include_ppc_sites=False)
        
        self.posterior_samples = {k: np.array(v) for k, v in self.mcmc.get_samples().items()}
        print("Fitting complete")
        return self
    
    def get_subject_params(self, param_name: str, hdi_prob: float = 0.94) -> pd.DataFrame:
        """
        Get subject-level parameter estimates with HDI.
        
        Parameters
        ----------
        param_name : str
            Parameter name: 'k', 'z', or 'beta'
        hdi_prob : float
            Probability mass for HDI (default 0.94)
            
        Returns
        -------
        DataFrame
            Columns: subject, mean, median, std, hdi_lower, hdi_upper
        """
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted first")
        if param_name not in self.posterior_samples:
            raise ValueError(f"Parameter '{param_name}' not found")
        
        samples = self.posterior_samples[param_name]
        records = []
        for i, subj in enumerate(self.subjects):
            subj_samples = samples[:, i]
            hdi_low, hdi_high = _compute_hdi(subj_samples, hdi_prob)
            records.append({
                'subject': subj,
                'mean': np.mean(subj_samples),
                'median': np.median(subj_samples),
                'std': np.std(subj_samples),
                'hdi_lower': hdi_low,
                'hdi_upper': hdi_high,
            })
        return pd.DataFrame(records)
    
    def get_population_params(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        """
        Get population-level parameter estimates with HDI.
        
        Parameters
        ----------
        hdi_prob : float
            Probability mass for HDI (default 0.94)
            
        Returns
        -------
        DataFrame
            Columns: parameter, mean, median, std, hdi_lower, hdi_upper
        """
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted first")
        
        pop_params = ['tau', 'mu_k', 'sigma_k', 'mu_z', 'sigma_z', 'mu_beta', 'sigma_beta', 'mu_beta_log', 'sigma_beta_log']
        records = []
        for param in pop_params:
            if param in self.posterior_samples:
                samples = self.posterior_samples[param]
                hdi_low, hdi_high = _compute_hdi(samples, hdi_prob)
                records.append({
                    'parameter': param,
                    'mean': np.mean(samples),
                    'median': np.median(samples),
                    'std': np.std(samples),
                    'hdi_lower': hdi_low,
                    'hdi_upper': hdi_high,
                })
        return pd.DataFrame(records)
    
    def summary(self):
        """Print model summary."""
        if self.posterior_samples is None:
            print("Model not fitted yet.")
            return
        
        print(f"\n{'='*60}")
        print(f"Model Summary: {self.__class__.__name__}")
        print(f"{'='*60}")
        
        pop_df = self.get_population_params()
        print("\nPopulation Parameters:")
        for _, row in pop_df.iterrows():
            print(f"  {row['parameter']:12s}: {row['mean']:.3f} [{row['hdi_lower']:.3f}, {row['hdi_upper']:.3f}]")
        
        print(f"\nSubject-level Parameters (n={self.n_subjects}):")
        for param in ['k', 'z', 'beta']:
            if param in self.posterior_samples:
                df = self.get_subject_params(param)
                print(f"  {param}: mean={df['mean'].mean():.3f}, range=[{df['mean'].min():.3f}, {df['mean'].max():.3f}]")
