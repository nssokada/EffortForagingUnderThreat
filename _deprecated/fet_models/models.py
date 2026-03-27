"""
FET Model Implementations.
"""

import jax.numpy as jnp
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist

# CORRECT IMPORT: single dot means same package
from .base_model import BaseEffortThreatModel


class FETExponential(BaseEffortThreatModel):
    """Exponential effort discount: D(effort) = exp(-k * effort)"""
    
    def discount_function(self, effort, k):
        return jnp.exp(-k * effort)
    
    @property
    def use_effort_component(self):
        return True
    
    @property
    def use_threat_component(self):
        return True


class FETHyperbolic(BaseEffortThreatModel):
    """Hyperbolic effort discount: D(effort) = 1 / (1 + k * effort)"""
    
    def discount_function(self, effort, k):
        return 1 / (1 + k * effort)
    
    @property
    def use_effort_component(self):
        return True
    
    @property
    def use_threat_component(self):
        return True


class FETQuadratic(BaseEffortThreatModel):
    """Quadratic effort discount: D(effort) = max(0, 1 - k * effort^2)"""
    
    def discount_function(self, effort, k):
        return jnp.clip(1 - k * jnp.square(effort), 0, 1)
    
    @property
    def use_effort_component(self):
        return True
    
    @property
    def use_threat_component(self):
        return True


class FETLinear(BaseEffortThreatModel):
    """Linear discount with k constrained to [0, 1]."""
    
    def discount_function(self, effort, k):
        return 1 - k * effort
    
    @property
    def use_effort_component(self):
        return True
    
    @property
    def use_threat_component(self):
        return True
    
    def model(self, subj_idx, threat, distance_H, distance_L, effort_H, effort_L,
              choice=None, R_H=5.0, R_L=1.0, C=5.0, n_subjects=None, include_ppc_sites=False):
        """Override to constrain k to [0, 1]."""
        if n_subjects is None:
            raise ValueError("n_subjects must be provided")
        
        mu_z_raw = numpyro.sample('mu_z_raw', dist.Normal(0, 0.5))
        sigma_z = numpyro.sample('sigma_z', dist.HalfNormal(0.3))
        mu_z = numpyro.deterministic('mu_z', jnp.exp(mu_z_raw))
        
        mu_k = numpyro.sample('mu_k', dist.Beta(2, 2))
        sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.15))
        
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0, 0.5))
        tau = numpyro.deterministic('tau', jnp.clip(jnp.exp(tau_raw), 0.1, 10.0))
        
        with numpyro.plate('subjects', n_subjects):
            z_raw = numpyro.sample('z_raw', dist.Normal(0, 1))
            z_log = mu_z_raw + sigma_z * z_raw
            z = numpyro.deterministic('z', jnp.clip(jnp.exp(z_log), 0.1, 3.0))
            
            k_raw = numpyro.sample('k_raw', dist.Normal(0, 1))
            k_logit = jnp.log(mu_k / (1 - mu_k)) + sigma_k * k_raw
            k = numpyro.deterministic('k', jnn.sigmoid(k_logit))
        
        z_i = z[subj_idx]
        k_i = k[subj_idx]
        
        discount_H = self.discount_function(effort_H, k_i)
        discount_L = self.discount_function(effort_L, k_i)
        effective_R_H = R_H * discount_H
        effective_R_L = R_L * discount_L
        
        S_u_H = self.survival_function(distance_H, threat, z_i)
        S_u_L = self.survival_function(distance_L, threat, z_i)
        
        SV_H = effective_R_H * S_u_H - (1 - S_u_H) * C
        SV_L = effective_R_L * S_u_L - (1 - S_u_L) * C
        
        if include_ppc_sites:
            numpyro.deterministic('discount_H', discount_H)
            numpyro.deterministic('discount_L', discount_L)
            numpyro.deterministic('S_u_H', S_u_H)
            numpyro.deterministic('S_u_L', S_u_L)
            numpyro.deterministic('SV_H', SV_H)
            numpyro.deterministic('SV_L', SV_L)
            p_high = jnn.sigmoid(jnp.clip((SV_H - SV_L) / tau, -20, 20))
            numpyro.deterministic('p_high', p_high)
        
        SV_diff = jnp.clip(SV_H - SV_L, -20, 20)
        logit_p = jnp.clip(SV_diff / tau, -20, 20)
        
        n_trials = len(threat) if choice is None else len(choice)
        with numpyro.plate('trials', n_trials):
            numpyro.sample('obs', dist.Bernoulli(logits=logit_p), obs=choice)


class FETExponentialBias(BaseEffortThreatModel):
    """Exponential discount + threat-induced choice bias (beta)."""
    
    def discount_function(self, effort, k):
        return jnp.exp(-k * effort)
    
    @property
    def use_effort_component(self):
        return True
    
    @property
    def use_threat_component(self):
        return True
    
    def model(self, subj_idx, threat, distance_H, distance_L, effort_H, effort_L,
              choice=None, R_H=5.0, R_L=1.0, C=5.0, n_subjects=None, include_ppc_sites=False):
        """Model with subject-level k, z, and beta."""
        if n_subjects is None:
            raise ValueError("n_subjects must be provided")
        
        mu_z_raw = numpyro.sample('mu_z_raw', dist.Normal(0, 0.5))
        sigma_z = numpyro.sample('sigma_z', dist.HalfNormal(0.3))
        mu_z = numpyro.deterministic('mu_z', jnp.exp(mu_z_raw))
        
        mu_k_raw = numpyro.sample('mu_k_raw', dist.Normal(0, 0.5))
        sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.3))
        mu_k = numpyro.deterministic('mu_k', jnp.exp(mu_k_raw))
        
        mu_beta_log = numpyro.sample('mu_beta_log', dist.Normal(-0.5, 0.5))
        sigma_beta_log = numpyro.sample('sigma_beta_log', dist.HalfNormal(0.15))  

        
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0, 0.5))
        tau = numpyro.deterministic('tau', jnp.clip(jnp.exp(tau_raw), 0.1, 10.0))
        
        with numpyro.plate('subjects', n_subjects):
            z_raw = numpyro.sample('z_raw', dist.Normal(0, 1))
            z_log = mu_z_raw + sigma_z * z_raw
            z = numpyro.deterministic('z', jnp.clip(jnp.exp(z_log), 0.1, 3.0))
            
            k_raw = numpyro.sample('k_raw', dist.Normal(0, 1))
            k_log = mu_k_raw + sigma_k * k_raw
            k = numpyro.deterministic('k', jnp.clip(jnp.exp(k_log), 0.01, 5.0))
            
            beta_log = numpyro.sample('beta_log', dist.Normal(mu_beta_log, sigma_beta_log))
            beta = numpyro.deterministic('beta', jnp.exp(beta_log))

        
        z_i = z[subj_idx]
        k_i = k[subj_idx]
        beta_i = beta[subj_idx]
        
        discount_H = self.discount_function(effort_H, k_i)
        discount_L = self.discount_function(effort_L, k_i)
        effective_R_H = R_H * discount_H
        effective_R_L = R_L * discount_L
        
        S_u_H = self.survival_function(distance_H, threat, z_i)
        S_u_L = self.survival_function(distance_L, threat, z_i)
        
        SV_H = effective_R_H * S_u_H - (1 - S_u_H) * C
        SV_L = effective_R_L * S_u_L - (1 - S_u_L) * C
        
        SV_diff = jnp.clip(SV_H - SV_L, -20, 20)
        bias_term = beta_i * threat
        logit_p = jnp.clip((SV_diff - bias_term) / tau, -20, 20)

        
        if include_ppc_sites:
            numpyro.deterministic('discount_H', discount_H)
            numpyro.deterministic('discount_L', discount_L)
            numpyro.deterministic('S_u_H', S_u_H)
            numpyro.deterministic('S_u_L', S_u_L)
            numpyro.deterministic('SV_H', SV_H)
            numpyro.deterministic('SV_L', SV_L)
            numpyro.deterministic('bias_term', bias_term)
            p_high = jnn.sigmoid(logit_p)
            numpyro.deterministic('p_high', p_high)
        
        n_trials = len(threat) if choice is None else len(choice)
        with numpyro.plate('trials', n_trials):
            numpyro.sample('obs', dist.Bernoulli(logits=logit_p), obs=choice)


class ThreatOnly(BaseEffortThreatModel):
    """Threat-only ablation model (no effort component)."""
    
    def discount_function(self, effort, k):
        return jnp.ones_like(effort)
    
    @property
    def use_effort_component(self):
        return False
    
    @property
    def use_threat_component(self):
        return True


class EffortOnly(BaseEffortThreatModel):
    """Effort-only ablation model (no threat component)."""
    
    def __init__(self, discount: str = 'exponential'):
        super().__init__()
        if discount not in ['exponential', 'linear', 'quadratic', 'hyperbolic']:
            raise ValueError(f"Unknown discount type: {discount}")
        self.discount_type = discount
    
    def discount_function(self, effort, k):
        if self.discount_type == 'exponential':
            return jnp.exp(-k * effort)
        elif self.discount_type == 'linear':
            return jnp.clip(1 - k * effort, 0, 1)
        elif self.discount_type == 'quadratic':
            return jnp.clip(1 - k * jnp.square(effort), 0, 1)
        elif self.discount_type == 'hyperbolic':
            return 1 / (1 + k * effort)
    
    @property
    def use_effort_component(self):
        return True
    
    @property
    def use_threat_component(self):
        return False


MODEL_REGISTRY = {
    'FETExponential': FETExponential,
    'FETHyperbolic': FETHyperbolic,
    'FETQuadratic': FETQuadratic,
    'FETLinear': FETLinear,
    'FETExponentialBias': FETExponentialBias,
    'ThreatOnly': ThreatOnly,
    'EffortOnly': EffortOnly,
}


def get_model(name: str, **kwargs):
    """Get a model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())
