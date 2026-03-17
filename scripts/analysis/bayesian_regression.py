"""
Bayesian Regression Analysis Module
====================================
A flexible module for running Bayesian multiple regression analyses with PyMC.

Author: Your Name
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class BayesianRegression:
    """
    A flexible Bayesian regression analysis class for multiple outcomes and predictors.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all variables
    outcomes : list
        List of outcome variable names
    predictors : list
        List of predictor variable names
    outcome_labels : list, optional
        Display labels for outcomes
    predictor_labels : list, optional
        Display labels for predictors
    standardize : bool, default=True
        Whether to standardize variables before analysis
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 outcomes: List[str],
                 predictors: List[str],
                 outcome_labels: Optional[List[str]] = None,
                 predictor_labels: Optional[List[str]] = None,
                 standardize: bool = True):
        
        self.data = data.copy()
        self.outcomes = outcomes
        self.predictors = predictors
        self.outcome_labels = outcome_labels or outcomes
        self.predictor_labels = predictor_labels or predictors
        self.standardize = standardize
        
        # Storage for results
        self.models = {}
        self.traces = {}
        self.results_df = None
        self.X_scaled = None
        self.Y_scaled = None
        
        # Validate inputs
        self._validate_inputs()
        
        # Prepare data
        self._prepare_data()
    
    def _validate_inputs(self):
        """Validate that all variables exist in the dataframe."""
        missing_outcomes = [v for v in self.outcomes if v not in self.data.columns]
        missing_predictors = [v for v in self.predictors if v not in self.data.columns]
        
        if missing_outcomes:
            raise ValueError(f"Outcomes not found in data: {missing_outcomes}")
        if missing_predictors:
            raise ValueError(f"Predictors not found in data: {missing_predictors}")
        
        if len(self.outcome_labels) != len(self.outcomes):
            raise ValueError("outcome_labels must have same length as outcomes")
        if len(self.predictor_labels) != len(self.predictors):
            raise ValueError("predictor_labels must have same length as predictors")
    
    def _prepare_data(self):
        """Prepare and optionally standardize the data."""
        # Remove any rows with missing data
        subset = self.outcomes + self.predictors
        self.data_clean = self.data[subset].dropna()
        
        print(f"Total subjects after removing missing data: {len(self.data_clean)}")
        
        # Extract X and Y
        X = self.data_clean[self.predictors].values
        Y = self.data_clean[self.outcomes].values
        
        if self.standardize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            self.X_scaled = scaler_x.fit_transform(X)
            self.Y_scaled = scaler_y.fit_transform(Y)
        else:
            self.X_scaled = X
            self.Y_scaled = Y
    
    def _build_model(self, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """
        Build a Bayesian regression model.
        
        Parameters
        ----------
        X : np.ndarray
            Predictor matrix
        y : np.ndarray
            Outcome vector
        
        Returns
        -------
        pm.Model
            PyMC model object
        """
        n_obs, n_pred = X.shape
        
        with pm.Model() as model:
            # Priors
            intercept = pm.Normal('intercept', mu=0, sigma=1)
            betas = pm.Normal('betas', mu=0, sigma=1, shape=n_pred)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear model
            mu = intercept + pm.math.dot(X, betas)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        return model
    
    def fit(self, 
            draws: int = 2000, 
            tune: int = 2000, 
            chains: int = 4,
            target_accept: float = 0.9,
            progressbar: bool = True,
            verbose: bool = True):
        """
        Fit Bayesian regression models for all outcomes.
        
        Parameters
        ----------
        draws : int, default=2000
            Number of sampling draws
        tune : int, default=2000
            Number of tuning steps
        chains : int, default=4
            Number of MCMC chains
        target_accept : float, default=0.9
            Target acceptance probability
        progressbar : bool, default=True
            Whether to show progress bar
        verbose : bool, default=True
            Whether to print progress messages
        """
        posterior_results = []
        
        if verbose:
            print("\n" + "="*60)
            print("FITTING BAYESIAN REGRESSION MODELS")
            print("="*60)
        
        # Fit a model for each outcome
        for i, (outcome, outcome_label) in enumerate(zip(self.outcomes, self.outcome_labels)):
            if verbose:
                print(f"\nFitting model: {outcome_label} ~ {' + '.join(self.predictor_labels)}")
            
            # Build and sample model
            model = self._build_model(self.X_scaled, self.Y_scaled[:, i])
            
            with model:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    return_inferencedata=True,
                    progressbar=progressbar
                )
                
                # Add posterior predictive
                pm.sample_posterior_predictive(trace, extend_inferencedata=True, progressbar=False)
            
            self.models[outcome_label] = model
            self.traces[outcome_label] = trace
            
            # Extract posterior statistics
            for j, predictor_label in enumerate(self.predictor_labels):
                posterior_samples = trace.posterior['betas'][:, :, j].values.flatten()
                
                mean_beta = np.mean(posterior_samples)
                sd_beta = np.std(posterior_samples)
                hdi = az.hdi(posterior_samples, hdi_prob=0.94)
                
                prob_positive = np.mean(posterior_samples > 0)
                prob_negative = np.mean(posterior_samples < 0)
                prob_nonzero = max(prob_positive, prob_negative)
                
                posterior_results.append({
                    'outcome': outcome_label,
                    'predictor': predictor_label,
                    'beta_mean': mean_beta,
                    'beta_sd': sd_beta,
                    'hdi_lower': hdi[0],
                    'hdi_upper': hdi[1],
                    'prob_positive': prob_positive,
                    'prob_negative': prob_negative,
                    'prob_nonzero': prob_nonzero,
                    'posterior_samples': posterior_samples
                })
        
        self.results_df = pd.DataFrame(posterior_results)
        
        if verbose:
            self._print_convergence_diagnostics()
    
    def _print_convergence_diagnostics(self):
        """Print convergence diagnostics for all models."""
        print("\n" + "="*60)
        print("MODEL CONVERGENCE DIAGNOSTICS")
        print("="*60)
        
        for outcome_label in self.outcome_labels:
            trace = self.traces[outcome_label]
            summary = az.summary(trace, var_names=['betas', 'sigma'])
            max_rhat = summary['r_hat'].max()
            min_ess = min(summary['ess_bulk'].min(), summary['ess_tail'].min())
            
            print(f"\n{outcome_label} model:")
            print(f"  Max R-hat: {max_rhat:.3f} (should be < 1.01)")
            print(f"  Min ESS: {min_ess:.0f} (should be > 400)")
            
            if max_rhat > 1.01:
                print("  ⚠️ Warning: Convergence issues detected")
    
    def plot_coefficients(self, 
                         figsize: Tuple[int, int] = (12, 7),
                         colors: Optional[Dict[str, str]] = None,
                         rope: float = 0.1,
                         save_path: Optional[str] = None):
        """
        Create a forest plot of regression coefficients.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 7)
            Figure size
        colors : dict, optional
            Dictionary mapping predictor labels to colors
        rope : float, default=0.1
            Region of Practical Equivalence (±rope)
        save_path : str, optional
            Path to save the figure
        """
        if self.results_df is None:
            raise ValueError("Must call fit() before plotting")
        
        # Default colors
        if colors is None:
            color_palette = sns.color_palette("husl", len(self.predictor_labels))
            colors = {pred: color for pred, color in zip(self.predictor_labels, color_palette)}
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot settings
        x_positions = np.arange(len(self.outcomes))
        width = 0.8 / len(self.predictors)  # Dynamic width based on number of predictors
        
        # Create grouped plot
        for j, predictor_label in enumerate(self.predictor_labels):
            x_pos = x_positions + (j - len(self.predictors)/2 + 0.5) * width
            
            # Get data for this predictor
            predictor_data = self.results_df[self.results_df['predictor'] == predictor_label]
            
            means = []
            lower_bounds = []
            upper_bounds = []
            prob_nonzero = []
            
            for outcome_label in self.outcome_labels:
                row = predictor_data[predictor_data['outcome'] == outcome_label].iloc[0]
                means.append(row['beta_mean'])
                lower_bounds.append(row['hdi_lower'])
                upper_bounds.append(row['hdi_upper'])
                prob_nonzero.append(row['prob_nonzero'])
            
            means = np.array(means)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            
            # Calculate error bars
            yerr_lower = means - lower_bounds
            yerr_upper = upper_bounds - means
            
            # Plot points and error bars
            ax.errorbar(x_pos, means, 
                       yerr=[yerr_lower, yerr_upper],
                       fmt='o', 
                       markersize=8,
                       capsize=5, 
                       capthick=2,
                       label=predictor_label,
                       color=colors[predictor_label],
                       elinewidth=2,
                       markeredgecolor='black',
                       markeredgewidth=0.5,
                       alpha=0.9)
            
            # Add significance markers
            for k, (x, mean, prob) in enumerate(zip(x_pos, means, prob_nonzero)):
                if prob > 0.95:
                    symbol = '***'
                elif prob > 0.90:
                    symbol = '**'
                elif prob > 0.85:
                    symbol = '*'
                else:
                    symbol = ''
                
                if symbol:
                    y_pos = upper_bounds[k] + 0.05
                    ax.text(x, y_pos, symbol, ha='center', fontsize=12, fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Model Parameter', fontsize=13, fontweight='bold')
        ax.set_ylabel('Posterior Mean Beta (Standardized)' if self.standardize else 'Posterior Mean Beta', 
                     fontsize=13, fontweight='bold')
        ax.set_title('Bayesian Regression: Effect Estimates\n(94% HDI shown)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(self.outcome_labels, fontsize=12)
        
        # Add legend
        ax.legend(title='Predictors', title_fontsize=11, fontsize=10,
                 loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhspan(-rope, rope, alpha=0.1, color='gray', label=f'ROPE (±{rope})')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_posteriors(self, 
                       figsize: Tuple[int, int] = (15, 12),
                       colors: Optional[Dict[str, str]] = None,
                       save_path: Optional[str] = None):
        """
        Plot posterior distributions for all coefficients.
        
        Parameters
        ----------
        figsize : tuple, default=(15, 12)
            Figure size
        colors : dict, optional
            Dictionary mapping predictor labels to colors
        save_path : str, optional
            Path to save the figure
        """
        if self.results_df is None:
            raise ValueError("Must call fit() before plotting")
        
        # Default colors
        if colors is None:
            color_palette = sns.color_palette("husl", len(self.predictor_labels))
            colors = {pred: color for pred, color in zip(self.predictor_labels, color_palette)}
        
        n_outcomes = len(self.outcome_labels)
        n_predictors = len(self.predictor_labels)
        
        fig, axes = plt.subplots(n_outcomes, n_predictors, figsize=figsize)
        if n_outcomes == 1:
            axes = axes.reshape(1, -1)
        if n_predictors == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Posterior Distributions of Regression Coefficients', fontsize=16)
        
        for i, outcome_label in enumerate(self.outcome_labels):
            for j, predictor_label in enumerate(self.predictor_labels):
                ax = axes[i, j]
                
                # Get posterior samples
                row = self.results_df[(self.results_df['outcome'] == outcome_label) & 
                                     (self.results_df['predictor'] == predictor_label)].iloc[0]
                samples = row['posterior_samples']
                
                # Plot histogram
                ax.hist(samples, bins=50, density=True, alpha=0.7, 
                       color=colors[predictor_label], edgecolor='black', linewidth=0.5)
                
                # Add mean and HDI
                ax.axvline(row['beta_mean'], color='red', linestyle='--', 
                          label=f"Mean: {row['beta_mean']:.3f}")
                ax.axvline(row['hdi_lower'], color='red', linestyle=':', alpha=0.5)
                ax.axvline(row['hdi_upper'], color='red', linestyle=':', alpha=0.5)
                ax.axvspan(row['hdi_lower'], row['hdi_upper'], alpha=0.1, color='red')
                
                # Add zero line
                ax.axvline(0, color='black', linestyle='-', alpha=0.5)
                
                # Labels
                ax.set_title(f"{outcome_label} ~ {predictor_label}", fontsize=11)
                ax.set_xlabel('Beta coefficient', fontsize=9)
                ax.set_ylabel('Density', fontsize=9)
                
                # Add probability text
                prob_text = f"P(β>0) = {row['prob_positive']:.2f}"
                ax.text(0.95, 0.95, prob_text, transform=ax.transAxes,
                       ha='right', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_summary(self, significance_level: float = 0.95) -> pd.DataFrame:
        """
        Get a summary of the regression results.
        
        Parameters
        ----------
        significance_level : float, default=0.95
            Probability threshold for significance
        
        Returns
        -------
        pd.DataFrame
            Summary of significant effects
        """
        if self.results_df is None:
            raise ValueError("Must call fit() before getting summary")
        
        # Create summary tables
        summary_pivot = self.results_df.pivot(index='outcome', columns='predictor', values='beta_mean')
        prob_pivot = self.results_df.pivot(index='outcome', columns='predictor', values='prob_nonzero')
        
        print("\n" + "="*60)
        print("BAYESIAN REGRESSION RESULTS SUMMARY")
        print("="*60)
        
        print("\nPosterior Mean Beta Coefficients:")
        print(summary_pivot.round(3).to_string())
        
        print("\n\nProbability of Non-Zero Effect:")
        print(prob_pivot.round(3).to_string())
        
        # Find significant effects
        print("\n" + "="*60)
        print(f"EFFECTS WITH STRONG EVIDENCE (P(β≠0) > {significance_level})")
        print("="*60)
        
        significant = self.results_df[self.results_df['prob_nonzero'] > significance_level].sort_values('prob_nonzero', ascending=False)
        
        if len(significant) > 0:
            for _, row in significant.iterrows():
                direction = "positive" if row['beta_mean'] > 0 else "negative"
                print(f"{row['predictor']:15} → {row['outcome']:10}: "
                      f"β = {row['beta_mean']:+.3f} [{row['hdi_lower']:.3f}, {row['hdi_upper']:.3f}], "
                      f"P(β≠0) = {row['prob_nonzero']:.3f} ({direction})")
        else:
            print(f"No effects with >{significance_level*100}% probability of being non-zero")
        
        return significant
    
    def save_results(self, filepath: str):
        """
        Save regression results to CSV.
        
        Parameters
        ----------
        filepath : str
            Path to save the CSV file
        """
        if self.results_df is None:
            raise ValueError("Must call fit() before saving results")
        
        save_df = self.results_df[['outcome', 'predictor', 'beta_mean', 'beta_sd', 
                                   'hdi_lower', 'hdi_upper', 'prob_nonzero']]
        save_df.to_csv(filepath, index=False)
        print(f"\nResults saved to '{filepath}'")


def quick_bayesian_regression(data: pd.DataFrame,
                             outcomes: List[str],
                             predictors: List[str],
                             outcome_labels: Optional[List[str]] = None,
                             predictor_labels: Optional[List[str]] = None,
                             standardize: bool = True,
                             plot: bool = True,
                             save_results: bool = False,
                             output_prefix: str = "bayesian_regression") -> BayesianRegression:
    """
    Quick function to run a complete Bayesian regression analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all variables
    outcomes : list
        List of outcome variable names
    predictors : list
        List of predictor variable names
    outcome_labels : list, optional
        Display labels for outcomes
    predictor_labels : list, optional
        Display labels for predictors
    standardize : bool, default=True
        Whether to standardize variables
    plot : bool, default=True
        Whether to create plots
    save_results : bool, default=False
        Whether to save results to files
    output_prefix : str, default="bayesian_regression"
        Prefix for output files
    
    Returns
    -------
    BayesianRegression
        Fitted BayesianRegression object
    
    Example
    -------
    >>> # Basic usage
    >>> results = quick_bayesian_regression(
    ...     data=df,
    ...     outcomes=['y1', 'y2'],
    ...     predictors=['x1', 'x2', 'x3']
    ... )
    
    >>> # With custom labels and saving
    >>> results = quick_bayesian_regression(
    ...     data=df,
    ...     outcomes=['z_mean', 'alpha_mean'],
    ...     predictors=['stress', 'anxiety'],
    ...     outcome_labels=['Z', 'α'],
    ...     predictor_labels=['Stress', 'Anxiety'],
    ...     save_results=True,
    ...     output_prefix="mental_health_analysis"
    ... )
    """
    # Create and fit the model
    model = BayesianRegression(
        data=data,
        outcomes=outcomes,
        predictors=predictors,
        outcome_labels=outcome_labels,
        predictor_labels=predictor_labels,
        standardize=standardize
    )
    
    # Fit the models
    model.fit(verbose=True)
    
    # Create plots if requested
    if plot:
        save_path1 = f"{output_prefix}_coefficients.png" if save_results else None
        model.plot_coefficients(save_path=save_path1)
        
        save_path2 = f"{output_prefix}_posteriors.png" if save_results else None
        model.plot_posteriors(save_path=save_path2)
    
    # Get summary
    model.get_summary()
    
    # Save results if requested
    if save_results:
        model.save_results(f"{output_prefix}_results.csv")
    
    return model