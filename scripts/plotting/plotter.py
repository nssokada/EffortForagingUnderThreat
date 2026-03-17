"""
Plotter Module - Publication-quality visualizations for computational psychiatry research
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Optional SciPy imports
try:
    from scipy.stats import pearsonr, t as student_t, gaussian_kde
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    pearsonr = None
    student_t = None
    gaussian_kde = None


# ============================================================================
# COLOR PALETTE
# ============================================================================

class Colors:
    """Central color palette for consistent styling"""
    # Neutrals
    INK = '#6B7280'
    GREY = '#EBEBEB'
    DARK_GREY = '#191919'
    WHITE = '#FFFFFF'
    LIGHT_GRAY = '#EBEBEB'
    
    # Primary colors
    RUBY1 = '#D4145A'       # vigor + capture markers
    CERULEAN2 = '#1A93FF'   # choice
    PERSIMMON3 = '#FAA70C'  # cumulative score
    SLATE = '#9CA3AF'
    
    # Extended palette colors (for evidence zones)
    MANTIS1 = '#80C55F'     # good evidence (green)
    AMBER = '#FAA71C'       # moderate evidence (amber/yellow)
    RED = '#F15A24'         # poor evidence
    EMERALD = '#80C55F'     # Alternative green
    
    # Extended palette
    PALETTE = {
        "black":       "#000000",
        "dark_gray":   "#191919",
        "white":       "#FFFFFF",
        "light_gray":  "#EBEBEB",
        "mantlis_1":   "#80C55F",
        "mantlis_2":   "#B5D741",
        "mantlis_3":   "#D5E844",
        "icerime_1":   "#F6F946",
        "persimmon_1": "#F15A24",
        "persimmon_2": "#F6811B",
        "persimmon_3": "#FAA71C",
        "persimmon_4": "#FFEC00",
        "cerulean_1":  "#22C6FF",
        "cerulean_2":  "#1A93FF",
        "cerulean_3":  "#76BBFF",
        "cerulean_4":  "#B5E5FF",
        "ruby_1":      "#D4145A",
        "ruby_2":      "#DA203D",
        "magenta_1":   "#E03DBF",
        "magenta_2":   "#E56FF2",
    }


# ============================================================================
# GLOBAL STYLE SETTINGS
# ============================================================================

def set_plot_style():
    """Apply consistent matplotlib style settings"""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.family": "sans-serif",
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
        "grid.linewidth": 0.8,
    })


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def style_axis(ax, ylabel=None, xlabel=None, ygrid=True, ylim=None, yticks=None):
    """Apply consistent axis styling."""
    ax.grid(ygrid, axis='y', color=Colors.GREY, alpha=0.55)
    ax.grid(False, axis='x')
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(colors=Colors.INK, labelsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=Colors.INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=Colors.INK)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)


def _gradient_fill_under_curve(ax, x, y, *, baseline=0.5, color="#D41454", alpha_top=0.25):
    """
    Stocks/Robinhood-style vertical fade under (x,y) down to `baseline`.
    Alpha is strongest right under the line and fades toward the baseline.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Clip polygon: line -> baseline (in data coords)
    verts = np.vstack([
        np.column_stack([x, y]),
        np.column_stack([x[::-1], np.full_like(x, baseline)[::-1]])
    ])
    clip_path = Path(verts)
    clip_patch = PathPatch(clip_path, facecolor="none", edgecolor="none", transform=ax.transData)
    ax.add_patch(clip_patch)

    # Image extent spans exactly from baseline to the (local) max line height.
    y0 = min(baseline, np.nanmin(y))
    y1 = max(baseline, np.nanmax(y))

    # Build an RGBA image whose alpha is 0 at bottom (baseline) -> 1 at top (line)
    n = 256
    fade = np.linspace(0.0, 1.0, n).reshape(n, 1)
    r, g, b = mcolors.to_rgb(color)
    rgba = np.dstack([np.full((n, 1), r),
                      np.full((n, 1), g),
                      np.full((n, 1), b),
                      fade * alpha_top])

    ax.imshow(
        rgba,
        extent=[x.min(), x.max(), y0, y1],
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
        clip_path=clip_patch,
        clip_on=True,
    )


def _gradient_fill_under_curve_oriented(ax, x, y, *, baseline, color="#D41454", alpha_top=0.30):
    """
    Same as _gradient_fill_under_curve, but if the baseline is above the curve
    (i.e., we're filling downward), the gradient is flipped so alpha is strongest
    at the curve and fades toward the baseline.
    """
    x = np.asarray(x); y = np.asarray(y)

    verts = np.vstack([
        np.column_stack([x, y]),
        np.column_stack([x[::-1], np.full_like(x, baseline)[::-1]])
    ])
    clip_path = Path(verts)
    clip_patch = PathPatch(clip_path, facecolor="none", edgecolor="none", transform=ax.transData)
    ax.add_patch(clip_patch)

    y0 = min(baseline, np.nanmin(y))
    y1 = max(baseline, np.nanmax(y))

    n = 256
    fade = np.linspace(0.0, 1.0, n).reshape(n, 1)
    # If filling downward (baseline above curve), reverse so alpha is max at the curve
    if baseline > np.nanmedian(y):
        fade = fade[::-1]

    r, g, b = mcolors.to_rgb(color)
    rgba = np.dstack([np.full((n, 1), r),
                      np.full((n, 1), g),
                      np.full((n, 1), b),
                      fade * alpha_top])

    ax.imshow(
        rgba,
        extent=[x.min(), x.max(), y0, y1],
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
        clip_path=clip_patch,
        clip_on=True,
    )


def _gradient_fill_ci(ax, x, y_lower, y_upper, y_center, *, color="#1A93FF", 
                      alpha_edge=0.05, alpha_center=0.35, glow_strength=5):
    """
    Create a gradient-filled confidence interval band with a smooth glow effect.
    Alpha fades from the center line outward to the CI bounds using a Gaussian-like curve.
    """
    x = np.asarray(x)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)
    y_center = np.asarray(y_center)
    
    # Create polygon for CI band
    verts = np.vstack([
        np.column_stack([x, y_upper]),
        np.column_stack([x[::-1], y_lower[::-1]])
    ])
    clip_path = Path(verts)
    clip_patch = PathPatch(clip_path, facecolor="none", edgecolor="none", transform=ax.transData)
    ax.add_patch(clip_patch)
    
    # Get bounds with a bit of padding for smoother edges
    y_min = np.nanmin(y_lower)
    y_max = np.nanmax(y_upper)
    
    # Use higher resolution for smoother gradients
    n = 512
    y_vals = np.linspace(y_min, y_max, n)
    
    # Create 2D alpha grid for smooth gradients
    alpha_grid = np.zeros((n, len(x)))
    
    for j, x_val in enumerate(x):
        center_val = y_center[j]
        lower_val = y_lower[j]
        upper_val = y_upper[j]
        
        # Calculate distance from center to bounds at this x position
        dist_to_lower = center_val - lower_val
        dist_to_upper = upper_val - center_val
        max_dist = max(dist_to_lower, dist_to_upper)
        
        if max_dist > 0:
            for i, y_val in enumerate(y_vals):
                # Distance from this y to the center line
                dist = abs(y_val - center_val)
                
                # Normalize distance
                norm_dist = dist / max_dist
                
                # Apply smooth Gaussian-like falloff for glowy effect
                falloff = np.exp(-(norm_dist ** 2) * glow_strength)
                
                # Interpolate between center and edge alpha using smooth falloff
                alpha_grid[i, j] = alpha_center * falloff + alpha_edge * (1 - falloff)
        else:
            alpha_grid[:, j] = alpha_center
    
    # Average alpha values across x for each y (creates smooth horizontal bands)
    alpha_vals = np.mean(alpha_grid, axis=1)
    
    # Build RGBA image
    r, g, b = mcolors.to_rgb(color)
    rgba = np.dstack([np.full((n, 1), r),
                      np.full((n, 1), g),
                      np.full((n, 1), b),
                      alpha_vals.reshape(n, 1)])
    
    ax.imshow(
        rgba,
        extent=[x.min(), x.max(), y_min, y_max],
        origin="lower",
        aspect="auto",
        interpolation="gaussian",
        zorder=0.5,
        clip_path=clip_patch,
        clip_on=True,
    )


def _kde_gaussian_1d(x, grid, bw=None):
    """Gaussian KDE on 'grid' with Silverman's rule if bw is None."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid)

    if bw is None:
        std = np.std(x, ddof=1) if x.size > 1 else 1.0
        bw = 1.06 * std * (x.size ** (-1/5))

    diffs = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diffs**2).sum(axis=1) / (x.size * bw * np.sqrt(2*np.pi))
    return dens



def _polish_parameter_axes(
    ax, y_positions, *, facecolor="#FCFCFD", xgrid_major_alpha=0.25,
    xgrid_minor_alpha=0.12, yref_alpha=0.15, left_margin=0.12,
    right_margin=0.98, top_margin=0.98, bottom_margin=0.10, 
    show_horizontal_lines=False
):
    """Light gridlines + baselines + breathing room."""
    ax.set_facecolor(facecolor)
    ax.grid(True, which="major", axis="x", color="#E5E7EB",
            linewidth=0.8, alpha=xgrid_major_alpha, zorder=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", axis="x", color="#ECEFF3",
            linewidth=0.6, alpha=xgrid_minor_alpha, zorder=0)
    ax.set_axisbelow(True)

    # Optional horizontal baseline for each parameter (disabled by default)
    if show_horizontal_lines:
        for y in y_positions:
            ax.axhline(y, color="#DCE1E7", linewidth=0.5, alpha=yref_alpha, zorder=0)

    ax.tick_params(axis='x', colors=Colors.INK, labelsize=10, length=4)
    ax.tick_params(axis='y', colors=Colors.INK, labelsize=10)
    ax.margins(x=0.02)
    plt.subplots_adjust(left=left_margin, right=right_margin,
                        top=top_margin, bottom=bottom_margin)

def _pearson_r_p(x, y, w=None):
    """
    Pearson correlation r and two-sided p.
    Uses SciPy if available; otherwise returns p=np.nan.
    Supports optional non-negative weights (approx via weighted corr + t-test on neff).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if w is None:
        if _HAS_SCIPY:
            r, p = pearsonr(x, y)
            return r, p, m.sum()
        r = np.corrcoef(x, y)[0, 1] if x.size > 1 else np.nan
        return r, np.nan, m.sum()
    else:
        w = np.asarray(w, float)[m]
        if w.ndim != 1 or w.size != x.size:
            raise ValueError("point_weights must be same length as x/y.")
        if np.any(w < 0):
            raise ValueError("point_weights must be non-negative.")
        W = np.sum(w)
        if W <= 0 or x.size < 2:
            return np.nan, np.nan, x.size
        mx = np.sum(w * x) / W
        my = np.sum(w * y) / W
        cov = np.sum(w * (x - mx) * (y - my)) / W
        vx  = np.sum(w * (x - mx) ** 2) / W
        vy  = np.sum(w * (y - my) ** 2) / W
        r = cov / (np.sqrt(vx * vy) + 1e-12)

        # Kish neff for rough p
        neff = (W**2) / (np.sum(w**2) + 1e-12)
        if _HAS_SCIPY and neff > 3:
            df = max(1, int(round(neff - 2)))
            tval = r * np.sqrt(df / (max(1e-12, 1 - r**2)))
            p = 2 * (1 - student_t.cdf(np.abs(tval), df))
        else:
            p = np.nan
        return float(r), float(p), int(round(neff))


def _ols_fit_ci_band(x, y, w, xs, alpha=0.05):
    """
    OLS (or weighted least squares) with analytic pointwise CI for the fitted mean.
    Returns: yhat(xs), lower(xs), upper(xs), (slope, intercept)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Design matrix with intercept
    X = np.column_stack([np.ones_like(x), x])
    Xs = np.column_stack([np.ones_like(xs), xs])

    if w is None:
        W = np.eye(len(x))
        WX = X
        Wy = y
    else:
        w = np.asarray(w, float)
        if np.any(w < 0):
            raise ValueError("point_weights must be non-negative.")
        W_sqrt = np.sqrt(w)
        WX = X * W_sqrt[:, None]
        Wy = y * W_sqrt

    # Coefficients via normal equations on weighted system
    XtWX = WX.T @ WX
    XtWy = WX.T @ Wy
    beta = np.linalg.solve(XtWX, XtWy)  # [b0, b1]

    # Residual variance estimate (sigma^2)
    yhat = X @ beta
    if w is None:
        rss = np.sum((y - yhat) ** 2)
        df = max(1, len(x) - X.shape[1])
        s2 = rss / df
    else:
        # Weighted residual sum of squares (using weights w)
        rss = np.sum(w * (y - yhat) ** 2)
        # Use Kish neff to be conservative on df
        neff = (np.sum(w)**2) / (np.sum(w**2) + 1e-12)
        df = max(1, int(round(neff - X.shape[1])))
        s2 = rss / df

    # Covariance matrix of beta
    XtWX_inv = np.linalg.inv(XtWX)
    # Standard error for fitted mean at each xs
    se_mean = np.sqrt(np.sum((Xs @ XtWX_inv) * Xs, axis=1) * s2)

    # t critical
    if _HAS_SCIPY:
        tcrit = student_t.ppf(1 - alpha/2, df)
    else:
        # Normal approx if SciPy not available
        tcrit = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96

    yhat_xs = Xs @ beta
    lo = yhat_xs - tcrit * se_mean
    hi = yhat_xs + tcrit * se_mean
    return yhat_xs, lo, hi, (beta[1], beta[0])


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_subject_trials(df, subject, max_trials=None):
    """
    Plot individual subject trial data with vigor, choice, and cumulative score.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: subj/participant_int, trial/trialID, log_vigor, choice, outcome
    subject : int
        Subject ID to plot
    max_trials : int, optional
        Maximum number of trials to display
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    set_plot_style()
    
    subj_col = 'subj' if 'subj' in df.columns else 'participant_int'
    trial_col = 'trialID' if 'trialID' in df.columns else 'trial'

    subj_data = df[df[subj_col] == subject].copy()
    if len(subj_data) == 0:
        raise ValueError(f"No data found for subject {subject}")

    subj_data = subj_data.sort_values(trial_col).reset_index(drop=True)
    if max_trials is not None:
        subj_data = subj_data.iloc[:max_trials]

    # Vigor normalization
    subj_data['vigor'] = np.exp(subj_data['log_vigor'])
    vmin, vmax = subj_data['vigor'].min(), subj_data['vigor'].max()
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    subj_data['vigor_norm'] = (subj_data['vigor'] - vmin) / denom

    # Scores + captures
    subj_data['score'] = np.where(
        subj_data['choice'].eq(1),
        np.where(subj_data['outcome'].eq(0), 5, -5),
        np.where(subj_data['outcome'].eq(0), 1, -5),
    )
    subj_data['cumulative_score'] = subj_data['score'].cumsum()
    capture_mask = subj_data['outcome'].eq(1)

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.35}
    )
    trial_nums = subj_data[trial_col].to_numpy()

    # --- Top: Vigor with gradient fill ---
    ax1 = axes[0]
    _gradient_fill_under_curve(ax1, trial_nums, subj_data['vigor_norm'], 
                               baseline=0.5, color=Colors.RUBY1)
    ax1.plot(trial_nums, subj_data['vigor_norm'], color=Colors.RUBY1, linewidth=2, zorder=3)
    style_axis(ax1, ylabel='Vigor', ylim=(-0.05, 1.05), yticks=[0, 0.5, 1])

    # --- Middle: Choice (step) ---
    ax2 = axes[1]
    ax2.step(trial_nums, subj_data['choice'], where='mid', color=Colors.CERULEAN2, linewidth=2)
    ax2.plot(trial_nums, subj_data['choice'], marker='o', markersize=3.5,
             linestyle='None', color=Colors.CERULEAN2, alpha=0.9)
    style_axis(ax2, ylabel='Choice', ylim=(-0.1, 1.1), yticks=[0, 1])

    # --- Bottom: Cumulative score with gradient + capture markers ---
    ax3 = axes[2]
    ymin = subj_data['cumulative_score'].min()
    ymax = subj_data['cumulative_score'].max()
    pad = 0.06 * (ymax - ymin if ymax != ymin else 1)

    ax3.plot(trial_nums, subj_data['cumulative_score'], 
             color=Colors.PERSIMMON3, linewidth=2.2, zorder=2)

    _gradient_fill_under_curve(
        ax3, trial_nums, subj_data['cumulative_score'],
        baseline=ymin - pad,
        color=Colors.PERSIMMON3,
        alpha_top=0.22
    )

    # Capture markers (hollow)
    if capture_mask.any():
        ax3.plot(
            trial_nums[capture_mask.values],
            subj_data.loc[capture_mask, 'cumulative_score'],
            linestyle='None', marker='o', markersize=6,
            markerfacecolor='white', markeredgecolor=Colors.PERSIMMON3, 
            markeredgewidth=2, zorder=3, label='Capture'
        )
        legend = ax3.legend(loc='upper left', frameon=True, fontsize=9.5, 
                       labelcolor=Colors.INK, edgecolor='#D1D5DB', fancybox=True)
        legend.get_frame().set_facecolor(Colors.GREY)
        legend.get_frame().set_alpha(0.45)
        legend.get_frame().set_linewidth(1)

    style_axis(ax3, ylabel='Cumulative Score', xlabel='Trial', 
               ylim=(ymin - pad, ymax + pad))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes





def plot_ppc_subject_trials(df, df_predicted=None, subjects=None, n=6, max_trials=None,
                            features=['vigor', 'choice', 'delta_sv'],
                            legend_position='outside_right', subject_label_position='above',
                            inter_subject_gap=0.25, intra_feature_gap=0.08,
                            figsize=None):
    """
    Plot posterior predictive check for individual subject trials with vertical stacking.
    
    Uses nested GridSpec for proper spacing control:
    - Larger gaps between subjects
    - Tighter gaps between features within a subject
    - Single "Trial" xlabel at the bottom only
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing both observed and predicted values (if df_predicted is None)
        OR just observed data (if df_predicted is provided)
    df_predicted : pd.DataFrame, optional
        Predicted data. If None, assumes df contains both observed and predicted
    subjects : list, optional
        Specific subjects to plot
    n : int
        Number of subjects if not specified
    max_trials : int, optional
        Maximum trials per subject
    features : list of str
        Features to plot. Can include 'vigor', 'choice', 'delta_sv'
    legend_position : str
        'outside_right', 'outside_top', 'upper_right', 'upper_left'
    subject_label_position : str
        'above', 'inside_left', 'inside_right', 'title'
    inter_subject_gap : float
        Vertical gap between subjects (0-1, fraction of subplot height)
    intra_feature_gap : float
        Vertical gap between features within a subject (0-1)
    figsize : tuple, optional
        Figure size. If None, auto-calculated.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes (2D array: [subject, feature])
    """
    # Import Colors and helpers from the main module (or define locally)
    class Colors:
        INK = '#6B7280'
        GREY = '#EBEBEB'
        RUBY1 = '#D4145A'
        CERULEAN2 = '#1A93FF'
        PERSIMMON3 = '#FAA70C'
        SLATE = '#9CA3AF'
    
    def style_axis(ax, ylabel=None, xlabel=None, ygrid=True, ylim=None, yticks=None):
        ax.grid(ygrid, axis='y', color=Colors.GREY, alpha=0.55)
        ax.grid(False, axis='x')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('#D1D5DB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.tick_params(colors=Colors.INK, labelsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11, color=Colors.INK)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11, color=Colors.INK)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
    
    # Set plot style
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.family": "sans-serif",
        "figure.dpi": 140,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    
    # Validate features
    valid_features = ['vigor', 'choice', 'delta_sv', 'survival']
    features = [f for f in features if f in valid_features]
    if not features:
        raise ValueError(f"No valid features specified. Choose from {valid_features}")
    
    n_features = len(features)
    
    # Column name detection
    subj_col = 'subj' if 'subj' in df.columns else 'participant_int'
    trial_col = 'trial' if 'trial' in df.columns else 'trialID'
    
    # Handle case where observed and predicted are in same dataframe
    if df_predicted is None:
        df_predicted = df.copy()
    
    # Select subjects
    if subjects is None:
        all_subjects = df[subj_col].unique()
        subjects = np.random.choice(all_subjects, size=min(n, len(all_subjects)), 
                                   replace=False)
    
    n_subjects = len(subjects)
    
    # Calculate figure size
    if figsize is None:
        fig_width = 10 if legend_position == 'outside_right' else 8
        # Account for inter-subject gaps in height calculation
        base_height_per_feature = 1.3
        subject_block_height = n_features * base_height_per_feature
        total_height = n_subjects * subject_block_height + (n_subjects - 1) * inter_subject_gap * subject_block_height
        figsize = (fig_width, total_height)
    
    # Create figure with nested GridSpec
    fig = plt.figure(figsize=figsize)
    
    # Outer GridSpec: one row per subject, with gaps between subjects
    outer_gs = GridSpec(
        n_subjects, 1,
        figure=fig,
        hspace=inter_subject_gap,  # Gap between subjects
        left=0.08,
        right=0.92 if legend_position == 'outside_right' else 0.98,
        top=0.95,
        bottom=0.08
    )
    
    # Store all axes in a 2D structure: axes[subject_idx][feature_idx]
    axes = []
    
    for idx, subject in enumerate(subjects):
        # Inner GridSpec for this subject's features
        inner_gs = GridSpecFromSubplotSpec(
            n_features, 1,
            subplot_spec=outer_gs[idx],
            hspace=intra_feature_gap  # Tight spacing within subject
        )
        
        subject_axes = []
        
        # Extract subject data
        obs = df[df[subj_col] == subject].copy()
        pred = df_predicted[df_predicted[subj_col] == subject].copy()
        
        if len(obs) == 0:
            # Create empty axes
            for feat_idx in range(n_features):
                ax = fig.add_subplot(inner_gs[feat_idx])
                subject_axes.append(ax)
            axes.append(subject_axes)
            continue
            
        obs = obs.sort_values(trial_col).reset_index(drop=True)
        pred = pred.sort_values(trial_col).reset_index(drop=True)
        
        if max_trials is not None:
            obs = obs.iloc[:max_trials]
            pred = pred.iloc[:max_trials]
        
        trial_nums = obs[trial_col].to_numpy()
        
        # Normalize vigor if needed
        if 'vigor' in features:
            if 'vigor_likelihood' in pred.columns:
                if 'log_vigor' in obs.columns:
                    obs['vigor'] = np.exp(obs['log_vigor'])
                    pred['vigor_pred'] = pred['vigor_likelihood']
                elif 'vigor' in obs.columns:
                    pred['vigor_pred'] = pred['vigor_likelihood']
                
                vmin = min(obs['vigor'].min(), pred['vigor_pred'].min())
                vmax = max(obs['vigor'].max(), pred['vigor_pred'].max())
                denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
                
                obs['vigor_norm'] = (obs['vigor'] - vmin) / denom
                pred['vigor_norm'] = (pred['vigor_pred'] - vmin) / denom
            else:
                obs['vigor'] = np.exp(obs['log_vigor']) if 'log_vigor' in obs.columns else obs['vigor']
                pred['vigor'] = np.exp(pred['log_vigor']) if 'log_vigor' in pred.columns else pred['vigor']
                
                vmin = min(obs['vigor'].min(), pred['vigor'].min())
                vmax = max(obs['vigor'].max(), pred['vigor'].max())
                denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
                
                obs['vigor_norm'] = (obs['vigor'] - vmin) / denom
                pred['vigor_norm'] = (pred['vigor'] - vmin) / denom
                
                if 'log_vigor_lower' in pred.columns and 'log_vigor_upper' in pred.columns:
                    pred['vigor_lower_norm'] = (np.exp(pred['log_vigor_lower']) - vmin) / denom
                    pred['vigor_upper_norm'] = (np.exp(pred['log_vigor_upper']) - vmin) / denom
        
        # Loop through each requested feature
        for feat_idx, feature in enumerate(features):
            ax = fig.add_subplot(inner_gs[feat_idx])
            subject_axes.append(ax)
            
            # Key fix: only the VERY LAST panel gets xlabel
            is_first_panel_of_subject = (feat_idx == 0)
            is_last_panel_overall = (idx == n_subjects - 1) and (feat_idx == n_features - 1)
            is_last_panel_of_subject = (feat_idx == n_features - 1)
            
            if feature == 'vigor':
                # Observed (grey, solid)
                ax.plot(trial_nums, obs['vigor_norm'], color=Colors.INK, 
                        linewidth=2.2, label='Observed', zorder=2, alpha=0.7)
                
                # Predicted (colored, solid)
                ax.plot(trial_nums, pred['vigor_norm'], color=Colors.RUBY1, 
                        linewidth=2.2, label='Predicted', zorder=3)
                
                # Mean line
                ax.axhline(0.5, color=Colors.INK, linestyle='--', linewidth=1, 
                           alpha=0.4, zorder=1)
                
                ylabel = 'Vigor'
                ylim = (-0.05, 1.05)
                yticks = [0, 0.5, 1]

            elif feature == 'choice':
                # Predicted probs
                if 'p_high' in pred.columns:
                    pred_values = pred['p_high']
                    pred_label = 'P(high effort)'
                elif 'choice_prob' in pred.columns:
                    pred_values = pred['choice_prob']
                    pred_label = 'P(choice=1)'
                else:
                    pred_values = pred['choice']
                    pred_label = 'Predicted'

                # Rolling mean of observed choices
                obs_probs = (
                    obs['choice']
                    .rolling(window=4, center=True, min_periods=1)
                    .mean()
                )

                # Predicted line
                ax.plot(trial_nums, pred_values,
                        color=Colors.CERULEAN2, linewidth=2.2,
                        label=pred_label, zorder=3)

                # Smoothed observed line
                ax.plot(trial_nums, obs_probs,
                        color=Colors.INK, linewidth=2.2,
                        alpha=0.7, linestyle='-',
                        label='Observed (smoothed)', zorder=2)

                # Raw observed dots
                ax.plot(trial_nums, obs['choice'].to_numpy(),
                        marker='o', markersize=3,
                        linestyle='None', color=Colors.INK,
                        alpha=0.4, label='Observed (trials)', zorder=1)

                ylabel = 'Choice'
                ylim = (-0.05, 1.05)
                yticks = [0, 0.5, 1]

            elif feature == 'delta_sv':
                ax.axhline(0, color=Colors.GREY, linewidth=2, zorder=1)
                
                # Observed (grey, solid)
                ax.plot(trial_nums, obs['delta_sv'], color=Colors.INK, 
                        linewidth=2.2, label='Observed', zorder=2, alpha=0.7)
                
                # Predicted (grey but slightly different, solid)
                ax.plot(trial_nums, pred['delta_sv'], color='#9CA3AF', 
                        linewidth=2.2, label='Predicted', zorder=3)
                
                # Calculate ylim centered on 0
                abs_max = max(abs(obs['delta_sv'].min()), abs(obs['delta_sv'].max()),
                             abs(pred['delta_sv'].min()), abs(pred['delta_sv'].max()))
                
                if 'delta_sv_lower' in pred.columns and 'delta_sv_upper' in pred.columns:
                    abs_max = max(abs_max, 
                                 abs(pred['delta_sv_lower'].min()),
                                 abs(pred['delta_sv_upper'].max()))
                
                pad = 0.1 * abs_max if abs_max > 0 else 1
                
                ylabel = 'ΔSV'
                ylim = (-abs_max - pad, abs_max + pad)
                yticks = None

            elif feature == 'survival':
                    # Look for survival column
                    surv_col = None
                    for col_name in ['S_u_H']:
                        if col_name in pred.columns:
                            surv_col = col_name
                            break
                    
                    # Compute danger = 1 - survival
                    pred_danger = 1 - pred[surv_col]
                    
                    ax.plot(trial_nums, pred_danger, color=Colors.RUBY1, linewidth=2.2, 
                            label='Predicted', zorder=3)
                    ax.axhline(0.5, color=Colors.INK, linestyle='--', linewidth=1, alpha=0.4)
                    # ax.axhspan(0.5, 1.0, color=Colors.RUBY1, alpha=0.08, zorder=0)  # Danger zone
                    
                    ylabel = 'Danger'
                    ylim = (-0.05, 1.05)
                    yticks = [0, 0.5, 1]
            
            # Apply styling - xlabel ONLY on the very last panel
            xlabel = 'Trial' if is_last_panel_overall else None
            style_axis(ax, ylabel=ylabel, xlabel=xlabel, ylim=ylim, yticks=yticks)
            
            # Share x-axis: hide tick labels except for the very last panel
            if not is_last_panel_overall:
                ax.tick_params(axis='x', labelbottom=False)
            
            # Add legend to first panel of first subject only
            if idx == 0 and is_first_panel_of_subject:
                if legend_position == 'outside_right':
                    legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                                      frameon=True, fontsize=9,
                                      labelcolor=Colors.INK, edgecolor='#D1D5DB', fancybox=True)
                elif legend_position == 'outside_top':
                    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), 
                                      ncol=3, frameon=True, fontsize=9,
                                      labelcolor=Colors.INK, edgecolor='#D1D5DB', fancybox=True)
                elif legend_position == 'upper_left':
                    legend = ax.legend(loc='upper left', frameon=True, fontsize=9,
                                      labelcolor=Colors.INK, edgecolor='#D1D5DB', fancybox=True)
                else:
                    legend = ax.legend(loc='upper right', frameon=True, fontsize=9,
                                      labelcolor=Colors.INK, edgecolor='#D1D5DB', fancybox=True)
                
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
                legend.get_frame().set_linewidth(1)
            
            # Add subject label to first panel
            if is_first_panel_of_subject:
                if subject_label_position == 'above':
                    # Use title instead of text annotation for cleaner spacing
                    ax.set_title(f'Subject {subject}', fontsize=12, 
                                color=Colors.INK, weight='bold', pad=8,
                                bbox=dict(boxstyle='round,pad=0.4', 
                                         facecolor='white', alpha=0.95, edgecolor='#D1D5DB'))
                elif subject_label_position == 'title':
                    ax.set_title(f'Subject {subject}', fontsize=12, color=Colors.INK, 
                                 weight='bold', pad=10)
                elif subject_label_position == 'inside_left':
                    ax.text(0.02, 0.98, f'Subject {subject}', 
                            transform=ax.transAxes, fontsize=11, 
                            color=Colors.INK, weight='bold', 
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round,pad=0.4', 
                                     facecolor='white', alpha=0.9, edgecolor='#D1D5DB'))
                else:  # 'inside_right'
                    ax.text(0.98, 0.98, f'Subject {subject}', 
                            transform=ax.transAxes, fontsize=11, 
                            color=Colors.INK, weight='bold', 
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.4', 
                                     facecolor='white', alpha=0.9, edgecolor='#D1D5DB'))
                    
               
        
        axes.append(subject_axes)
    
    return fig, axes

def plot_parameter_half_violin_glow(
    df, params=("z","theta","beta"), *, mean_suffix="_mean",
    labels=None, colors=None, side="lower", ridge_height=0.6,
    point_alpha=0.10, point_size=14, jitter_sd=0.05,
    whisk=(5,95), iqr=(25,75), xlabel="Estimate", figsize=None,
    bw=None, gridsize=600, pad_frac=0.06, density_cut=0.01,
    gradient_alpha_top=0.30, show_baseline_lines=False
):
    """
    Half-violin (split) KDE per parameter with gradient fill.
    
    Parameters
    ----------
    df : pd.DataFrame
        Parameter data
    params : tuple
        Parameter names to plot
    mean_suffix : str
        Suffix for mean columns
    labels : dict, optional
        Custom labels for parameters
    colors : dict, optional
        Custom colors for parameters
    side : str
        'lower' or 'upper' for violin side
    ridge_height : float
        Height of violin ridge
    point_alpha : float
        Alpha for jittered points
    point_size : float
        Size of points
    jitter_sd : float
        Standard deviation for jitter
    whisk : tuple
        Percentiles for whiskers
    iqr : tuple
        Percentiles for box
    xlabel : str
        X-axis label
    figsize : tuple, optional
        Figure size
    bw : float, optional
        Bandwidth for KDE
    gridsize : int
        Grid resolution for KDE
    pad_frac : float
        Padding fraction for x-range
    density_cut : float
        Threshold to trim KDE tails
    gradient_alpha_top : float
        Top alpha for gradient
    show_baseline_lines : bool
        Whether to show horizontal grey lines at parameter baselines
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    set_plot_style()
    
    params = list(params)
    if labels is None:
        labels = {p: p for p in params}
    if colors is None:
        colors = {p: {'z': Colors.RUBY1, 'k': Colors.CERULEAN2, 
                     'beta': Colors.SLATE, 'gamma': Colors.PERSIMMON3}.get(p, Colors.SLATE) 
                 for p in params}

    # Build global x-limits only for framing
    all_vals = []
    for p in params:
        col = f"{p}{mean_suffix}"
        if col in df.columns:
            all_vals.append(df[col].to_numpy())
    all_vals = np.concatenate(all_vals)
    all_vals = all_vals[np.isfinite(all_vals)]
    gmin, gmax = np.min(all_vals), np.max(all_vals)
    gpad = 0.08 * (gmax - gmin + 1e-9)
    xlim_global = (gmin - gpad, gmax + gpad)

    n = len(params)
    if figsize is None:
        figsize = (8.2, 1.2 + 1.3*n)
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False, axis='y')

    ax.set_xlim(*xlim_global)
    ax.set_ylim(-0.8, n-1+0.8)
    ax.set_xlabel(xlabel, fontsize=11, color=Colors.INK)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([labels.get(p, p) for p in params], color=Colors.INK, fontsize=10)
    ax.spines['left'].set_color('#D1D5DB'); ax.spines['bottom'].set_color('#E5E7EB')
    ax.invert_yaxis()

    sign = -1.0 if side.lower() == "lower" else 1.0

    for yi, p in enumerate(params):
        col = f"{p}{mean_suffix}"
        if col not in df.columns:
            continue
        vals = df[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        color = colors[p]

        # Per-parameter x-range limited to data + small pad
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        span = vmax - vmin + 1e-9
        lo = vmin - pad_frac * span
        hi = vmax + pad_frac * span
        grid = np.linspace(lo, hi, gridsize)

        # KDE & trimming to avoid tiny tails
        dens = _kde_gaussian_1d(vals, grid, bw=bw)
        if dens.max() > 0:
            dens = dens / dens.max()
        mask = dens >= density_cut
        if mask.any():
            grid_eff = grid[mask]
            dens_eff  = dens[mask]
        else:
            grid_eff, dens_eff = grid, dens

        thickness = ridge_height * dens_eff
        edge = yi + sign * thickness

        # Gradient fill to the baseline
        _gradient_fill_under_curve_oriented(
            ax, grid_eff, edge,
            baseline=yi, color=color, alpha_top=gradient_alpha_top
        )

        # Edge stroke
        ax.plot(grid_eff, edge, color=color, alpha=0.8, lw=1.6, zorder=1.1)

        # Glow points (subject means)
        rng = np.random.default_rng(42 + yi)
        yj = yi + rng.normal(0, jitter_sd, size=vals.size)
        ax.scatter(vals, yj, s=point_size, color=color, alpha=point_alpha,
                   edgecolors='none', zorder=2)

        # Transparent box & whiskers
        q5, q25, q50, q75, q95 = np.percentile(vals, [whisk[0], iqr[0], 50, iqr[1], whisk[1]])
        ax.hlines(yi, q5, q95, color=Colors.INK, lw=1.3, zorder=3)
        ax.vlines([q5, q95], yi-0.12, yi+0.12, color=Colors.INK, lw=1.1, zorder=3)

        box_h = 0.26
        rect = Rectangle((q25, yi - box_h/2), q75 - q25, box_h,
                         facecolor='none', edgecolor=Colors.INK, lw=1.3, zorder=3.2)
        ax.add_patch(rect)
        ax.vlines(q50, yi - box_h/2, yi + box_h/2, color=Colors.INK, lw=1.8, zorder=3.3)

    ax.set_ylim(-0.6, n - 1 + 0.6)

    # Apply polish
    y_positions = np.arange(n)
    _polish_parameter_axes(ax, y_positions, show_horizontal_lines=show_baseline_lines)
    
    plt.tight_layout()
    return fig, ax


def plot_calibration(
    y_true, y_pred, *, n_bins=10, colorset_hex="#C41E3A",
    kind="prob", figsize=(6, 6), 
    show_bars=True,
    bar_style='shaded',  # 'shaded', 'caps', or 'both'
    bar_alpha=0.20,
    ci_level=0.95, 
    smooth_band=True,  # interpolate for smoother shaded bands
    xlabel=None, ylabel=None, title=None
):
    """
    Calibration plot: predicted vs observed with 45° reference line.
    
    Parameters
    ----------
    y_true : array-like
        Observed values (0/1 for prob; continuous for reg)
    y_pred : array-like
        Model predictions (probabilities [0,1] or continuous)
    n_bins : int
        Number of bins for grouping predictions (used for prob only)
    colorset_hex : str
        Color for calibration curve and points
    kind : {"prob", "reg"}
        "prob" for probability calibration, "reg" for regression calibration
    figsize : tuple
        Figure size
    show_bars : bool
        Show uncertainty bands/bars
    bar_style : {"shaded", "caps", "both"}
        Style for uncertainty visualization:
        - 'shaded': smooth filled band (default, matches module aesthetic)
        - 'caps': traditional error bars with caps
        - 'both': overlay both styles
    bar_alpha : float
        Alpha for shaded band (or error bar alpha for 'caps')
    smooth_band : bool
        If True and bar_style includes 'shaded', interpolate between points
        for a smoother band. Recommended when n_bins > 5.
    xlabel, ylabel, title : str or None
        Axis labels and title
        
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    set_plot_style()
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Remove NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if y_true.size == 0:
        raise ValueError("No valid data points after removing NaN/Inf")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # --- Styling ---
    ax.set_facecolor("#FCFCFD")
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Light grid
    ax.grid(True, which="major", color="#E5E7EB", linewidth=0.8, alpha=0.25, zorder=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", color="#ECEFF3", linewidth=0.6, alpha=0.12, zorder=0)
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)
    
    # --- Determine axis range ---
    if kind == "prob":
        ax_min, ax_max = 0, 1
    else:
        all_vals = np.concatenate([y_true, y_pred])
        ax_min = np.min(all_vals)
        ax_max = np.max(all_vals)
        pad = 0.05 * (ax_max - ax_min)
        ax_min -= pad
        ax_max += pad
    
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect('equal', adjustable='box')
    
    # --- 45° reference line (perfect calibration) ---
    ax.plot([ax_min, ax_max], [ax_min, ax_max], 
            color='#9CA3AF', linestyle='--', linewidth=1.5, 
            alpha=0.6, zorder=1, label='Perfect calibration')
    
    # --- Binning and calibration curve ---
    if kind == "prob":
        # Bin edges for predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        mean_pred = []
        mean_true = []
        se_true = []
        counts = []
        
        for i in range(n_bins):
            bin_mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                bin_mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
            
            if bin_mask.sum() > 0:
                mean_pred.append(y_pred[bin_mask].mean())
                mean_true.append(y_true[bin_mask].mean())
                # Standard error
                se = y_true[bin_mask].std(ddof=1) / np.sqrt(bin_mask.sum()) if bin_mask.sum() > 1 else 0
                se_true.append(se)
                counts.append(bin_mask.sum())
        
        mean_pred = np.array(mean_pred)
        mean_true = np.array(mean_true)
        se_true = np.array(se_true)
        counts = np.array(counts)

        if _HAS_SCIPY:
            from scipy.stats import norm
            z = norm.ppf(1 - (1 - ci_level) / 2)
        else:
            # Fallback for common values
            z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(ci_level, 1.96)
        
        ci_half_width = z * se_true
        upper = mean_true + ci_half_width
        lower = mean_true - ci_half_width
        
        # --- Uncertainty visualization ---
        if show_bars and len(mean_pred) > 1:         
            # Shaded band
            if bar_style in ('shaded', 'both'):
                if smooth_band and len(mean_pred) >= 3:
                    # Interpolate for smoother band
                    try:
                        from scipy.interpolate import PchipInterpolator
                        
                        # Sort by mean_pred for interpolation
                        sort_idx = np.argsort(mean_pred)
                        x_sorted = mean_pred[sort_idx]
                        y_sorted = mean_true[sort_idx]
                        upper_sorted = upper[sort_idx]
                        lower_sorted = lower[sort_idx]
                        
                        # Create smooth x values
                        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                        
                        # Interpolate upper and lower bounds
                        interp_upper = PchipInterpolator(x_sorted, upper_sorted)
                        interp_lower = PchipInterpolator(x_sorted, lower_sorted)
                        
                        upper_smooth = interp_upper(x_smooth)
                        lower_smooth = interp_lower(x_smooth)
                        
                        # Clip to valid probability range
                        upper_smooth = np.clip(upper_smooth, 0, 1)
                        lower_smooth = np.clip(lower_smooth, 0, 1)
                        
                        ax.fill_between(
                            x_smooth, lower_smooth, upper_smooth,
                            color=colorset_hex, alpha=bar_alpha,
                            linewidth=0, zorder=1.5
                        )
                    except ImportError:
                        # Fall back to simple fill_between
                        ax.fill_between(
                            mean_pred, lower, upper,
                            color=colorset_hex, alpha=bar_alpha,
                            linewidth=0, zorder=1.5
                        )
                else:
                    # Simple fill between points
                    ax.fill_between(
                        mean_pred, lower, upper,
                        color=colorset_hex, alpha=bar_alpha,
                        linewidth=0, zorder=1.5
                    )
            
            # Traditional error bars with caps
            if bar_style in ('caps', 'both'):
                caps_alpha = bar_alpha if bar_style == 'caps' else bar_alpha * 0.6
                ax.errorbar(
                    mean_pred, mean_true, yerr=se_true,
                    fmt='none', ecolor=colorset_hex, alpha=caps_alpha,
                    linewidth=1.5, capsize=3, capthick=1.5, zorder=2
                )
        
        # Calibration curve
        if len(mean_pred) > 1:
            # Sort for proper line drawing
            sort_idx = np.argsort(mean_pred)
            ax.plot(
                mean_pred[sort_idx], mean_true[sort_idx], 
                color=colorset_hex, linewidth=2.5, alpha=0.85, 
                zorder=3, label='Calibration curve'
            )
        
        # Points
        ax.scatter(
            mean_pred, mean_true, s=80, color=colorset_hex,
            edgecolors='white', linewidths=1.5, alpha=0.95, zorder=4
        )
        
    else:  # kind == "reg"
        # Scatter all points with low alpha
        ax.scatter(
            y_pred, y_true, s=20, color=colorset_hex,
            alpha=0.15, edgecolors='none', zorder=2
        )
    
    # --- Labels ---
    if xlabel is None:
        xlabel = "Mean predicted probability" if kind == "prob" else "Predicted value"
    if ylabel is None:
        ylabel = "Observed rate" if kind == "prob" else "Observed value"
    
    ax.set_xlabel(xlabel, fontsize=11, color=Colors.INK)
    ax.set_ylabel(ylabel, fontsize=11, color=Colors.INK)
    
    if title:
        ax.set_title(title, fontsize=12, color=Colors.INK, pad=12, fontweight=500)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=False, 
             edgecolor='#E5E7EB', framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    return fig, ax


def plot_model_comparison(df, metric='waic', figsize=(10, 6), orientation='vertical'):
    """
    Plot model comparison with gradient-shaded bars and solid edges.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'model', metric (e.g. 'waic' or 'loo'), 
        and optionally 'se' for standard error
    metric : str
        Column name for the metric to plot ('waic' or 'loo')
    figsize : tuple
        Figure size
    orientation : str
        'vertical' or 'horizontal'
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    set_plot_style()
    
    # Prepare data
    plot_df = df.copy()
    plot_df = plot_df.sort_values(metric, ascending=True)  # Best (lowest) first
    
    # Identify winner (lowest value)
    best_idx = plot_df[metric].idxmin()
    plot_df['is_winner'] = plot_df.index == best_idx
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_models = len(plot_df)
    positions = np.arange(n_models)
    values = plot_df[metric].values
    models = plot_df['model'].values
    is_winner = plot_df['is_winner'].values
    
    bar_width = 0.7
    
    if orientation == 'vertical':
        # Vertical bars
        for i, (pos, val, winner) in enumerate(zip(positions, values, is_winner)):
            edge_color = Colors.RUBY1 if winner else Colors.INK
            
            # Gradient fill inside bar
            if winner:
                gradient = np.linspace(0.25, 0.6, 256).reshape(256, 1)
                rgba = np.dstack([
                    np.full((256, 1), 0.831),  # R for RUBY1
                    np.full((256, 1), 0.078),  # G
                    np.full((256, 1), 0.353),  # B
                    gradient * 0.85
                ])
            else:
                gradient = np.linspace(0.2, 0.6, 256).reshape(256, 1)
                rgba = np.dstack([
                    np.full((256, 1), 0.420),  # R for INK
                    np.full((256, 1), 0.447),  # G
                    np.full((256, 1), 0.502),  # B
                    gradient * 0.75
                ])
            
            ax.imshow(
                rgba,
                extent=[pos - bar_width/2, pos + bar_width/2, 0, val],
                aspect='auto', origin='lower', zorder=1, interpolation='bicubic'
            )
            
            # Solid edge rectangle
            rect = Rectangle(
                (pos - bar_width/2, 0), bar_width, val,
                facecolor='none', edgecolor=edge_color, 
                linewidth=2, zorder=2
            )
            ax.add_patch(rect)
        
        # Error bars if available
        if 'se' in plot_df.columns:
            ax.errorbar(positions, values, yerr=plot_df['se'].values,
                       fmt='none', ecolor=Colors.INK, elinewidth=1.5, 
                       capsize=4, capthick=1.5, alpha=0.6, zorder=3)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_xlim(-0.5, n_models - 0.5)
        ax.set_ylim(0, values.max() * 1.05)
        
        # Apply polished styling without horizontal gridlines
        ax.set_facecolor("#FCFCFD")
        ax.grid(True, which="major", axis="y", color="#E5E7EB",
                linewidth=0.8, alpha=0.25, zorder=0)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", axis="y", color="#ECEFF3",
                linewidth=0.6, alpha=0.12, zorder=0)
        ax.grid(False, axis='x')
        ax.set_axisbelow(True)
        
        ax.spines['left'].set_color('#D1D5DB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)
        
        ax.set_ylabel(metric.upper(), fontsize=11, color=Colors.INK)
        ax.set_xlabel('Model', fontsize=11, color=Colors.INK)
        
    else:  # horizontal
        for i, (pos, val, winner) in enumerate(zip(positions, values, is_winner)):
            edge_color = Colors.RUBY1 if winner else Colors.INK
            
            # Gradient fill inside bar (left to right)
            if winner:
                gradient = np.linspace(0.2, 0.6, 256).reshape(1, 256)
                rgba = np.dstack([
                    np.full((1, 256), 0.831),
                    np.full((1, 256), 0.078),
                    np.full((1, 256), 0.353),
                    gradient * 0.85
                ])
            else:
                gradient = np.linspace(0.2, 0.6, 256).reshape(1, 256)
                rgba = np.dstack([
                    np.full((1, 256), 0.420),
                    np.full((1, 256), 0.447),
                    np.full((1, 256), 0.502),
                    gradient * 0.75
                ])
            
            ax.imshow(
                rgba,
                extent=[0, val, pos - bar_width/2, pos + bar_width/2],
                aspect='auto', origin='lower', zorder=1, interpolation='bicubic'
            )
            
            # Solid edge rectangle
            rect = Rectangle(
                (0, pos - bar_width/2), val, bar_width,
                facecolor='none', edgecolor=edge_color,
                linewidth=2, zorder=2
            )
            ax.add_patch(rect)
        
        # Error bars if available
        if 'se' in plot_df.columns:
            ax.errorbar(values, positions, xerr=plot_df['se'].values,
                       fmt='none', ecolor=Colors.INK, elinewidth=1.5,
                       capsize=4, capthick=1.5, alpha=0.6, zorder=3)
        
        ax.set_yticks(positions)
        ax.set_yticklabels(models)
        ax.set_ylim(-0.5, n_models - 0.5)
        ax.set_xlim(0, values.max() * 1.05)
        
        # Apply polished styling without horizontal gridlines
        ax.set_facecolor("#FCFCFD")
        ax.grid(True, which="major", axis="x", color="#E5E7EB",
                linewidth=0.8, alpha=0.25, zorder=0)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", axis="x", color="#ECEFF3",
                linewidth=0.6, alpha=0.12, zorder=0)
        ax.grid(False, axis='y')
        ax.set_axisbelow(True)
        
        ax.spines['left'].set_color('#D1D5DB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)
        
        ax.set_xlabel(metric.upper(), fontsize=11, color=Colors.INK)
        ax.set_ylabel('Model', fontsize=11, color=Colors.INK)
    
    plt.tight_layout()
    return fig, ax

# def plot_delta_ic_comparison(
#     results,
#     metric='loo',
#     figsize=(14, 6),
#     *,
#     show_absolute=True,
#     show_delta=True
# ):
#     """
#     Compare absolute IC and ΔIC with unified styling.
#     - Winner highlighted in red, others grey.
#     - Grey background (#FCFCFD) on all shown panels.
#     - Left padding on ΔIC so the winner at 0 is clearly visible.
#     - Toggle panels via show_absolute / show_delta.

#     Returns
#     -------
#     fig, axes_dict
#         axes_dict keys: 'absolute' and/or 'delta' depending on which are shown.
#     """
#     if not (show_absolute or show_delta):
#         raise ValueError("At least one of show_absolute or show_delta must be True.")
#     set_plot_style()

#     # ---------- helpers ----------
#     def apply_axes_style(ax):
#         ax.set_facecolor("#FCFCFD")
#         ax.grid(True, which="major", axis="x", color="#E5E7EB",
#                 linewidth=0.8, alpha=0.25, zorder=0)
#         ax.grid(False, axis='y')
#         ax.set_axisbelow(True)
#         ax.spines['left'].set_color('#D1D5DB')
#         ax.spines['bottom'].set_color('#E5E7EB')
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.tick_params(axis='both', colors=Colors.INK, labelsize=9, length=4)

#     def draw_points_with_errors(ax, xvals, yvals, winners, sevals=None):
#         for i, (x, y) in enumerate(zip(xvals, yvals)):
#             c = Colors.RUBY1 if winners[i] else Colors.INK
#             ax.scatter(x, y, s=120, c=c, edgecolors='white',
#                        linewidths=2, zorder=3, alpha=0.95, marker='o')
#             if sevals is not None:
#                 ax.errorbar(x, y, xerr=sevals[i], fmt='none',
#                             ecolor=c, elinewidth=2, capsize=5,
#                             capthick=2, alpha=0.5, zorder=2)

#     # ---------- prep data ----------
#     ready_key = f'{metric}'
#     delta_key = f'd{metric}'
#     if ready_key not in results or delta_key not in results:
#         raise ValueError(f"Results must contain '{ready_key}' and '{delta_key}'")

#     df_ready = results[ready_key].copy().sort_values(metric, ascending=True)
#     df_delta = results[delta_key].copy().sort_values(metric, ascending=True)

#     models = df_ready['model'].values
#     abs_values = df_ready[metric].values
#     delta_values = df_delta[metric].values
#     se_abs = df_ready['se'].values if 'se' in df_ready.columns else None
#     se_delta = df_delta['se'].values if 'se' in df_delta.columns else None

#     n_models = len(df_ready)
#     positions = np.arange(n_models)

#     winners_mask = np.zeros(n_models, dtype=bool)
#     if n_models > 0:
#         winners_mask[0] = True

#     # ---------- figure & axes ----------
#     panels = [p for p, flag in (('absolute', show_absolute), ('delta', show_delta)) if flag]
#     ncols = len(panels)

#     # If only one panel, make a single axes; otherwise side-by-side
#     if ncols == 1:
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#         axes = {panels[0]: ax}
#     else:
#         fig, axs = plt.subplots(1, ncols, figsize=figsize)
#         if ncols == 2:
#             ax_map = dict(zip(panels, axs))
#         else:  # future proofing, though ncols ∈ {1,2} here
#             ax_map = {p: axs[i] for i, p in enumerate(panels)}
#         axes = ax_map

#     # ---------- draw selected panels ----------
#     if show_absolute:
#         ax1 = axes['absolute']
#         draw_points_with_errors(ax1, abs_values, positions, winners_mask, se_abs)
#         ax1.set_yticks(positions)
#         ax1.set_yticklabels(models, fontsize=10)
#         ax1.set_xlabel(f'{metric.upper()} (absolute)', fontsize=11, fontweight='bold', color=Colors.INK)
#         ax1.set_ylabel('Model', fontsize=11, fontweight='bold', color=Colors.INK)
#         ax1.set_title('Absolute Information Criterion', fontsize=12, fontweight='bold', pad=15)
#         apply_axes_style(ax1)

#     if show_delta:
#         ax2 = axes['delta']
#         draw_points_with_errors(ax2, delta_values, positions, winners_mask, se_delta)

#         # reference line at 0
#         ax2.axvline(0, color=Colors.RUBY1, linewidth=2, linestyle='--', alpha=0.7, zorder=1)

#         ax2.set_yticks(positions)
#         # If both panels shown, omit y-labels on the right to avoid duplication
#         if show_absolute:
#             ax2.set_yticklabels([''] * n_models)
#         else:
#             ax2.set_yticklabels(models, fontsize=10)

#         ax2.set_xlabel(f'Δ{metric.upper()} (relative to best)', fontsize=11, fontweight='bold', color=Colors.INK)
#         ax2.set_title('Relative Model Support', fontsize=12, fontweight='bold', pad=15)

#         # Left padding so the 0-point winner isn't flush with the edge
#         max_delta = float(np.nanmax(delta_values)) if len(delta_values) else 0.0
#         right = max(10.0, max_delta * 1.15 if max_delta > 0 else 10.0)
#         left_pad = max(0.02 * right, 0.5)
#         ax2.set_xlim(-left_pad, right)

#         apply_axes_style(ax2)

#     plt.tight_layout()
#     return fig, axes
def plot_delta_ic_comparison(
    df,
    metric='WAIC',
    figsize=(14, 6),
    *,
    show_absolute=True,
    show_delta=True
):
    """
    Compare absolute IC and ΔIC with unified styling.
    
    Parameters
    ----------
    df : DataFrame
        Must contain columns: 'Model', metric (e.g. 'WAIC'), 
        'd{metric}' or 'dWAIC', and '{metric}_se' or 'WAIC_se'
    metric : str
        Column name for the IC metric (default 'WAIC')
    """
    if not (show_absolute or show_delta):
        raise ValueError("At least one of show_absolute or show_delta must be True.")
    set_plot_style()

    # ---------- helpers ----------
    def apply_axes_style(ax):
        ax.set_facecolor("#FCFCFD")
        ax.grid(True, which="major", axis="x", color="#E5E7EB",
                linewidth=0.8, alpha=0.25, zorder=0)
        ax.grid(False, axis='y')
        ax.set_axisbelow(True)
        ax.spines['left'].set_color('#D1D5DB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', colors=Colors.INK, labelsize=9, length=4)

    def draw_points_with_errors(ax, xvals, yvals, winners, sevals=None):
        for i, (x, y) in enumerate(zip(xvals, yvals)):
            c = Colors.RUBY1 if winners[i] else Colors.INK
            ax.scatter(x, y, s=120, c=c, edgecolors='white',
                       linewidths=2, zorder=3, alpha=0.95, marker='o')
            if sevals is not None:
                ax.errorbar(x, y, xerr=sevals[i], fmt='none',
                            ecolor=c, elinewidth=2, capsize=5,
                            capthick=2, alpha=0.5, zorder=2)

    # ---------- resolve column names ----------
    # Handle flexible column naming
    model_col = 'Model' if 'Model' in df.columns else 'model'
    delta_col = f'd{metric}' if f'd{metric}' in df.columns else f'd{metric.upper()}'
    se_col = f'{metric}_se' if f'{metric}_se' in df.columns else f'{metric.upper()}_se'
    
    # Validate required columns exist
    if metric not in df.columns:
        raise ValueError(f"DataFrame must contain '{metric}' column")
    if delta_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{delta_col}' column")

    # ---------- prep data ----------
    df_sorted = df.copy().sort_values(metric, ascending=True).reset_index(drop=True)

    models = df_sorted[model_col].values
    abs_values = df_sorted[metric].values
    delta_values = df_sorted[delta_col].values
    se_values = df_sorted[se_col].values if se_col in df_sorted.columns else None

    n_models = len(df_sorted)
    positions = np.arange(n_models)

    winners_mask = np.zeros(n_models, dtype=bool)
    if n_models > 0:
        winners_mask[0] = True

    # ---------- figure & axes ----------
    panels = [p for p, flag in (('absolute', show_absolute), ('delta', show_delta)) if flag]
    ncols = len(panels)

    if ncols == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = {panels[0]: ax}
    else:
        fig, axs = plt.subplots(1, ncols, figsize=figsize)
        axes = dict(zip(panels, axs))

    # ---------- draw selected panels ----------
    if show_absolute:
        ax1 = axes['absolute']
        draw_points_with_errors(ax1, abs_values, positions, winners_mask, se_values)
        ax1.set_yticks(positions)
        ax1.set_yticklabels(models, fontsize=10)
        ax1.set_xlabel(f'{metric} (absolute)', fontsize=11, fontweight='bold', color=Colors.INK)
        ax1.set_ylabel('Model', fontsize=11, fontweight='bold', color=Colors.INK)
        ax1.set_title('Absolute Information Criterion', fontsize=12, fontweight='bold', pad=15)
        apply_axes_style(ax1)

    if show_delta:
        ax2 = axes['delta']
        draw_points_with_errors(ax2, delta_values, positions, winners_mask, se_values)

        ax2.axvline(0, color=Colors.RUBY1, linewidth=2, linestyle='--', alpha=0.7, zorder=1)

        ax2.set_yticks(positions)
        if show_absolute:
            ax2.set_yticklabels([''] * n_models)
        else:
            ax2.set_yticklabels(models, fontsize=10)

        ax2.set_xlabel(f'Δ{metric} (relative to best)', fontsize=11, fontweight='bold', color=Colors.INK)
        ax2.set_title('Relative Model Support', fontsize=12, fontweight='bold', pad=15)

        max_delta = float(np.nanmax(delta_values)) if len(delta_values) else 0.0
        right = max(10.0, max_delta * 1.15 if max_delta > 0 else 10.0)
        left_pad = max(0.02 * right, 0.5)
        ax2.set_xlim(-left_pad, right)

        apply_axes_style(ax2)

    plt.tight_layout()
    return fig, axes

def plot_kde(data, color=None, title=None, xlabel='Value', ylabel='Density', 
             figsize=(8, 5), alpha_fill=0.25, bw_method='scott'):
    """
    Plot kernel density estimation with gradient fill.
    
    Parameters
    ----------
    data : array-like or dict
        If array-like: single distribution to plot
        If dict: multiple distributions with keys as labels
    color : str or list, optional
        Hex color(s) for the KDE line(s)
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    alpha_fill : float
        Alpha transparency for gradient fill
    bw_method : str or float
        Bandwidth method for KDE ('scott', 'silverman', or scalar)
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    set_plot_style()
    
    if not _HAS_SCIPY or gaussian_kde is None:
        raise ImportError("SciPy is required for KDE plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle single distribution
    if not isinstance(data, dict):
        data = {'': np.asarray(data)}
        show_legend = False
    else:
        show_legend = True
    
    # Handle colors
    if color is None:
        color = Colors.RUBY1
    
    if isinstance(color, str):
        if len(data) == 1:
            colors = [color]
        else:
            # Generate different shades for multiple distributions
            colors = [Colors.RUBY1, Colors.CERULEAN2, Colors.PERSIMMON3, 
                     Colors.INK, Colors.GREY][:len(data)]
    else:
        colors = color
    
    for (label, values), col in zip(data.items(), colors):
        values = np.asarray(values)
        values = values[~np.isnan(values)]  # Remove NaNs
        
        if len(values) == 0:
            continue
        
        # Compute KDE
        kde = gaussian_kde(values, bw_method=bw_method)
        
        # Create smooth x-axis
        x_min, x_max = values.min(), values.max()
        x_range = x_max - x_min
        x_eval = np.linspace(x_min - 0.1 * x_range, 
                            x_max + 0.1 * x_range, 500)
        density = kde(x_eval)
        
        # Gradient fill under curve
        _gradient_fill_under_curve(
            ax, x_eval, density, 
            baseline=0, 
            color=col, 
            alpha_top=alpha_fill
        )
        
        # Solid line on top
        line_label = label if show_legend and label else None
        ax.plot(x_eval, density, color=col, linewidth=2.5, 
                label=line_label, zorder=3)
    
    # Add legend if multiple distributions
    if show_legend:
        legend = ax.legend(loc='best', frameon=True, fontsize=10,
                          labelcolor=Colors.INK, edgecolor='#D1D5DB', fancybox=True)
        legend.get_frame().set_facecolor(Colors.GREY)
        legend.get_frame().set_alpha(0.45)
        legend.get_frame().set_linewidth(1)
    
    # Polished styling without horizontal gridlines
    ax.set_facecolor("#FCFCFD")
    ax.grid(True, which="major", axis="x", color="#E5E7EB",
            linewidth=0.8, alpha=0.25, zorder=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", axis="x", color="#ECEFF3",
            linewidth=0.6, alpha=0.12, zorder=0)
    ax.grid(False, axis='y')  # No horizontal gridlines
    ax.set_axisbelow(True)
    
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)
    
    ax.set_ylabel(ylabel, fontsize=11, color=Colors.INK)
    ax.set_xlabel(xlabel, fontsize=11, color=Colors.INK)
    ax.set_ylim(bottom=0)
    
    if title:
        ax.set_title(title, fontsize=13, color=Colors.INK, pad=15, weight='semibold')
    
    plt.tight_layout()
    return fig, ax


def plot_corr(
    x, y, colorset_hex=None, *, point_alpha=0.35, point_size=24,
    line_lw=2.0, band_alpha=0.18, ci=0.95, xlabel=None, ylabel=None,
    point_weights=None, ax=None
):
    """
    Clean correlation scatter with OLS fit and shaded CI band.
    
    Parameters
    ----------
    x, y : array-like
        Data to plot
    colorset_hex : str, optional
        Color for points and line
    point_alpha : float
        Alpha for scatter points
    point_size : float
        Size of scatter points
    line_lw : float
        Line width for fit
    band_alpha : float
        Alpha for confidence band
    ci : float
        Confidence level for band (e.g., 0.95)
    xlabel, ylabel : str, optional
        Axis labels
    point_weights : array-like, optional
        Weights for WLS
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on
    
    Returns
    -------
    fig, ax, (r, p, neff) : figure, axis, and statistics
        If ax provided, returns ax, (r, p, neff)
    """
    set_plot_style()
    
    if colorset_hex is None:
        colorset_hex = "#C41E3A"
    
    # Convert & mask NaNs
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    w = None if point_weights is None else np.asarray(point_weights, float)[mask]

    if x.size < 2:
        raise ValueError("Need at least two finite points.")

    # Axis labels
    if xlabel is None:
        xlabel = "x"
    if ylabel is None:
        ylabel = "y"

    # Axes
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        created_ax = True

    # Scatter
    ax.scatter(x, y, s=point_size, alpha=point_alpha,
               color=colorset_hex, edgecolors='none', zorder=2)

    # Fit + CI band
    xmin, xmax = np.min(x), np.max(x)
    xr = xmax - xmin + 1e-12
    xs = np.linspace(xmin - 0.04 * xr, xmax + 0.04 * xr, 200)

    alpha = 1 - ci
    yhat, lo, hi, (slope, intercept) = _ols_fit_ci_band(x, y, w, xs, alpha=alpha)

    # Shaded band first (behind the line)
    ax.fill_between(xs, lo, hi, color=colorset_hex, alpha=band_alpha, 
                   linewidth=0, zorder=1)

    # Fit line
    ax.plot(xs, yhat, color=colorset_hex, lw=line_lw, zorder=3)

    # Polished styling
    ax.set_facecolor("#FCFCFD")
    ax.grid(True, which="major", axis="x", color="#E5E7EB",
            linewidth=0.8, alpha=0.25, zorder=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", axis="x", color="#ECEFF3",
            linewidth=0.6, alpha=0.12, zorder=0)
    ax.grid(False, axis='y')  # No horizontal gridlines
    ax.set_axisbelow(True)
    
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)
    
    # Axes limits
    ymin = np.nanmin([y.min(), lo.min()])
    ymax = np.nanmax([y.max(), hi.max()])
    pad_y = 0.06 * (ymax - ymin + 1e-12)

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_xlabel(xlabel, color=Colors.INK, fontsize=11)
    ax.set_ylabel(ylabel, color=Colors.INK, fontsize=11)

    # Correlation stats
    r, p, neff = _pearson_r_p(x, y, w=w)
    if np.isnan(p):
        stats_text = f"r = {r:0.3f}\np = (n/a)"
    else:
        if p < 1e-4:
            p_str = "< 1e-4"
        elif p < 0.001:
            p_str = f"{p:.1e}"
        else:
            p_str = f"{p:.3f}"
        stats_text = f"r = {r:0.3f}\np = {p_str}"

    bbox_props = dict(boxstyle="round,pad=0.28",
                      facecolor="white", edgecolor="#E5E7EB", linewidth=0.9, alpha=0.95)
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=Colors.INK, bbox=bbox_props)
    
    if created_ax:
        plt.tight_layout()
        return ax.figure, ax, (r, p, neff)
    else:
        return ax, (r, p, neff)


def plot_subject_calibration(
    mean_pred,
    observed,
    *,
    colorset_hex=Colors.RUBY1,
    xlabel="Mean predicted P(Choose High)",
    ylabel="Observed P(Choose High)",
    title="Subject-Level Calibration",
    figsize=(5.6, 6.0),
    ax=None
):
    """
    Subject-level calibration: each point is a subject.

    - x: mean predicted P(Choose High) per subject
    - y: observed P(Choose High) per subject
    - shows 45° perfect-calibration line
    - no regression line; just r / p in a box
    - hollow circle markers for points
    """
    set_plot_style()

    x = np.asarray(mean_pred, float)
    y = np.asarray(observed, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if x.size < 2:
        raise ValueError("Need at least two subjects for calibration plot.")

    # Compute correlation (for the little stats box)
    r, p, neff = _pearson_r_p(x, y)

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 6.0))
        created_ax = True
    else:
        fig = ax.figure

    # --- base styling (match other plots) ---
    ax.set_facecolor("#FCFCFD")
    ax.grid(True, which="major", axis="both", color="#E5E7EB",
            linewidth=0.8, alpha=0.25, zorder=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", axis="both", color="#ECEFF3",
            linewidth=0.6, alpha=0.12, zorder=0)
    ax.set_axisbelow(True)

    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)

    # --- 45° perfect calibration line ---
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color=Colors.SLATE,
        linewidth=1.5,
        alpha=0.8,
        zorder=1,
    )

    # --- subject points: hollow circles ---
    ax.scatter(
        x, y,
        s=50,
        facecolors=colorset_hex,
        edgecolors=colorset_hex,
        linewidths=1.5,
        alpha=0.3,
        zorder=2
    )

    # limits & aspect
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    # labels
    ax.set_xlabel(xlabel, fontsize=11, color=Colors.INK)
    ax.set_ylabel(ylabel, fontsize=11, color=Colors.INK)

    # title with r
    if title is not None:
        ax.set_title(f"{title} (r = {r:.3f})",
                     fontsize=13, color=Colors.INK,
                     pad=12, weight="semibold")


    # Legend text
    p_str = "< 1e-4" if p < 1e-4 else f"{p:.3g}"
    legend_label = f"r = {r:.3f}, p {p_str}"

    # Combine into one legend entry
    ax.plot(
        [], color=Colors.SLATE, linewidth=1.5, label=legend_label
    )

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)  # Use the parameter here
        created_ax = True
    else:
        fig = ax.figure

 
    return fig, ax, (r, p, neff)
    
def plot_scatter(
    x, y, colorset_hex=None, *, point_alpha=0.35, point_size=50,
    xlabel=None, ylabel=None, title="", ax=None, figsize=(5.6, 4.6)
):
    """
    Clean scatter plot with consistent styling.
    
    Parameters
    ----------
    x, y : array-like
        Data to plot
    colorset_hex : str, optional
        Color for points
    point_alpha, point_size : float
        Point appearance
    xlabel, ylabel : str, optional
        Axis labels
    ax : matplotlib.axes.Axes, optional
        Existing axis
    figsize : tuple
        Figure size if creating new axis
    
    Returns
    -------
    fig, ax (or just ax if ax was provided)
    """
    set_plot_style()
    
    if colorset_hex is None:
        colorset_hex = "#C41E3A"
    
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if xlabel is None:
        xlabel = "x"
    if ylabel is None:
        ylabel = "y"

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, s=point_size, alpha=point_alpha,
               color=colorset_hex, edgecolors=colorset_hex,
               linewidths=1.5, zorder=2)

    # Standard styling
    ax.set_facecolor("#FCFCFD")
    ax.grid(True, which="major", axis="x", color="#E5E7EB",
            linewidth=0.8, alpha=0.25, zorder=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", axis="x", color="#ECEFF3",
            linewidth=0.6, alpha=0.12, zorder=0)
    ax.grid(False, axis='y')
    ax.set_axisbelow(True)
    
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)
    
    # Auto limits with padding
    pad_x = 0.06 * (x.max() - x.min() + 1e-12)
    pad_y = 0.06 * (y.max() - y.min() + 1e-12)
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)

    ax.set_xlabel(xlabel, color=Colors.INK, fontsize=11)
    ax.set_ylabel(ylabel, color=Colors.INK, fontsize=11)
    
    if title is not None:
        ax.set_title(f"{title}",
                        fontsize=13, color=Colors.INK,
                        pad=12, weight="semibold")
    
    if created_ax:
        plt.tight_layout()
        return ax.figure, ax
    return ax


def plot_histogram_kde(
    data,
    *,
    n_bins=15,
    color=None,
    title=None,
    xlabel="Accuracy",
    ylabel="Count",
    figsize=(8, 5),
    hist_alpha=0.6,
    kde_alpha=0.25,
    kde_line_alpha=0.85,
    show_kde=True,
    show_mean=True,
    show_median=False,
    mean_color=None,
    median_color=None,
    xlim=None,
    density=False,
    bw_method="scott",
):
    """
    Histogram with optional KDE overlay and reference lines.
    
    Ideal for visualizing participant accuracy distributions or any
    continuous measure across subjects.
    
    Parameters
    ----------
    data : array-like
        Values to plot (e.g., participant accuracies)
    n_bins : int
        Number of histogram bins
    color : str, optional
        Primary color (defaults to Colors.CERULEAN2)
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label (automatically changes to 'Density' if density=True)
    figsize : tuple
        Figure size
    hist_alpha : float
        Alpha for histogram bars
    kde_alpha : float
        Alpha for KDE gradient fill
    kde_line_alpha : float
        Alpha for KDE line
    show_kde : bool
        Whether to overlay KDE curve
    show_mean : bool
        Whether to show vertical mean line
    show_median : bool
        Whether to show vertical median line
    mean_color : str, optional
        Color for mean line (defaults to Colors.RUBY1)
    median_color : str, optional
        Color for median line (defaults to Colors.PERSIMMON3)
    xlim : tuple, optional
        X-axis limits (e.g., (0, 1) for accuracy)
    density : bool
        If True, normalize histogram to density (integrates to 1)
    bw_method : str or float
        Bandwidth method for KDE ('scott', 'silverman', or scalar)
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    stats : dict
        Dictionary with 'mean', 'median', 'std', 'n'
    
    Examples
    --------
    >>> accuracies = df.groupby('subj')['correct'].mean().values
    >>> fig, ax, stats = plot_histogram_kde(
    ...     accuracies,
    ...     xlabel='Accuracy',
    ...     title='Participant Accuracy Distribution',
    ...     xlim=(0, 1),
    ...     show_mean=True
    ... )
    """
    set_plot_style()
    
    # Handle optional scipy import for KDE
    if show_kde and not _HAS_SCIPY:
        import warnings
        warnings.warn("SciPy not available; disabling KDE overlay")
        show_kde = False
    
    # Defaults
    if color is None:
        color = Colors.CERULEAN2
    if mean_color is None:
        mean_color = Colors.RUBY1
    if median_color is None:
        median_color = Colors.PERSIMMON3
    
    # Clean data
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    
    if data.size == 0:
        raise ValueError("No finite data points to plot")
    
    # Compute stats
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data, ddof=1) if data.size > 1 else 0.0,
        'n': data.size,
    }
    
    # Update ylabel if density mode
    if density:
        ylabel = "Density"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine bin edges
    if xlim is not None:
        bin_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    else:
        bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
    
    # Plot histogram
    counts, edges, patches = ax.hist(
        data,
        bins=bin_edges,
        density=density,
        color=color,
        alpha=hist_alpha,
        edgecolor='white',
        linewidth=1.2,
        zorder=2,
    )
    
    # Add gradient effect to bars (subtle darker bottom)
    for patch in patches:
        patch.set_edgecolor('white')
        patch.set_linewidth(1.2)
    
    # KDE overlay
    if show_kde and data.size >= 2:
        from scipy.stats import gaussian_kde
        
        kde = gaussian_kde(data, bw_method=bw_method)
        
        # Extend x range slightly for smoother tails
        if xlim is not None:
            x_kde = np.linspace(xlim[0], xlim[1], 500)
        else:
            x_range = data.max() - data.min()
            x_kde = np.linspace(
                data.min() - 0.05 * x_range,
                data.max() + 0.05 * x_range,
                500
            )
        
        kde_vals = kde(x_kde)
        
        # Scale KDE to match histogram if not density
        if not density:
            bin_width = edges[1] - edges[0]
            kde_vals = kde_vals * data.size * bin_width
        
        # Gradient fill under KDE
        _gradient_fill_under_curve(
            ax, x_kde, kde_vals,
            baseline=0,
            color=color,
            alpha_top=kde_alpha,
        )
        
        # KDE line
        ax.plot(
            x_kde, kde_vals,
            color=color,
            linewidth=2.5,
            alpha=kde_line_alpha,
            zorder=3,
        )
    
    # Reference lines
    y_max = counts.max() if not density else kde_vals.max() if show_kde else counts.max()
    
    if show_mean:
        ax.axvline(
            stats['mean'],
            color=mean_color,
            linewidth=2,
            linestyle='--',
            alpha=0.85,
            zorder=4,
            label=f"Mean = {stats['mean']:.3f}",
        )
    
    if show_median:
        ax.axvline(
            stats['median'],
            color=median_color,
            linewidth=2,
            linestyle=':',
            alpha=0.85,
            zorder=4,
            label=f"Median = {stats['median']:.3f}",
        )
    
    # Legend if reference lines shown
    if show_mean or show_median:
        legend = ax.legend(
            loc='upper right',
            frameon=True,
            fontsize=10,
            labelcolor=Colors.INK,
            edgecolor='#D1D5DB',
            fancybox=True,
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(1)
    
    # Polished styling
    ax.set_facecolor("#FCFCFD")
    ax.grid(True, which="major", axis="y", color="#E5E7EB",
            linewidth=0.8, alpha=0.25, zorder=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", axis="y", color="#ECEFF3",
            linewidth=0.6, alpha=0.12, zorder=0)
    ax.grid(False, axis='x')
    ax.set_axisbelow(True)
    
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)
    
    ax.set_xlabel(xlabel, fontsize=11, color=Colors.INK)
    ax.set_ylabel(ylabel, fontsize=11, color=Colors.INK)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    
    if title:
        ax.set_title(title, fontsize=13, color=Colors.INK, pad=15, weight='semibold')
    
    # Add N annotation
    ax.text(
        0.97, 0.97,
        f"n = {stats['n']}",
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        color=Colors.INK,
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='#E5E7EB',
            alpha=0.9,
        ),
    )
    
    return fig, ax, stats


def plot_choice_by_threat_effort(
    data,
    *,
    subject_data=None,
    threat_col='threat',
    delta_effort_col='delta_effort',
    observed_col='p_high_effort_obs',
    predicted_col='p_high_effort_pred',
    subject_col='subj',
    subject_obs_col='obs_subj',
    subject_pred_col='pred_subj',
    threat_levels=None,
    delta_effort_levels=None,
    threat_colors=None,
    observed_label='Observed',
    predicted_label='Predicted',
    figsize=(12, 4),
    bar_width=0.35,
    ylabel='P(choose high effort)',
    xlabel='ΔEffort',
    title=None,
    panel_title_prefix='Threat = ',
    ylim=(0, 1),
    show_legend=True,
    legend_loc='upper right',
    error_col_obs=None,
    error_col_pred=None,
    capsize=3,
    hatch_pattern='///',
    hatch_linewidth=0.5,
    show_subject_points=False,
    subject_point_size=25,
    subject_point_alpha=0.5,
    subject_jitter=0.06,
    show_connecting_lines=False,
    connecting_line_alpha=0.15,
    connecting_line_width=0.8,
):
    """
    Create a publication-quality figure showing how choice behavior depends 
    on threat and effort difference, and how well the model reproduces that pattern.
    
    The plot is a 3-panel row of grouped bar charts:
    - One panel per threat level (e.g., 0.1, 0.5, 0.9)
    - Colors vary by threat level (blue→grey→red)
    - Within each panel, bars are grouped by ΔEffort (e.g., 0.2, 0.4, 0.6)
    - Observed bars: hatched/textured (unfilled)
    - Predicted bars: solid filled
    - Error bars (SE) can be shown for observed and/or predicted means.
    - Optional: Individual subject points overlaid on bars
    
    Parameters
    ----------
    data : pd.DataFrame
        Aggregated data with columns: threat_col, delta_effort_col, observed_col, predicted_col
    subject_data : pd.DataFrame, optional
        Subject-level data for scatter overlay. Must contain:
        - subject_col (e.g., 'subj')
        - threat_col (e.g., 'threat')
        - delta_effort_col (e.g., 'delta_effort')
        - subject_obs_col (e.g., 'obs_subj') - subject's observed P(high)
        - subject_pred_col (e.g., 'pred_subj') - subject's predicted P(high)
    show_subject_points : bool
        If True and subject_data provided, show individual subject points on bars
    subject_point_size : float
        Size of subject scatter points
    subject_point_alpha : float
        Alpha transparency for subject points
    subject_jitter : float
        Amount of horizontal jitter for subject points (fraction of bar width)
    show_connecting_lines : bool
        If True, draw lines connecting each subject's observed to predicted point
    connecting_line_alpha : float
        Alpha for connecting lines
    connecting_line_width : float
        Line width for connecting lines
    """
    set_plot_style()
    
    # --- Local helper for panel styling ---
    def _apply_panel_style(ax):
        """Apply polished axis styling consistent with plotter.py"""
        ax.set_facecolor("#FCFCFD")
        ax.grid(True, which="major", axis="y", color="#E5E7EB",
                linewidth=0.8, alpha=0.25, zorder=0)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", axis="y", color="#ECEFF3",
                linewidth=0.6, alpha=0.12, zorder=0)
        ax.grid(False, axis='x')
        ax.set_axisbelow(True)
        
        ax.spines['left'].set_color('#D1D5DB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', colors=Colors.INK, labelsize=10, length=4)
    
    # --- Handle input data ---
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    else:
        data = data.copy()
    
    # Validate required columns
    required_cols = [threat_col, delta_effort_col, observed_col, predicted_col]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate subject_data if show_subject_points is True
    if show_subject_points and subject_data is not None:
        subject_data = subject_data.copy()
        required_subj_cols = [subject_col, threat_col, delta_effort_col, subject_obs_col, subject_pred_col]
        missing_subj = [c for c in required_subj_cols if c not in subject_data.columns]
        if missing_subj:
            raise ValueError(f"Missing required subject_data columns: {missing_subj}")
    elif show_subject_points and subject_data is None:
        import warnings
        warnings.warn("show_subject_points=True but subject_data not provided. Skipping subject points.")
        show_subject_points = False
    
    # --- Determine levels ---
    if threat_levels is None:
        threat_levels = np.sort(data[threat_col].unique())
    else:
        threat_levels = np.asarray(threat_levels)
    
    if delta_effort_levels is None:
        delta_effort_levels = np.sort(data[delta_effort_col].unique())
    else:
        delta_effort_levels = np.asarray(delta_effort_levels)
    
    n_threats = len(threat_levels)
    n_efforts = len(delta_effort_levels)
    
    # --- Default threat colors: blue → grey → red ---
    if threat_colors is None:
        threat_colors = {}
        sorted_threats = np.sort(threat_levels)
        if len(sorted_threats) == 1:
            threat_colors[sorted_threats[0]] = Colors.SLATE
        elif len(sorted_threats) == 2:
            threat_colors[sorted_threats[0]] = Colors.CERULEAN2
            threat_colors[sorted_threats[1]] = Colors.RUBY1
        else:
            # For 3+ levels, use blue for lowest, red for highest, grey for middle
            threat_colors[sorted_threats[0]] = Colors.CERULEAN2
            threat_colors[sorted_threats[-1]] = Colors.RUBY1
            for t in sorted_threats[1:-1]:
                threat_colors[t] = Colors.SLATE
    
    # --- Create figure ---
    fig, axes = plt.subplots(1, n_threats, figsize=figsize, sharey=True)
    
    # Handle single panel case
    if n_threats == 1:
        axes = np.array([axes])
    
    # --- Summary statistics ---
    summary = {
        'threat_levels': threat_levels,
        'delta_effort_levels': delta_effort_levels,
        'n_panels': n_threats,
        'n_groups': n_efforts,
        'threat_colors': threat_colors,
    }
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # --- Plot each panel ---
    for panel_idx, threat in enumerate(threat_levels):
        ax = axes[panel_idx]
        
        # Get color for this threat level
        panel_color = threat_colors.get(threat, Colors.SLATE)
        
        # Filter data for this threat level
        panel_data = data[data[threat_col] == threat]
        
        # Get values for each delta effort level
        obs_vals = []
        pred_vals = []
        obs_errs = []
        pred_errs = []
        
        for de in delta_effort_levels:
            row = panel_data[panel_data[delta_effort_col] == de]
            if len(row) > 0:
                obs_vals.append(row[observed_col].values[0])
                pred_vals.append(row[predicted_col].values[0])
                
                # Error bars if specified
                if error_col_obs is not None and error_col_obs in data.columns:
                    obs_errs.append(row[error_col_obs].values[0])
                else:
                    obs_errs.append(np.nan)
                if error_col_pred is not None and error_col_pred in data.columns:
                    pred_errs.append(row[error_col_pred].values[0])
                else:
                    pred_errs.append(np.nan)
            else:
                obs_vals.append(np.nan)
                pred_vals.append(np.nan)
                obs_errs.append(np.nan)
                pred_errs.append(np.nan)
        
        obs_vals = np.asarray(obs_vals, dtype=float)
        pred_vals = np.asarray(pred_vals, dtype=float)
        obs_errs = np.asarray(obs_errs, dtype=float)
        pred_errs = np.asarray(pred_errs, dtype=float)
        
        # Bar positions
        x = np.arange(n_efforts)
        offset = bar_width / 2
        
        # --- Plot OBSERVED bars (hatched, unfilled) ---
        bars_obs = ax.bar(
            x - offset, obs_vals,
            width=bar_width,
            facecolor='white',
            edgecolor=panel_color,
            linewidth=1.8,
            hatch=hatch_pattern,
            zorder=2
        )
        for bar in bars_obs:
            bar.set_edgecolor(panel_color)
        
        plt.rcParams['hatch.linewidth'] = hatch_linewidth
        
        # --- Plot PREDICTED bars (solid filled) ---
        bars_pred = ax.bar(
            x + offset, pred_vals,
            width=bar_width,
            facecolor=panel_color,
            edgecolor=panel_color,
            alpha=0.85,
            linewidth=1.2,
            zorder=2
        )
        
        # --- Subject-level scatter points ---
        if show_subject_points and subject_data is not None:
            # Filter subject data for this threat level
            panel_subj_data = subject_data[subject_data[threat_col] == threat]
            
            # Set random seed for reproducible jitter
            np.random.seed(42)
            
            for de_idx, de in enumerate(delta_effort_levels):
                cell_subj = panel_subj_data[panel_subj_data[delta_effort_col] == de]
                
                if len(cell_subj) == 0:
                    continue
                
                n_subj = len(cell_subj)
                
                # Jitter positions
                jitter_obs = np.random.uniform(-subject_jitter, subject_jitter, n_subj)
                jitter_pred = np.random.uniform(-subject_jitter, subject_jitter, n_subj)
                
                x_obs = (de_idx - offset) + jitter_obs * bar_width
                x_pred = (de_idx + offset) + jitter_pred * bar_width
                
                y_obs = cell_subj[subject_obs_col].values
                y_pred = cell_subj[subject_pred_col].values
                
                # Draw connecting lines first (behind points)
                if show_connecting_lines:
                    for i in range(n_subj):
                        ax.plot(
                            [x_obs[i], x_pred[i]],
                            [y_obs[i], y_pred[i]],
                            color=panel_color,
                            alpha=connecting_line_alpha,
                            linewidth=connecting_line_width,
                            zorder=3,
                        )
                
                # Observed points: hollow circles
                ax.scatter(
                    x_obs, y_obs,
                    s=subject_point_size,
                    facecolors='white',
                    edgecolors=panel_color,
                    linewidths=1.0,
                    alpha=subject_point_alpha,
                    zorder=4,
                )
                
                # Predicted points: filled circles
                ax.scatter(
                    x_pred, y_pred,
                    s=subject_point_size,
                    facecolors=panel_color,
                    edgecolors='white',
                    linewidths=0.5,
                    alpha=subject_point_alpha,
                    zorder=4,
                )
        
        # --- Error bars ---
        if np.any(np.isfinite(obs_errs)):
            ax.errorbar(
                x - offset,
                obs_vals,
                yerr=obs_errs,
                fmt='none',
                ecolor=panel_color,
                alpha=0.7,
                elinewidth=1.5,
                capsize=capsize,
                capthick=1.5,
                zorder=5,
            )
        
        if np.any(np.isfinite(pred_errs)):
            ax.errorbar(
                x + offset,
                pred_vals,
                yerr=pred_errs,
                fmt='none',
                ecolor=panel_color,
                alpha=0.7,
                elinewidth=1.5,
                capsize=capsize,
                capthick=1.5,
                zorder=5,
            )
        
        # --- Panel styling ---
        _apply_panel_style(ax)
        
        # X-axis
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f'{de:.1f}' if isinstance(de, float) else str(de)
             for de in delta_effort_levels]
        )
        
        # Panel title
        if isinstance(threat, float):
            panel_title = f'{panel_title_prefix}{threat:.1f}'
        else:
            panel_title = f'{panel_title_prefix}{threat}'
        ax.set_title(
            panel_title,
            fontsize=12,
            color=Colors.INK,
            pad=10,
            fontweight='semibold'
        )
        
        # Y-axis (only leftmost panel)
        if panel_idx == 0:
            ax.set_ylabel(ylabel, fontsize=11, color=Colors.INK)
        
        # X-axis label
        ax.set_xlabel(xlabel, fontsize=11, color=Colors.INK)
        
        # Y-limits
        if ylim is not None:
            ax.set_ylim(ylim)
    
    # --- Create legend with proxy artists ---
    if show_legend:
        legend_color = Colors.INK
        
        handles = []
        labels = []
        
        # Bar patches
        obs_patch = Patch(
            facecolor='white',
            edgecolor=legend_color,
            hatch=hatch_pattern,
            linewidth=1.5,
        )
        handles.append(obs_patch)
        labels.append(observed_label)
        
        pred_patch = Patch(
            facecolor=legend_color,
            edgecolor=legend_color,
            alpha=0.85,
            linewidth=1.2,
        )
        handles.append(pred_patch)
        labels.append(predicted_label)
        
        # Subject point markers (if shown)
        if show_subject_points and subject_data is not None:
            obs_marker = Line2D(
                [0], [0],
                marker='o',
                color='white',
                markerfacecolor='white',
                markeredgecolor=legend_color,
                markeredgewidth=1.0,
                markersize=6,
                linestyle='None',
            )
            handles.append(obs_marker)
            labels.append('Subject (obs)')
            
            pred_marker = Line2D(
                [0], [0],
                marker='o',
                color='white',
                markerfacecolor=legend_color,
                markeredgecolor='white',
                markeredgewidth=0.5,
                markersize=6,
                linestyle='None',
            )
            handles.append(pred_marker)
            labels.append('Subject (pred)')
        
        legend = axes[0].legend(
            handles=handles,
            labels=labels,
            loc=legend_loc,
            frameon=True,
            fontsize=9,
            labelcolor=Colors.INK,
            edgecolor='#D1D5DB',
            fancybox=True
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(1)
    
    # --- Overall title ---
    if title:
        fig.suptitle(
            title,
            fontsize=14,
            color=Colors.INK,
            fontweight='semibold',
            y=1.02
        )
    
    plt.tight_layout()
    return fig, axes, summary

def plot_ppc_three_panels(
    df_ppc,
    *,
    threats=(0.1, 0.5, 0.9),
    threat_col="threat",
    delta_effort_col="delta_effort",
    mean_col="ppc_mean",
    observed_col="observed_p_high",
    hdi95_cols=("ppc_hdi_low", "ppc_hdi_high"),
    hdi50_cols=("ppc_hdi50_low", "ppc_hdi50_high"),
    threat_colors=None,
    figsize=(12, 4),
    ylim=(0.0, 1.0),
    yticks=(0.0, 0.5, 1.0),
    title=None,
):
    """
    Three-panel posterior predictive check:
    P(choose high) vs ΔEffort for each threat level.
    
    HDI bands are colored by threat level:
    - Low threat (0.1): Blue (CERULEAN2)
    - Medium threat (0.5): Grey (SLATE)
    - High threat (0.9): Red (RUBY1)

    Parameters
    ----------
    df_ppc : pd.DataFrame
        Must contain columns:
        - threat_col (e.g. 'threat')
        - delta_effort_col (e.g. 'delta_effort')
        - mean_col (e.g. 'ppc_mean')
        - observed_col (e.g. 'observed_p_high')
        - hdi95_cols: (low, high)
        - hdi50_cols: (low, high)
    threats : sequence
        Threat levels to show (one panel per threat).
    threat_colors : dict, optional
        Mapping of threat level -> hex color for HDI bands. If None, uses defaults:
        {0.1: CERULEAN2 (blue), 0.5: SLATE (grey), 0.9: RUBY1 (red)}
    figsize : tuple
        Figure size.
    """
    set_plot_style()

    threats = list(threats)
    n_panels = len(threats)

    # Default threat colors for HDI bands: blue (low) → grey (mid) → red (high)
    if threat_colors is None:
        threat_colors = {}
        sorted_threats = sorted(threats)
        if len(sorted_threats) == 1:
            threat_colors[sorted_threats[0]] = Colors.SLATE
        elif len(sorted_threats) == 2:
            threat_colors[sorted_threats[0]] = Colors.CERULEAN2  # low = blue
            threat_colors[sorted_threats[1]] = Colors.RUBY1      # high = red
        else:
            # 3+ levels: blue for lowest, red for highest, grey for middle
            threat_colors[sorted_threats[0]] = Colors.CERULEAN2   # low = blue
            threat_colors[sorted_threats[-1]] = Colors.RUBY1      # high = red
            for t in sorted_threats[1:-1]:
                threat_colors[t] = Colors.SLATE                   # mid = grey

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=figsize,
        sharey=True,
        constrained_layout=False
    )

    # Handle single axis case
    if n_panels == 1:
        axes = np.array([axes])

    # Convenience
    hdi95_low, hdi95_high = hdi95_cols
    hdi50_low, hdi50_high = hdi50_cols

    for panel_idx, (ax, t) in enumerate(zip(axes, threats)):
        sub = (
            df_ppc[df_ppc[threat_col] == t]
            .sort_values(delta_effort_col)
            .copy()
        )

        if sub.empty:
            ax.set_visible(False)
            continue

        x = sub[delta_effort_col].to_numpy()
        mean = sub[mean_col].to_numpy()
        obs = sub[observed_col].to_numpy()
        lo95 = sub[hdi95_low].to_numpy()
        hi95 = sub[hdi95_high].to_numpy()
        lo50 = sub[hdi50_low].to_numpy()
        hi50 = sub[hdi50_high].to_numpy()

        # Get color for this threat level (for HDI bands only)
        hdi_color = threat_colors.get(t, Colors.SLATE)

        # Background + grid style to match other functions
        ax.set_facecolor("#FCFCFD")

        # 95% HDI band (wider, lighter) - threat-colored
        ax.fill_between(
            x,
            lo95,
            hi95,
            color=hdi_color,
            alpha=0.18,
            linewidth=0.0,
            label="95% HDI" if panel_idx == 0 else None,
            zorder=1,
        )

        # 50% HDI band (narrower, slightly darker) - threat-colored
        ax.fill_between(
            x,
            lo50,
            hi50,
            color=hdi_color,
            alpha=0.35,
            linewidth=0.0,
            label="50% HDI" if panel_idx == 0 else None,
            zorder=2,
        )

        # Posterior predictive mean - keep grey dashed
        ax.plot(
            x,
            mean,
            "--",
            lw=2,
            color=Colors.INK,
            label="PPC mean" if panel_idx == 0 else None,
            zorder=3,
        )

        # Observed mean P(high) - keep black
        ax.scatter(
            x,
            obs,
            color="k",
            s=40,
            zorder=4,
            label="Observed" if panel_idx == 0 else None,
        )

        # Clean tick labels for ΔEffort
        unique_x = np.unique(x)
        ax.set_xticks(unique_x)
        ax.set_xticklabels(
            [f"{v:.1f}" if isinstance(v, float) else str(v) for v in unique_x]
        )

        # Panel title
        if isinstance(t, float):
            panel_title = f"Threat = {t:.1f}"
        else:
            panel_title = f"Threat = {t}"
        ax.set_title(
            panel_title,
            fontsize=12,
            color=Colors.INK,
            pad=10,
            fontweight="semibold",
        )

        # Axis styling (only left panel gets ylabel)
        ylabel = "P(choose high)" if ax is axes[0] else None
        style_axis(
            ax,
            ylabel=ylabel,
            xlabel="Δ Effort (H - L)",
            ylim=ylim,
            yticks=list(yticks) if yticks is not None else None,
        )

    # Legend only on first axis
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend = axes[0].legend(
            loc="upper right",
            frameon=True,
            fontsize=9,
            labelcolor=Colors.INK,
            edgecolor="#D1D5DB",
            fancybox=True,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(1)

    if title:
        fig.suptitle(
            title,
            fontsize=13,
            color=Colors.INK,
            fontweight="semibold",
            y=1.02,
        )

    plt.tight_layout()
    return fig, axes


# Optional: Create a convenience function to set all styles at once
def apply_publication_style():
    """Apply all publication-quality styling settings"""
    set_plot_style()
    return Colors  # Return Colors class for easy access to palette


# Export main components
__all__ = [
    'Colors',
    'set_plot_style',
    'apply_publication_style',
    'plot_subject_trials',
    'plot_ppc_subject_trials',
    'plot_parameter_half_violin_glow',
    'plot_calibration',
    'plot_model_comparison',
    'plot_delta_ic_comparison',
    'plot_subject_calibration',
    'plot_choice_by_threat_effort',
    'plot_kde',
    'plot_corr',
    'plot_scatter',
    'plot_ppc_three_panels',
    'style_axis'

]