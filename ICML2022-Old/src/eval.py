import math
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------

def gaussian_loss(y, m, v):
    epsilon = np.finfo(float).eps
    return (
        0.5 * (y-m)**2 / (v+epsilon) +
        0.5 * np.log(v+epsilon)
      ).mean()


# ----------------------------------------------------------------------------

def mean_calibration_error(p_exp, p_obs, n_obs):
    n_obs_bins = np.maximum(np.diff(n_obs),0)
    p_obs_bins = np.maximum(np.diff(p_obs), 0)
    p_exp_bins = np.maximum(np.diff(p_exp), 0)

    d_bins = ((p_exp_bins - p_obs_bins) ** 2) * n_obs_bins
    return np.sqrt(d_bins.mean())

def mean_calibration_error2(p_exp, p_obs, n_obs):
    n_obs_bins = np.maximum(np.diff(n_obs),0)
    p_obs_bins = np.maximum(np.diff(p_obs), 0)
    p_exp_bins = np.maximum(np.diff(p_exp), 0)

    d_bins = ((p_exp_bins - p_obs_bins) ** 2) * n_obs_bins
    return np.sqrt(d_bins.mean())

def pinball_loss(y_true, q_pred, alphas):
    """Pinball loss for one alpha
    Args:
        y_true: shape=[batch_size] - vector of labels
        q_pred: shape=[batch_size, n_alphas] - vector of predictions
        alphas: shape=[n_alphas]
    Returns:
        loss
    """
    loss_values = np.array([
        pinball_loss_alpha(y_true, q_pred[:,i], alpha) 
        for i, alpha in enumerate(alphas)
    ])
    return np.mean(loss_values)


def pinball_loss_alpha(y_true, q_pred, alpha):
    """Pinball loss for one alpha
    Args:
        y_true: shape=[batch_size] - vector of labels
        q_pred: shape=[batch_size] - vector of predictions
        alpha: shape=[1] or [batch_size]
    Returns:
        loss
    """
    loss_vector = np.maximum(
      alpha * (y_true - q_pred), 
      (alpha - 1) * (y_true - q_pred)
    )
    return np.mean(loss_vector)

# ----------------------------------------------------------------------------

def distributional_calibration_gaussian_map(y, mean, var):
    steps = np.arange(0,1,0.1)+0.1
    bins = var.min() * (1.-steps) + steps * (var.max()*1.01)
    bin_assignments = np.digitize(var, bins)
    
    empirical_var = np.zeros(len(bins),)
    centered_y = y.flatten() - mean.flatten()
    for idx in range(len(bins)):
        empirical_var[idx] = np.var(centered_y[bin_assignments==idx])
    
    return bins, bin_assignments, empirical_var

def distributional_calibration_gaussian_plot(y, mean, var):
    bins, bin_assignments, empirical_var = (
        distributional_calibration_gaussian_map(y, mean, var)
    )
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # plot true vs. predicted
    ax0 = plt.subplot(gs[0])
    ax0.plot(bins, empirical_var)

    # plot bucket sizes
    ax1 = plt.subplot(gs[1])
    ax1.hist(bin_assignments)

    plt.plot()

def distributional_calibration_error(y, mean, var):
    bins, bin_assignments, empirical_var = (
        distributional_calibration_gaussian_map(y, mean, var)
    )
    return np.sqrt(np.mean((bins-empirical_var)**2))