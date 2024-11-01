import numpy as np
import pandas as pd

from scipy.stats import norm

from sklearn.base import BaseEstimator, RegressorMixin

# ----------------------------------------------------------------------------

class GaussianModelWrapper(BaseEstimator, RegressorMixin):
  """Wraps around Gaussian keras model.

  For use in evaluation codebase.
  
  Model should be already fit.
  """

  def __init__(
    self, 
    base_model,
    recalibrator,
    quantiles=[0.2, 0.4, 0.5, 0.6, 0.8],
  ):
    self.quantiles = quantiles

    def regressor_fn(X_in):
      y_pred, v_pred = base_model.predict(X_in)
      return y_pred, v_pred

    self.regressor_ = regressor_fn

  def predict(self, X):
    """ A reference implementation of a prediction for a classifier.
    Parameters
    ----------
    X : dataframe of shape = [n_rows, n_cols]
        The input samples.
    Returns
    -------
    df_pred : dataframe of shape = [n_rows, 2 + n_quantiles]
        Contains two columns: 'ds' and 'y_pred'
    """
    y, var = self.regressor_(X)

    sorted_q = sorted(self.quantiles)
    l_out = [
      y.flatten() + norm.ppf(q) * np.sqrt(var.flatten()) for q in sorted_q
    ]
    X_out = np.array(l_out).T
    df_pred = pd.DataFrame(X_out, columns=[str(q) for q in sorted_q])

    # make the mean an extra column (y_pred)
    df_pred['y_pred'] = y.flatten()

    # return output
    return df_pred

  def predict_quantile(self, X, q=0.5):
    """ Predict quantile using posterior uncertainty.
    Parameters
    ----------
    X : dataframe of shape = [n_rows, n_cols]
        The input samples.
    Returns
    -------
    y : dataframe of shape = [n_rows, 2]
        Contains two columns: 'ds' and 'y_pred', where y_pred is the quantile
    """
    y, variance = self.regressor_(X)

    df_pred = pd.DataFrame({
      'y_pred' : (y + norm.ppf(q) * np.sqrt(variance)).flatten()
    })

    # return output
    return df_pred

  def predict_proba(self, X, y):
    """ Predict cumulative probability p(y' <= y|x).
    Parameters
    ----------
    X : dataframe of shape = [n_rows, n_cols]
        The input samples.
    Returns
    -------
    y : dataframe of shape = [n_rows, 2]
        Contains two columns: 'ds' and 'y_pred', where y_pred is the density
    """
    mean, variance = self.regressor_.predict(X)

    # compute cumulative probability of observed points
    cum_prob = norm.cdf(y, loc=mean, scale=np.sqrt(variance))
    df_pred = pd.DataFrame({'y_pred' : cum_prob})

    # return output
    return df_pred

# ----------------------------------------------------------------------------

class MDNModelWrapper(BaseEstimator, RegressorMixin):
  """Wraps around Gaussian keras model.

  For use in evaluation codebase.
  
  Model should be already fit.
  """

  def __init__(
    self, 
    base_model,
    quantiles=[0.2, 0.4, 0.5, 0.6, 0.8],
  ):
    self.quantiles = quantiles

    def regressor_fn(X_in):
      y_pred, v_pred, w_pred = base_model.predict(X_in)
      return y_pred, v_pred, w_pred

    self.regressor_ = regressor_fn

  def predict(self, X):
    """ A reference implementation of a prediction for a classifier.
    Parameters
    ----------
    X : dataframe of shape = [n_rows, n_cols]
        The input samples.
    Returns
    -------
    df_pred : dataframe of shape = [n_rows, 2 + n_quantiles]
        Contains two columns: 'ds' and 'y_pred'
    """
    mu, var, w = self.regressor_(X)
    y = (mu * w).sum(axis=1)

    sorted_q = sorted(self.quantiles)
    # l_out = [y + norm.ppf(q) * np.sqrt(var) for q in sorted_q]
    # X_out = np.array(l_out).T
    df_pred = pd.DataFrame(columns=[str(q) for q in sorted_q])

    # make the mean an extra column (y_pred)
    df_pred['y_pred'] = y.flatten()

    # return output
    return df_pred

  def predict_quantile(self, X, q=0.5):
    """ Predict quantile using posterior uncertainty.
    Parameters
    ----------
    X : dataframe of shape = [n_rows, n_cols]
        The input samples.
    Returns
    -------
    y : dataframe of shape = [n_rows, 2]
        Contains two columns: 'ds' and 'y_pred', where y_pred is the quantile
    """
    mu, var, w = self.regressor_(X)

    # compute max scale of each distribution
    search_min = np.min(mu - 2*np.sqrt(var), axis=1)
    search_max = np.max(mu + 2*np.sqrt(var), axis=1)

    # create search bounds for each element
    n_steps = 1000
    search_steps = (1./n_steps) * np.arange(0,n_steps+1)[np.newaxis, :]
    search_list = (
      search_min[:, np.newaxis] * (1.-search_steps) 
      + search_max[:, np.newaxis] * search_steps
    )

    # compute the cdf for each element in search list
    K = mu.shape[1]
    cdf_vals_all = np.zeros(list(search_list.shape) + [K])
    for k in range(K):
        cdf_vals_all[:,:,k] = norm.cdf(search_list, loc=mu[:,[k]], scale=np.sqrt(var[:,[k]]))
    cdf_vals = np.sum(cdf_vals_all * w[:,np.newaxis,:], axis=2)

    # find the value in the search list that's closest to the query quantile
    q_idx = np.argmin(np.abs(cdf_vals - q), axis=1)
    q_vals = search_list[range(search_list.shape[0]), q_idx]

    # create output dataframe
    df_pred = pd.DataFrame({'y_pred' : q_vals})

    # return output
    return df_pred
