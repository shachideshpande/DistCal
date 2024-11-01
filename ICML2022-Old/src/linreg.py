"""
Bayesian Linear Regession.
"""

import numpy as np
import pandas as pd

from scipy.stats import norm

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import BayesianRidge

from afresh.engine.prediction.features.metrics import mean_rel_l1_error

# ----------------------------------------------------------------------------

class BayesianLinearRegressor(BaseEstimator, RegressorMixin):
  """ Bayesian Linear Regression forecaster.

  Parameters
  ----------
  quantiles : array-like, optional, default=[0.2, 0.4, 0.5, 0.6, 0.8]
      List of quantiles to estimate using the posterior uncertainty
  """
  def __init__(
    self,  
    quantiles=[0.2, 0.4, 0.5, 0.6, 0.8],
  ):
    self.quantiles = quantiles

  def fit(self, X, y):
    """ Fit estimator and choose hyper-parameters by cross-validation.
    Parameters
    ----------
    X : array-like or sparse matrix of shape = [n_samples, n_features]
        The training input samples.
    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        The target values.
    Returns
    -------
    self : object
        Returns self.
    """

    # initialize model
    self.regressor_ = BayesianRidge()

    # fit the pipeline
    self.regressor_.fit(X, y)
    
    # Return the estimator
    return self 

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
    # Check is fit had been called
    check_is_fitted(self, ['regressor_'])

    # predict using the regressor
    y, stdv = self.regressor_.predict(X, return_std=True)
    variance = stdv**2

    sorted_q = sorted(self.quantiles)
    l_out = [y + norm.ppf(q) * np.sqrt(variance) for q in sorted_q]
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
    # Check is fit had been called
    check_is_fitted(self, ['regressor_'])

    # predict using the regressor
    y, stdv = self.regressor_.predict(X, return_std=True)
    variance = stdv**2

    df_pred = pd.DataFrame({
      'y_pred' : y + norm.ppf(q) * np.sqrt(variance)
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
    # Check is fit had been called
    check_is_fitted(self, ['regressor_'])

    # predict mean and variance using the regressor
    mean, stdv = self.regressor_.predict(X, return_std=True)
    variance = stdv**2

    # compute cumulative probability of observed points
    cum_prob = norm.cdf(y, loc=mean, scale=np.sqrt(variance))
    df_pred = pd.DataFrame({'y_pred' : cum_prob})

    # return output
    return df_pred

  def predict_mv(self, X):
    """ Predict mean and variance
    Parameters
    ----------
    X : dataframe of shape = [n_rows, n_cols]
        The input samples.
    """
    # Check is fit had been called
    check_is_fitted(self, ['regressor_'])

    # predict mean and variance using the regressor
    mean, stdv = self.regressor_.predict(X, return_std=True)
    variance = stdv**2
    mean, variance = mean.flatten(), variance.flatten()

    # return output
    return mean, variance
