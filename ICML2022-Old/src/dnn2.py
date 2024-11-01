"""
Bayesian Dense Neural Network implemented in Keras.
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

import keras.backend as K
from keras.layers import Input, Dense, Dropout, PReLU, Add
from keras.models import Model
from keras.optimizers import Adam

from afresh.engine.prediction.features.metrics import mean_rel_l1_error
from afresh.engine.prediction.features import FeatureTransformer

ksoftplus = lambda s: K.log(1+K.exp(s))

# ----------------------------------------------------------------------------

class CalibratedBayesianDNNForecaster(BaseEstimator, RegressorMixin):
  """ Bayesian Dense Neural Network forecaster.

  Parameters
  ----------
  features : array-like, optional, default=['ar', 'dates', 'holiday']
      List of feature extractors to use.

  ...

  quantiles : array-like, optional, default=[0.2, 0.4, 0.5, 0.6, 0.8]
      List of quantiles to estimate using the posterior uncertainty

  mc_samples : int, default=20
      Number of MC dropout samples to use to estimate confidence intervals.

  ar_indicators : bool (optional), default=True
      Include binary feature indicators of whether AR value was imputed.

  scale_x : bool (optional), default=True
      Scale input features to look like normal distribution

  scale_y : bool (optional), default=False
      Scale outputs to look like normal distribution
  """
  def __init__(
    self, 
    features=['ar', 'dates', 'holiday'], 
    quantiles=[0.2, 0.4, 0.5, 0.6, 0.8],
    mc_samples=20,
    hidden_layer_sizes=[[100]],
    learning_rate=[3e-4],
    batch_size=[100],
    ar_indicators=True,
    scale_x=True,
    scale_y=False,
  ):
    self.features = features
    self.quantiles = quantiles
    self.mc_samples = mc_samples
    self.hidden_layer_sizes = hidden_layer_sizes
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.ar_indicators = ar_indicators
    self.scale_x = scale_x
    self.scale_y = scale_y

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
    # initialize base model
    base_model = BayesianDNNRegressor(X.shape[1])
    base_model.fit(X,y)
    self.base_model_ = base_model

    # get training set predictions
    y_pred, v_pred = base_model.predict(X)
    
    # initialize recalibration model
    recalibrator = self.create_recalibrator()
    self.recalibrator_ = recalibrator

    # create recalibration dataset
    X_R = np.concatenate([y_pred[:,np.newaxis], v_pred[:,np.newaxis]], axis=1)
    y_R = y.copy()
    recalibrator.fit(
      # X_R, y_R, batch_size=100, epochs=500, verbose=1
      X_R, y_R, batch_size=100, epochs=1, verbose=0
    )

    # chain these model
    def regressor_fn(X_in):
      y_pred, v_pred = self.base_model_.predict(X_in)
      X_R = np.concatenate(
        [y_pred[:,np.newaxis], v_pred[:,np.newaxis]], axis=1
      )
      r_pred = recalibrator.predict(X_R)
      return y_pred, v_pred
      # return  r_pred[:,0], 0*r_pred[:,1]

    # save model
    self.regressor_ = regressor_fn
    
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
    if self.scale_y:
      raise NotImplementedError()
      # y_scaled = self.regressor_.predict(X)

      # # unscale the inputs
      # y = \
      # self.y_scaler_.inverse_transform(y_scaled.reshape([-1,1])).flatten()
    else:
      y, logvariance = self.regressor_(X)

    sorted_q = sorted(self.quantiles)
    l_out = [y + norm.ppf(q) * np.sqrt(np.exp(logvariance)) for q in sorted_q]
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
    if self.scale_y:
      raise NotImplementedError()
    else:
      y, variance = self.regressor_(X)

    df_pred = pd.DataFrame({
      'y_pred' : y + norm.ppf(q) * np.sqrt(variance)
    })

    # return output
    return df_pred

  def predict_density(self, X, y):
    """ Predict density p(y|x) using posterior uncertainty.
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

    # predict using the regressor
    if self.scale_y:
      raise NotImplementedError()
    else:
      mean, variance = self.regressor_.predict(X)

    # compute density of observed points
    # density = np.exp(-.5*(y-mean)**2 / variance) / np.sqrt(2*3.14*variance)
    density = norm.pdf(y, loc=mean, scale=np.sqrt(variance))
    df_pred = pd.DataFrame({'y_pred' : density})

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
    if self.scale_y:
      raise NotImplementedError()
    else:
      mean, variance = self.regressor_.predict(X)

    # compute cumulative probability of observed points
    cum_prob = norm.cdf(y, loc=mean, scale=np.sqrt(variance))
    df_pred = pd.DataFrame({'y_pred' : cum_prob})

    # return output
    return df_pred

  def create_recalibrator(self):
    inputs = Input(shape=(2,))

    x = Dense(32, activation=None)(inputs)
    x = PReLU()(x)

    x = Dense(32, activation=None)(x)
    x = PReLU()(x)

    predictions = Dense(2, activation=None)(x)
    outputs = Add()([predictions, inputs])

    recalibrator = Model(inputs=inputs, outputs=predictions)
    gaussian_loss = lambda y_true, y_pred: K.mean(
        0.5 * K.square(y_true-y_pred[0]) / (ksoftplus(y_pred[1])+1e-4) +
        0.5 * K.log(ksoftplus(y_pred[1])+1e-4)
      )
    # gaussian_loss = lambda y_true, y_pred: K.mean(K.square(y_true-y_pred[0]))
    recalibrator.compile(optimizer=Adam(lr=3e-4), loss=gaussian_loss)

    return recalibrator



# ----------------------------------------------------------------------------

class BayesianDNNRegressor(BaseEstimator, RegressorMixin):
  """ Sklearn wrapper for Bayesian Dense Neural Networks.

  We use dropout as an approximation for bayesian inference.
  See the paper by Gal and Gharamani for full details:
  https://arxiv.org/pdf/1506.02157.pdf
  https://stackoverflow.com/questions/43529931/
  how-to-calculate-prediction-uncertainty-using-keras

  Parameters
  ----------
  n_dim : int
      Number of input dimensions

  mc_samples : int, default=20
      Number of MC dropout samples to use to estimate confidence intervals.

  learning_rate : float, default=3e-4
      Learning rate

  batch_size : int, default=50
      Batch size

  n_epochs : int, default=50
      Number of training epochs
  """
  def __init__(
    self, 
    n_dim,
    mc_samples=20,
    learning_rate=3e-4,
    batch_size=50,
    n_epochs=400
  ):
    self.n_dim = n_dim
    self.mc_samples = mc_samples
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.n_epochs = n_epochs

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
    # initialize classifier
    inputs = Input(shape=(self.n_dim,))

    x = Dense(128, activation=None)(inputs)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation=None)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    predictions = Dense(2, activation=None)(x)

    base_clf = Model(inputs=inputs, outputs=predictions)
    optimizer = Adam(lr=self.learning_rate)
    gaussian_loss = lambda y_true, y_pred: K.mean(
        0.5 * K.square(y_true-y_pred[0]) / (ksoftplus(y_pred[1])+1e-4) +
        0.5 * K.log(ksoftplus(y_pred[1])+1e-4)
      )
    # base_clf.compile(optimizer=optimizer, loss='mean_squared_error')
    base_clf.compile(optimizer=optimizer, loss=gaussian_loss)

    base_clf.fit(X, y, batch_size=self.batch_size, epochs=self.n_epochs, verbose=1)

    self.regressor_ = K.function(
      [inputs, K.learning_phase()],
      [base_clf.layers[-1].output]
    )
    
    # Return the estimator
    return self 

  def predict(self, X):
    """ A reference implementation of a prediction for a classifier.
    Parameters
    ----------
    X : array of shape = [n_rows, n_cols]
        The input samples.
    Returns
    -------
    prediction : array of shape = [n_rows]
        Mean prediction

    variance : array of shape = [n_rows]
        Variance of posterior distribution
    """
    # Check is fit had been called
    check_is_fitted(self, ['regressor_'])
    out = self.regressor_([X,0])[0]
    prediction, variance = out[:,0], out[:,1]

    # result = np.zeros([self.mc_samples, len(X)])

    # for sample in range(self.mc_samples):
    #   result[sample] = self.regressor_([X, 1])[0].flatten()

    # prediction = result.mean(axis=0)
    # variance = result.var(axis=0)

    # return output
    return prediction, variance