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
from keras.layers import Input, Dense, Dropout, PReLU, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from afresh_engine.prediction.features.metrics import mean_rel_l1_error
from afresh_engine.prediction.features import FeatureTransformer

# ----------------------------------------------------------------------------

class BayesianDNNForecaster(BaseEstimator, RegressorMixin):
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
    # initialize feature extractor
    base_clf = BayesianDNNRegressor(X.shape[1])
    base_clf.fit(X,y)
    self.regressor_ = base_clf
    
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
      y, variance = self.regressor_.predict(X)

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
    if self.scale_y:
      raise NotImplementedError()
    else:
      y, variance = self.regressor_.predict(X)

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
    n_epochs=300
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

    x = ConcreteDropout(Dense(128, activation=None))(inputs)
    x = PReLU()(x)

    x = ConcreteDropout(Dense(128, activation=None))(x)
    x = PReLU()(x)

    predictions = Dense(1, activation=None)(x)

    base_clf = Model(inputs=inputs, outputs=predictions)
    optimizer = Adam(lr=self.learning_rate)
    base_clf.compile(optimizer=optimizer, loss='mean_squared_error')

    base_clf.fit(X, y, batch_size=self.batch_size, epochs=self.n_epochs, verbose=0)

    self.regressor_ = K.function(
      [inputs, K.learning_phase()],
      base_clf.layers[-1].output
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

    result = np.zeros([self.mc_samples, len(X)])

    for sample in range(self.mc_samples):
      result[sample] = self.regressor_([X, 1]).flatten()

    prediction = result.mean(axis=0)
    variance = result.var(axis=0)

    # return output
    return prediction, variance

import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Dense, Lambda, Wrapper


class ConcreteDropout(Wrapper):
  """This wrapper allows to learn the dropout probability for any given input layer.
  ```python
      # as the first layer in a model
      model = Sequential()
      model.add(ConcreteDropout(Dense(8), input_shape=(16)))
      # now model.output_shape == (None, 8)
      # subsequent layers: no need for input_shape
      model.add(ConcreteDropout(Dense(32)))
      # now model.output_shape == (None, 32)
  ```
  `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
  for instance with a `Conv2D` layer:
  ```python
      model = Sequential()
      model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                input_shape=(299, 299, 3)))
  ```
  # Arguments
      layer: a layer instance.
      weight_regularizer:
          A positive number which satisfies
              $weight_regularizer = l**2 / (\tau * N)$
          with prior lengthscale l, model precision $\tau$ (inverse observation noise),
          and N the number of instances in the dataset.
          Note that kernel_regularizer is not needed.
      dropout_regularizer:
          A positive number which satisfies
              $dropout_regularizer = 2 / (\tau * N)$
          with model precision $\tau$ (inverse observation noise) and N the number of
          instances in the dataset.
          Note the relation between dropout_regularizer and weight_regularizer:
              $weight_regularizer / dropout_regularizer = l**2 / 2$
          with prior lengthscale l. Note also that the factor of two should be
          ignored for cross-entropy loss, and used only for the eculedian loss.
  """

  def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
               init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
      assert 'kernel_regularizer' not in kwargs
      super(ConcreteDropout, self).__init__(layer, **kwargs)
      self.weight_regularizer = weight_regularizer
      self.dropout_regularizer = dropout_regularizer
      self.is_mc_dropout = is_mc_dropout
      self.supports_masking = True
      self.p_logit = None
      self.p = None
      self.init_min = np.log(init_min) - np.log(1. - init_min)
      self.init_max = np.log(init_max) - np.log(1. - init_max)

  def build(self, input_shape=None):
      self.input_spec = InputSpec(shape=input_shape)
      if not self.layer.built:
          self.layer.build(input_shape)
          self.layer.built = True
      super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

      # initialise p
      self.p_logit = self.layer.add_weight(name='p_logit',
                                          shape=(1,),
                                          initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                          trainable=True)
      self.p = K.sigmoid(self.p_logit[0])

      # initialise regulariser / prior KL term
      input_dim = np.prod(input_shape[1:])  # we drop only last dim
      weight = self.layer.kernel
      kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
      dropout_regularizer = self.p * K.log(self.p)
      dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
      dropout_regularizer *= self.dropout_regularizer * input_dim
      regularizer = K.sum(kernel_regularizer + dropout_regularizer)
      self.layer.add_loss(regularizer)

  def compute_output_shape(self, input_shape):
      return self.layer.compute_output_shape(input_shape)

  def concrete_dropout(self, x):
      '''
      Concrete dropout - used at training time (gradients can be propagated)
      :param x: input
      :return:  approx. dropped out input
      '''
      eps = K.cast_to_floatx(K.epsilon())
      temp = 0.1

      unif_noise = K.random_uniform(shape=K.shape(x))
      drop_prob = (
          K.log(self.p + eps)
          - K.log(1. - self.p + eps)
          + K.log(unif_noise + eps)
          - K.log(1. - unif_noise + eps)
      )
      drop_prob = K.sigmoid(drop_prob / temp)
      random_tensor = 1. - drop_prob

      retain_prob = 1. - self.p
      x *= random_tensor
      x /= retain_prob
      return x

  def call(self, inputs, training=None):
      if self.is_mc_dropout:
          return self.layer.call(self.concrete_dropout(inputs))
      else:
          def relaxed_dropped_inputs():
              return self.layer.call(self.concrete_dropout(inputs))
          return K.in_train_phase(relaxed_dropped_inputs,
                                  self.layer.call(inputs),
                                  training=training)