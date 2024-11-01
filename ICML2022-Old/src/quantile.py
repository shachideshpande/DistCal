import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, Dense, Dropout, PReLU, Add, Lambda, Concatenate, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

# ----------------------------------------------------------------------------

def quantile_loss(y_true, y_pred, alpha):
  # note: alpha is a vector of same dim as y_true and y_pred
  loss_vector = tf.maximum(
      alpha * (y_true - y_pred), 
      (alpha - 1) * (y_true - y_pred)
  )
  return loss_vector

def make_quantile_recalibrator(n_inputs):
    q_in = Input(shape=(n_inputs,))
    a_in = Input(shape=(1,))
    inputs = Concatenate()([q_in, a_in])
    x1 = Dense(22, activation=None)(inputs)
    x1 = PReLU()(x1)
    x1 = Concatenate()([x1, Concatenate([inputs, inputs])])

    x2 = Dense(22, activation=None)(x1)
    x2 = PReLU()(x2)
    x2 = Concatenate()([x2, x1])

    q_out = Dense(1, activation=None)(x2)

    quantile_recalibrator = Model(
        inputs=[q_in, a_in], outputs=[q_out]
    )

    return quantile_recalibrator

class QuantileRegressor(keras.Model):
  def __init__(self, **kwargs):
    super(QuantileRegressor, self).__init__(**kwargs)
    self.dense1 = Dense(20, activation=None)
    self.prelu1 = PReLU()
    self.dense2 = Dense(20, activation=None)
    self.prelu2 = PReLU()
    self.dense3 = Dense(1, activation=None)
    
    # self.loss_tracker = keras.metrics.Mean(name="loss")
    self.mae_metric = keras.metrics.mean_absolute_error

  def call(self, inputs):
    q_in, a_in = inputs
    inputs = Concatenate()([q_in, a_in])
    d1 = self.dense1(inputs)
    x1 = self.prelu1(d1)
    x1 = Concatenate()([x1, Concatenate()([inputs, inputs])])
    x2 = self.prelu2(self.dense2(x1))
    x2 = Concatenate()([x1, x2])
    q_out = self.dense3(x2)

    return q_out

  def train_step(self, data):
    # unpack data
    x, y = data
    # generate alphas
    alpha = tf.random.uniform(shape=(y.shape[0],1))
    with tf.GradientTape() as tape:
      y_pred = self([x, alpha], training=True)
      loss = quantile_loss(y, y_pred, alpha)

    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))

    # compute median predictions
    alpha_median = 0.5*tf.ones(shape=(y.shape[0],1))
    y_pred_median = self([x, alpha_median], training=True)

    self.mae_metric.update_state(y, y_pred_median)
    self.loss_tracker.update_state(loss)
    return {
        # "loss": self.loss_tracker.result(), 
        "mae": self.mae_metric.result()
      }

  def test_step(self, data):
    # Unpack the data
    x, y = data
    # generate alphas
    alpha = tf.random.uniform(shape=(self.batch_size,1))
    y_pred = self([x, alpha], training=True)
    loss = quantile_loss(y, y_pred, alpha)

    # compute median predictions
    alpha_median = 0.5*tf.ones(shape=(self.batch_size,1))
    y_pred_median = self([x, alpha_median], training=True)

    # Update the metrics.
    self.mae_metric.update_state(y, y_pred_median)
    self.loss_tracker.update_state(loss)
    return {
        "loss": self.loss_tracker.result(), 
        "mae": self.mae_metric.result()
      }
