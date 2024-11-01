import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, Dense, Dropout, PReLU, Add, Lambda, Concatenate, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from .eval import pinball_loss

def gaussian_loss_tensor(y_true, mu, var):
    # c.f. https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf
    return (
        0.5 * K.square(y_true-mu) / (var+K.epsilon()) +
        0.5 * K.log(var+K.epsilon()) +
        0.5 * np.log(2*np.pi)
      )

def mdn_loss_tensor(y_true, mu, var, w):
    # c.f. https://ig248.gitlab.io/post/2019-02-16-deepquantiles-p1/
    inv_sigma_2 = 1 / (var + K.epsilon())
    phi = inv_sigma_2 * K.exp(-inv_sigma_2 * K.square(y_true - mu))
    return -K.log(K.sum(w * phi, axis=1) + K.epsilon())

def mdn_loss_tensor_v2(y_true, mu, var, w):
    # https://github.com/aalmah/ift6266amjad/blob/master/experiments/mdn.py
    exponent = -0.5 * K.square((y_true - mu)) / var
    normalizer = (2 * np.pi * var)
    exponent = exponent + K.log(w) - (.5)*K.log(normalizer) 
    max_exponent = K.max(exponent ,axis=1, keepdims=True)
    mod_exponent = exponent - max_exponent
    gauss_mix = K.sum(K.exp(mod_exponent),axis=1)
    log_gauss = max_exponent + K.log(gauss_mix) 
    # res = -K.mean(log_gauss)
    return -log_gauss

def make_gaussian_base_model(n_dim):
    inputs = Input(shape=(n_dim,))

    x = Dense(128, activation=None)(inputs)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation=None)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    mu = Dense(1, activation=None)(x)
    var = Dense(1, activation='softplus')(x)

    base_model = Model(inputs=inputs, outputs=[mu, var])

    return base_model, mu, var, inputs

def make_gaussian_loss_model(base_model, mu, var, inputs, learning_rate):
    # hack: https://ig248.gitlab.io/post/2019-02-16-deepquantiles-p1/
    GaussianLossLayer = Lambda(lambda args: gaussian_loss_tensor(*args))
    dummy_loss = lambda y_true, y_pred: K.mean(y_pred)

    # now we can define a loss model
    input_labels = Input((1, ), name='y')
    loss_output = GaussianLossLayer([input_labels, mu, var])

    loss_input = (
        inputs + [input_labels] if isinstance(inputs, list) 
        else [inputs, input_labels]
    )
    loss_model = Model(loss_input, loss_output)
    optimizer = Adam(lr=learning_rate)
    loss_model.compile(optimizer=optimizer, loss=dummy_loss)

    return loss_model

def make_gaussian_models(n_dim, learning_rate=3e-4):
    gaussian_model, mu, var, inputs = make_gaussian_base_model(n_dim)
    loss_model = make_gaussian_loss_model(
        gaussian_model, mu, var, inputs, learning_rate
    )

    return gaussian_model, loss_model

# ----------------------------------------------------------------------------

def make_gaussian_recalibrator():
    mu_in = Input(shape=(1,))
    var_in = Input(shape=(1,))
    inputs = Concatenate()([mu_in, var_in])

    x1 = Dense(32, activation=None)(inputs)
    # x1 = BatchNormalization()(x1)
    x1 = PReLU()(x1)

    x2 = Dense(32, activation=None)(x1)
    # x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    x2 = Concatenate()([x2, x1])

    # x = Dense(8, activation=None)(var_in)
    # x = PReLU()(x)

    # x = Dense(16, activation=None)(x)
    # x = PReLU()(x)

    mu_r = Dense(1, activation=None)(x2)
    # mu_r = Lambda(lambda x: x)(mu_in)
    var_r = Dense(1, activation='softplus')(x2)
    # var_r = Dense(1, activation='softplus')(x)
    
    mu_r = Add()([mu_r, mu_in])
    var_r = Add()([var_r, var_in])

    gaussian_recalibrator = Model(
        inputs=[mu_in, var_in], outputs=[mu_r, var_r]
    )

    return gaussian_recalibrator

def make_recalibrated_model(n_dim, base_model, recalibrator):
    inputs = Input(shape=(n_dim,))
    outputs = recalibrator(base_model(inputs))
    recalibrated_model = Model(inputs=inputs, outputs=outputs)
    return recalibrated_model

def make_gaussian_recalibrator_loss(
    recalibrated_model, base_model, learning_rate
):
    # hack: https://ig248.gitlab.io/post/2019-02-16-deepquantiles-p1/
    GaussianLossLayer = Lambda(lambda args: gaussian_loss_tensor(*args))
    dummy_loss = lambda y_true, y_pred: K.mean(y_pred)

    # now we can define a loss model
    input_labels = Input((1, ), name='y')
    mu_r, var_r = recalibrated_model.outputs
    loss_output = GaussianLossLayer([input_labels, mu_r, var_r])

    loss_model = Model(recalibrated_model.inputs + [input_labels], loss_output)
    optimizer = Adam(lr=learning_rate)
    base_model.trainable = False
    loss_model.compile(optimizer=optimizer, loss=dummy_loss)

    return loss_model

def make_recalibrated_gaussian_models(
    n_dim,
    learning_rate=3e-4,
    recalibrator_learning_rate=1e-3
):
    # create base models
    gaussian_model, loss_model = make_gaussian_models(n_dim, learning_rate)

    # create recalibrated model
    gaussian_recalibrator = make_gaussian_recalibrator()
    recalibrated_model = make_recalibrated_model(
        n_dim, gaussian_model, gaussian_recalibrator
    )
    
    # create loss model for the recalibrator
    gaussian_recalibrator_loss \
        = make_gaussian_recalibrator_loss(
            recalibrated_model, gaussian_model, recalibrator_learning_rate
        )

    return (
        gaussian_model,
        loss_model,
        recalibrated_model,
        gaussian_recalibrator_loss
    )

# ----------------------------------------------------------------------------

def make_mdn_recalibrator(n_components=2):
    mu_in = Input(shape=(1,))
    var_in = Input(shape=(1,))
    inputs = Concatenate()([mu_in, var_in])
    x1 = Dense(32, activation=None)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = PReLU()(x1)

    x2 = Dense(32, activation=None)(x1)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    # x2 = Concatenate()([x2, x1])

    mu_r = Dense(n_components, activation=None, kernel_initializer='orthogonal')(x2)
    var_r = Dense(n_components, activation=K.exp, kernel_initializer='orthogonal')(x2)
    w_r = Dense(n_components, activation='softmax', kernel_initializer='orthogonal')(x2)
    
    # mu_r = Add()([mu_r, mu_in])
    # var_r = Add()([mu_r, var_in])

    mdn_recalibrator = Model(
        inputs=[mu_in, var_in], outputs=[mu_r, var_r, w_r]
    )

    return mdn_recalibrator

def make_mdn_recalibrator_loss(
    recalibrated_model, base_model, learning_rate
):
    # hack: https://ig248.gitlab.io/post/2019-02-16-deepquantiles-p1/
    MDNLossLayer = Lambda(lambda args: mdn_loss_tensor_v2(*args))
    dummy_loss = lambda y_true, y_pred: K.mean(y_pred)

    # now we can define a loss model
    input_labels = Input((1, ), name='y')
    mu_r, var_r, w_r = recalibrated_model.outputs
    loss_output = MDNLossLayer([input_labels, mu_r, var_r, w_r])

    loss_model = Model(recalibrated_model.inputs + [input_labels], loss_output)
    optimizer = Adam(lr=learning_rate)
    if base_model is not None: base_model.trainable = False
    loss_model.compile(optimizer=optimizer, loss=dummy_loss)

    return loss_model

def make_recalibrated_mdn_models(
    n_dim,
    learning_rate=3e-4,
    recalibrator_learning_rate=1e-3
):
    # create base models
    gaussian_model, loss_model = make_gaussian_models(n_dim, learning_rate)

    # create recalibrated model
    mdn_recalibrator = make_mdn_recalibrator()
    recalibrated_model = make_recalibrated_model(
        n_dim, gaussian_model, mdn_recalibrator
    )
    
    # create loss model for the recalibrator
    mdn_recalibrator_loss = make_mdn_recalibrator_loss(
        recalibrated_model, gaussian_model, recalibrator_learning_rate
    )

    return (
        gaussian_model,
        loss_model,
        recalibrated_model,
        mdn_recalibrator_loss
    )

# ----------------------------------------------------------------------------

def make_regular_model(n_dim, learning_rate=3e-4):
    inputs = Input(shape=(n_dim,))

    x = Dense(128, activation=None)(inputs)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation=None)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation=None)(x)

    base_model = Model(inputs=inputs, outputs=predictions)
    optimizer = Adam(lr=learning_rate)
    gaussian_loss = lambda y_true, y_pred: K.mean(
        0.5 * K.square(y_true-y_pred[0]) / (ksoftplus(y_pred[1])+1e-4) +
        0.5 * K.log(ksoftplus(y_pred[1])+1e-4)
      )
    base_model.compile(optimizer=optimizer, loss='mean_squared_error')

    return base_model

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
  def __init__(self, batch_size, **kwargs):
    super(QuantileRegressor, self).__init__(**kwargs)
    self.batch_size=batch_size
    self.dense1 = Dense(20, activation=None)
    self.prelu1 = PReLU()
    self.dense2 = Dense(20, activation=None)
    self.prelu2 = PReLU()
    self.dense3 = Dense(1, activation=None)
    
    self.loss_tracker = keras.metrics.Mean(name="loss")
    self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
    self.pbl_tracker = keras.metrics.Mean(name="pbl")

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

  @property
  def metrics(self):
    return [
        self.loss_tracker,
        self.mae_metric,
    ]

  def train_step(self, data):
    # unpack data
    x, y = data
    # generate alphas
    alpha = tf.random.uniform(shape=(self.batch_size,1))
    with tf.GradientTape() as tape:
      y_pred = self([x, alpha], training=True)
      loss = quantile_loss(y, y_pred, alpha)

    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))

    # compute median predictions
    alpha_median = 0.5*tf.ones(shape=(self.batch_size,1))
    y_pred_median = self([x, alpha_median], training=True)

    self.mae_metric.update_state(y, y_pred_median)
    self.loss_tracker.update_state(loss)
    return {
        "loss": self.loss_tracker.result(), 
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

    # alphas=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # q_pred=[np.empty([y.shape[0], alphas.shape[0]])]
    # for i, alpha in enumerate(alphas):
    #     a_vec = alpha*tf.ones(shape=(self.batch_size,1))
    #     q_pred[:,i] = np.array(self([x, a_vec])).flatten()
    # pbl = pinball_loss(y, q_pred, alphas)

    # Update the metrics.
    self.mae_metric.update_state(y, y_pred_median)
    self.loss_tracker.update_state(loss)
    self.pbl_tracker.update_state(pbl)
    return {
        "loss": self.loss_tracker.result(), 
        "pbl": self.pbl_tracker.result(), 
        "mae": self.mae_metric.result()
      }
