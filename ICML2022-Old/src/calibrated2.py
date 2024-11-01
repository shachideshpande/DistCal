"""Model recalibration.

Applies Platt scaling or isotonic regression to recalibrate regression model.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from scipy.stats import norm

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.simple import (
    make_gaussian_recalibrator, make_gaussian_loss_model,
    make_mdn_recalibrator, make_mdn_recalibrator_loss,
    QuantileRegressor
)


class DistributionCalibratedQuantileRegressor():
    """Calibrated forecaster.

    Wraps around sklearn-style models and calibrates them.

    Parameters
    ----------
    model : object
        Baseline sklearn model that will be trained incrementally
    """

    def __init__(self, model) -> None:
        self.model = model
        self.representation_quantiles = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]

    def fit(self, X, y, Xval, yval) -> 'CalibratedRegressor':
        """Fit estimator on a calibration set.

        Estimator need predict_proba function for outputting q estimates.

        Parameters
        ----------
        X : pandas.DataFrame of shape = [n_samples, n_features]
            The training input samples. Must have a ds column.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # compute mean and variance from model
        # mean, var = self.model.predict_mv(X)

        # compute list of quantiles from base model
        input_repr = np.zeros([y.shape[0], len(self.representation_quantiles)])
        input_repr_val = np.zeros([yval.shape[0], len(self.representation_quantiles)])
        for i, alpha in enumerate(self.representation_quantiles):
            input_repr[:, i] = self.model.predict_quantile(X, alpha).to_numpy().flatten()
            input_repr_val[:, i] = self.model.predict_quantile(Xval, alpha).to_numpy().flatten()

        # create data
        # X_recal = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])
        X_recal = input_repr
        X_recal = np.array(X_recal, dtype=np.float32)
        X_recal_val = input_repr_val
        X_recal_val = np.array(X_recal_val, dtype=np.float32)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        yval = np.array(yval, dtype=np.float32)

        # set up recalibrator
        batch_size = 32
        self.recalibrator_ = QuantileRegressor(batch_size=batch_size)
        a_vals = 0.5*np.ones([X_recal.shape[0],1], dtype=np.float32)
        self.recalibrator_.predict([X_recal, a_vals])

        
        dataset = tf.data.Dataset.from_tensor_slices((X_recal, y[:,np.newaxis]))
        dataset = dataset.shuffle(400).repeat(100).batch(batch_size, drop_remainder=True)
        self.recalibrator_.compile(optimizer=keras.optimizers.Adam(1e-3))

        self.recalibrator_.fit(
            dataset, epochs=10, validation_data=(X_recal_val, yval), 
            batch_size=batch_size
        )

        # self.recalibrator_.fit(X_recal, y[:,np.newaxis], epochs=200)
            # steps_per_epoch=dataset_size//batch_size)


    def predict(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict targets

        Parameters
        ----------
        X : pandas.DataFrame
            Input data. Must contain column 'ds', as well as extra feature
            columns.

        Returns
        -------
        df_pred : pandas.DataFrame
            Contains three columns: 'ds', 'horizon', and 'y_pred' and one
            column for each predicted quantile. The column 'horizon' always
            equals one. The column 'y_pred' is a point forecast that is
            allowed to be different for each predictor (e.g. could be the
            mean or the median).
        """
        # Check is fit had been called
        check_is_fitted(self, ['recalibrator_'])

        # compute mean and variance from model
        # mean, var = self.model.predict_mv(X)

        # compute list of quantiles from base model
        X_recal = np.zeros([X.shape[0], len(self.representation_quantiles)])
        for i, alpha in enumerate(self.representation_quantiles):
            X_recal[:, i] = self.model.predict_quantile(X, alpha).to_numpy().flatten()
        X_recal = np.array(X_recal, dtype=np.float32)
        X = np.array(X, dtype=np.float32)

        # create data
        # X_recal = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])

        # run recalibrator
        a_vals = 0.5*np.ones([X_recal.shape[0],1], dtype=np.float32)
        Y_pred = self.recalibrator_([X_recal, a_vals])
        q_vals = np.array(Y_pred)

        # return output
        return q_vals

    def predict_quantile(
        self, X: pd.DataFrame, q: float = 0.5
    ) -> pd.DataFrame:
        """Predict quantile of distribution.

        Parameters
        ----------
        X : pandas.DataFrame of shape = [n_rows, n_cols]
            The input samples.
        q : float or array-like, shape = [n_quantiles]
            One or more quantiles at which we want to make predictions

        Returns
        -------
        df_pred : pandas.DataFrame of shape = [n_rows, 1 + n_quantiles]
            Contains two columns: 'ds' and 'y_pred'
        """
        # stack input data with previously seen training data
        check_is_fitted(self, ['recalibrator_'])

        # compute mean and variance from model
        # mean, var = self.model.predict_mv(X)

        # compute list of quantiles from base model
        X_recal = np.zeros([X.shape[0], len(self.representation_quantiles)])
        for i, alpha in enumerate(self.representation_quantiles):
            X_recal[:, i] = self.model.predict_quantile(X, alpha).to_numpy().flatten()
        X_recal = np.array(X_recal, dtype=np.float32)

        # create data
        # X_recal = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])

        # run recalibrator
        a_vals = q*np.ones([X_recal.shape[0],1], dtype=np.float32)
        q_vals = self.recalibrator_([X_recal, a_vals])
        q_vals = np.array(q_vals)

        # return output
        df_pred = pd.DataFrame({'y_pred' : q_vals.flatten()})
        return df_pred



class DistributionCalibratedRegressor():
    """Calibrated forecaster.

    Wraps around sklearn-style models and calibrates them.

    Parameters
    ----------
    model : object
        Baseline sklearn model that will be trained incrementally
    """

    def __init__(self, model) -> None:
        self.model = model

    def fit(self, X, y) -> 'CalibratedRegressor':
        """Fit estimator on a calibration set.

        Estimator need predict_proba function for outputting q estimates.

        Parameters
        ----------
        X : pandas.DataFrame of shape = [n_samples, n_features]
            The training input samples. Must have a ds column.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # train base model if ncessary
        try:
            mean, var = self.model.predict_mv(X)
        except NotFittedError:
            self.model.fit(X, y)
            mean, var = self.model.predict_mv(X)

        # create data
        X_recal = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])

        # set up recalibrator
        self.recalibrator_ = make_gaussian_recalibrator()
        self.recalibrator_loss_ = make_gaussian_loss_model(
            base_model=self.recalibrator_, 
            mu=self.recalibrator_.outputs[0], 
            var=self.recalibrator_.outputs[1], 
            inputs=self.recalibrator_.inputs, 
            learning_rate=1e-2
        )

        # fit recalibrator
        self.recalibrator_loss_.fit(
            [X_recal[:,0], X_recal[:,1], y], np.empty(y.shape),
            batch_size=y.shape[0], epochs=5000, verbose=1
        )
        perf_number = self.recalibrator_loss_.evaluate(
            [X_recal[:,0], X_recal[:,1], y], np.empty(y.shape)
        )

        return perf_number

    def predict(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict targets

        Parameters
        ----------
        X : pandas.DataFrame
            Input data. Must contain column 'ds', as well as extra feature
            columns.

        Returns
        -------
        df_pred : pandas.DataFrame
            Contains three columns: 'ds', 'horizon', and 'y_pred' and one
            column for each predicted quantile. The column 'horizon' always
            equals one. The column 'y_pred' is a point forecast that is
            allowed to be different for each predictor (e.g. could be the
            mean or the median).
        """
        # Check is fit had been called
        check_is_fitted(self, ['recalibrator_', 'recalibrator_loss_'])

        # base model predictions
        mean, var = self.model.predict_mv(X)

        # create data for R
        X_out = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])

        # run R
        mean2, var2 = self.recalibrator_.predict([X_out[:,0], X_out[:,1]])
        mean2, var2 = mean2.flatten(), var2.flatten()

        # create output dataframe
        X_out2 = np.hstack([mean2[:, np.newaxis], var2[:, np.newaxis]])
        df_pred = pd.DataFrame(X_out2, columns=['y_pred', 'y_var'])

        # return output
        return df_pred

    def predict_quantile(
        self, X: pd.DataFrame, q: float = 0.5
    ) -> pd.DataFrame:
        """Predict quantile of distribution.

        Parameters
        ----------
        X : pandas.DataFrame of shape = [n_rows, n_cols]
            The input samples.
        q : float or array-like, shape = [n_quantiles]
            One or more quantiles at which we want to make predictions

        Returns
        -------
        df_pred : pandas.DataFrame of shape = [n_rows, 1 + n_quantiles]
            Contains two columns: 'ds' and 'y_pred'
        """
        # stack input data with previously seen training data
        check_is_fitted(self, ['recalibrator_', 'recalibrator_loss_'])

        # get model predictions
        df_pred = self.predict(X)
        y, variance = df_pred['y_pred'].values, df_pred['y_var'].values

        if isinstance(q, float):
            # make predictions
            df_pred = pd.DataFrame({
              'y_pred' : (y + norm.ppf(q) * np.sqrt(variance)).flatten()
            })
        else:
            l_out = []
            if 0.5 not in q:
                q += [0.5]
            for qi in sorted(q):
                l_out += [
                    (y + norm.ppf(qi) * np.sqrt(variance)).flatten()
                ]

            X_out = np.array(l_out).T
            df_pred = pd.DataFrame(X_out, columns=[str(q) for q in sorted(q)])
            df_pred['y_pred'] = df_pred['0.5']

        # return output
        return df_pred


class DistributionCalibratedMDNRegressor():
    """Calibrated forecaster.

    Wraps around sklearn-style models and calibrates them.

    Parameters
    ----------
    model : object
        Baseline sklearn model that will be trained incrementally
    """

    def __init__(self, model) -> None:
        self.model = model

    def fit(self, X, y) -> 'CalibratedRegressor':
        """Fit estimator on a calibration set.

        Estimator need predict_proba function for outputting q estimates.

        Parameters
        ----------
        X : pandas.DataFrame of shape = [n_samples, n_features]
            The training input samples. Must have a ds column.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # train base model if ncessary
        try:
            mean, var = self.model.predict_mv(X)
        except NotFittedError:
            self.model.fit(X, y)
            mean, var = self.model.predict_mv(X)

        # create data
        X_recal = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])

        # set up recalibrator
        self.recalibrator_ = make_mdn_recalibrator(n_components=3)
        self.recalibrator_loss_ = make_mdn_recalibrator_loss(
            self.recalibrator_, None, 5e-3
        )

        # fit recalibrator
        self.recalibrator_loss_.fit(
            [X_recal[:,0], X_recal[:,1], y], np.empty(y.shape),
            batch_size=16, epochs=200, verbose=1
        )
        perf_number = self.recalibrator_loss_.evaluate(
            [X_recal[:,0], X_recal[:,1], y], np.empty(y.shape)
        )

        return perf_number

    def predict(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict targets

        Parameters
        ----------
        X : pandas.DataFrame
            Input data. Must contain column 'ds', as well as extra feature
            columns.

        Returns
        -------
        df_pred : pandas.DataFrame
            Contains three columns: 'ds', 'horizon', and 'y_pred' and one
            column for each predicted quantile. The column 'horizon' always
            equals one. The column 'y_pred' is a point forecast that is
            allowed to be different for each predictor (e.g. could be the
            mean or the median).
        """
        # Check is fit had been called
        check_is_fitted(self, ['recalibrator_', 'recalibrator_loss_'])

        # base model predictions
        mean, var = self.model.predict_mv(X)

        # create data for R
        X_out = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])

        # run R
        mean2, var2, w2 = self.recalibrator_.predict([X_out[:,0], X_out[:,1]])

        # return output
        return mean2, var2, w2

    def predict_quantile(
        self, X: pd.DataFrame, q: float = 0.5
    ) -> pd.DataFrame:
        """Predict quantile of distribution.

        Parameters
        ----------
        X : pandas.DataFrame of shape = [n_rows, n_cols]
            The input samples.
        q : float or array-like, shape = [n_quantiles]
            One or more quantiles at which we want to make predictions

        Returns
        -------
        df_pred : pandas.DataFrame of shape = [n_rows, 1 + n_quantiles]
            Contains two columns: 'ds' and 'y_pred'
        """
        # stack input data with previously seen training data
        check_is_fitted(self, ['recalibrator_', 'recalibrator_loss_'])

        # get model predictions
        mu, var, w = self.predict(X)

        if isinstance(q, float):
            # compute max scale of each distribution
            search_min = np.min(mu - 2*np.sqrt(var), axis=1)
            search_max = np.max(mu + 2*np.sqrt(var), axis=1)

            # create search bounds for each element
            n_steps = 10000
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
        else:
            raise NotImplementedError

        # return output
        return df_pred



class DistributionCalibratedRegressor():
    """Calibrated forecaster.

    Wraps around sklearn-style models and calibrates them.

    Parameters
    ----------
    model : object
        Baseline sklearn model that will be trained incrementally
    """

    def __init__(self, model) -> None:
        self.model = model

    def fit(self, X, y) -> 'CalibratedRegressor':
        """Fit estimator on a calibration set.

        Estimator need predict_proba function for outputting q estimates.

        Parameters
        ----------
        X : pandas.DataFrame of shape = [n_samples, n_features]
            The training input samples. Must have a ds column.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # train base model if ncessary
        try:
            mean, var = self.model.predict_mv(X)
        except NotFittedError:
            self.model.fit(X, y)
            mean, var = self.model.predict_mv(X)

        # create data
        X_recal = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])

        # set up recalibrator
        self.recalibrator_ = make_gaussian_recalibrator()
        self.recalibrator_loss_ = make_gaussian_loss_model(
            base_model=self.recalibrator_, 
            mu=self.recalibrator_.outputs[0], 
            var=self.recalibrator_.outputs[1], 
            inputs=self.recalibrator_.inputs, 
            learning_rate=1e-2
        )

        # fit recalibrator
        self.recalibrator_loss_.fit(
            [X_recal[:,0], X_recal[:,1], y], np.empty(y.shape),
            batch_size=y.shape[0], epochs=5000, verbose=1
        )
        perf_number = self.recalibrator_loss_.evaluate(
            [X_recal[:,0], X_recal[:,1], y], np.empty(y.shape)
        )

        return perf_number

    def predict(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict targets

        Parameters
        ----------
        X : pandas.DataFrame
            Input data. Must contain column 'ds', as well as extra feature
            columns.

        Returns
        -------
        df_pred : pandas.DataFrame
            Contains three columns: 'ds', 'horizon', and 'y_pred' and one
            column for each predicted quantile. The column 'horizon' always
            equals one. The column 'y_pred' is a point forecast that is
            allowed to be different for each predictor (e.g. could be the
            mean or the median).
        """
        # Check is fit had been called
        check_is_fitted(self, ['recalibrator_', 'recalibrator_loss_'])

        # base model predictions
        mean, var = self.model.predict_mv(X)

        # create data for R
        X_out = np.hstack([mean[:, np.newaxis], var[:, np.newaxis]])

        # run R
        mean2, var2 = self.recalibrator_.predict([X_out[:,0], X_out[:,1]])
        mean2, var2 = mean2.flatten(), var2.flatten()

        # create output dataframe
        X_out2 = np.hstack([mean2[:, np.newaxis], var2[:, np.newaxis]])
        df_pred = pd.DataFrame(X_out2, columns=['y_pred', 'y_var'])

        # return output
        return df_pred

    def predict_quantile(
        self, X: pd.DataFrame, q: float = 0.5
    ) -> pd.DataFrame:
        """Predict quantile of distribution.

        Parameters
        ----------
        X : pandas.DataFrame of shape = [n_rows, n_cols]
            The input samples.
        q : float or array-like, shape = [n_quantiles]
            One or more quantiles at which we want to make predictions

        Returns
        -------
        df_pred : pandas.DataFrame of shape = [n_rows, 1 + n_quantiles]
            Contains two columns: 'ds' and 'y_pred'
        """
        # stack input data with previously seen training data
        check_is_fitted(self, ['recalibrator_', 'recalibrator_loss_'])

        # get model predictions
        df_pred = self.predict(X)
        y, variance = df_pred['y_pred'].values, df_pred['y_var'].values

        if isinstance(q, float):
            # make predictions
            df_pred = pd.DataFrame({
              'y_pred' : (y + norm.ppf(q) * np.sqrt(variance)).flatten()
            })
        else:
            l_out = []
            if 0.5 not in q:
                q += [0.5]
            for qi in sorted(q):
                l_out += [
                    (y + norm.ppf(qi) * np.sqrt(variance)).flatten()
                ]

            X_out = np.array(l_out).T
            df_pred = pd.DataFrame(X_out, columns=[str(q) for q in sorted(q)])
            df_pred['y_pred'] = df_pred['0.5']

        # return output
        return df_pred
