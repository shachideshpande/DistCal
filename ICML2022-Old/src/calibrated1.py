"""Model recalibration.

Applies Platt scaling or isotonic regression to recalibrate regression model.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.density import KernelDensity
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


# ----------------------------------------------------------------------------

class CalibratedRegressor():
    """ Calibrated forecaster.

    Wraps around sklearn-style models and calibrates them.

    Parameters
    ----------
    model : object
        Baseline sklearn model that will be trained incrementally
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        """ Fit estimator on a calibration set

        Estimator need predict_proba function for outputting q estimates.

        Parameters
        ----------
        X : dataframe of shape = [n_samples, n_features]
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
            cdf_pred = self.model.predict_proba(X, y).y_pred.values
        except NotFittedError:
            self.model.fit(X, y)
            cdf_pred = self.model.predict_proba(X, y).y_pred.values

        # set up recalibrators
        self.pred2true_ = IsotonicRegression()
        self.true2pred_ = IsotonicRegression()

        # create data
        cdf_true = [
            len([1 for cdf_i in cdf_pred if cdf_i <= cdf_lvl]) / float(
                len(cdf_pred))
            for cdf_lvl in cdf_pred
        ]

        # convert these to quantiles
        q_pred = cdf_pred
        q_true = np.array(cdf_true)

        # fit estimators
        self.pred2true_.fit(q_pred, q_true)
        self.true2pred_.fit(q_true, q_pred)

        return self

    def predict(self, X, days_idx=None):
        """ Predict targets on a set of specific days

        Parameters
        ----------
        X : pandas dataframe
            Input data. Must contain column 'ds', as well as extra feature
            columns.

        days_idx : Pandas MultiIndex, default=None
            The (store, item, days) on which we make predictions.
            If none, these are taken from the indices of y (if present) or
            X (otherwise).  See 'test_transform_by_days' unit test.
            Should be a subset of the index of X (and y).
            xgb-banana-forecasting-geary.ipynb shows how to use days_idx.

        Returns
        -------
        df_pred : pandas dataframe
            Contains three columns: 'ds', 'horizon', and 'y_pred' and one
            column for each predicted quantile. The column 'horizon' always
            equals one. The column 'y_pred' is a point forecast that is
            allowed to be different for each predictor (e.g. could be the
            mean or the median).
        """
        # validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas dataframe.")

        # Check is fit had been called
        check_is_fitted(self, ['pred2true_', 'true2pred_'])

        # compute the quantiles:
        sorted_q = sorted(self.model.quantiles)
        # pylint: disable=maybe-no-member
        l_out = []
        for qi in sorted_q:
            q_actual = self.true2pred_.predict([qi])[0]
            l_out += [
                self.model.predict_quantile(X, q=q_actual).y_pred.values]

        # hack to make sure they don't interleave: just sort the quantiles
        X_out = np.array(l_out).T
        X_out.sort(axis=1)

        # make it a dataframe
        df_pred = pd.DataFrame(X_out, columns=['q%0.2f' % q for q in sorted_q])

        # make the median an extra column (y_pred)
        df_pred['y_pred'] = df_pred['q0.50']

        # add horizon
        df_pred['horizon'] = 1

        # put back index
        df_pred.set_index(X.index, inplace=True)

        # return output
        return df_pred

    def predict_quantile(self, X, q=0.5):
        """ Predict quantile of distribution

        Parameters
        ----------
        X : dataframe of shape = [n_rows, n_cols]
            The input samples.
        q : float or array-like, shape = [n_quantiles]
            One or more quantiles at which we want to make predictions

        Returns
        -------
        df_pred : dataframe of shape = [n_rows, 1 + n_quantiles]
            Contains two columns: 'ds' and 'y_pred'
        """

        # stack input data with previously seen training data
        check_is_fitted(self, ['pred2true_', 'true2pred_'])

        if isinstance(q, float):
            # make predictions
            q_actual = self.true2pred_.predict([q])[0]
            df_pred = self.model.predict_quantile(X, q=q_actual)
        else:
            l_out = []
            if 0.5 not in q:
                q += [0.5]
            for qi in sorted(q):
                q_actual = self.true2pred_.predict([qi])[0]
                l_out += [
                    self.model.predict_quantile(X, q=q_actual).y_pred.values]

            X_out = np.array(l_out).T
            df_pred = pd.DataFrame(X_out, columns=[str(q) for q in sorted(q)])
            if isinstance(X, pd.DataFrame):
                df_pred['ds'] = X.ds.values
            df_pred['y_pred'] = df_pred['0.5']

        # return output
        return df_pred