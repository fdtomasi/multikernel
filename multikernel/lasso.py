"""Multiple Kernel Learning (MKL) with logistic regression.

Learn kernel weights with a linear model.
"""
from __future__ import division, print_function

import warnings

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearClassifierMixin, LinearModel
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import squared_norm

from regain.prox import soft_thresholding_sign as soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence


def lasso_objective(Y, w, K, w_1, lamda):
    obj = squared_norm(Y - np.tensordot(w, K, axes=1))
    obj += lamda * np.abs(w_1).sum()
    return obj


def lasso_kernel_admm(
        K, y, lamda=0.01, rho=1., max_iter=100, verbose=0, rtol=1e-4,
        tol=1e-4, return_n_iter=True, update_rho_options=None,
        sample_weight=None):
    """Elastic Net kernel learning.

    Solve the following problem via ADMM:
        min sum_{i=1}^p 1/2 ||y_i - alpha_i * sum_{k=1}^{n_k} w_k * K_{ik}||^2
        + lamda ||w||_1 + beta sum_{j=1}^{c_i}||alpha_j||_2^2
    """
    n_kernels, n_samples, n_features = K.shape
    coef = np.ones(n_kernels)

    # alpha = [np.zeros(K[j].shape[2]) for j in range(n_patients)]
    # u = [np.zeros(K[j].shape[1]) for j in range(n_patients)]
    w_1 = coef.copy()
    u_1 = np.zeros(n_kernels)

    # x_old = [np.zeros(K[0].shape[1]) for j in range(n_patients)]
    w_1_old = w_1.copy()
    Y = y[:, None].dot(y[:, None].T)

    checks = []
    for iteration_ in range(max_iter):
        # update w
        KK = 2 * np.tensordot(K, K.T, axes=([1, 2], [0, 1]))
        yy = 2 * np.tensordot(Y, K, axes=([0, 1], [1, 2]))
        yy += rho * (w_1 - u_1)
        coef = _solve_cholesky_kernel(KK, yy[..., None], rho).ravel()

        w_1 = soft_thresholding(coef + u_1, lamda / rho)
        # w_2 = prox_laplacian(coef + u_2, beta / rho)

        u_1 += coef - w_1

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(coef - w_1))
        snorm = rho * np.sqrt(
            squared_norm(w_1 - w_1_old))

        obj = lasso_objective(Y, coef, K, w_1, lamda)
        check = convergence(
            obj=obj, rnorm=rnorm, snorm=snorm,
            e_pri=np.sqrt(coef.size) * tol + rtol * max(
                    np.sqrt(squared_norm(coef)),
                    np.sqrt(squared_norm(w_1))),
            e_dual=np.sqrt(coef.size) * tol + rtol * rho * (
                    np.sqrt(squared_norm(u_1))))

        w_1_old = w_1.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual and iteration_ > 1:
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_,
                             **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        u_1 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [coef]
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LassoKernelLearning(LinearModel, RegressorMixin):
    """Multiple kernel learning via Lasso model.

    Parameters
    ----------
    ...
    """
    # Ensure consistent split
    _pairwise = True

    def __init__(self, lamda=0.01, gamma=1, max_iter=100, verbose=False,
                 tol=1e-4, mode='gd', rho=1, rtol=1e-4):
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.lamda = lamda
        self.gamma = gamma
        self.mode = mode
        self.rho = rho
        self.rtol = rtol

    def fit(self, K, y):
        """Learn weights for kernels.

        Parameters:
        K : array-like
            K should be a list of kernels (if self.kernel.kernel is
            'precomputed'). Otherwise is a 2-d data matrix.
        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.

        """
        if self.verbose > 1:
            print(self.get_params())
        self.coef_, self.n_iter_ = lasso_kernel_admm(
            K, y, lamda=self.lamda,
            tol=self.tol, rtol=self.rtol, max_iter=self.max_iter,
            verbose=self.verbose, return_n_iter=True, rho=self.rho)
        self.intercept_ = 0.
        return self

    def predict(self, K):
        """Predict using the kernel ridge model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        # check_is_fitted(self, ["X_fit_", "dual_coef_"])
        # K = self._get_kernel(X, self.X_fit_)
        # return np.dot(K, self.dual_coef_)
        # return [np.dot(self.alpha_[j], K[j].T.dot(self.coef_))
        #         for j in range(len(K))]
        return [super(LassoKernelLearning, self).predict(
            K[j].dot(self.alpha_[j]).T) for j in range(len(K))]

    def score(self, K, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(K)
        if sample_weight is None:
            return np.mean([r2_score(
                y[j], y_pred[j], multioutput='variance_weighted')
                for j in range(len(K))])
        else:
            return np.mean([
                r2_score(y[j], y_pred[j], sample_weight=sample_weight[j],
                         multioutput='variance_weighted')
                for j in range(len(K))])


class LassoKernelLearningClassifier(LassoKernelLearning, LinearClassifierMixin):
    """Multiple kernel learning classifier.

    Parameters
    ----------
    ...
    """
    def fit(self, K, y):
        """Learn weights for kernels.

        Parameters:
        K : array-like
            K should be a list of kernels (if self.kernel.kernel is
            'precomputed'). Otherwise is a 2-d data matrix.
        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.

        """
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y).ravel()
        if self._label_binarizer.y_type_.startswith('multilabel'):
            # we don't (yet) support multi-label classification in ENet
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))

        # Y = column_or_1d(Y, warn=True)
        super(LassoKernelLearningClassifier, self).fit(K, Y)
        if self.classes_.shape[0] > 2:
            ndim = self.classes_.shape[0]
        else:
            ndim = 1
        self.coef_ = self.coef_.reshape(ndim, -1)

        return self

    def predict(self, K):
        """Predict using the kernel ridge model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        # check_is_fitted(self, ["X_fit_", "dual_coef_"])
        # K = self._get_kernel(X, self.X_fit_)
        # return np.dot(K, self.dual_coef_)
        # return [super(ElasticNetKernelLearningClassifier, self).predict(
        #     np.dot(self.alpha_[j], K[j].T)) for j in range(len(K))]
        return [LinearClassifierMixin.predict(
            self, K[j].dot(self.alpha_[j]).T) for j in range(len(K))]

    def score(self, K, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(K)
        if sample_weight is None:
            return np.mean([accuracy_score(
                y[j], y_pred[j]) for j in range(len(K))])
        else:
            return np.mean([
                accuracy_score(y[j], y_pred[j], sample_weight=sample_weight[j])
                for j in range(len(K))])

    @property
    def classes_(self):
        return self._label_binarizer.classes_
