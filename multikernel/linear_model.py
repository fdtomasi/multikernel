"""Multiple Kernel Learning (MKL) with linear models.

Learn kernel weights with a linear model.
"""
from __future__ import division, print_function

import warnings

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import squared_norm

from regain.prox import prox_laplacian
from regain.prox import soft_thresholding_sign as soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence


def objective(K, y, alpha, lamda, beta, w):
    """Objective function for lasso kernel learning."""
    obj = .5 * sum(squared_norm(
        alpha[j].dot(K[j].T.dot(w)) - y[j]) for j in range(len(K)))
    obj += lamda * np.abs(w).sum()
    obj += beta * sum(squared_norm(a) for a in alpha)
    return obj


def objective_admm(K, y, alpha, lamda, beta, w, w1, w2):
    """Objective function for lasso kernel learning."""
    obj = .5 * sum(squared_norm(
        np.dot(alpha[j], K[j].T.dot(w)) - y[j]) for j in range(len(K)))
    obj += lamda * np.abs(w1).sum()
    obj += beta * squared_norm(w2)
    return obj


def objective_admm2(x, y, alpha, lamda, beta, w1):
    """Objective function for lasso kernel learning."""
    obj = .5 * sum(squared_norm(x[j] - y[j]) for j in range(len(x)))
    obj += lamda * np.abs(w1).sum()
    obj += beta * sum(squared_norm(a) for a in alpha)
    return obj


def enet_kernel_learning(
        K, y, lamda=0.01, beta=0.01, gamma='auto', max_iter=100, verbose=0,
        tol=1e-4, return_n_iter=True):
    """Elastic Net kernel learning.

    Solve the following problem via alternating minimisation:
        min sum_{i=1}^p 1/2 ||alpha_i * w * K_i - y_i||^2 + lamda ||w||_1 +
        + beta||w||_2^2
    """
    n_patients = len(K)
    n_kernels = len(K[0])
    coef = np.ones(n_kernels)

    alpha = [np.zeros(K[j].shape[2]) for j in range(n_patients)]
    # KKT = [K[j].T.dot(K[j]) for j in range(len(K))]
    # print(KKT[0].shape)
    if gamma == 'auto':
        lipschitz_constant = np.array([
            sum(np.linalg.norm(K_j[i].dot(K_j[i].T))
                for i in range(K_j.shape[0]))
            for K_j in K])
        gamma = 1. / (lipschitz_constant)

    objective_new = 0
    for iteration_ in range(max_iter):
        w_old = coef.copy()
        alpha_old = [a.copy() for a in alpha]
        objective_old = objective_new

        # update w
        A = [K[j].dot(alpha[j]) for j in range(n_patients)]
        alpha_coef_K = [alpha[j].dot(K[j].T.dot(coef))
                        for j in range(n_patients)]
        gradient = sum((alpha_coef_K[j] - y[j]).dot(A[j].T)
                       for j in range(n_patients))

        # gradient_2 = coef.dot(sum(
        #     np.dot(K[j].dot(alpha[j]), K[j].dot(alpha[j]).T)
        #     for j in range(len(K)))) - sum(
        #         y[j].dot(K[j].dot(alpha[j]).T) for j in range(len(K)))

        # gradient = coef.dot(sum(
        #     alpha[j].dot(KKT[j].dot(alpha[j])) for j in range(len(K)))) - sum(
        #         y[j].dot(K[j].dot(alpha[j]).T) for j in range(len(K)))

        # gradient += 2 * beta * coef
        coef = soft_thresholding(coef - gamma * gradient, lamda=lamda * gamma)

        # update alpha
        # for j in range(len(K)):
        #     alpha[j] = _solve_cholesky_kernel(
        #         K[j].T.dot(coef), y[j][..., None], lamda).ravel()
        A = [K[j].T.dot(coef) for j in range(n_patients)]
        alpha_coef_K = [alpha[j].dot(K[j].T.dot(coef))
                        for j in range(n_patients)]
        gradient = [(alpha_coef_K[j] - y[j]).dot(A[j].T) + 2 * beta * alpha[j]
                    for j in range(n_patients)]
        alpha = [alpha[j] - gamma * gradient[j] for j in range(n_patients)]

        objective_new = objective(K, y, alpha, lamda, beta, coef)
        objective_difference = abs(objective_new - objective_old)
        snorm = np.sqrt(squared_norm(coef - w_old) + sum(
            squared_norm(a - a_old) for a, a_old in zip(alpha, alpha_old)))

        obj = objective(K, y, alpha, lamda, beta, coef)

        if verbose and iteration_ % 10 == 0:
            print("obj: %.4f, snorm: %.4f" % (obj, snorm))

        if snorm < tol and objective_difference < tol:
            break
        if np.isnan(snorm) or np.isnan(objective_difference):
            raise ValueError('assdgg')
    else:
        warnings.warn("Objective did not converge.")

    return_list = [alpha, coef]
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


def enet_kernel_learning_admm(
        K, y, lamda=0.01, beta=0.01, rho=1., max_iter=100, verbose=0, rtol=1e-4,
        tol=1e-4, return_n_iter=True, update_rho_options=None):
    """Elastic Net kernel learning.

    Solve the following problem via ADMM:
        min sum_{i=1}^p 1/2 ||alpha_i * w * K_i - y_i||^2 + lamda ||w||_1 +
        + beta||w||_2^2
    """
    n_patients = len(K)
    n_kernels = len(K[0])
    coef = np.ones(n_kernels)
    u_1 = np.zeros(n_kernels)
    u_2 = np.zeros(n_kernels)
    w_1 = np.zeros(n_kernels)
    w_2 = np.zeros(n_kernels)

    w_1_old = w_1.copy()
    w_2_old = w_2.copy()

    checks = []
    for iteration_ in range(max_iter):
        # update alpha
        # solve (AtA + 2I)^-1 (Aty) with A = wK
        A = [K[j].T.dot(coef) for j in range(n_patients)]
        KK = [A[j].dot(A[j].T) for j in range(n_patients)]
        yy = [y[j].dot(A[j]) for j in range(n_patients)]

        alpha = [_solve_cholesky_kernel(
            KK[j], yy[j][..., None], 2).ravel() for j in range(n_patients)]
        # alpha = [_solve_cholesky_kernel(
        #     K_dot_coef[j], y[j][..., None], 0).ravel() for j in range(n_patients)]

        w_1 = soft_thresholding(coef + u_1, lamda / rho)
        w_2 = prox_laplacian(coef + u_2, beta / rho)

        # equivalent to alpha_dot_K
        # solve (sum(AtA) + 2*rho I)^-1 (sum(Aty) + rho(w1+w2-u1-u2))
        # with A = K * alpha
        A = [K[j].dot(alpha[j]) for j in range(n_patients)]
        KK = sum(A[j].dot(A[j].T) for j in range(n_patients))
        yy = sum(y[j].dot(A[j].T) for j in range(n_patients))
        yy += rho * (w_1 + w_2 - u_1 - u_2)

        coef = _solve_cholesky_kernel(KK, yy[..., None], 2 * rho).ravel()

        # update residuals
        u_1 += coef - w_1
        u_2 += coef - w_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(squared_norm(coef - w_1) + squared_norm(coef - w_2))
        snorm = rho * np.sqrt(
            squared_norm(w_1 - w_1_old) + squared_norm(w_2 - w_2_old))

        obj = objective_admm(K, y, alpha, lamda, beta, coef, w_1, w_2)

        check = convergence(
            obj=obj, rnorm=rnorm, snorm=snorm,
            e_pri=np.sqrt(2 * coef.size) * tol + rtol * max(
                np.sqrt(squared_norm(coef) + squared_norm(coef)),
                np.sqrt(squared_norm(w_1) + squared_norm(w_2))),
            e_dual=np.sqrt(2 * coef.size) * tol + rtol * rho * (
                np.sqrt(squared_norm(u_1) + squared_norm(u_2))))

        w_1_old = w_1.copy()
        w_2_old = w_2.copy()

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
        u_2 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [alpha, coef]
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


def enet_kernel_learning_admm2(
        K, y, lamda=0.01, beta=0.01, rho=1., max_iter=100, verbose=0, rtol=1e-4,
        tol=1e-4, return_n_iter=True, update_rho_options=None):
    """Elastic Net kernel learning.

    Solve the following problem via ADMM:
        min sum_{i=1}^p 1/2 ||y_i - alpha_i * sum_{k=1}^{n_k} w_k * K_{ik}||^2
        + lamda ||w||_1 + beta sum_{j=1}^{c_i}||alpha_j||_2^2
    """
    n_patients = len(K)
    n_kernels = len(K[0])
    coef = np.ones(n_kernels)
    alpha = [np.zeros(K[j].shape[2]) for j in range(n_patients)]

    u = [np.zeros(K[j].shape[1]) for j in range(n_patients)]
    u_1 = np.zeros(n_kernels)
    w_1 = np.zeros(n_kernels)

    x_old = [np.zeros(K[0].shape[1]) for j in range(n_patients)]
    w_1_old = w_1.copy()
    # w_2_old = w_2.copy()

    checks = []
    for iteration_ in range(max_iter):
        # update x
        A = [K[j].T.dot(coef) for j in range(n_patients)]
        x = [prox_laplacian(y[j] + rho * (A[j].T.dot(alpha[j]) - u[j]), rho / 2.)
             for j in range(n_patients)]

        # update alpha
        # solve (AtA + 2I)^-1 (Aty) with A = wK
        KK = [rho * A[j].dot(A[j].T) for j in range(n_patients)]
        yy = [rho * A[j].dot(x[j] + u[j]) for j in range(n_patients)]
        alpha = [_solve_cholesky_kernel(
            KK[j], yy[j][..., None], 2 * beta).ravel() for j in range(n_patients)]
        # equivalent to alpha_dot_K
        # solve (sum(AtA) + 2*rho I)^-1 (sum(Aty) + rho(w1+w2-u1-u2))
        # with A = K * alpha
        A = [K[j].dot(alpha[j]) for j in range(n_patients)]
        KK = sum(A[j].dot(A[j].T) for j in range(n_patients))
        yy = sum(A[j].dot(x[j] + u[j]) for j in range(n_patients))
        yy += w_1 - u_1
        coef = _solve_cholesky_kernel(KK, yy[..., None], 1).ravel()

        w_1 = soft_thresholding(coef + u_1, lamda / rho)
        # w_2 = prox_laplacian(coef + u_2, beta / rho)

        # update residuals
        alpha_coef_K = [
            alpha[j].dot(K[j].T.dot(coef)) for j in range(n_patients)]
        residuals = [x[j] - alpha_coef_K[j] for j in range(n_patients)]
        u = [u[j] + residuals[j] for j in range(n_patients)]
        u_1 += coef - w_1

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(coef - w_1) +
            sum(squared_norm(residuals[j]) for j in range(n_patients)))
        snorm = rho * np.sqrt(
            squared_norm(w_1 - w_1_old) +
            sum(squared_norm(x[j] - x_old[j]) for j in range(n_patients)))

        obj = objective_admm2(x, y, alpha, lamda, beta, w_1)
        check = convergence(
            obj=obj, rnorm=rnorm, snorm=snorm,
            e_pri=np.sqrt(coef.size + sum(
                x[j].size for j in range(n_patients))) * tol + rtol * max(
                    np.sqrt(squared_norm(coef) + sum(squared_norm(
                        alpha_coef_K[j]) for j in range(n_patients))),
                    np.sqrt(squared_norm(w_1) + sum(squared_norm(
                        x[j]) for j in range(n_patients)))),
            e_dual=np.sqrt(coef.size + sum(
                x[j].size for j in range(n_patients))) * tol + rtol * rho * (
                    np.sqrt(squared_norm(u_1) + sum(squared_norm(
                        u[j]) for j in range(n_patients)))))

        w_1_old = w_1.copy()
        x_old = [x[j].copy() for j in range(n_patients)]

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual and iteration_ > 1:
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_,
                             **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        u = [u[j] * (rho / rho_new) for j in range(n_patients)]
        u_1 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [alpha, coef]
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class ElasticNetKernelLearning(LinearModel, RegressorMixin):
    """Multiple kernel learning via Lasso model.

    Parameters
    ----------
    ...
    """
    # Ensure consistent split
    _pairwise = True

    def __init__(self, lamda=0.01, beta=0.01, gamma=1, max_iter=100, verbose=False,
                 tol=1e-4, mode='gd', rho=1, rtol=1e-4):
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.lamda = lamda
        self.beta = beta
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
        if self.mode == 'gd':
            self.alpha_, self.coef_, self.n_iter_ = enet_kernel_learning(
                K, y, lamda=self.lamda, beta=self.beta, gamma=self.gamma,
                tol=self.tol, max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True)
        else:
            self.alpha_, self.coef_, self.n_iter_ = enet_kernel_learning_admm2(
                K, y, lamda=self.lamda, beta=self.beta,
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
        return [super(ElasticNetKernelLearning, self).predict(
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


class ElasticNetKernelLearningClassifier(ElasticNetKernelLearning, LinearClassifierMixin):
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
        Y = [self._label_binarizer.fit_transform(j).ravel() for j in y]
        if self._label_binarizer.y_type_.startswith('multilabel'):
            # we don't (yet) support multi-label classification in ENet
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))

        # Y = column_or_1d(Y, warn=True)
        super(ElasticNetKernelLearningClassifier, self).fit(K, Y)
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
