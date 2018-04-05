"""
Logistic Regression
"""
from __future__ import division

import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model.logistic import _intercept_dot, _logistic_loss
from sklearn.linear_model.logistic import \
    _logistic_loss_and_grad as _loglossgrad
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from multikernel.lasso import LinearClassifierMixin, squared_norm
from multikernel.logistic import LogisticRegressionMultipleKernel
from multikernel.logistic import logistic_loss as single_logloss
from regain.prox import soft_thresholding_sign as soft_thresholding


def logistic_loss(K, y, alpha, coef, lamda, beta):
    return sum(single_logloss(K[i], y[i], alpha[i], coef, lamda, beta)
               for i in range(len(K)))


def logistic_objective(K, y, alpha, coef, lamda, beta):
    obj = sum(_logistic_loss(alpha[i], np.tensordot(coef, K[i], axes=1), y[i],
              lamda) for i in range(len(K)))
    obj += beta * np.abs(coef).sum()
    return obj


def _logistic_loss_and_grad(w, alpha, X, y, lamda, sample_weight=None):
    """Computes the logistic loss and gradient.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    out : float
        Logistic loss.

    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """
    n_patients = len(X)
    out = 0.
    grad = np.zeros_like(w)
    sample_weight_orig = sample_weight.copy() if sample_weight is not None \
        else None

    for i in range(n_patients):
        n_kernels, n_samples, n_features = X[i].shape
        x_i = np.tensordot(w, X[i], axes=1)
        alpha_i, c, yz = _intercept_dot(alpha[i], x_i, y[i])

        if sample_weight_orig is None:
            sample_weight = np.ones(n_samples)

        # Logistic loss is the negative of the log of the logistic function.
        out += -np.sum(sample_weight * log_logistic(yz))

        z = expit(yz)
        z0 = sample_weight * (z - 1) * y[i]

        grad += safe_sparse_dot(X[i].dot(alpha_i), z0)

        # alpha_i, c_i, x_i = _intercept_dot(alpha[i][:-1], X[i], 1.)
        # out_i, grad_i = _loglossgrad(
        #     np.append(w, alpha[i][-1]), x_i.T, y[i], 0,
        #     sample_weight=sample_weight)
        # out += out_i
        # grad += grad_i[:n_kernels]

    out += .5 * lamda * np.dot(w, w)
    grad += lamda * w
    return out, grad


def logistic_alternating(K, y, lamda=0.01, beta=0.01, gamma=.5,
                         max_iter=100, l1_ratio_lamda=0.1, l1_ratio_beta=0.1,
                         deep=True, verbose=0, tol=1e-4, return_n_iter=True,
                         fit_intercept=True, lr_p2=None):
    # multiple patient
    n_patients = len(K)
    n_kernels = len(K[0])
    coef = np.random.rand(n_kernels)
    alpha = [np.zeros(K[j].shape[2]) for j in range(n_patients)]
    # intercepts = [np.zeros(K[j].shape[1]) for j in range(n_patients)]
    obj = 0

    max_iter_deep = max_iter // 3 if deep else 1

    if lr_p2 is None:
        raise ValueError("lr_p2 cant be None")

    for iteration_ in range(max_iter):
        w_old = coef.copy()
        alpha_old = [a.copy() for a in alpha]
        objective_old = obj

        alpha, intercepts, alpha_intercept = [], [], []
        for i in range(n_patients):
            # lr_p2[i].partial_fit(np.tensordot(coef, K[i], axes=1), y[i],
            #                      classes=np.unique(y[i]))
            l_i = lr_p2[i].fit(np.tensordot(coef, K[i], axes=1), y[i])
            a = soft_thresholding(l_i.coef_.ravel(), lamda * l1_ratio_lamda)
            alpha.append(a)
            c = soft_thresholding(l_i.intercept_.ravel(), lamda * l1_ratio_lamda)
            intercepts.append(c)
            alpha_intercept.append(np.hstack((a, c)))

        # alpha = [log.coef_.ravel() for log in lr_p2]
        # intercepts = [log.intercept_.ravel() for log in lr_p2]
        # alpha_intercept = [np.hstack((a, c)) for a, c in zip(alpha, intercepts)]

        # X = np.tensordot(alpha, K, axes=([0], [2])).T
        # X = sum(K[j].dot(alpha[j]).T for j in range(n_patients))
        # coef = lr_p1.fit(X, y).coef_.ravel()

        for it in range(max_iter_deep):
            coef_old = coef.copy()

            l2_reg = beta * (1 - l1_ratio_beta)
            loss, gradient = _logistic_loss_and_grad(
                coef, alpha_intercept, K, y, l2_reg)
            l1_reg = beta * l1_ratio_beta
            coef = soft_thresholding(coef - gamma * gradient, gamma * l1_reg)
            coef = np.maximum(coef, 0.)

            if np.linalg.norm(coef - coef_old) < tol:
                break

        obj = logistic_objective(K, y, alpha, coef, lamda, beta)
        objective_difference = abs(obj - objective_old)
        # snorm = np.sqrt(squared_norm(coef - w_old) +
        #                 squared_norm(alpha - alpha_old))

        diff_w = np.linalg.norm(coef - w_old)
        diff_a = np.sqrt(
            sum(squared_norm(a - a_old) for a, a_old in zip(alpha, alpha_old)))

        if verbose:
            print("obj: %.4f, loss: %.4f, diff_w: %.4f, diff_a: %.4f" % (
                obj, logistic_loss(K, y, alpha, coef, lamda, beta), diff_w,
                diff_a))

        if diff_a < tol and objective_difference < tol:
            break
        if np.isnan(diff_w) or np.isnan(diff_a) or np.isnan(objective_difference):
            raise ValueError('something is nan')
    else:
        warnings.warn("Objective did not converge.")
    return_list = [alpha, coef, intercepts]
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class MultipleLogisticRegressionMultipleKernel(
        LogisticRegressionMultipleKernel, LogisticRegression, LinearClassifierMixin):
    # Ensure consistent split
    _pairwise = True

    def __init__(self, penalty='l2', dual=False, tol=1e-4,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1,
                 l1_ratio_lamda=0.1, l1_ratio_beta=0.1, deep=True,
                 lamda=0.01, gamma=1, rho=1, rtol=1e-4, beta=0.01):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

        self.lamda = lamda
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.rtol = rtol
        self.l1_ratio_lamda = l1_ratio_lamda
        self.l1_ratio_beta = l1_ratio_beta
        self.deep = deep

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self : object
            Returns self.
        """
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = [self._label_binarizer.fit_transform(yy).ravel() for yy in y]

        # self.lr_p2 = [SGDClassifier(
        #     loss='log', l1_ratio=self.l1_ratio_lamda,
        #     fit_intercept=self.fit_intercept, shuffle=False,
        #     penalty='elasticnet', alpha=self.lamda, warm_start=True,
        #     max_iter=(self.max_iter // 3 if self.deep else 1) + 0)
        #     for i in range(len(X))]
        self.lr_p2 = [LogisticRegression(
            fit_intercept=self.fit_intercept, penalty='l2', solver='lbfgs',
            C=1. / (self.lamda * (1 - self.l1_ratio_lamda)), warm_start=True,
            max_iter=(self.max_iter // 3 if self.deep else 1) + 5)
            for i in range(len(X))]

        self.alpha_, self.coef_, self.intercept_, self.n_iter_ = \
            logistic_alternating(
                X, Y, lamda=self.lamda, beta=self.beta, gamma=self.gamma,
                max_iter=self.max_iter, verbose=self.verbose, tol=self.tol,
                return_n_iter=True, deep=self.deep,
                lr_p2=self.lr_p2, l1_ratio_beta=self.l1_ratio_beta,
                l1_ratio_lamda=self.l1_ratio_lamda,  # unused
                fit_intercept=self.fit_intercept  # unused
            )

        if self.classes_.shape[0] > 2:
            # ndim = self.classes_.shape[0]
            raise ValueError("too many classes")
        else:
            ndim = 1

        self.coef_ = self.coef_.reshape(ndim, -1)
        # self.alpha_ = [alpha.reshape(ndim, -1) for alpha in self.alpha_]

        self.y_train_ = Y

        return self

    def predict(self, X):
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
        check_is_fitted(self, ["alpha_", "coef_"])
        # return [LinearClassifierMixin.predict(
        #     self, np.tensordot(k, a, axes=1)) for a, k in zip(
        #         self.alpha_, X)]
        return [self.lr_p2[i].predict(np.tensordot(
            self.coef_.ravel(), X[i], axes=1)) for i in range(len(X))]

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

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self, ["alpha_", "coef_"])
        # return [LinearClassifierMixin._predict_proba_lr(
        #     self, np.tensordot(k, a, axes=1)) for a, k in zip(
        #         self.alpha_, X)]
        return [self.lr_p2[i].predict_proba(np.tensordot(
            self.coef_.ravel(), X[i], axes=1)) for i in range(len(X))]

    def predict_log_proba(self, X):
        """Log of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        proba = self.predict_proba(X)
        return [np.log(p) for p in proba]
