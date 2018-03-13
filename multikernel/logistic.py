"""
Logistic Regression
"""
from __future__ import division

import warnings

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model.logistic import _logistic_loss
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from multikernel.lasso import LinearClassifierMixin


def logistic_loss(K, y, alpha, coef, lamda, beta):
    X = np.tensordot(coef, K, axes=1)
    return _logistic_loss(alpha, X, y, lamda) - .5 * lamda * np.dot(alpha, alpha)


def logistic_objective(K, y, alpha, coef, lamda, beta):
    X = np.tensordot(coef, K, axes=1)
    return _logistic_loss(alpha, X, y, lamda) + beta * np.abs(coef).sum()


def logistic_alternating(K, y, lamda=0.01, beta=0.01, gamma=.5, max_iter=100,
                         verbose=0, tol=1e-4, return_n_iter=True):
    # single patient
    n_kernels, n_samples, n_dimensions = K.shape

    objective_new = 0
    coef = np.ones(n_kernels)
    alpha = np.zeros(n_dimensions)

    lr_p2 = LogisticRegression(
        verbose=verbose, penalty='l2', C=1 / lamda, warm_start=True, max_iter=1)
    lr_p1 = SGDClassifier(
        loss='log', l1_ratio=0.9,
        verbose=0, penalty='elasticnet', alpha=beta, warm_start=True, max_iter=1)

    for iteration_ in range(max_iter):
        w_old = coef.copy()
        alpha_old = alpha.copy()
        objective_old = objective_new

        X = np.tensordot(coef, K, axes=1)
        alpha = lr_p2.fit(X, y).coef_.ravel()

        X = np.tensordot(alpha, K, axes=([0], [2])).T
        coef = lr_p1.fit(X, y).coef_.ravel()

        # if verbose:
        #     print("n_iter alpha %d" % lr_p2.n_iter_)
        #     print("n_iter coef %d" % lr_p1.n_iter_)

        obj = logistic_objective(K, y, alpha, coef, lamda, beta)
        objective_difference = abs(objective_new - objective_old)
        # snorm = np.sqrt(squared_norm(coef - w_old) +
        #                 squared_norm(alpha - alpha_old))

        diff_w = np.linalg.norm(coef - w_old)
        diff_a = np.linalg.norm(alpha - alpha_old)

        if verbose:# and iteration_ % 10 == 0:
            # print("obj: %.4f, snorm: %.4f" % (obj, snorm))
            print("obj: %.4f, loss: %.4f, diff_w: %.4f, diff_a: %.4f" % (
                obj, logistic_loss(K, y, alpha, coef, lamda, beta), diff_w,
                diff_a))

        if diff_w < tol and diff_a < tol and objective_difference < tol:
            break
        if np.isnan(diff_w) or np.isnan(diff_a) or np.isnan(objective_difference):
            raise ValueError('something is nan')
    else:
        warnings.warn("Objective did not converge.")
    return_list = [alpha, coef]
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LogisticRegressionMultipleKernel(LogisticRegression, LinearClassifierMixin):
    # Ensure consistent split
    _pairwise = True

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1,
                 lamda=0.01, gamma=1, rho=1, rtol=1e-4, beta=0.01):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
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
        Y = self._label_binarizer.fit_transform(y).ravel()
        if self._label_binarizer.y_type_.startswith('multilabel'):
            # we don't (yet) support multi-label classification in ENet
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))

        # Y = column_or_1d(Y, warn=True)
        self.alpha_, self.coef_, self.n_iter_ = logistic_alternating(
            X, Y, lamda=self.lamda, beta=self.beta, gamma=self.gamma,
            max_iter=self.max_iter, verbose=self.verbose, tol=self.tol,
            return_n_iter=True)

        if self.classes_.shape[0] > 2:
            ndim = self.classes_.shape[0]
        else:
            ndim = 1
        self.coef_ = self.coef_.reshape(ndim, -1)
        self.y_train_ = Y

        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_

    def decision_function(self, X):
        """Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})

        # X = check_array(X, accept_sparse='csr')

        # n_features = self.coef_.shape[1]
        # if X.shape[0] != n_features:
        #     raise ValueError("X has %d features per sample; expecting %d"
        #                      % (X.shape[1], n_features))
        #
        scores = np.tensordot(self.coef_, X, axes=1)# + self.intercept_
        return scores.ravel() # if scores.shape[1] == 1 else scores

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
        check_is_fitted(self, ["alpha_", "coef_"])
        X = np.tensordot(K, self.alpha_, axes=1)
        return LinearClassifierMixin.predict(self, X)  #.reshape(*K.shape[1:])

    # def predict_proba(self, X):
    #     """Probability estimates.
    #
    #     The returned estimates for all classes are ordered by the
    #     label of classes.
    #
    #     For a multi_class problem, if multi_class is set to be "multinomial"
    #     the softmax function is used to find the predicted probability of
    #     each class.
    #     Else use a one-vs-rest approach, i.e calculate the probability
    #     of each class assuming it to be positive using the logistic function.
    #     and normalize these values across all the classes.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape = [n_samples, n_features]
    #
    #     Returns
    #     -------
    #     T : array-like, shape = [n_samples, n_classes]
    #         Returns the probability of the sample for each class in the model,
    #         where classes are ordered as they are in ``self.classes_``.
    #     """
    #     if not hasattr(self, "coef_"):
    #         raise NotFittedError("Call fit before prediction")
    #     calculate_ovr = self.coef_.shape[0] == 1 or self.multi_class == "ovr"
    #     if calculate_ovr:
    #         return super(LogisticRegression, self)._predict_proba_lr(X)
    #     else:
    #         return softmax(self.decision_function(X), copy=False)
    #
    # def predict_log_proba(self, X):
    #     """Log of probability estimates.
    #
    #     The returned estimates for all classes are ordered by the
    #     label of classes.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape = [n_samples, n_features]
    #
    #     Returns
    #     -------
    #     T : array-like, shape = [n_samples, n_classes]
    #         Returns the log-probability of the sample for each class in the
    #         model, where classes are ordered as they are in ``self.classes_``.
    #     """
    #     return np.log(self.predict_proba(X))
