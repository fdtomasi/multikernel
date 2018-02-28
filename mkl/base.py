"""Multiple Kernel Learning (MKL).

Fit a kernel-based model using multiple kernels, learning the weights.
"""
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _combine_kernels(kernels, gammas, mode='linear'):
    if mode != 'linear':
        raise NotImplementedError(mode)
    K = 0
    for gamma, Ki in zip(gammas, kernels):
        K += gamma * Ki
    return K


class MultipleKernelLearning(BaseEstimator):
    """General class for multiple kernel learning.

    Parameters
    ----------
    kernel : string, optional
        List of precomputed kernels.
    p : float, optional
        ???
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    """

    def __init__(self, kernels=None, p=1., max_iter=10, verbose=False, tol=1e-5,
                 kernel=None, combination='linear'):
        self.p = p
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.combination = combination
        self.kernel = kernel
        self.kernels = kernels

    @property
    def _is_precomputed(self):
        return self.kernel.kernel == 'precomputed'

    def fit(self, X, y, **params):
        """Fit the MKL and learn the kernel.

        Parameters:
        X : array-like
            X should be a list of kernels (if self.kernel.kernel is
            'precomputed'). Otherwise is a 2-d data matrix.
        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        self.set_params(**params)

        # X = np.atleast_2d(X)
        y = y.ravel()

        n_kernels = len(X)
        # kernel weights
        gammas = (1.0 / n_kernels) ** (1.0 / self.p) * np.ones(n_kernels)

        for it in range(self.max_iter):
            if self.verbose:
                print("Gammas : %s" % gammas)

            # svc = svm.SVC(kernel=multi_kernel, C=self.C)
            # svc.fit(X, y)
            if self._is_precomputed:
                kernels_ = X
            else:
                kernels_ = [kernel(X) for kernel in self.kernels]

            K = _combine_kernels(kernels_, gammas, self.combination)
            self.kernel.fit(K, y)

            gammas_ = self._update_kernel_weights(X, y)
            if (gammas - gammas_).max() < self.tol:
                if self.verbose:
                    print("Converged after %d interations" % it)
                break

            # multi_kernel.gammas = gammas_
            gammas = gammas_
        else:
            if self.verbose:
                print("Did NOT converge after %d interations" % it)

        self.gammas_ = gammas
        return self

    def predict(self, X):
        """Predict."""
        check_is_fitted(self, 'gammas_')
        if self._is_precomputed:
            kernels_ = X
        else:
            kernels_ = [kernel(X) for kernel in self.kernels]

        K = _combine_kernels(kernels_, self.gammas_, self.combination)
        return self.kernel.predict(K)

    @abstractmethod
    def _update_kernel_weights(self, kernels, y):
        raise NotImplementedError("abstract method")


class MultiKernelRegressor(MultipleKernelLearning, RegressorMixin):
    """Multiple Kernel Learning (MKL) for regression problems."""

    def _update_kernel_weights(self, kernels, y):
        """Following Qiu and Lane 2009."""
        from sklearn.metrics import mean_squared_error
        scores = np.array([mean_squared_error(y, self.kernel.predict(Ki))
                           for Ki in kernels])
        return (np.sum(scores) - scores) / ((scores.size - 1) * np.sum(scores))


class MultiKernelClassifier(MultipleKernelLearning, ClassifierMixin):
    """Multiple Kernel Learning (MKL) for classification problems."""

    def _update_kernel_weights(self, kernels, y):
        """Following Tanabe et al. (2008).

        delta is the threshold that should be less than or
        equal to the minimum of the accuracies obtained from single-kernel
        learners.
        """
        from sklearn.metrics import accuracy_score
        scores = np.array([accuracy_score(y, self.kernel.predict(Ki))
                           for Ki in kernels])
        threshold = np.min(scores)
        scores -= threshold
        return scores / np.sum(scores)
