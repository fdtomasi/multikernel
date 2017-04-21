"""
=============================
Multiple Kernel Learning (MKL)
=============================

Computes an SVM where the kernel is learnt
as a linear combination of precomputed kernels.

"""
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Marie Szafranski <marie.szafranski@ibisc.fr>
#
# License: BSD Style.

import numpy as np

from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin

class RBF(object):
    """docstring for RBF"""
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        XX = np.sum(X * X, axis=1)[:, np.newaxis]
        if Y is None:
            Y = X
            YY = XX.T
        else:
            YY = np.sum(Y * Y, axis=1)[np.newaxis, :]
        distances = XX + YY  # Using broadcasting
        distances -= 2 * np.dot(X, Y.T)
        distances = np.maximum(distances, 0)
        return np.exp(- self.gamma * distances)


def linear(X, Y=None):
    """Linear kernel"""
    if Y is None:
        Y = X
    return np.dot(X, Y.T)


class MultiKernelLinear(object):
    def __init__(self, kernels, gammas, X=None, kernel='precomputed'):
        self.kernels = kernels
        self.gammas = gammas
        self.X = X
        self.Ks = None
        self.kernel = kernel
        if X is not None:  # Precompute kernels
            self.Ks = [kernel(X) for kernel in kernels]

    def __call__(self, X, Y=None):
        """Construct kernel by linear combination"""
        K = 0
        if X is self.X and (Y is X or Y is None):
            for gamma, Ki in zip(self.gammas, self.Ks):
                if gamma > 0.0:
                    K += gamma * Ki
        else:
            for gamma, kernel in zip(self.gammas, self.kernels):
                if gamma > 0.0:
                    K += gamma * kernel(X, Y)
        return K

    @property
    def _precomputed(self):
        return self.kernel == "precomputed"


class MultiKernelSVC(BaseEstimator, ClassifierMixin):
    """Lp - Multiple Kernel Learning (MKL).

    2 classes only

    Parameters
    ----------

    kernel : string, optional
        List of precomputed kernels.

    p : float, optional
        ???

    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    Notes
    -----
    Add ref Szafranski 2010 Machine Learning Journal
    """

    def __init__(self, kernels, p=1, maxit=10, C=1, verbose=False, tol=1e-5,
                 store_objective=False):
        self.kernels = kernels
        self.p = p
        self.maxit = maxit
        self.C = C
        self.verbose = verbose
        self.tol = tol
        self.store_objective = store_objective

    def fit(self, X, y, **params):
        """Fit the MKL and learn the kernel."""
        self.set_params(**params)

        X = np.atleast_2d(X)
        y = y.ravel()

        classes = np.unique(y)
        n_classes = classes.size
        assert n_classes == 2

        y = np.array(y, dtype=np.int)
        y[y == classes[0]] = -1
        y[y == classes[1]] = 1

        p = float(self.p)
        kernels = self.kernels
        C = self.C

        n_kernels = len(self.kernels)
        # kernel weights
        gammas = (1.0 / n_kernels) ** (1.0 / p) * np.ones(n_kernels)

        # Construct kernel by linear combination
        multi_kernel = MultiKernelLinear(kernels, gammas, X)
        Ks = multi_kernel.Ks

        norms = np.empty(n_kernels)
        maxit = self.maxit

        objective = []

        for it in range(maxit):
            if self.verbose:
                print "Gammas : %s" % multi_kernel.gammas

            svc = svm.SVC(kernel=multi_kernel, C=C)
            svc.fit(X, y)
            dual_coef_ = svc.dual_coef_.ravel()
            support_ = np.array(svc.support_, dtype=np.int).ravel() - 1

            # Update kernel weights
            for i, (gamma, K) in enumerate(zip(multi_kernel.gammas, Ks)):
                norms[i] = (gamma * gamma) * np.dot(
                    dual_coef_, np.dot(K[support_][:, support_], dual_coef_))

            if self.store_objective:
                dual_obj = -0.5 * np.dot(
                    dual_coef_, np.dot(
                        multi_kernel(X[support_]), dual_coef_)) + (
                            dual_coef_ * y[support_]).sum()
                objective.append(dual_obj)

            # print norms
            norms = norms ** (1.0 / (1.0 + p))
            scaling = np.sum(norms ** p) ** (1.0 / p)
            gammas_ = norms / scaling

            gammas_[gammas_ < 1e-6 * gammas_.max()] = 0.0

            if (gammas_ - multi_kernel.gammas).max() < self.tol:
                if self.verbose:
                    print "Converged after %d interations" % it
                break

            multi_kernel.gammas = gammas_
        else:
            if self.verbose:
                print "Did NOT converge after %d interations" % it

        self._svc = svc
        self.gammas_ = multi_kernel.gammas
        self.objective = objective
        return self

    def predict(self, X):
        return self._svc.predict(X)
