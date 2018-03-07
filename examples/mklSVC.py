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
import pylab as pl

from multikernel.mSVC import RBF, MultiKernelSVC, linear

if __name__ == '__main__':
    xx, yy = np.meshgrid(np.linspace(-5, 5, 40), np.linspace(-5, 5, 40))
    np.random.seed(0)
    X = np.random.randn(300, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    # y = X[:,0]>0

    y = np.array(y, dtype=np.int)
    y[y == 0] = -1

    # Define the kernels
    kernels = [RBF(10 ** k) for k in range(-5, 0)]  # some RBF kernels
    kernels.append(linear)  # Add linear kernel

    # fit the model
    clf = MultiKernelSVC(kernels=kernels, C=1e6, verbose=True, maxit=100,
                         tol=1e-5, p=1, store_objective=True)
    clf.fit(X, y)

    objective = clf.objective

    pl.close('all')
    pl.figure()
    pl.plot(objective)
    pl.xlabel('Iterations')
    pl.ylabel('Dual objective')

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # pl.close('all')
    pl.figure()
    pl.set_cmap(pl.cm.Paired)
    pl.pcolormesh(xx, yy, Z)
    pl.scatter(X[:,0], X[:,1], c=y)

    pl.axis('tight')
    pl.show()
