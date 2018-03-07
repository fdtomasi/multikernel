"""An example for regression problems."""
import matplotlib.pyplot as pl
import numpy as np
import seaborn as sns
from sklearn.svm import SVR

from multikernel.base import MultiKernelRegressor

xx, yy = np.meshgrid(np.linspace(-5, 5, 40), np.linspace(-5, 5, 40))
np.random.seed(0)
X = np.random.randn(300, 2)
y = np.linspace(0, 200, X.shape[0])
# y = X[:,0]>0

K1 = 1. * np.dot(X, X.T)
K2 = 2.0 * np.dot(X, X.T)
kernels = [K1, K2]

clf = MultiKernelRegressor(
    verbose=1, maxit=100,
    tol=1e-5, p=1, kernel=SVR(kernel='precomputed'))
#   store_objective=True, kernel=KernelRidge(kernel='precomputed', alpha=.5))
clf.fit(kernels, y)

print("Prediction score:", clf.score(kernels, y))

# pl.close('all')
# pl.figure()
# pl.plot(objective)
# pl.xlabel('Iterations')
# pl.ylabel('Dual objective')

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


p =1; n_kernels = 2
gammas = (1.0 / n_kernels) ** (1.0 / p) * np.ones(n_kernels)

pl.close('all')
sns.pointplot(X[:, 0], X[:,1], y, linestyle='')
pl.show()
