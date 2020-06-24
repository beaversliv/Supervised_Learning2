import numpy as np
from scipy.stats import mode

from scipy.stats import mode

# use this if you want it to be slow
class KernelKNN:

    def __init__(self, X, y, kernel, k=4):
        self.k = k
        self.X = X
        self.y = y
        self.c = len(np.unique(y))
        self.m, self.n = X.shape
        assert (len(y) == self.m)

        self.kernel = kernel
        self.K = self.kernel(self.X, self.X)

    def get_feature_distances(self, x):
        Kxt = self.kernel(x.reshape(1, -1), self.X).squeeze()
        Kxx = self.kernel(x.reshape(1, -1), x.reshape(1, -1))[0, 0]
        Ktt = np.diagonal(self.K)
        distances = np.sqrt(Kxx + Ktt - 2 * Kxt)
        return distances

    def predict(self, x):
        distances = self.get_feature_distances(x)
        smallest_idx = np.argpartition(distances, self.k)[:self.k]
        classes = self.y[smallest_idx]
        return mode(classes)[0][0]

# use this if you want it to be monstrously fast
class VectorizedKernelKNN:

    def __init__(self, X, y, kernel, k=1):
        self.k = k
        self.X = X
        self.y = y
        self.c = len(np.unique(y))
        self.m, self.n = X.shape
        assert (len(y) == self.m)

        self.kernel = kernel
        self.K = self.kernel(self.X, self.X)

    # vertorize everything to make this thing rapid
    def get_feature_distances(self, X):
        # num_examples * num_training points
        # = K(x, x_t)
        Kxt = self.kernel(X, self.X)

        # num_examples * 1 = K(x, x)
        Kxx = np.diagonal(self.kernel(X, X))
        Kxx = np.broadcast_to(Kxx, (Kxt.shape[1], Kxt.shape[0])).T

        # num_training_points * 1 = K(x_t, x_t)
        Ktt = np.diagonal(self.K)
        Ktt = np.broadcast_to(Ktt, Kxt.shape)

        return np.sqrt((Kxx + Ktt) - 2 * Kxt)

    def predict_all(self, X):
        distances = self.get_feature_distances(X)
        smallest_idx = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        classes = self.y[smallest_idx]
        return mode(classes, axis=1)[0].squeeze()