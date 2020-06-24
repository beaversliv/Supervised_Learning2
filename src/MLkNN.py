import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

def one_hot(y):
    classes = 10
    values_train = y.reshape(-1)
    enc_y = np.eye(classes)[values_train]
    return enc_y


class MLkNN():

    def __init__(self, X, y, k, s):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.labels = len(np.unique(self.y))
        assert (len(y) == self.m)

        self.k = k
        self.s = s
        self.ph1 = np.zeros(self.labels)
        self.ph0 = np.zeros(self.labels)
        self.peH1 = np.zeros((self.labels, self.k + 1))
        self.peH0 = np.zeros((self.labels, self.k + 1))

    def fit(self):
        # calculate the prior distribution
        self.e_y = one_hot(self.y)
        self.ph1 = (1 + self.e_y.sum(axis=0)) / (self.s * 2 + len(self.e_y))
        self.ph0 = 1 - self.ph1

        # calculate the posterior distribution
        self.model = KNeighborsClassifier(self.k + 1)
        self.model.fit(self.X, self.y)
        self.neighbors = []
        for i in range(self.m):
            self.neighbors.append(self.model.kneighbors(self.X[i].reshape(1, -1), self.k + 1)[1][0])
        self.neighbors = np.delete(self.neighbors, 0, 1)
        for lab in range(self.labels):
            self.c1 = np.zeros(self.k + 1)
            self.c0 = np.zeros(self.k + 1)
            for j in range(self.m):
                self.nn = self.neighbors[j]
                deltas = 0
                for a in self.nn:
                    if self.e_y[a][lab] == 1:
                        deltas += 1
                if self.e_y[j][lab] == 1.0:
                    self.c1[deltas] += 1
                else:
                    self.c0[deltas] += 1

            for k in range(self.k + 1):
                self.peH1[lab][k] = (1 + self.c1[k]) / ((self.k + 1) + self.c1.sum())
                self.peH0[lab][k] = (1 + self.c0[k]) / ((self.k + 1) + self.c0.sum())

    def predict(self, test_X):
        test_size = test_X.shape[0]
        self.label_pred = np.zeros((test_size, self.labels))
        self.pred_list = []
        for i in range(test_size):
            self.pred_list.append(self.model.kneighbors(test_X[i].reshape(1, -1), self.k + 1)[1][0])
        self.pred_list = np.delete(self.pred_list, 0, 1)
        for i in range(test_size):
            nn = self.pred_list[i]
            for lab in range(self.labels):
                c = 0
                for ks in range(self.k):
                    if self.e_y[nn[ks]][lab] == 1:
                        c += 1
                y1 = self.ph1[lab] * self.peH1[lab][c]
                y0 = self.ph0[lab] * self.peH0[lab][c]
                if y1 > y0:
                    self.label_pred[i][lab] = 1
                else:
                    self.label_pred[i][lab] = 0
        return self.label_pred