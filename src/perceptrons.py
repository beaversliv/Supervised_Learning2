#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm


class AbstractKernelPerceptron():

    def predict(self):
        raise NotImplementedError("You have to implement behaviour for prediction.")

    def run_training_epoch(self):
        raise NotImplementedError("You have to implement behaviour for a training epoch.")

    def measure(self, maybe):
        if maybe:
            return lambda x: tqdm(x)
        else:
            return lambda x: x

    def train_for_epochs(self, epochs=1, progress=False):
        for epoch in self.measure(progress)((list(range(epochs)))):
            self.run_training_epoch()


class KernelPerceptron(AbstractKernelPerceptron):

    def __init__(self, X, y, kernel):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        assert (len(y) == self.m)

        self.kernel = kernel
        self.K = self.kernel(X, X)

        self.alphas = np.zeros(self.m)

    # K_x is the vector correspond to the kernel evaluated at x against all X i.e. k(x, X)
    def f(self, K_x):
        return self.alphas.dot(K_x)

    def predict_training_pt(self, t):
        K_x = self.K[t, :]
        return 1 if self.f(K_x) >= 0 else -1

    def predict_magnitude(self, x):
        K_x = self.kernel(x.reshape(1, -1), self.X).squeeze()
        return self.f(K_x)

    def predict(self, x):
        return 1 if self.predict_magnitude(x) >= 0 else -1

    def run_training_epoch(self):
        # evaluate all training examples and update alpha
        for t in range(self.m):
            y_prime = self.predict_training_pt(t)

            # make a mistake => change alpha
            if y_prime != self.y[t]:
                self.alphas[t] += self.y[t]


class VectorizedOneVsAllKernelPerceptron(AbstractKernelPerceptron):

    def __init__(self, X, y, kernel):
        self.X = X
        self.y = y
        self.c = len(np.unique(y))
        self.m, self.n = X.shape
        assert (len(y) == self.m)

        self.kernel = kernel
        self.K = self.kernel(X, X)

        # store alphas in a matrix
        self.alphas = np.zeros((self.c, self.m))

    # K_x is the vector corresponding to the kernel evaluated at x against all X i.e. k(x, X)
    def f(self, K_x):
        return self.alphas.dot(K_x)

    def predict_training_pt(self, t):
        K_xt = self.K[t, :]
        y_prime = self.f(K_xt)
        predictions = np.ones(self.c)
        predictions[y_prime < 0] = -1
        return predictions

    def predict_magnitude(self, x):
        K_x = self.kernel(x.reshape(1, -1), self.X).squeeze()
        return self.f(K_x)

    def predict_mag(self, X):
        K_x = self.kernel(self.X, X)
        return self.f(K_x)

    def predict(self, x):
        return np.argmax(self.predict_magnitude(x))

    def predict_all(self, X):
        K_x = self.kernel(self.X, X)
        magnitudes = self.f(K_x)
        return np.argmax(magnitudes, axis=0)

    def run_training_epoch(self):
        # evaluate all training examples and update alpha
        for t in range(self.m):
            y_actual = -1*np.ones(self.c)
            y_actual[self.y[t]] = 1

            # generate predctions for all perceptrons
            predictions = self.predict_training_pt(t)
            incorrect = y_actual != predictions

            # make a mistake => change alpha
            self.alphas[incorrect, t] += y_actual[incorrect]


class VectorizedOneVsOneKernelPerceptron(AbstractKernelPerceptron):

    def __init__(self, X, y, kernel):
        self.X = X
        self.y = y
        self.m, self.n = X.shape

        self.c = len(np.unique(y))

        self.kernel = kernel
        self.K = self.kernel(X, X)

        # store alphas for i vs j perceptron at self.alphas[i][j]
        # then self.alphas.dot(K_x)[i][j] is magnitude of belief in i over j
        # note that this matrix is equal to the negative of its transpose, as are all
        # the predictions - and its leading diagonal is zero, this is equivalent
        # to training 45 perceptrons, except we're doing it with a single matrix
        # which allows us to compute the majority vote by simply taking the row num at the end.
        self.alphas = np.zeros((self.c, self.c, self.m))

    # K_x is the vector corresponding to the kernel evaluated at x against all X i.e. k(x, X)
    def f(self, K_x):
        return np.tensordot(self.alphas, K_x, axes=1)

    def d(self, K_x):
        magnitudes = self.f(K_x)
        votes = np.zeros(magnitudes.shape)

        # calculate sign of magnitude for each i-v-j classifier
        votes[magnitudes > 0] = 1.
        votes[magnitudes < 0] = -1.

        return votes

    def predict_training_pt(self, t):
        K_xt = self.K[t, :]
        return self.d(K_xt)

    def predict_votes(self, x):
        K_x = self.kernel(x.reshape(1, -1), self.X).squeeze()
        votes = self.d(K_x)

        # calculate signed sum of votes
        return votes.sum(axis=1)

    def predict(self, x):
        # get highest voted
        return np.argmax(self.predict_votes(x))

    def predict_all(self, X):
        K_x = self.kernel(self.X, X)
        votes = self.d(K_x)
        votes = votes.sum(axis=1)
        return np.argmax(votes, axis=0)

    def run_training_epoch(self):
        # evaluate all training examples and update alphas for relevant perceptron
        for t in range(self.m):
            y_actual = np.zeros((self.c, self.c))
            # -1 corresponds to voting for the column class
            y_actual[:, self.y[t]] = -1.
            # 1 corresponds to voting for the row class
            y_actual[self.y[t], :] = 1.
            y_actual[self.y[t], self.y[t]] = 0.

            # generate voting/prediction matrix with our perceptrons
            predictions = self.predict_training_pt(t)
            incorrect = y_actual != predictions

            # make a mistake => change alpha
            self.alphas[incorrect, t] += y_actual[incorrect]
