#!/usr/bin/env python3
import numpy as np
import sklearn
import sklearn.datasets
from pprint import pprint
import matplotlib.pyplot as plt


class Sigmoid(object):
    @staticmethod
    def func(X):
        return 1.0 / (1.0 + np.exp(-X))

    @staticmethod
    def deriv(X):
        s = Sigmoid.func(X)
        return s * (1.0 - s)


class MSE(object):
    @staticmethod
    def loss(X, Y):
        diffs = Y - X
        return 0.5 * np.sum(diffs ** 2, axis=0)

    @staticmethod
    def deriv(X, Y):
        return X - Y


class FCNN(object):
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.weights = [
            np.random.randn(outsize, insize)
            for insize, outsize in zip(layer_sizes, layer_sizes[1:])
        ]

        self.biases = [
            np.random.randn(outsize, 1)
            for outsize in layer_sizes[1:]
        ]

    def forward(self, X):
        """
        input: X: d x n matrix
        output: d_out x n matrix of labels
        """
        for (weight, bias) in zip(self.weights, self.biases):
            X = Sigmoid.func(np.dot(weight, X) + bias)

        return X

    def backward(self, X, Y):
        """
        input: X: d x n matrix
               Y: d_out x n matrix
        output: (dLoss_dWeights, dLoss_dBiases) where dLoss_dWeights is a
                num_layers long list of matrices the size of each weight and
                similarly for biases
        """

        activations = [X]
        zs = []

        for (weight, bias) in zip(self.weights, self.biases):
            z = np.dot(weight, X) + bias
            zs.append(z)

            X = Sigmoid.func(z)
            activations.append(X)

        dL_dWs = [np.zeros(w.shape) for w in self.weights]
        dL_dbs = [np.zeros(b.shape) for b in self.biases]

        dL_da = MSE.deriv(X, Y)
        da_dz = Sigmoid.deriv(zs[-1])
        # 1 x n
        delta = dL_da * da_dz

        dL_dWs[-1] = delta.dot(activations[-2].T)
        dL_dbs[-1] = np.sum(delta, axis=1, keepdims=True)

        for li in range(2, self.num_layers):
            z = zs[-li]
            delta = self.weights[-li + 1].T.dot(delta) * Sigmoid.deriv(z)
            
            dL_dWs[-li] = delta.dot(activations[-li - 1].T)
            dL_dbs[-li] = np.sum(delta, axis=1, keepdims=True)

        return (dL_dWs, dL_dbs)

    def update_mini_batch(self, X, Y, eta):
        """
        input: X is d_in x n, Y is d_out x n, eta is learning rate (float)
        effect: updates weights and biases by deltas
        """
        dW, db = self.backward(X, Y)

        self.weights = [w - (eta / len(X)) * d
                        for w, d in zip(self.weights, dW)]

        self.biases = [b - (eta / len(X)) * d
                       for b, d in zip(self.biases, db)]

        
    def SGD(self, X, Y, epochs, mini_batch_size, eta):
        n = len(X)
        for i in range(epochs):
            ixs = np.arange(n)
            mini_batch_ixs = np.random.permutation(ixs)[:mini_batch_size]
            mini_X = X[mini_batch_ixs]
            mini_Y = Y[mini_batch_ixs]

            self.update_mini_batch(mini_X.T, mini_Y[None], eta)
        

if __name__ == "__main__":
    X, y = sklearn.datasets.make_moons(100, noise=0.2)
    n = len(X)
    splits = [0.8, 0.9]
    split_ixs = list(map(lambda s: int(s * n), splits))
    nTr = split_ixs[0]
    xTr, yTr = X[:split_ixs[0]], y[:split_ixs[0]]
    xVa, yVa = X[split_ixs[0]:split_ixs[1]], y[split_ixs[0]:split_ixs[1]]
    xTe, yTe = X[split_ixs[1]:], y[split_ixs[1]:]

#    plt.scatter(X[:,0], X[:,1], c=y)
#    plt.show()

    nn = FCNN([2, 4, 4, 1])
    preds = nn.forward(xVa.T)
    print(1/2 * np.sum((yVa[None] - preds) ** 2))

    nn.SGD(xTr, yTr, 100, 100, 0.1)

    preds = nn.forward(xVa.T)
    print(1/2 * np.sum((yVa[None] - preds) ** 2))

    
    plt.subplot(224)
    plt.scatter(X[:,0], X[:,1], c=y)
    
    plt.subplot(221)
    preds1 = nn.forward(xTr.T)
    print(preds1)
    plt.scatter(xTr[:,0], xTr[:,1], c = np.round(preds1)[0].astype(int))
    plt.subplot(222)
    preds2 = nn.forward(xVa.T)
    plt.scatter(xVa[:,0], xVa[:,1], c = np.round(preds2)[0].astype(int))
    plt.subplot(223)
    preds3 = nn.forward(xTe.T)
    plt.scatter(xTe[:,0], xTe[:,1], c = np.round(preds3)[0].astype(int))
    plt.show()

