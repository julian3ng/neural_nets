#!/usr/bin/env python3
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pprint import pprint
import matplotlib.pyplot as plt


def one_hot(Y, encoding_size):
    arr = np.zeros((len(Y), encoding_size))
    arr[np.arange(len(Y)), Y] = 1
    return arr


def un_one_hot(Y_oh):
    return np.argmax(Y_oh, axis=1)


class Sigmoid(object):
    @staticmethod
    def f(X):
        
        return np.where(X >= 0, 1.0 / (1.0 + np.exp(-X)), np.exp(X) / (1.0 + np.exp(X)))

    @staticmethod
    def df(X):
        s = Sigmoid.f(X)
        return s * (1.0 - s)


class Relu(object):
    @staticmethod
    def f(X):
        return X * (X > 0.0)

    @staticmethod
    def df(X):
        return 1.0 * (X > 0.0)


class Tanh(object):
    @staticmethod
    def f(X):
        return np.tanh(X)

    @staticmethod
    def df(X):
        return 1 - np.tanh(X) ** 2


class MSE(object):
    @staticmethod
    def f(X, Y):
        diff = Y - X
        return np.sum(0.5 * np.sum(diff ** 2, axis=1, keepdims=True), axis=0)

    @staticmethod
    def df(X, Y):
        return X - Y


class FCNN(object):
    def __init__(self, layer_sizes, activation):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(*w)
                        for w in zip(layer_sizes, layer_sizes[1:])]
        self.biases = [np.random.randn(1, d) for d in layer_sizes[1:]]
        self.activation = activation

    def forward(self, X, debug=False):
        for (weight, bias) in zip(self.weights[:-1], self.biases[:-1]):
            X = self.activation.f(X.dot(weight) + bias)

        X = X.dot(self.weights[-1]) + self.biases[-1]
        if debug:
            print(np.round(X, 2))

        return X

    def predict(self, X, debug=False):
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X, Y):
        preds = self.predict(X)
        Y = un_one_hot(Y)
        differences = np.sum(preds == Y)
        return differences / len(Y) * 100

    def backward(self, X, Y, debug=False):
        acts = [X]
        zs   = []

        for (weight, bias) in zip(self.weights[:-1], self.biases[:-1]):
            z = X.dot(weight) + bias
            zs.append(z)

            X = self.activation.f(z)
            acts.append(X)

        acts.append(X.dot(self.weights[-1]) + self.biases[-1])

        dWs = [np.zeros(w.shape) for w in self.weights]
        dbs = [np.zeros(b.shape) for b in self.biases]

        delta = MSE.df(acts[-1], Y)
        dWs[-1] = acts[-2].T.dot(delta)
        dbs[-1] = np.sum(delta, axis=0, keepdims=True)

        for li in range(2, self.num_layers):
            z = zs[-li + 1]
            delta = delta.dot(self.weights[-li + 1].T) * self.activation.df(z)
            dWs[-li] = acts[-li - 1].T.dot(delta)
            dbs[-li] = np.sum(delta, axis=0, keepdims=True)

        return (dWs, dbs)

    def update_mini_batch(self, mini_X, mini_Y, eta, debug=False):
        m = len(mini_X)
        dWs, dbs = self.backward(mini_X, mini_Y, debug=debug)
        self.weights = [w - (eta / m) * dw
                        for w, dw in zip(self.weights, dWs)]
        self.biases = [b - (eta / m) * db
                       for b, db in zip(self.biases, dbs)]

    def SGD(self, X, Y, epochs, mini_batch_size, eta, debug=False):
        n = len(X)
        print("Starting SGD with batch size {}, learning rate {}".format(
            mini_batch_size, eta))
        for i in range(epochs):
            ixs = np.arange(n)
            np.random.shuffle(ixs)
            Xshuf, Yshuf = X[ixs], Y[ixs]

            for j in range(0, n, mini_batch_size):
                self.update_mini_batch(Xshuf[j: j + mini_batch_size],
                                       Yshuf[j: j + mini_batch_size],
                                       eta, debug=debug)
            if debug:
                print("Epoch {} complete, loss: {}".
                      format(i, np.round(MSE.f(self.forward(X), Y), 2)))
            else:
                if i % (epochs // 10) == 0:
                    print("Epoch {} complete, loss: {}".
                          format(i,
                                 np.round(np.sum(MSE.f(self.forward(X), Y)),
                                          2)))

import pickle
import gzip

if __name__ == "__main__":
    np.random.seed(0)

    with gzip.open("./data/mnist.pkl.gz", "rb") as f:
        train, valid, test = pickle.load(f, encoding='latin1')

    xTr, yTr = train
    xVa, yVa = valid
    xTe, yTe = test

    yTr = one_hot(yTr, 10)
    yVa = one_hot(yVa, 10)
    yTe = one_hot(yTe, 10)    

    nn = FCNN([784, 15, 10], Sigmoid)

    nn.forward(xTr, debug=True)
    nn.SGD(xTr, yTr, 30, 100, 1, debug=True)

    print("Train accuracy: {}".format(nn.evaluate(xTr, yTr)))
    print("Test accuracy: {}".format(nn.evaluate(xTe, yTe)))
    preds = nn.predict(xTe)
    print(confusion_matrix(un_one_hot(yTe), preds))
