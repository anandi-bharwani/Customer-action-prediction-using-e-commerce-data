import numpy as np
import matplotlib.pyplot as plt
from ecommerce_project.process import get_data

def softmax(A):
    expA = np.exp(A)
    ret = expA/expA.sum(axis=1, keepdims=True)
    return ret

def forward(X, W, b):
    A = X.dot(W) + b
    return softmax(A)

def classification_rate(P, Y):
    return np.mean(Y==P)

def predict(P):
    return np.argmax(P, axis=1)

def cross_entropy_cost(Y, T):
    c = T * np.log(Y)
    return c.sum()

def y2indicator(Y):
    ind = np.zeros([len(Y), len(set(Y))])
    for i in range(len(Y)):
        ind[i, Y[i]] = 1
    return ind

def derivative_w(Y, T, X):
    return X.T.dot(T-Y)

def derivative_b(Y, T):
    return (T-Y).sum(axis = 0)

X, Y = get_data()
#X,Y = shuffle(X,Y)
Y = Y.astype(np.int32)
N, D = X.shape
K = len(set(Y))

W = np.random.rand(D, K)
b = 0

XTrain = X[:-100]
YTrain = Y[:-100]
YTrain_ind = y2indicator(YTrain)
XTest = X[-100:]
YTest = Y[-100:]
YTest_ind = y2indicator(YTest)
learning_rate = 0.001

for i in range(10000):
    pYTrain = forward(XTrain, W, b)
    pYTest = forward(XTest, W, b)

    cTrain = cross_entropy_cost(pYTrain, YTrain_ind)
    cTest = cross_entropy_cost(pYTest, YTest_ind)

    W += learning_rate * derivative_w(pYTrain, YTrain_ind, XTrain)
    b += learning_rate * derivative_b(pYTrain, YTrain_ind)
    if(i%10==0):
        print(cTrain, cTest)

print("Training Classification rate: ", classification_rate(YTrain, predict(pYTrain)))
print("Testing Classification rate: ", classification_rate(YTest, predict(pYTest)))