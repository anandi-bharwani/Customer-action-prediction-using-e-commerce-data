import numpy as np
from ecommerce_project.process import get_data

X, Y = get_data()

M = 5
N, D  = X.shape
K = len(set(Y))

W1 = np.random.randn(D,M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)

def softmax(A):
    expA = np.exp(A)
    P = expA / expA.sum(axis=1, keepdims=True)
    return P

def forward_prop(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) + b1))
    A = Z.dot(W2) + b2
    return softmax(A)

def classification_rate(Y, P):
    return np.mean(Y == P)

Py_X = forward_prop(X, W1, b1, W2, b2)
P = np.argmax(Py_X, axis =1)
assert(len(Y) == len(P))
print("Classification rate: ", classification_rate(Y, P))