import numpy as np
import matplotlib.pyplot as plt
from ecommerce_project.process import get_data


def y2indicator(Y):
    ind = np.zeros([len(Y), len(set(Y))])
    for i in range(len(Y)):
        ind[i, Y[i]] = 1
    return ind

def predict(P):
    return np.argmax(P, axis=1)

def forward_prop(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    expA  = np.exp(A)
    return expA/expA.sum(axis=1, keepdims=True), Z

def cross_entropy_cost(pY, T):
    c = T * np.log(pY)
    return -c.mean()

def classification_rate(Y, P):
    return np.mean(P == Y)

def derivative_w2(T, Y, Z):
    return Z.T.dot(T - Y)

def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

def derivative_w1(T, Y, Z, w2, X):
    dZ = (T - Y).dot(w2.T) * (1-Z*Z)
    return X.T.dot(dZ)

def derivative_b1(T, Y, Z, w2):
    return ((T - Y).dot(w2.T) * Z * Z).sum(axis=0)


#Get data and for train and test matrices
X, Y = get_data()
Y = Y.astype(np.int32)
XTrain = X[:-100]
YTrain = Y[:-100]
YTrain_ind = y2indicator(YTrain)
XTest = X[-100:]
YTest = Y[-100:]
YTest_ind = y2indicator(YTest)

#Initialize random weights
M = 5                       #No of hidden layers
D = X.shape[1]              #No of input features
K = len(set(Y))             #No of classes
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

learning_rate = 0.001

for i in range(10000):
    #Do forward prop and calculate cost
    pYTrain, Z = forward_prop(XTrain, W1, b1, W2, b2)
    pYTest, ZTest = forward_prop(XTest, W1, b1, W2, b2)
    cTrain = cross_entropy_cost(pYTrain, YTrain_ind)
    cTest = cross_entropy_cost(pYTest
                               , YTest_ind)

    #Print Training cost and classififcation rate
    if(i%100 == 0):
        YPredict = predict(pYTrain)
        print("Training cost: ", cTrain, "Classification Rate: ", classification_rate(YPredict, YTrain))

    #Backprop
    W2 += learning_rate * derivative_w2(YTrain_ind, pYTrain, Z)
    b2 += learning_rate * derivative_b2(YTrain_ind, pYTrain)
    W1 += learning_rate * derivative_w1(YTrain_ind, pYTrain, Z, W2, XTrain)
    b1 += learning_rate * derivative_b1(YTrain_ind, pYTrain, Z, W2)


print("Training Classification rate: ", classification_rate(YTrain, predict(pYTrain)))
print("Testing Classification rate: ", classification_rate(YTest, predict(pYTest)))
