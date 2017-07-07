import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv('C:\Projects\Deep learning using python Part 1\ecommerce_project\ecommerce_data.csv')
    data = df.as_matrix()
    X = data[:,:-1]
    Y = data[:,-1]

    #Normalize data
    X[:,1] =  (X[:,1] - X[:,1].mean()) / X[:,1].std()               #is mobile or not no of products viewed
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()            #no of products viewed

    N,D = X.shape
    X2 = np.zeros([N, D+3])
    X2[:, 0:D-1] = X[:, 0:D-1]

    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1
    return X2,Y

def get_binary_data():
    #For binary classification
    X,Y = get_data()
    X2 = X[Y<=1]
    Y2 = Y[Y<=1]
    return X2, Y2

X, Y = get_data()
print(X.shape, Y.shape)


