import pandas as pd
import numpy as np

def get_data():

    '''
    create a random example data set
    '''
    # create data
    np.random.seed(1234)
    N = 10_000
    k = 4
    G = 25
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    X = pd.DataFrame(X)
    X[1] = np.random.choice(list(range(0, 50)), N, True)
    X[2] = np.random.choice(list(range(0, 100)), N, True)
    X[3] = np.random.choice(list(range(0, 1000)), N, True)

    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(0,G)), N)

    Y = pd.DataFrame(Y)
    Y.rename(columns = {0:'Y'}, inplace = True)
    X = pd.DataFrame(X)

    data = pd.concat([Y, X], axis = 1)
    data.rename(columns = {0:'X1', 1:'X2', 2:'X3', 3:'X4'}, inplace = True)
    data['X4'] = data['X4'].astype('category').astype(str)
    data['X3'] = data['X3'].astype('category').astype(str)
    data['X2'] = data['X2'].astype('category').astype(str)
    data['group_id'] = cluster.astype(str)
    data['Y2'] = data.Y + np.random.normal(0, 1, N)

    return data
