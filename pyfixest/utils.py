import numpy as np
import pandas as pd

def get_data():

    '''
    create a random example data set
    '''

    # create data

    np.random.seed(1231)

    N = 10000
    k = 5
    G = 25
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    X = pd.DataFrame(X)
    X[1] = np.random.choice(list(range(0, 5)), N, True)
    X[2] = np.random.choice(list(range(0, 10)), N, True)
    X[3] = np.random.choice(list(range(0, 10)), N, True)
    X[4] = np.random.normal(0, 1, N)

    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(0,G)), N)

    Y = pd.DataFrame(Y)
    Y.rename(columns = {0:'Y'}, inplace = True)
    X = pd.DataFrame(X)

    data = pd.concat([Y, X], axis = 1)
    data.rename(columns = {0:'X1', 1:'X2', 2:'X3', 3:'X4', 4:'X5'}, inplace = True)

    data['group_id'] = cluster
    data['Y2'] = data.Y + np.random.normal(0, 1, N)

    data['Y'][0] = np.nan
    data['X1'][1] = np.nan
    #data['X2'][2] = np.nan


    return data
