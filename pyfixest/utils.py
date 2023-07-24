import numpy as np
import pandas as pd

def get_data(seed = 1234):

    '''
    create a random example data set
    '''

    # create data

    np.random.seed(seed)

    N = 2000
    k = 20
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

    data["Z1"] = data["X1"] + np.random.normal(0, 1, data.shape[0])
    data["Z2"] = data["X2"] + np.random.normal(0, 1, data.shape[0])
    data["Z3"] = data["X3"] + np.random.normal(0, 1, data.shape[0])

    #data['X2'][2] = np.nan


    return data



def get_poisson_data(N = 1000, seed = 4320):

    '''
    Generate data following a Poisson regression dgp.
    Args:
        N: number of observations
        seed: seed for the random number generator
    Returns:
        data: a pandas data frame
    '''

    # create data
    np.random.seed(seed)
    X1 = np.random.normal(0, 1, N)
    X2 =  np.random.choice([0, 1], N, True)
    X3 = np.random.choice([0, 1, 2, 3, 4, 5, 6], N, True)
    X4 = np.random.choice([0, 1], N, True)
    beta = np.array([1, 0, 1, 0])
    u = np.random.normal(0,1,N)
    mu = np.exp(1 + X1 * beta[0] + X2 * beta[1] + X3 * beta[2] + X4 * beta[3] + u)

    Y = np.random.poisson(mu, N)

    data = pd.DataFrame({'Y':Y, 'X1':X1, 'X2':X2, 'X3':X3, 'X4':X4})

    return data
