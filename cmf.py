import numpy as np

def get_error(S, S_hat):
    N = S.shape[0]*S.shape[1]
    return np.sqrt(np.sum((S - S_hat)**2)/N)

def split_train_test(X, Y, Z=0, train_fraction=0.8, seed = 8635):
    """Randomly splits two numpy matrices into train and test data using the same indices

    Args:
        X: Matrix with rows from S, dimension (n, d)
        Y: Matrix with rows from M, dimension (n, p)
        train_fraction: persent of training data

    Returns: Four matrices
    """

    np.random.seed(seed)
    indices = np.random.permutation(X.shape[0])
    numtrain = int(np.floor(train_fraction*X.shape[0]))
     
    X_train, X_test = X[indices[:numtrain],:], X[indices[numtrain:],:]
    Y_train, Y_test = Y[indices[:numtrain],:], Y[indices[numtrain:],:]

    if Z != 0:
        Z_train, Z_test = Z[indices[:numtrain],:], Z[indices[numtrain:],:]
        return X_train, X_test, Y_train, Y_test, Z_train, Z_test
    
    return X_train, X_test, Y_train, Y_test


def cmf(S1, S2, M1, M2, n_factors, lambdas, n_iterations=20):

    """Collective matrix factorization using ALS
    Warm start solution using matrix M = (M1, M2) to predict missing rows of matrix S = (S1, S2).

    Args:
        S1: Matrix with rows from S, dimension (n, d)
        S2: Matrix with rows from S, dimension (k, d)
        M1: Matrix with rows from M, dimension (n, p)
        M2: Matrix with rows from M, dimension (k, p)
        lambdas: regularization parameters, list of length 3
        n_iterations: number of iterations

    Returns: Predicted values of of S2
    """

    U1 = np.random.rand(S1.shape[0], n_factors)
    U2 = np.random.rand(M2.shape[0], n_factors)

    Vs = np.random.rand(n_factors, S1.shape[1])
    Vm = np.random.rand(n_factors, M1.shape[1])

    for ii in range(n_iterations):

        U1 = np.linalg.solve(np.dot(Vs, Vs.T) + np.dot(Vm, Vm.T)  + lambdas[0] * np.eye(n_factors), 
                            np.dot(Vs, S1.T) + np.dot(Vm, M1.T)).T

        U2 = np.linalg.solve(np.dot(Vm, Vm.T) + lambdas[0] * np.eye(n_factors), 
                        np.dot(Vm, M2.T)).T

        Vs = np.linalg.solve(np.dot(U1.T, U1) + lambdas[1] * np.eye(n_factors),
                            np.dot(U1.T, S1))
        
        U_all = np.concatenate((U1, U2), axis=0)
        M_all = np.concatenate((M1, M2), axis=0)
        Vm = np.linalg.solve(np.dot(U_all.T, U_all) + lambdas[2] * np.eye(n_factors),
                            np.dot(U_all.T, M_all))
        
    return np.dot(U2, Vs)


def graph_cmf(S1, S2, M1, M2, L1, L2, n_factors, lambdas, n_iterations=20):

    """Collective matrix factorization using ALS. Graph regularized with Laplacian L.
    (We constrain users to be similar to their friends, based on strength of ties.)
    Warm start solution using matrix M = (M1, M2) to predict missing rows of matrix S = (S1, S2).

    Args:
        S1: Matrix with rows from S, dimension (n, d)
        S2: Matrix with rows from S, dimension (k, d)
        M1: Matrix with rows from M, dimension (n, p)
        M2: Matrix with rows from M, dimension (k, p)
        L1: Matrix with rows from L, dimension (n, n+k)
        L1: Matrix with rows from L, dimension (k, n+k)
        lambdas: regularization parameters, list of length 4
        n_iterations: number of iterations

    Returns: Predicted values of of S2
    """

    U1 = np.random.rand(S1.shape[0], n_factors)
    U2 = np.random.rand(M2.shape[0], n_factors)

    Vs = np.random.rand(n_factors, S1.shape[1])
    Vm = np.random.rand(n_factors, M1.shape[1])

    for ii in range(n_iterations):

        U1 = np.linalg.solve(np.dot(Vs, Vs.T) + np.dot(Vm, Vm.T) + 0.5*lambdas[3]*(L1 + L1.T) + lambdas[0] * np.eye(n_factors), np.dot(Vs, S1.T) + np.dot(Vm, M1.T)).T

        U2 = np.linalg.solve(np.dot(Vm, Vm.T) + lambdas[3]*(L2 + L2.T) + lambdas[0] * np.eye(n_factors), 
                        np.dot(Vm, M2.T)).T

        Vs = np.linalg.solve(np.dot(U1.T, U1) + lambdas[1] * np.eye(n_factors),
                            np.dot(U1.T, S1))
        
        U_all = np.concatenate((U1, U2), axis=0)
        M_all = np.concatenate((M1, M2), axis=0)
        Vm = np.linalg.solve(np.dot(U_all.T, U_all) + lambdas[2] * np.eye(n_factors),
                            np.dot(U_all.T, M_all))
    return np.dot(U2, Vs)


def main():
    S = np.genfromtxt('data/shopping.csv', delimiter=',')
    M = np.genfromtxt('data/mobility.csv', delimiter=',')
    S1, S2, M1, M2 = split_train_test(S, M)
    S_hat = cmf(S1=S1, S2=S2, M1=M1, M2=M2, n_factors=10, lambdas=[10, 10, 10], n_iterations=20)
    get_error(S2, S_hat)