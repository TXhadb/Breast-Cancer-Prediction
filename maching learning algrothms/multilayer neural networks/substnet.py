import pandas as pd
import numpy as np
import sys
import time

def substitute(Wx):
    return np.log(1 + np.exp(Wx))

def logistic(Wx):
    return np.exp(Wx).div(np.ones(len(Wx)) @ np.exp(Wx))

def substnet(train_file_name, test_file_name):

    start = time.time()

    YX = pd.read_csv(train_file_name)

    Y = pd.get_dummies(YX[YX.columns[0]])
    X = YX[YX.columns[1:]]
    N = len(YX)
    X['line'] = np.ones((N, 1))

    J = len(Y.columns)
    K = 5
    V = len(X.columns)
    L = 2

    W = [None, pd.DataFrame(np.random.rand(K, V), range(K), X.columns), \
         pd.DataFrame(np.random.rand(J, K), Y.columns, range(K))]

    f = {}
    f[0] = lambda x: x
    f[1] = lambda x: substitute(W[1] @ f[0](x))
    f[2] = lambda x: logistic(W[2] @ f[1](x))

    fx = {}
    df_dWf = {}
    dC_dWf = {}

    for i in range(100):
        for n in range(N):
            for l in range(L + 1):
                fx[l] = f[l](X.iloc[[n]].T)
                if l > 0:
                    df_dWf[l] = np.diagflat((np.exp(W[l] @ fx[l - 1]) / (1 + np.exp(W[l] @ fx[l - 1]))).values)
            for l in range(L, 0, -1):
                if l == L:
                    dC_dWf[l] = (logistic(W[l] @ fx[l - 1]) - Y.iloc[[n]].T).T
                else:
                    dC_dWf[l] = dC_dWf[l + 1] @ W[l + 1] @ df_dWf[l]
                W[l] -= 1 / N * dC_dWf[l].T @ fx[l - 1].T

    print('Training time for substnet over 100 iterations is ' + str(time.time() - start) + ' seconds')

    X = pd.read_csv(test_file_name)
    X['line'] = np.ones((len(X), 1))

    res = np.empty(len(X), dtype = 'str')
    res = res.astype('U256')
    
    for n in range(len(X)):
        res[n] = f[L](X.iloc[[n]].T).idxmax()[n]
    
    res = pd.DataFrame(res.reshape(-1, 1), columns = [YX.columns[0]])
    res.to_csv('substnet-predictions.csv', index = False)

if __name__ == '__main__':
    substnet(sys.argv[1], sys.argv[2])
