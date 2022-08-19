import pandas as pd
import numpy as np
import sys
import time

def logisticReg(train_file_name, test_file_name):

    start = time.time()

    YX = pd.read_csv(train_file_name)
    N = len(YX)

    Y_name = YX.columns[0]
    Y = pd.get_dummies(YX[Y_name])
    X = YX[YX.columns[1:]]
    X['line'] = np.ones(N)
    K = len(Y.columns)
    V = len(X.columns)

    W = pd.DataFrame(np.random.rand(K, V), Y.columns, X.columns)

    for i in range(1, 1001):
        W = W + 1 / N * (Y.T - np.exp(W @ X.T) @ np.linalg.inv(np.diag(np.ones(K).T @ np.exp(W @ X.T)))) @ X

    print('Training time for logisticReg over 1000 iterations is ' + str(time.time() - start) + ' seconds')

    X = pd.read_csv(test_file_name)
    X['line'] = np.ones(len(X))
    Y = X @ W.T

    res = np.empty(len(Y), dtype = str)
    res = res.astype('U256')

    for i in range(len(Y)):
        Y.iloc[i] = np.exp(Y.iloc[i])
        tot = np.sum(Y.iloc[i])
        Y.iloc[i] /= tot
        res[i] = Y.columns[Y.iloc[i].argmax()]
    
    res = pd.DataFrame(res.reshape(-1, 1), columns = [Y_name])
    res.to_csv('logisticReg-predictions.csv', index = False)

if __name__ == '__main__':
    logisticReg(sys.argv[1], sys.argv[2])
