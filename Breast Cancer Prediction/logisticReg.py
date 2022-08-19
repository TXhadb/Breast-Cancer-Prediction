import pandas as pd
import numpy as np
import time
import sys

def fit_predict(train_data, test_observables, iterations, learning_rate):

    YX = train_data
    N = len(YX)
    num_iterations = iterations
    alpha = learning_rate

    Y_name = YX.columns[0]
    Y = pd.get_dummies(YX[Y_name])
    X = YX[YX.columns[1:]]
    X['line'] = np.ones(N)
    K = len(Y.columns)
    V = len(X.columns)

    W = pd.DataFrame(np.random.rand(K, V), Y.columns, X.columns)

    for i in range(num_iterations):
        W += alpha / N * (Y.T - np.exp(W @ X.T) @ np.linalg.inv(np.diag(np.ones(K).T @ np.exp(W @ X.T)))) @ X
    
    X = test_observables
    X['line'] = np.ones(len(X))
    Y = X @ W.T

    res = np.empty(len(Y))

    for i in range(len(Y)):
        Y.iloc[i] = np.exp(Y.iloc[i])
        tot = np.sum(Y.iloc[i])
        Y.iloc[i] /= tot
        res[i] = Y.columns[Y.iloc[i].argmax()]
    
    res = pd.DataFrame(res.reshape(-1, 1), columns = [Y_name])

    return res

if __name__ == '__main__':
    train_data = pd.read_csv(sys.argv[1])
    test_observables = pd.read_csv(sys.argv[2])
    iterations = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    start = time.time()
    res = fit_predict(train_data, test_observables, iterations, learning_rate)
    print('The time cost for logisticReg is ' + str(round(time.time() - start, 2)) + ' seconds.')
    res.to_csv('logisticReg_predictions.csv', index = False)
    
