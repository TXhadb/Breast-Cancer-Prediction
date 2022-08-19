import numpy as np
import pandas as pd
import time
import sys
import cvxopt
cvxopt.solvers.options['show_progress'] = False

def fit_predict(train_data, test_observables, degree):

    YX = train_data
    N = len(YX)
    alpha = degree

    y = YX[YX.columns[0]].to_frame()
    X = YX[YX.columns[1:]]
    
    H = (np.diagflat(y.values) @ (X @ X.T / (100 ** alpha)) ** alpha @ np.diagflat(y.values)).values
    
    a = np.array(cvxopt.solvers.qp(cvxopt.matrix(H, tc = 'd'), \
                                   cvxopt.matrix(-np.ones((N, 1))), \
                                   cvxopt.matrix(-np.eye(N)), \
                                   cvxopt.matrix(np.zeros(N)), \
                                   cvxopt.matrix(y.T.values, tc = 'd'), \
                                   cvxopt.matrix(np.zeros(1)))['x'])
    
    n = np.argmax(a * y)
    b = (y.T)[n] - ((a * y).T @ (X @ X.T[n] / (100 ** alpha)) ** alpha)
    
    X_test = test_observables
    
    Y = np.sign((X_test @ X.T / (100 ** alpha)) ** alpha @ (a * y) + b)
    
    return Y

if __name__ == '__main__':
    train_data = pd.read_csv(sys.argv[1])
    test_observables = pd.read_csv(sys.argv[2])
    degree = float(sys.argv[3])
    start = time.time()
    res = fit_predict(train_data, test_observables, degree)
    print('The time cost for homoPolySVM is ' + str(round(time.time() - start, 2)) + ' seconds.')
    res.to_csv('homoPolySVM_predictions.csv', index = False)
