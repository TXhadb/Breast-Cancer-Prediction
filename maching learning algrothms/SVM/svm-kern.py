import numpy as np
import pandas as pd
import sys
import cvxopt
cvxopt.solvers.options['show_progress'] = False

def svm_kern(train_file_name, test_file_name):

    YX = pd.read_csv(train_file_name)
    N = len(YX)

    y = YX[YX.columns[0]].to_frame()
    X = YX[YX.columns[1:]] / 10
    
    H = (np.diagflat(y.values) @ (X @ X.T + 1) ** 3 @ np.diagflat(y.values)).values
    
    a = np.array(cvxopt.solvers.qp(cvxopt.matrix(H, tc = 'd'), \
                                   cvxopt.matrix(-np.ones((N, 1))), \
                                   cvxopt.matrix(-np.eye(N)), \
                                   cvxopt.matrix(np.zeros(N)), \
                                   cvxopt.matrix(y.T.values, tc = 'd'), \
                                   cvxopt.matrix(np.zeros(1)))['x'])
    
    n = np.argmax(a * y)
    b = (y.T)[n] - ((a * y).T @ (X @ (X.T)[n] + 1) ** 3)
    
    X_test = pd.read_csv(test_file_name) / 10
    
    Y = np.sign((X_test @ X.T + 1) ** 3 @ (a * y) + b)
    
    Y.to_csv('svm-kern-predictions.csv', index = False)

if __name__ == '__main__':
    svm_kern(sys.argv[1], sys.argv[2])
