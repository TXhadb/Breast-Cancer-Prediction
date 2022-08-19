import pandas as pd
import numpy as np
import sys

def sensitive(file_name):

    file = pd.read_csv(file_name)
    N = len(file)
    Y = pd.DataFrame(file.loc[:, ['Y']], columns = ['Y'])
    X = pd.DataFrame(np.ones((N, 2)), columns = ['x^1', 'x^0'])
    X['x^1'] = file['X']
    
    w = pd.DataFrame(np.random.rand(len(X.columns), 1), X.columns, ['Y'])
    a = 0.001

    for i in range(4000):
        w += 4 / N * a * X.T @ (Y - X @ w) ** 3
    
    w.to_csv('sensitive-weights.csv')

if __name__ == '__main__':
    sensitive(sys.argv[1])
