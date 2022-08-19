import numpy as np
import pandas as pd
import sys

def sweet(train_file_name, test_file_name):
    
    train_file = pd.read_csv(train_file_name)
    y = pd.DataFrame(train_file.loc[:, train_file.columns[0]], columns = [train_file.columns[0]])
    X = pd.DataFrame(np.ones((len(train_file), 3)), columns = ['x^2', 'x^1', 'x^0'])
    X['x^1'] = train_file.loc[:, train_file.columns[1]]
    X['x^2'] = X['x^1'] ** 2
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    test_file = pd.read_csv(test_file_name)
    X = pd.DataFrame(np.ones((len(test_file), 3)), columns = ['x^2', 'x^1', 'x^0'])
    X['x^1'] = test_file.loc[:, test_file.columns[0]]
    X['x^2'] = X['x^1'] ** 2
    y = X @ w.to_numpy()
    y.columns = ['Y']
    y.to_csv('sweet-predictions.csv', index = False)

if __name__ == '__main__':
    sweet(sys.argv[1], sys.argv[2])
