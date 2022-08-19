import pandas as pd
import numpy as np
import sys

def I(p):
    return -p * np.log2(p) if p > 0.0 else 0.0

def H(Y):
    return sum([I(len(Y[Y == y]) / len(Y)) for y in Y.unique()])

def split(D):
    YandXvars = D.columns.values.tolist()
    res = []
    for v in YandXvars[1:]:
        if v == 'birthyr':
            vals = sorted(list(set(D[v].to_numpy())))
            res.append((H(D[YandXvars[0]]), v, float('inf')))
            for i in range(len(vals) - 1):
                j = (vals[i] + vals[i + 1]) / 2
                res.append((len(D[D[v] < j]) / len(D) * H(D[D[v] < j][YandXvars[0]]) \
                            + len(D[D[v] > j]) / len(D) * H(D[D[v] > j][YandXvars[0]]), v, j))
        else:
            for j in D[v].unique():
                res.append((len(D[D[v] == j]) / len(D) * H(D[D[v] == j][YandXvars[0]]) \
                            + len(D[D[v] != j]) / len(D) * H(D[D[v] != j][YandXvars[0]]), v, j))
    return min(res)

def train(D, d = 0):
    if len(D) > 0 and d < 3:
        h, v, j = split(D)
        if j == float('inf'):
            y = D.columns.values.tolist()[0]
            return D[y].value_counts().idxmax()
        elif v == 'birthyr':
            return (v, j, train(D[D[v] < j], d + 1), train(D[D[v] > j], d + 1))
        return (v, j, train(D[D[v] == j], d + 1), train(D[D[v] != j], d + 1))
    y = D.columns.values.tolist()[0]
    return D[y].value_counts().idxmax()

def predict(item, tree):
    if tree[0] == 'birthyr':
        if item.at[tree[0]] < tree[1]:
            if type(tree[2]) == tuple:
                return predict(item, tree[2])
            return tree[2]
        else:
            if type(tree[3]) == tuple:
                return predict(item, tree[3])
            return tree[3]
    else:
        if item.at[tree[0]] == tree[1]:
            if type(tree[2]) == tuple:
                return predict(item, tree[2])
            return tree[2]
        else:
            if type(tree[3]) == tuple:
                return predict(item, tree[3])
            return tree[3]

def estimates(D, tree, column_name, data_type):
    res = np.empty(len(D))
    for pos in range(len(D)):
        res[pos] = predict(D.iloc[pos, :], tree)
    df = pd.DataFrame(res.reshape(-1, 1), columns = [column_name], dtype = data_type)
    df.to_csv('numerical-decision-tree-estimates.csv', index = False)

if __name__ == '__main__':
    train_file = pd.read_csv(sys.argv[1])
    tree = train(train_file)
    test_file = pd.read_csv(sys.argv[2])
    estimates(test_file, tree, train_file.columns[0], train_file.dtypes[train_file.columns[0]])




