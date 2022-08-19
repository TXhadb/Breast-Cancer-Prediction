import numpy as np
import pandas as pd
import sys

def longitopic(train_file_name):

    pd.set_option('display.max_columns', None)

    X = pd.read_csv(train_file_name)
    N = len(X)
    V = np.unique(X)
    K = 2

    M_E = pd.DataFrame(np.random.dirichlet(np.ones(K))).T
    M_L = pd.DataFrame(np.random.dirichlet(np.ones(K), K))
    M_X = pd.DataFrame(np.random.dirichlet(np.ones(len(V)), K), columns = V)

    xT = {}
    for n in range(N):
        for w in X:
            xT[n, w] = pd.DataFrame(np.zeros((1, len(V))), columns = V)
            xT[n, w][X[w][n]] += 1

    for i in range(10):
        
        b_XL = [M_X @ xT[n, X.columns[1]].T for n in range(N)]
        b_LE = [M_L @ b_XL[n] for n in range(N)]
        b_XE = [M_X @ xT[n, X.columns[0]].T for n in range(N)]
        
        eT, lT = {}, {}
        for n in range(N):
            e_distrib = M_E * b_LE[n].T * b_XE[n].T
            e_distrib /= e_distrib @ np.ones((K, K))
            eT[n] = np.random.multinomial(1, e_distrib.values.flatten()).reshape((1, K))
            l_distrib = eT[n] @ M_L * b_XL[n].T
            l_distrib /= l_distrib @ np.ones((K, K))
            lT[n] = np.random.multinomial(1, l_distrib.values.flatten()).reshape((1, K))
        
        hparams = 1 + np.add.reduce([eT[n] for n in range(N)])
        M_E = pd.DataFrame(np.random.dirichlet(hparams.flatten()).reshape(1, K), columns = range(K))
        hparams = 1 + np.add.reduce([eT[n].T @ lT[n] for n in range(N)])
        M_L = pd.DataFrame(np.stack([np.random.dirichlet(hparams[k]) for k in range(K)]), columns = range(K))
        hparams = 1 + np.add.reduce([eT[n].T @ xT[n, X.columns[0]] + lT[n].T @ xT[n, X.columns[1]] for n in range(N)])
        M_X = pd.DataFrame(np.stack([np.random.dirichlet(hparams[k]) for k in range(K)]), columns = V)
    
    print('distribution for E:')
    print(M_E)
    print('\n')

    print('condition matrix for L:')
    print(M_L)
    print('\n')

    print('condition matrix for X:')
    print(M_X)
    print('\n')

    print('distribution for L:')
    print(M_E @ M_L)
    print('\n')

    print('distribution for X at E time:')
    print(M_E @ M_X)
    print('\n')

    print('distribution for X at L time:')
    print(M_E @ M_L @ M_X)
    print('\n')
        

if __name__ == '__main__':
    longitopic(sys.argv[1])
