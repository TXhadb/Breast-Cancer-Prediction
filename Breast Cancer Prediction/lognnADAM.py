import pandas as pd
import numpy as np
import tensorflow as tf
import time
import sys

def fit_predict(train_data, test_observables, hidden_variables, num_epochs):

    YX = train_data

    Y = pd.get_dummies(YX[YX.columns[0]])
    X = YX[YX.columns[1:]]
    N = len(YX)
    X['line'] = np.ones((N, 1))

    J = len(Y.columns)
    K = hidden_variables
    V = len(X.columns)
    L = 2

    X0 = tf.constant(X.T, dtype = tf.float32)
    Y0 = tf.constant(Y.T, dtype = tf.float32)

    W1 = tf.Variable(tf.random.uniform([K, V], dtype = tf.float32), trainable = True)
    W2 = tf.Variable(tf.random.uniform([J, K], dtype = tf.float32), trainable = True)

    def cost():
        H1 = tf.nn.softmax(W1 @ X0, axis = 0)
        Yhat = tf.nn.softmax(W2 @ H1, axis = 0)
        return -1 / N * tf.reduce_sum(tf.math.log(tf.reduce_sum(Y0 * Yhat, axis = 0)))

    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
    epochs = num_epochs
    for epoch in range(1, epochs + 1):
        opt.minimize(cost, var_list = [W1, W2])

    X = test_observables
    X['line'] = np.ones((len(X), 1))

    X0 = tf.constant(X.T, dtype = tf.float32)

    Y0 = tf.nn.softmax(W2 @ tf.nn.softmax(W1 @ X0, axis = 0), axis = 0)
    Y0 = tf.transpose(Y0)

    res = np.empty(len(X))

    for n in range(len(X)):
        res[n] = Y.columns[np.argmax(Y0[n])]

    res = pd.DataFrame(res.reshape(-1, 1), columns = [YX.columns[0]])

    return res

if __name__ == '__main__':
    train_data = pd.read_csv(sys.argv[1])
    test_observables = pd.read_csv(sys.argv[2])
    hidden_variables = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    start = time.time()
    res = fit_predict(train_data, test_observables, hidden_variables, num_epochs)
    print('The time cost for lognnADAM is ' + str(round(time.time() - start, 2)) + ' seconds.')
    res.to_csv('lognnADAM_predictions.csv', index = False)
