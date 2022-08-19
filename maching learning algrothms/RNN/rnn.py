import numpy as np
import pandas as pd
import tensorflow as tf

targets = pd.read_csv('MovementAAL/dataset/MovementAAL_target.csv')
targets_train = targets[targets['#sequence_ID'] % 2 == 0]
targets_train.index = range(len(targets_train))
Y_train = pd.get_dummies(targets_train[' class_label'])
N = len(Y_train)
Y0 = tf.constant(Y_train, dtype = tf.float32)

W_H = tf.Variable(tf.random.uniform([20, 25], dtype = tf.float32), trainable = True)
W_Y = tf.Variable(tf.random.uniform([len(Y_train.columns), 20], dtype = tf.float32), trainable = True)

def estimate():
    Yhat = []
    h = tf.Variable(np.zeros((20, 1)), dtype = tf.float32)
    for i in range(2, 315, 2):
        X = pd.read_csv('MovementAAL/dataset/MovementAAL_RSS_' + str(i) + '.csv')
        X['line'] = np.ones((len(X), 1))
        X0 = tf.constant(X, dtype = tf.float32)
        for j in range(len(X)):
            xT = tf.reshape(X0[j], (5, 1))
            h = tf.nn.softmax(W_H @ tf.concat([h, xT], axis = 0), axis = 0)
        Yhat.append(tf.reshape(tf.nn.softmax(W_Y @ h, axis = 0), (len(Y_train.columns),)))
    return tf.stack(Yhat)

def cost():
    return - 1 / N * tf.reduce_sum(tf.math.log(tf.reduce_sum(Y0 * estimate(), axis = 1)))

opt = tf.keras.optimizers.Adam(learning_rate = 1.0)

for epoch in range(10):
    opt.minimize(cost, var_list = [W_H, W_Y])

res = []
for i in range(1, 314, 2):
    X = pd.read_csv('MovementAAL/dataset/MovementAAL_RSS_' + str(i) + '.csv')
    X['line'] = np.ones((len(X), 1))
    X0 = tf.constant(X, dtype = tf.float32)
    h = tf.Variable(np.zeros((20, 1)), dtype = tf.float32)
    for j in range(len(X)):
        xT = tf.reshape(X0[j], (5, 1))
        h = tf.nn.softmax(W_H @ tf.concat([h, xT], axis = 0), axis = 0)
    y = tf.reshape(tf.nn.softmax(W_Y @ h, axis = 0), (len(Y_train.columns),))
    res.append(Y_train.columns[np.argmax(y)])

res = pd.DataFrame(res, columns = [' class_lable'])
res.to_csv('aal-predictions.csv', index = False)
