import pandas as pd
import sys

def less_naive_bayes(train_file, test_file):

    # training data...
    print('training...')
    print('===========')
    
    training = pd.read_csv(train_file)

    Y_name, X1_name, X2_name, X3_name = list(training.columns)

    Y = dict()
    for y in training[Y_name].unique():
        Y[y] = len(training[training[Y_name] == y]) / len(training)

    X1_given_Y = dict()
    for y in training[Y_name].unique():
        for x1 in training[X1_name].unique():
            X1_given_Y[y, x1] = len(training[(training[Y_name] == y) & (training[X1_name] == x1)]) \
                                / len(training[training[Y_name] == y])
    
    X2_given_Y_and_X1 = dict()
    for y in training[Y_name].unique():
        for x1 in training[X1_name].unique():
            for x2 in training[X2_name].unique():
                X2_given_Y_and_X1[y, x1, x2] = len(training[(training[Y_name] == y) & (training[X1_name] == x1) & (training[X2_name] == x2)]) \
                                               / len(training[(training[Y_name] == y) & (training[X1_name] == x1)])
    
    X3_given_Y = dict()
    for y in training[Y_name].unique():
        for x3 in training[X3_name].unique():
            X3_given_Y[y, x3] = len(training[(training[Y_name] == y) & (training[X3_name] == x3)]) \
                                / len(training[training[Y_name] == y])


    # print full joint probabilities from training data...
    print('full joint probabilities:')
    for y in training[Y_name].unique():
        for x1 in training[X1_name].unique():
            for x2 in training[X2_name].unique():
                for x3 in training[X3_name].unique():
                    print(x1 + ', ' + x2 + ', ' + x3 + ', ' + y + ': ' \
                          + str(Y[y] * X1_given_Y[y, x1] * X2_given_Y_and_X1[y, x1, x2] * X3_given_Y[y, x3]))
    print('\n')

    # print predicting probabilities for each possible values of the hidden variable according to the three observed variables
    print('predicting...')
    print('=============')
    
    testing = pd.read_csv(test_file)

    for n, (x1, x2, x3) in testing.iterrows():
        for y in Y:
            print(x1 + ', ' + x2 + ', ' + x3 + ', ' + y + ': ' \
                  + str(Y[y] * X1_given_Y[y, x1] * X2_given_Y_and_X1[y, x1, x2] * X3_given_Y[y, x3]))
   
if __name__ == '__main__':
    less_naive_bayes(sys.argv[1], sys.argv[2])
    
