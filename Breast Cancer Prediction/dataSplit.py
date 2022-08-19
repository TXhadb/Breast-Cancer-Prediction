import pandas as pd
import sys

def split(data_file):

    df = data_file

    test_positive = df[df[df.columns[0]] == 1].sample(frac = 0.2, replace = False)
    test_negative = df[df[df.columns[0]] == -1].sample(frac = 0.2, replace = False)
    test = pd.concat([test_positive, test_negative])
    
    train = df.drop(test.index)

    train = train.sample(frac = 1, replace = False)
    train.index = range(len(train))

    test = test.sample(frac = 1, replace = False)
    test.index = range(len(test))
    test_responses = test[[test.columns[0]]]
    test_observables = test[test.columns[1:]]

    return train, test_observables, test_responses

if __name__ == '__main__':
    data_file = pd.read_csv(sys.argv[1])
    train, test_observables, test_responses = split(data_file)
    train.to_csv('train.csv', index = False)
    test_observables.to_csv('test_observables.csv', index = False)
    test_responses.to_csv('test_responses.csv', index = False)

