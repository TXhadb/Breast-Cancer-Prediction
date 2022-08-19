import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataSplit
import logisticReg
import ADAM
import homoPolySVM
import lognnADAM
import lognnSGD
import score
import sys

folds = int(sys.argv[1])

data_file = pd.read_csv('train.csv')

logisticReg_iterations = [200, 400, 600, 800, 1000, 2000, 3000, 4000]
logisticReg_learning_rates = [0.5, 0.1, 0.075, 0.05, 0.01]

ADAM_iterations = [100 ,200, 400, 600, 800, 1000, 2000, 3000]
ADAM_learning_rates = [0.5, 0.1, 0.075, 0.05, 0.01]

homoPolySVM_degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]

lognn_hidden_variables = [4, 5, 8, 9, 12, 13]
lognn_num_epochs = [200, 400, 600, 800, 1000, 2000, 4000, 6000, 7000, 8000]

logisticReg_valAccuracy = {}
logisticReg_valSensitivity = {}
ADAM_valAccuracy = {}
ADAM_valSensitivity = {}
homoPolySVM_valAccuracy = {}
homoPolySVM_valSensitivity = {}
lognnADAM_valAccuracy = {}
lognnADAM_valSensitivity = {}
lognnSGD_valAccuracy = {}
lognnSGD_valSensitivity = {}

for fold in range(folds):

    print('fold ' + str(fold + 1) + ' starts')

    train, validation_observables, validation_responses = dataSplit.split(data_file)

    print('logisticReg in progress...')
    for learning_rate in logisticReg_learning_rates:
        for iteration in logisticReg_iterations:
            train_data, test_observables, test_responses = train.copy(), validation_observables.copy(), validation_responses.copy()
            valPredictions = logisticReg.fit_predict(train_data, test_observables, iteration, learning_rate)
            valAccuracy, valSensitivity = score.score(test_responses, valPredictions)
            if (iteration, learning_rate) in logisticReg_valAccuracy:
                logisticReg_valAccuracy[(iteration, learning_rate)] += valAccuracy
            else:
                logisticReg_valAccuracy[(iteration, learning_rate)] = valAccuracy
            if (iteration, learning_rate) in logisticReg_valSensitivity:
                logisticReg_valSensitivity[(iteration, learning_rate)] += valSensitivity
            else:
                logisticReg_valSensitivity[(iteration, learning_rate)] = valSensitivity

    print('ADAM in progress...')
    for learning_rate in ADAM_learning_rates:
        for iteration in ADAM_iterations:
            train_data, test_observables, test_responses = train.copy(), validation_observables.copy(), validation_responses.copy()
            valPredictions = ADAM.fit_predict(train_data, test_observables, iteration, learning_rate)
            valAccuracy, valSensitivity = score.score(test_responses, valPredictions)
            if (iteration, learning_rate) in ADAM_valAccuracy:
                ADAM_valAccuracy[(iteration, learning_rate)] += valAccuracy
            else:
                ADAM_valAccuracy[(iteration, learning_rate)] = valAccuracy
            if (iteration, learning_rate) in ADAM_valSensitivity:
                ADAM_valSensitivity[(iteration, learning_rate)] += valSensitivity
            else:
                ADAM_valSensitivity[(iteration, learning_rate)] = valSensitivity

    print('homoPolySVM in progress...')
    for degree in homoPolySVM_degrees:
        train_data, test_observables, test_responses = train.copy(), validation_observables.copy(), validation_responses.copy()
        valPredictions = homoPolySVM.fit_predict(train_data, test_observables, degree)
        valAccuracy, valSensitivity = score.score(test_responses, valPredictions)
        if degree in homoPolySVM_valAccuracy:
            homoPolySVM_valAccuracy[degree] += valAccuracy
        else:
            homoPolySVM_valAccuracy[degree] = valAccuracy
        if degree in homoPolySVM_valSensitivity:
            homoPolySVM_valSensitivity[degree] += valSensitivity
        else:
            homoPolySVM_valSensitivity[degree] = valSensitivity

    print('lognnADAM in progress...')
    for hidden_variable in lognn_hidden_variables:
        for epoch in lognn_num_epochs:
            train_data, test_observables, test_responses = train.copy(), validation_observables.copy(), validation_responses.copy()
            valPredictions = lognnADAM.fit_predict(train_data, test_observables, hidden_variable, epoch)
            valAccuracy, valSensitivity = score.score(test_responses, valPredictions)
            if (epoch, hidden_variable) in lognnADAM_valAccuracy:
                lognnADAM_valAccuracy[(epoch, hidden_variable)] += valAccuracy
            else:
                lognnADAM_valAccuracy[(epoch, hidden_variable)] = valAccuracy
            if (epoch, hidden_variable) in lognnADAM_valSensitivity:
                lognnADAM_valSensitivity[(epoch, hidden_variable)] += valSensitivity
            else:
                lognnADAM_valSensitivity[(epoch, hidden_variable)] = valSensitivity

    print('lognnSGD in progress...')
    for hidden_variable in lognn_hidden_variables:
        for epoch in lognn_num_epochs:
            train_data, test_observables, test_responses = train.copy(), validation_observables.copy(), validation_responses.copy()
            valPredictions = lognnSGD.fit_predict(train_data, test_observables, hidden_variable, epoch)
            valAccuracy, valSensitivity = score.score(test_responses, valPredictions)
            if (epoch, hidden_variable) in lognnSGD_valAccuracy:
                lognnSGD_valAccuracy[(epoch, hidden_variable)] += valAccuracy
            else:
                lognnSGD_valAccuracy[(epoch, hidden_variable)] = valAccuracy
            if (epoch, hidden_variable) in lognnSGD_valSensitivity:
                lognnSGD_valSensitivity[(epoch, hidden_variable)] += valSensitivity
            else:
                lognnSGD_valSensitivity[(epoch, hidden_variable)] = valSensitivity

    print('fold ' + str(fold + 1) + ' ends\n')

colors = ['b', 'g', 'r', 'c', 'y', 'k']
shapes = ['o', '^', 's', 'P', 'X', 'D']

logisticReg_avgValAccuracy = pd.DataFrame(np.zeros((len(logisticReg_iterations), len(logisticReg_learning_rates))), index = logisticReg_iterations, columns = logisticReg_learning_rates)
for iteration, learning_rate in logisticReg_valAccuracy:
    logisticReg_avgValAccuracy.at[iteration, learning_rate] = round(logisticReg_valAccuracy[(iteration, learning_rate)] / folds, 4)
logisticReg_avgValAccuracy.to_csv('logisticReg_avgValAccuracy.csv')

plt.figure()
for i in range(len(logisticReg_learning_rates)):
    plt.plot(logisticReg_iterations, logisticReg_avgValAccuracy[logisticReg_learning_rates[i]],
             colors[i] + shapes[i] + ':', label = 'learning rate=' + str(logisticReg_learning_rates[i]))
plt.xscale('log')
plt.xlabel('iterations')
plt.ylabel('average validation accuracy')
plt.legend()
plt.savefig('logisticReg_avgValAccuracy.png')

logisticReg_avgValSensitivity = pd.DataFrame(np.zeros((len(logisticReg_iterations), len(logisticReg_learning_rates))), index = logisticReg_iterations, columns = logisticReg_learning_rates)
for iteration, learning_rate in logisticReg_valSensitivity:
    logisticReg_avgValSensitivity.at[iteration, learning_rate] = round(logisticReg_valSensitivity[(iteration, learning_rate)] / folds, 4)
logisticReg_avgValSensitivity.to_csv('logisticReg_avgValSensitivity.csv')

plt.figure()
for i in range(len(logisticReg_learning_rates)):
    plt.plot(logisticReg_iterations, logisticReg_avgValSensitivity[logisticReg_learning_rates[i]],
             colors[i] + shapes[i] + ':', label = 'learning rate=' + str(logisticReg_learning_rates[i]))
plt.xscale('log')
plt.xlabel('iterations')
plt.ylabel('average validation sensitivity')
plt.legend()
plt.savefig('logisticReg_avgValSensitivity.png')

ADAM_avgValAccuracy = pd.DataFrame(np.zeros((len(ADAM_iterations), len(ADAM_learning_rates))), index = ADAM_iterations, columns = ADAM_learning_rates)
for iteration, learning_rate in ADAM_valAccuracy:
    ADAM_avgValAccuracy.at[iteration, learning_rate] = round(ADAM_valAccuracy[(iteration, learning_rate)] / folds, 4)
ADAM_avgValAccuracy.to_csv('ADAM_avgValAccuracy.csv')

plt.figure()
for i in range(len(ADAM_learning_rates)):
    plt.plot(ADAM_iterations, ADAM_avgValAccuracy[ADAM_learning_rates[i]],
             colors[i] + shapes[i] + ':', label = 'learning rate=' + str(ADAM_learning_rates[i]))
plt.xscale('log')
plt.xlabel('iterations')
plt.ylabel('average validation accuracy')
plt.legend()
plt.savefig('ADAM_avgValAccuracy.png')

ADAM_avgValSensitivity = pd.DataFrame(np.zeros((len(ADAM_iterations), len(ADAM_learning_rates))), index = ADAM_iterations, columns = ADAM_learning_rates)
for iteration, learning_rate in ADAM_valSensitivity:
    ADAM_avgValSensitivity.at[iteration, learning_rate] = round(ADAM_valSensitivity[(iteration, learning_rate)] / folds, 4)
ADAM_avgValSensitivity.to_csv('ADAM_avgValSensitivity.csv')

plt.figure()
for i in range(len(ADAM_learning_rates)):
    plt.plot(ADAM_iterations, ADAM_avgValSensitivity[ADAM_learning_rates[i]],
             colors[i] + shapes[i] + ':', label = 'learning rate=' + str(ADAM_learning_rates[i]))
plt.xscale('log')
plt.xlabel('iterations')
plt.ylabel('average validation sensitivity')
plt.legend()
plt.savefig('ADAM_avgValSensitivity.png')

homoPolySVM_avgValAccuracy = pd.DataFrame(np.zeros((len(homoPolySVM_degrees), 1)), index = homoPolySVM_degrees, columns = ['avgValAccuracy'])
for degree in homoPolySVM_valAccuracy:
    homoPolySVM_avgValAccuracy.at[degree, 'avgValAccuracy'] = round(homoPolySVM_valAccuracy[degree] / folds, 4)
homoPolySVM_avgValAccuracy.to_csv('homoPolySVM_avgValAccuracy.csv')

plt.figure()
plt.plot(homoPolySVM_degrees, homoPolySVM_avgValAccuracy['avgValAccuracy'], colors[0] + shapes[0] + ':')
plt.xlabel('degree')
plt.ylabel('average validation accuracy')
plt.savefig('homoPolySVM_avgValAccuracy.png')

homoPolySVM_avgValSensitivity = pd.DataFrame(np.zeros((len(homoPolySVM_degrees), 1)), index = homoPolySVM_degrees, columns = ['avgValSensitivity'])
for degree in homoPolySVM_valSensitivity:
    homoPolySVM_avgValSensitivity.at[degree, 'avgValSensitivity'] = round(homoPolySVM_valSensitivity[degree] / folds, 4)
homoPolySVM_avgValSensitivity.to_csv('homoPolySVM_avgValSensitivity.csv')

plt.figure()
plt.plot(homoPolySVM_degrees, homoPolySVM_avgValSensitivity['avgValSensitivity'], colors[0] + shapes[0] + ':')
plt.xlabel('degree')
plt.ylabel('average validation sensitivity')
plt.savefig('homoPolySVM_avgValSensitivity.png')

lognnADAM_avgValAccuracy = pd.DataFrame(np.zeros((len(lognn_num_epochs), len(lognn_hidden_variables))), index = lognn_num_epochs, columns = lognn_hidden_variables)
for epoch, hidden_variable in lognnADAM_valAccuracy:
    lognnADAM_avgValAccuracy.at[epoch, hidden_variable] = round(lognnADAM_valAccuracy[(epoch, hidden_variable)] / folds, 4)
lognnADAM_avgValAccuracy.to_csv('lognnADAM_avgValAccuracy.csv')

plt.figure()
for i in range(len(lognn_hidden_variables)):
    plt.plot(lognn_num_epochs, lognnADAM_avgValAccuracy[lognn_hidden_variables[i]],
             colors[i] + shapes[i] + ':', label = 'hidden units=' + str(lognn_hidden_variables[i]))
plt.xscale('log')
plt.xlabel('epochs')
plt.ylabel('average validation accuracy')
plt.legend()
plt.savefig('lognnADAM_avgValAccuracy.png')

lognnADAM_avgValSensitivity = pd.DataFrame(np.zeros((len(lognn_num_epochs), len(lognn_hidden_variables))), index = lognn_num_epochs, columns = lognn_hidden_variables)
for epoch, hidden_variable in lognnADAM_valSensitivity:
    lognnADAM_avgValSensitivity.at[epoch, hidden_variable] = round(lognnADAM_valSensitivity[(epoch, hidden_variable)] / folds, 4)
lognnADAM_avgValSensitivity.to_csv('lognnADAM_avgValSensitivity.csv')

plt.figure()
for i in range(len(lognn_hidden_variables)):
    plt.plot(lognn_num_epochs, lognnADAM_avgValSensitivity[lognn_hidden_variables[i]],
             colors[i] + shapes[i] + ':', label = 'hidden units=' + str(lognn_hidden_variables[i]))
plt.xscale('log')
plt.xlabel('epochs')
plt.ylabel('average validation sensitivity')
plt.legend()
plt.savefig('lognnADAM_avgValSensitivity.png')

lognnSGD_avgValAccuracy = pd.DataFrame(np.zeros((len(lognn_num_epochs), len(lognn_hidden_variables))), index = lognn_num_epochs, columns = lognn_hidden_variables)
for epoch, hidden_variable in lognnSGD_valAccuracy:
    lognnSGD_avgValAccuracy.at[epoch, hidden_variable] = round(lognnSGD_valAccuracy[(epoch, hidden_variable)] / folds, 4)
lognnSGD_avgValAccuracy.to_csv('lognnSGD_avgValAccuracy.csv')

plt.figure()
for i in range(len(lognn_hidden_variables)):
    plt.plot(lognn_num_epochs, lognnSGD_avgValAccuracy[lognn_hidden_variables[i]],
             colors[i] + shapes[i] + ':', label = 'hidden units=' + str(lognn_hidden_variables[i]))
plt.xscale('log')
plt.xlabel('epochs')
plt.ylabel('average validation accuracy')
plt.legend()
plt.savefig('lognnSGD_avgValAccuracy.png')

lognnSGD_avgValSensitivity = pd.DataFrame(np.zeros((len(lognn_num_epochs), len(lognn_hidden_variables))), index = lognn_num_epochs, columns = lognn_hidden_variables)
for epoch, hidden_variable in lognnSGD_valSensitivity:
    lognnSGD_avgValSensitivity.at[epoch, hidden_variable] = round(lognnSGD_valSensitivity[(epoch, hidden_variable)] / folds, 4)
lognnSGD_avgValSensitivity.to_csv('lognnSGD_avgValSensitivity.csv')

plt.figure()
for i in range(len(lognn_hidden_variables)):
    plt.plot(lognn_num_epochs, lognnSGD_avgValSensitivity[lognn_hidden_variables[i]],
             colors[i] + shapes[i] + ':', label = 'hidden units=' + str(lognn_hidden_variables[i]))
plt.xscale('log')
plt.xlabel('epochs')
plt.ylabel('average validation sensitivity')
plt.legend()
plt.savefig('lognnSGD_avgValSensitivity.png')
        
