import pandas as pd
import sys

def score(true_responses, estimated_responses):
    
    truth = true_responses
    estimate = estimated_responses
    
    scores = pd.DataFrame({'score': truth[truth.columns[0]] == estimate[estimate.columns[0]]}, dtype = "int")

    accuracy = round(len(scores[scores['score'] == 1]) / len(scores), 4)

    positive_index = truth[truth[truth.columns[0]] == 1].index
    positive_scores = scores.iloc[positive_index]
    sensitivity = round(len(positive_scores[positive_scores['score'] == 1]) / len(positive_scores), 4)

    return accuracy, sensitivity

if __name__ == '__main__':
    true_responses = pd.read_csv(sys.argv[1])
    estimated_responses = pd.read_csv(sys.argv[2])
    accuracy, sensitivity = score(true_responses, estimated_responses)
    print('The accuracy is: ' + str(accuracy))
    print('The sensitivity is: ' + str(sensitivity))
