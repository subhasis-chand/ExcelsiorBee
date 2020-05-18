from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from flask import jsonify, send_from_directory
from sklearn.metrics import mean_squared_error, r2_score

def logistic_regression(percenatgeTraining, shouldShuffle, opColNo, normalizeData):
    percenatgeTraining = int(percenatgeTraining)
    if shouldShuffle == 'true':
        shouldShuffle = True
    else:
        shouldShuffle = False
    opColNo = int(opColNo)-1
    if normalizeData == 'true':
        normalizeData = True
    else:
        normalizeData = False

    data = np.genfromtxt('./temp/inputFile.csv', delimiter=',')

    tempCol = np.copy(data[:, -1])
    data[:, -1] = np.copy(data[:, opColNo])
    data[:, opColNo] = np.copy(tempCol)
    
    ipData = data[:, :-1]
    opData = data[:, -1]

    if normalizeData:
        maxVals = ipData.max(axis=0)
        maxVals[maxVals == 0] = 1
        ipData = ipData / maxVals
    
    x_train, x_test, y_train, y_test = train_test_split(
        ipData,
        opData,
        test_size=(100 - percenatgeTraining)/100.0,
        shuffle=shouldShuffle
        )
    
    reg = LogisticRegression(fit_intercept=True).fit(x_train, y_train)

    
    weight = reg.coef_
    np.savetxt('./temp/weights.csv', weight, fmt='%f')

    y_pred = reg.predict(x_test)
    cm = metrics.confusion_matrix(y_test, y_pred)

    cm_list = []
    for i in range(cm.shape[0]):
        cm_list.append([int(x) for x in cm[i]])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

    accuracy = ''
    precision = '' 
    recall = ''
    f1 = ''

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ( (precision * recall) / (precision + recall) )
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    return jsonify({
        'status': 'success',
        'confusion_matrix': cm_list,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
        })
