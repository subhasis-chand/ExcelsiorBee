from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from flask import jsonify, send_from_directory
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(percenatgeTraining, shouldShuffle, opColNo, normalizeData):
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
    
    if normalizeData:
        maxVals = data.max(axis=0)
        data = data / maxVals

    tempCol = np.copy(data[:, -1])
    data[:, -1] = np.copy(data[:, opColNo])
    data[:, opColNo] = np.copy(tempCol)
    
    ipData = data[:, :-1]
    opData = data[:, -1]
    
    x_train, x_test, y_train, y_test = train_test_split(
        ipData,
        opData,
        test_size=(100 - percenatgeTraining)/100.0,
        shuffle=shouldShuffle
        )
    
    reg = LinearRegression(fit_intercept=True).fit(x_train, y_train)

    
    weight = reg.coef_
    np.savetxt('./temp/weights.csv', weight, fmt='%f')

    y_pred = reg.predict(x_test)

    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)    
    
    perErr = 100.0 * (np.abs(y_pred - y_test) / y_test)

    y_pred_list = []
    y_test_list = []
    perErrList = []
    for i in range(y_test.shape[0]):
        y_pred_list.append(y_pred[i] * maxVals[-1])
        y_test_list.append(y_test[i] * maxVals[-1])
        perErrList.append(perErr[i])

    weight_list = []
    for w in weight:
        weight_list.append(w)
    

    return jsonify({
        'status': 'success',
        'weights': weight_list,
        'rmse': rmse,
        'r2': r2,
        'ytest': y_test_list,
        'ypred': y_pred_list,
        'perErr': perErrList,
        'maxPerErr': perErr.max(),
        'minPerErr': perErr.min(),
        'avgPerErr': np.average(perErr)
        })
