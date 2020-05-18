import os
from flask import jsonify, request
import numpy as np

UPLOAD_DIRECTORY = "./temp/"

def edit_file(deleteCols, deleteRows, removeHeader, removeNaN, saveInBE):
    inputFile = np.genfromtxt(UPLOAD_DIRECTORY + 'inputFile.csv', delimiter=',')

    if removeHeader:
        inputFile = inputFile[1:]
    if removeNaN:
        inputFile = inputFile[~np.isnan(inputFile).any(axis=1)]
    if len(deleteCols) > 0:
        deleteCols = deleteCols.replace(' ', '').strip(',').strip('').split(',')
        delList = []
        for i in range(len(deleteCols)):
            if int(deleteCols[i]) <= inputFile.shape[1] and int(deleteCols[i]) > 0:
                delList.append(int(deleteCols[i])-1)
        inputFile = np.delete(inputFile, delList, 1)
    if len(deleteRows) > 0:
        deleteRows = deleteRows.replace(' ', '').strip(',').split(',')
        delList = []
        for i in range(len(deleteRows)):
            if int(deleteRows[i]) <= inputFile.shape[0] and int(deleteRows[i]) > 0:
                delList.append(int(deleteRows[i])-1)
        inputFile = np.delete(inputFile, delList, 0)

    np.savetxt(UPLOAD_DIRECTORY + "inputFile.csv", inputFile, delimiter=",")

    returnList = []
    for i in range(10):
        returnList.append([x for x in inputFile[i]])
    
    return jsonify({'status': 'success', 'content': returnList})

