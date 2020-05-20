import os
import numpy as np
from flask import jsonify, request
from sklearn import decomposition

UPLOAD_DIRECTORY = "./temp/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def upload_input_file(filename):
    for filename in os.listdir('./temp'):
        file_path = os.path.join('./temp', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    file = request.files['file']
    file.save(os.path.join(UPLOAD_DIRECTORY, 'inputFile.csv'))
    s = os.popen('head -10 temp/inputFile.csv').read()
    content = s.split('\n')[:-1]
    content = [x.split(',') for x in content]
    return jsonify({'status': 'success', 'content': content})

def get_be_file(fileName):
    filePath = 'head -10 temp/' + fileName
    s = os.popen(filePath).read()
    content = s.split('\n')[:-1]
    content = [x.split(',') for x in content]
    for i in range(len(content)):
        for j in range(len(content[i])):
            content[i][j] = round(float(content[i][j]), 3)

    return jsonify({'status': 'success', 'content': content})

def apply_pca(opColNo, keepFeatures):
    if opColNo=='' or keepFeatures=='':
        return get_be_file('inputFile.csv')

    data = np.genfromtxt('./temp/inputFile.csv', delimiter=',')
    n_components = int(keepFeatures)
    opColNo = int(opColNo) - 1
    if n_components >= data.shape[1] - 1:
        return get_be_file('inputFile.csv')
    data = np.matrix(data)

    op = data[:, opColNo]
    if opColNo == 0:
        ipData = data[:, 1:]
    elif opColNo == data.shape[1] - 1:
        ipData = data[:, 0: -1]
    else:
        leftData = data[:, 0: opColNo]
        rightData = data[:, opColNo + 1 : ]
        ipData = np.hstack((leftData, rightData))

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(ipData)
    reducedData = pca.transform(ipData)
    reducedDataWithOp = np.hstack((op, reducedData))

    np.savetxt(UPLOAD_DIRECTORY + "reducedPCA.csv", reducedDataWithOp, delimiter=",")
     
    return get_be_file('reducedPCA.csv')
