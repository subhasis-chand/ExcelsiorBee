import os
import io 
from flask_cors import CORS
from flask import Flask, request, abort, jsonify, send_from_directory, url_for, redirect, render_template
import json

from src.FileUploads import upload_input_file, get_be_file, apply_pca
from src.EditFile import edit_file
from src.LinearRegression import linear_regression
from src.LogisticRegression import logistic_regression
from src.NeuralNet import build_nn, train_nn

UPLOAD_DIRECTORY = "./temp/"

api = Flask(__name__)
CORS(api)

@api.route('/')
def index():
   return render_template('Home.html')

@api.route('/download/<filename>', methods=["GET", "POST"])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIRECTORY, filename, as_attachment=True)

@api.route("/upload_input_file/<filename>", methods=["POST"])
def UploadInputFile(filename):
    return upload_input_file(filename)

@api.route("/edit_file/", methods=["POST"])
def EditFile():
    deleteCols = request.args.get('deleteCols')
    deleteRows = request.args.get('deleteRows')
    removeHeader = request.args.get('removeHeader')
    removeNaN = request.args.get('removeNaN')
    saveInBE = request.args.get('saveInBE')
    return edit_file(deleteCols, deleteRows, removeHeader, removeNaN, saveInBE)

@api.route("/get_be_file/", methods=["POST"])
def GetBEFile():
    ip = request.remote_addr
    return get_be_file('inputFile.csv')

@api.route("/linear_regression/", methods=["POST"])
def LinearRegression():
    percenatgeTraining = request.args.get('percenatgeTraining')
    shouldShuffle = request.args.get('shouldShuffle')
    opColNo = request.args.get('opColNo')
    normalizeData = request.args.get('normalizeData')
    return linear_regression(percenatgeTraining, shouldShuffle, opColNo, normalizeData)

@api.route("/logistic_regression/", methods=["POST"])
def LogisticRegression():
    percenatgeTraining = request.args.get('percenatgeTraining')
    shouldShuffle = request.args.get('shouldShuffle')
    opColNo = request.args.get('opColNo')
    normalizeData = request.args.get('normalizeData')
    return logistic_regression(percenatgeTraining, shouldShuffle, opColNo, normalizeData)

@api.route("/apply_pca/", methods=["POST"])
def ApplyPCA():
    opColNo = request.args.get('opColNo')
    keepFeatures = request.args.get('keepFeatures')
    return apply_pca(opColNo, keepFeatures)

@api.route("/build_nn/", methods=["POST"])
def BuildNN():
    newArr = request.args.get('newArr')
    networkArray = json.loads(newArr)
    net = build_nn(networkArray)
    f = io.StringIO()
    print(net, file=f)
    net_str = f.getvalue()
    f.close()
    return jsonify({'neuralNet': net_str })

@api.route("/train_nn/", methods=["POST"])
def TrainNN():
    percenatgeTraining = int(request.args.get('percenatgeTraining'))
    noOfEpochs = int(request.args.get('noOfEpochs'))
    learningRate = float(request.args.get('learningRate'))
    batchSize = int(request.args.get('batchSize'))
    opColNo = int(request.args.get('opColNo'))
    shouldShuffle = request.args.get('shouldShuffle')
    return train_nn(percenatgeTraining, noOfEpochs, learningRate, batchSize, opColNo, shouldShuffle)

    

if __name__ == "__main__":
    api.run(debug=True, port=5000)