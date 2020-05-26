import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flask import jsonify, send_from_directory
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn import metrics

def build_nn(netArr):
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      for i in range(len(netArr) - 1):
          layerName = 'fullyConnected' + str(i+1)
          setattr(self, layerName, nn.Linear(int(netArr[i]['noOfNodes']), int(netArr[i+1]['noOfNodes'])))

    def forward(self, x):
      for i in range(1, len(netArr)):
        layerName = 'fullyConnected' + str(i)
        layer = getattr(self, layerName)
        x = layer(x)
        if netArr[i]['activationFunction'] == 'relu':
          x = F.relu(x)
        elif netArr[i]['activationFunction'] == 'softmax':
          x = F.log_softmax(x)
        elif netArr[i]['activationFunction'] == 'sigmoid':
          x = F.sigmoid(x)
      return x

  net = Net()
  with open('./temp/NetworkBluePrint.json', 'w') as outfile:
    json.dump(netArr, outfile)
  return net

def train_nn(
  percenatgeTraining,
  noOfEpochs,
  learningRate,
  batchSize,
  opColNo,
  shouldShuffle,
  normalizeData,
  lossFunction
  ):

  if normalizeData == 'true':
    normalizeData = True
  else:
    normalizeData = False

  if shouldShuffle == 'true':
    shouldShuffle = True
  else:
    shouldShuffle = False

  with open('./temp/NetworkBluePrint.json') as json_file:
    modelBluePrint = json.load(json_file)
  net = build_nn(modelBluePrint)
  optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=0.9)
  
  if lossFunction=='nll':
    criterion = nn.NLLLoss()
  elif lossFunction=='mse':
    criterion = nn.MSELoss()
  else:
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

  #cross entropy and nll: 1d array expected

  opColNo = opColNo - 1
  data = np.genfromtxt('./temp/inputFile.csv', delimiter=',')

  opData = data[:, opColNo]
  if opColNo == 0:
    ipData = data[:, 1:]
  elif opColNo == data.shape[1] - 1:
    ipData = data[:, 0: -1]
  else:
    leftData = data[:, 0: opColNo]
    rightData = data[:, opColNo + 1 : ]
    ipData = np.hstack((leftData, rightData))

  if normalizeData:
    maxVals = ipData.max(axis=0)
    maxVals[maxVals == 0] = 1
    ipData = ipData / maxVals

  percenatgeTraining = percenatgeTraining/100.0

  ipTraining, ipTesting, opTraining, opTesting = train_test_split(
    ipData, opData, train_size=percenatgeTraining, shuffle= shouldShuffle)

  noOfRows = ipTraining.shape[0]

  for epoch in range(noOfEpochs):
    print(epoch)
    startingIndex = 0
    endingIndex = startingIndex + batchSize
    while True:
      if startingIndex == noOfRows:
        break
      ipTrainingTensor = torch.from_numpy(ipTraining[ startingIndex:endingIndex , :]).float()
      opTrainingTensor = torch.from_numpy(np.reshape(opTraining[ startingIndex:endingIndex], (opTraining[ startingIndex:endingIndex].shape[0], 1))).float()
      # opTrainingTensor = torch.from_numpy(opTraining[ startingIndex:endingIndex]).float()
      optimizer.zero_grad()
      net_out = net(ipTrainingTensor)
      loss = criterion(net_out, opTrainingTensor)
      loss.backward()
      optimizer.step()
      print("starting index: ", startingIndex, "endningindex: ", endingIndex, "loss", loss)
      startingIndex = endingIndex
      if startingIndex + batchSize > noOfRows:
        endingIndex = noOfRows
      else:
        endingIndex = startingIndex + batchSize
  trainingLoss = str(loss.data.item())


  ipTestingTensor = torch.from_numpy(ipTesting).float()
  predOut = net(ipTestingTensor)
  testingLoss = criterion(predOut, torch.from_numpy(np.reshape(opTesting, (opTesting.shape[0], 1))).float())
  testingLoss = str(testingLoss.data.item() / opTesting.shape[0] * batchSize)
  predOut = predOut.detach().numpy()
  print(predOut, opTesting)
  predOut[predOut < 0.5 ] = 0 #required in binary classification
  predOut[predOut >= 0.5 ] = 1
  # predOut = predOut.argmax(axis=1)
  confusionMatrix = metrics.confusion_matrix(opTesting, predOut)
  print("confusion matrix: ", confusionMatrix)

  accuracy = ''
  precision = '' 
  recall = ''
  f1 = ''
  correct = np.trace(confusionMatrix)

  if confusionMatrix.shape == (2, 2):
    tn, fp, fn, tp = confusionMatrix.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ( (precision * recall) / (precision + recall) )
    accuracy = (tp + tn) / (tp + tn + fp + fn)
  else:
    accuracy = correct / opTesting.shape[0] * 100

  cm_list = []
  for i in range(confusionMatrix.shape[0]):
    cm_list.append([int(x) for x in confusionMatrix[i]])
  return jsonify({
    'status': 'success',
    'confusion_matrix': cm_list,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'correct': str(correct),
    'total': str(opTesting.shape[0]),
    'trainingLoss': trainingLoss,
    'testingLoss': testingLoss
    })