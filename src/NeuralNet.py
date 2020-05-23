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

def train_nn(percenatgeTraining, noOfEpochs, learningRate, batchSize, opColNo, shouldShuffle):
  with open('./temp/NetworkBluePrint.json') as json_file:
    modelBluePrint = json.load(json_file)
  net = build_nn(modelBluePrint)
  optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=0.9)
  # criterion = nn.CrossEntropyLoss()  
  criterion = nn.NLLLoss()

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

  maxVals = ipData.max(axis=0)
  maxVals[maxVals == 0] = 1
  ipData = ipData / maxVals

  if shouldShuffle == 'true':
    shouldShuffle = True
  else:
    shouldShuffle = False
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
      opTrainingTensor = torch.from_numpy(opTraining[ startingIndex:endingIndex]).float()
      optimizer.zero_grad()
      net_out = net(ipTrainingTensor)
      loss = criterion(net_out, opTrainingTensor.long())
      loss.backward()
      optimizer.step()
      print("starting index: ", startingIndex, "endningindex: ", endingIndex, "loss", loss)
      startingIndex = endingIndex
      if startingIndex + batchSize > noOfRows:
        endingIndex = noOfRows
      else:
        endingIndex = startingIndex + batchSize


  ipTestingTensor = torch.from_numpy(ipTesting).float()
  predOut = net(ipTestingTensor)
  predOut = predOut.detach().numpy()
  predOut = predOut.argmax(axis=1)
  confusionMatrix = metrics.confusion_matrix(opTesting, predOut)

  accuracy = ''
  precision = '' 
  recall = ''
  f1 = ''

  if confusionMatrix.shape == (2, 2):
    tn, fp, fn, tp = confusionMatrix.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ( (precision * recall) / (precision + recall) )
    accuracy = (tp + tn) / (tp + tn + fp + fn)
  else:
    correct = np.trace(confusionMatrix)
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
    'f1': f1
    })