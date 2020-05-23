import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flask import jsonify, send_from_directory
import numpy as np
import json
from sklearn.model_selection import train_test_split

def build_nn(netArr):
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      for i in range(len(netArr) - 1):
          layerName = 'fullyConnected' + str(i+1)
          setattr(self, layerName, nn.Linear(int(netArr[i]['noOfNodes']), int(netArr[i+1]['noOfNodes'])))

    def forward(self, x):
      #print netArr and check... The range below has some problem
      for i in range(len(netArr) - 1):
        layerName = 'fullyConnected' + str(i+1)
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
  print("learning rate: ", learningRate)
  optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=0.9)
  # criterion = nn.CrossEntropyLoss()  
  criterion = nn.NLLLoss()

  opColNo = opColNo - 1
  data = np.genfromtxt('./temp/inputFile.csv', delimiter=',')
  maxVals = data.max(axis=0)
  maxVals[maxVals == 0] = 1
  data = data / maxVals

  opData = data[:, opColNo]
  if opColNo == 0:
    ipData = data[:, 1:]
  elif opColNo == data.shape[1] - 1:
    ipData = data[:, 0: -1]
  else:
    leftData = data[:, 0: opColNo]
    rightData = data[:, opColNo + 1 : ]
    ipData = np.hstack((leftData, rightData))

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

  

  return  jsonify({'success': 'success'})