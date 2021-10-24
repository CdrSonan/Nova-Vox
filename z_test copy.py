import csv
import torch
import numpy
from Backend.VB_Components.SpecCrfAi import RelLoss, SpecCrfAi

hiddenLayerNumberMax = 10

data = torch.load("data.dat")
testVBNumber = int(input("test Voicebank number? >>>"))
hiddenLayerNumber = int(input("hidden layer number start? >>>"))
device = input("training device? >>>")
if device == "GPU":
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
testData = data.pop(testVBNumber)
trainData = data.pop(0)
for i in range(len(data)):
    trainData = torch.cat(trainData, data.pop(0), 0)
trainData = trainData.to(device)
testData = testData.to(device)
for i in range(hiddenLayerNumber, hiddenLayerNumberMax):
    crfAi = SpecCrfAi(device, hiddenLayerCount = i)
    crfAi.train(trainData)
    trainResults = []
    testResults = []
    for j in trainData:
        encodedTypes = j[1]
        j = j[0]
        spectrum1 = j[0]
        spectrum2 = j[1]
        spectrum3 = j[-2]
        spectrum4 = j[-1]
        for k in range(j.size()[0]):
            factor = i / float(j.size()[0])
            spectrumTarget = j[k]
            output = torch.squeeze(crfAi.processData(spectrum1, spectrum2, spectrum3, spectrum4, factor))
            loss = RelLoss(output, spectrumTarget)
            trainResults.append(loss)
    trainAvg = numpy.mean(trainResults)
    trainStd = numpy.std(trainResults)
    del trainResults
    for j in testData:
        spectrum1 = j[0]
        spectrum2 = j[1]
        spectrum3 = j[-2]
        spectrum4 = j[-1]
        for k in range(j.size()[0]):
            factor = i / float(j.size()[0])
            spectrumTarget = j[k]
            output = torch.squeeze(crfAi.processData(spectrum1, spectrum2, spectrum3, spectrum4, factor))
            loss = RelLoss(output, spectrumTarget)
            testResults.append(loss)
    testAvg = numpy.mean(testResults)
    testStd = numpy.std(testResults)
    del testResults
    writer = csv.writer(open("results.csv", "w"))
    writer.writerow([testVBNumber, hiddenLayerNumber, testAvg, testStd])
    writer.close()