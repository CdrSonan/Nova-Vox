import csv
import torch
import global_consts
from Backend.VB_Components.SpecCrfAi import SpecCrfAi

data = torch.load("data.dat")
testVBNumber = int(input("test Voicebank number? >>>"))
testSampleBatchNumber = int(input("test sample batch? >>>"))

testVB = data.pop(testVBNumber)
dataTmp = data[testVBNumber + 1]
testSampleBatch = dataTmp.pop(testSampleBatchNumber)
data[testVBNumber + 1] = dataTmp

