# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 22:46:54 2021

@author: CdrSonan
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torchaudio
torchaudio.set_audio_backend("soundfile")
import matplotlib.pyplot as plt

class AudioSample:
    def __init__(self, filepath):
        loadedData = torchaudio.load(filepath)
        self.waveform = loadedData[0][0]
        self.sampleRate = loadedData[1]
        del loadedData
        self.pitchDeltas = torch.tensor([], dtype = int)
        self.pitchBorders = torch.tensor([], dtype = int)
        self.pitch = torch.tensor([0], dtype = int)
        self.spectra = torch.tensor([[]], dtype = float)
        self.spectrum = torch.tensor([], dtype = float)
        self.excitation = torch.tensor([], dtype = float)
        self.voicedExcitation = torch.tensor([], dtype = float)
        self.VoicedExcitations = torch.tensor([], dtype = float)
        
    def CalculatePitch(self, expectedPitch, searchRange = 0.2):
        batchSize = math.floor((1. + searchRange) * self.sampleRate / expectedPitch)
        lowerSearchLimit = math.floor((1. - searchRange) * self.sampleRate / expectedPitch)
        batchStart = 0
        while batchStart + batchSize <= self.waveform.size()[0] - batchSize:
            sample = torch.index_select(self.waveform, 0, torch.linspace(batchStart, batchStart + batchSize, batchSize, dtype = int))
            zeroTransitions = torch.tensor([], dtype = int)
            for i in range(lowerSearchLimit, batchSize):
                if (sample[i-1] < 0) and (sample[i] > 0):
                    zeroTransitions = torch.cat([zeroTransitions, torch.tensor([i])], 0)
            error = math.inf
            delta = math.floor(self.sampleRate / expectedPitch)
            for i in zeroTransitions:
                shiftedSample = torch.index_select(self.waveform, 0, torch.linspace(batchStart + i.item(), batchStart + batchSize + i.item(), batchSize, dtype = int))
                newError = torch.sum(torch.pow(sample - shiftedSample, 2))
                if error > newError:
                    delta = i.item()
                    error = newError
            self.pitchDeltas = torch.cat([self.pitchDeltas, torch.tensor([delta])])
            batchStart += delta
        nBatches = self.pitchDeltas.size()[0]
        self.pitchBorders = torch.zeros(nBatches + 1, dtype = int)
        for i in range(nBatches):
            self.pitchBorders[i+1] = self.pitchBorders[i] + self.pitchDeltas[i]
        self.pitch = torch.mean(self.pitchDeltas.float()).int()
        del batchSize
        del lowerSearchLimit
        del batchStart
        del sample
        del zeroTransitions
        del error
        del delta
        del shiftedSample
        del newError
        del nBatches
        
    def CalculateSpectra(self, iterations = 10, filterWidth = 10, preIterations = 2):
        tripleBatchSize = int(self.sampleRate / 25)
        BatchSize = int(self.sampleRate / 75)
        Window = torch.hann_window(tripleBatchSize)
        signals = torch.stft(self.waveform, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = Window, return_complex = True, onesided = True)
        signals = torch.transpose(signals, 0, 1)
        signalsAbs = signals.abs()
        
        workingSpectra = torch.sqrt(signalsAbs)
        
        workingSpectra = torch.max(workingSpectra, torch.tensor([-100]))
        self.spectra = torch.full_like(workingSpectra, -float("inf"), dtype=torch.float)
        
        for j in range(preIterations):
            workingSpectra = torch.max(workingSpectra, self.spectra)
            self.spectra = workingSpectra
            for i in range(filterWidth):
                self.spectra = torch.roll(workingSpectra, -i, dims = 1) + self.spectra + torch.roll(workingSpectra, i, dims = 1)
            self.spectra = self.spectra / (2 * filterWidth + 1)
        
        self.VoicedExcitations = torch.zeros_like(signals)
        for i in range(signals.size()[0]):
            for j in range(signals.size()[1]):
                if torch.sqrt(signalsAbs[i][j]) > self.spectra[i][j]:
                    self.VoicedExcitations[i][j] = signals[i][j]
                
        for j in range(iterations):
            workingSpectra = torch.max(workingSpectra, self.spectra)
            self.spectra = workingSpectra
            for i in range(filterWidth):
                self.spectra = torch.roll(workingSpectra, -i, dims = 1) + self.spectra + torch.roll(workingSpectra, i, dims = 1)
            self.spectra = self.spectra / (2 * filterWidth + 1)
        
        self.spectrum = torch.mean(self.spectra, 0)
        for i in range(self.spectra.size()[0]):
            self.spectra[i] = self.spectra[i] - self.spectrum
        #return torch.log(signalsAbs)
        del Window
        del signals
        del workingSpectra
        
    def CalculateExcitation(self, filterWidth = 10):
        tripleBatchSize = int(self.sampleRate / 25)
        BatchSize = int(self.sampleRate / 75)
        Window = torch.hann_window(tripleBatchSize)
        signals = torch.stft(self.waveform, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = Window, return_complex = True, onesided = True)
        signals = torch.transpose(signals, 0, 1)
        excitations = torch.empty_like(signals)
        for i in range(excitations.size()[0]):
            excitations[i] = signals[i] / torch.square(self.spectrum + self.spectra[i])
            self.VoicedExcitations[i] = self.VoicedExcitations[i] / torch.square(self.spectrum + self.spectra[i])
        
        VoicedExcitations = torch.transpose(self.VoicedExcitations, 0, 1)
        excitations = torch.transpose(excitations, 0, 1)
        self.excitation = torch.istft(excitations, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = Window, onesided = True)
        self.voicedExcitation = torch.istft(VoicedExcitations, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = Window, onesided = True)
        
        self.excitation = self.excitation - self.voicedExcitation
        self.excitation = torch.stft(self.excitation, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = Window, return_complex = True, onesided = True)
        self.voicedExcitation = torch.stft(self.voicedExcitation, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = Window, return_complex = True, onesided = True)
        self.excitation = torch.transpose(self.excitation, 0, 1)
        #self.voicedExcitation = torch.transpose(self.voicedExcitation, 0, 1)
        
        del Window
        del signals
        del excitations
        
class loadedAudioSample:
    def __init__(self, audioSample):
        self.pitchDeltas = audioSample.pitchDeltas
        self.pitch = audioSample.pitch
        self.spectra = audioSample.spectra
        self.spectrum = audioSample.spectrum
        self.excitation = audioSample.excitation
        self.voicedExcitation = audioSample.voicedExcitation
        
class RelLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(RelLoss, self).__init__()
 
    def forward(self, inputs, targets):    
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        differences = torch.abs(inputs - targets)
        sums = torch.abs(inputs + targets)
        out = (differences / sums).sum() / inputs.size()[0]
        return out
    
class SpecCrfAi(nn.Module):
    def __init__(self, learningRate=1e-4):
        super(SpecCrfAi, self).__init__()
        
        self.layer1 = torch.nn.Linear(3843, 3843)
        self.ReLu1 = nn.ReLU()
        self.layer2 = torch.nn.Linear(3843, 5763)
        self.ReLu2 = nn.ReLU()
        self.layer3 = torch.nn.Linear(5763, 3842)
        self.ReLu3 = nn.ReLU()
        self.layer4 = torch.nn.Linear(3842, 1921)
        
        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=0.)
        self.criterion = nn.L1Loss()
        #self.criterion = RelLoss()
        self.epoch = 0
        self.loss = None
        
    def forward(self, spectrum1, spectrum2, factor):
        fac = torch.tensor([factor])
        x = torch.cat((spectrum1, spectrum2, fac), dim = 0)
        x = x.float()#.unsqueeze(0).unsqueeze(0)
        x = self.layer1(x)
        x = self.ReLu1(x)
        x = self.layer2(x)
        x = self.ReLu2(x)
        x = self.layer3(x)
        x = self.ReLu3(x)
        x = self.layer4(x)
        return x
    
    def processData(self, spectrum1, spectrum2, factor):
        output = torch.square(torch.squeeze(self(torch.sqrt(spectrum1), torch.sqrt(spectrum2), factor)))
        return output
    
    def train(self, indata, epochs=1):
        if (self.epoch == 0) or self.epoch == epochs:
            self.epoch = epochs
        else:
            self.epoch = None
        
        for epoch in range(epochs):
            for data in self.dataLoader(indata):
                spectrum1 = data[0]
                spectrum2 = data[-1]
                indexList = np.arange(0, data.size()[0], 1)
                np.random.shuffle(indexList)
                for i in indexList:
                    factor = i / float(data.size()[0])
                    spectrumTarget = data[i]
                    output = torch.squeeze(self(spectrum1, spectrum2, factor))
                    loss = self.criterion(output, spectrumTarget)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, epochs, loss.data))
        
        self.loss = loss
            
    def dataLoader(self, data):
        return torch.utils.data.DataLoader(dataset=data, shuffle=True)
    
    def getState(self):
        AiState = {'epoch': self.crfAi.epoch,
                 'model_state_dict': self.crfAi.state_dict(),
                 'optimizer_state_dict': self.crfAi.optimizer.state_dict(),
                 'loss': self.crfAi.loss
                 }
        return AiState
    
class Voicebank:
    def __init__(self, filepath, newFile):
        self.filepath = filepath
        self.sampleRate = 48000
        self.phonemeDict = dict()
        self.crfAi = SpecCrfAi()
        self.parameters = []
        if newFile:
            self.save(self.filepath)
        else:
            self.loadSampleRate(self.filepath)
            self.loadPhonemeDict(self.filepath, False)
            self.loadCrfWeights(self.filepath)
            self.loadParameters(self.filepath, False)
        
    def save(self, filepath):
        #torch.save(vb.crfAi.state_dict(), "CrossfadeWeights.dat")
        torch.save({
            "sampleRate":self.sampleRate,
            "crfAiState":self.crfAi.getState(),
            "phonemeDict":self.phonemeDict,
            "Parameters":self.parameters
            }, filepath)
        
    def loadSampleRate(self, filepath):
        data = torch.load(filepath)
        return data["sampleRate"]
    
    def loadPhonemeDict(self, filepath, additive):
        data = torch.load(filepath)
        if additive:
            for i in data["phonemeDict"].keys:
                if i in self.phonemeDict.keys:
                    self.phonemeDict[i + "#"] = data["phonemeDict"][i]
                    print("phoneme " + i + " is already present in voicebank; its key has been changed to " + i + "#")
                else:
                    self.phonemeDict[i] = data["phonemeDict"][i]
        else:
            self.phonemeDict = data["phonemeDict"]
    
    def loadCrfWeights(self, filepath):
        data = torch.load(filepath)
        self.crfAi.epoch = data["crfAiState"]['epoch']
        self.crfAi.load_state_dict(data["crfAiState"]['model_state_dict'])
        self.crfAi.optimizer.load_state_dict(data["crfAiState"]['optimizer_state_dict'])
        self.crfAi.loss = data["crfAiState"]['loss']
        self.crfAi.eval()
        
    def loadParameters(self, filepath, additive):
        if additive:
            pass
        else:
            pass
        
"""class TempVB:
    def __init__(self):
        self.sampleRate = 48000
        self.phonemeDict = dict([])
        phonemeKeys = ["A", "E", "I", "O", "U", "G", "K", "N", "S", "T"]
        for key in phonemeKeys:
            self.phonemeDict[key] = AudioSample("Samples_rip/"+key+".wav")
            self.sampleRate = self.phonemeDict[key].sampleRate
            self.phonemeDict[key].CalculatePitch(249.)
            self.phonemeDict[key].CalculateSpectra(iterations = 15)
            self.phonemeDict[key].CalculateExcitation()
        self.crfAi = SpecCrfAi(learningRate=1e-4)"""
        
TrainingSamples = dict([])
TrainingKeys = ["A_E", "A_G", "A_I", "A_K", "A_N", "A_O", "A_S", "A_T", "A_U",
                   "E_A", "E_G", "E_I", "E_K", "E_N", "E_O", "E_S", "E_T", "E_U",
                   "I_A", "I_E", "I_G", "I_K", "I_N", "I_O", "I_S", "I_T", "I_U",
                   "O_A", "O_E", "O_G", "O_I", "O_K", "O_N", "O_S", "O_T", "O_U",
                   "U_A", "U_E", "U_G", "U_I", "U_K", "U_N", "U_O", "U_S", "U_T",
                   "G_A", "G_E", "G_I", "G_O", "G_U",
                   "K_A", "K_E", "K_I", "K_O", "K_U",
                   "N_A", "N_E", "N_I", "N_O", "N_U",
                   "S_A", "S_E", "S_I", "S_O", "S_U",
                   "T_A", "T_E", "T_I", "T_O", "T_U"
                  ]

for key in TrainingKeys:
    TrainingSamples[key] = AudioSample("Samples_rip/"+key+".wav")
    TrainingSamples[key].CalculatePitch(249.)
    TrainingSamples[key].CalculateSpectra(iterations = 25)
    TrainingSamples[key].CalculateExcitation()
    
trainSpectra = []
i = 0
for key in TrainingKeys:
    trainSpectra.append(torch.empty_like(TrainingSamples[key].spectra))
    for j in range(TrainingSamples[key].spectra.size()[0]):
        trainSpectra[i][j] = TrainingSamples[key].spectrum + TrainingSamples[key].spectra[j]
    i += 1

vb = TempVB()
for i in range(70):
    vb.crfAi.train(trainSpectra[i], epochs = 5)