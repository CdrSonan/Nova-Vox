#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import random
from math import floor, pi, ceil, log

import torch
import torch.nn as nn
import global_consts
from Backend.VB_Components.Ai.CrfAi import SpecCrfAi
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.DataHandler.VocalSegment import VocalSegment
from Backend.Resampler.Resamplers import getSpecharm
from Backend.Resampler.CubicSplineInter import interp
from Backend.Resampler.PhaseShift import phaseInterp
from Util import dec2bin

halfHarms = int(global_consts.nHarmonics / 2) + 1


class SpecPredAi(nn.Module):
    """Class for the Ai postprocessing/spectral prediction component.
    
    Methods:
        forward: processes a spectrum tensor, updating the internal states and returning the predicted next spectrum
        
        resetState: resets the hidden states and cell states of the internal LSTM layers"""


    def __init__(self, device:torch.device = None, learningRate:float=5e-5, recLayerCount:int=3, recSize:int=halfHarms + global_consts.halfTripleBatchSize + 1, regularization:float=1e-5) -> None:
        """basic constructor accepting the learning rate and other hyperparameters as input"""

        super().__init__()
        
        self.specLayerStart1 = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, int(global_consts.halfTripleBatchSize / 2 + recSize / 2), device = device)
        self.specReLuStart1 = nn.Sigmoid()
        self.specLayerStart2 = torch.nn.Linear(int(global_consts.halfTripleBatchSize / 2 + recSize / 2), recSize, device = device)
        self.specReLuStart2 = nn.Sigmoid()
        self.specRecurrentLayers = nn.LSTM(input_size = recSize, hidden_size = recSize, num_layers = recLayerCount, batch_first = True, dropout = 0.05, device = device)
        self.specLayerEnd1 = torch.nn.Linear(recSize, int(recSize / 2 + global_consts.halfTripleBatchSize / 2), device = device)
        self.specReLuEnd1 = nn.Sigmoid()
        self.specLayerEnd2 = torch.nn.Linear(int(recSize / 2 + global_consts.halfTripleBatchSize / 2), global_consts.halfTripleBatchSize + 1, device = device)
        self.specReLuEnd2 = nn.Sigmoid()
        
        self.harmLayerStart1 = torch.nn.Linear(halfHarms, int(halfHarms / 2 + recSize / 2), device = device)
        self.harmReLuStart1 = nn.Sigmoid()
        self.harmLayerStart2 = torch.nn.Linear(int(halfHarms / 2 + recSize / 2), recSize, device = device)
        self.harmReLuStart2 = nn.Sigmoid()
        self.harmRecurrentLayers = nn.LSTM(input_size = recSize, hidden_size = recSize, num_layers = recLayerCount, batch_first = True, dropout = 0.05, device = device)
        self.harmLayerEnd1 = torch.nn.Linear(recSize, int(recSize / 2 + halfHarms / 2), device = device)
        self.harmReLuEnd1 = nn.Sigmoid()
        self.harmLayerEnd2 = torch.nn.Linear(int(recSize / 2 + halfHarms / 2), halfHarms, device = device)
        self.harmReLuEnd2 = nn.Sigmoid()
        
        self.spec2harmIn = torch.nn.Linear(recSize, recSize, device = device)
        self.harm2specIn = torch.nn.Linear(recSize, recSize, device = device)
        self.spec2harmOut = torch.nn.Linear(recSize, recSize, device = device)
        self.harm2specOut = torch.nn.Linear(recSize, recSize, device = device)

        self.threshold = torch.nn.Threshold(0.001, 0.001)

        self.device = device
        self.learningRate = learningRate
        self.recLayerCount = recLayerCount
        self.recSize = recSize
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0

        self.specState = (torch.zeros(recLayerCount, 1, recSize, device = self.device), torch.zeros(recLayerCount, 1, recSize, device = self.device))
        self.harmState = (torch.zeros(recLayerCount, 1, recSize, device = self.device), torch.zeros(recLayerCount, 1, recSize, device = self.device))
        

    def forward(self, specharm:torch.Tensor, useJoints:bool = True) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next spectrum in a sequence"""

        phases = specharm[:, halfHarms:2 * halfHarms]
        spectrum = specharm[:, 2 * halfHarms:]
        harms = specharm[:, :halfHarms]
        x = spectrum.float().to(self.device)
        x = self.specLayerStart1(x)
        x = self.specReLuStart1(x)
        x = self.specLayerStart2(x)
        x = self.specReLuStart2(x)
        y = harms.float().to(self.device)
        y = self.harmLayerStart1(y)
        y = self.harmReLuStart1(y)
        y = self.harmLayerStart2(y)
        y = self.harmReLuStart2(y)
        
        if useJoints:
            x, y = (x + self.harm2specIn(y), y + self.spec2harmIn(x))
        
        x, self.specState = self.specRecurrentLayers(x.unsqueeze(0), self.specState)
        x = x.squeeze(dim = 0)
        y, self.harmState = self.harmRecurrentLayers(y.unsqueeze(0), self.harmState)
        y = y.squeeze(dim = 0)
        
        if useJoints:
            x, y = (x + self.harm2specOut(y), y + self.spec2harmOut(x))
        
        x = self.specLayerEnd1(x)
        x = self.specReLuEnd1(x)
        x = self.specLayerEnd2(x)
        x = self.specReLuEnd2(x)
        y = self.harmLayerEnd1(y)
        y = self.harmReLuEnd1(y)
        y = self.harmLayerEnd2(y)
        y = self.harmReLuEnd2(y)

        spectralFilterWidth = 2 * global_consts.filterTEEMult
        x = torch.fft.rfft(x, dim = 1)
        cutoffWindow = torch.zeros_like(x)
        cutoffWindow[:, 0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[:, int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        x = torch.fft.irfft(cutoffWindow * x, dim = 1, n = global_consts.halfTripleBatchSize + 1)
        x = self.threshold(x)
        
        y = torch.max(y, torch.tensor([0.0001,], device = self.device))

        return torch.cat((y + harms, phases, x + spectrum), 1)

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        self.specState = (torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device), torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device))
        self.harmState = (torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device), torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device))

class SpecPredDiscriminator(nn.Module):
    
    def __init__(self, device:torch.device = None, learningRate:float=5e-5, recLayerCount:int=3, recSize:int=halfHarms + global_consts.halfTripleBatchSize + 1, regularization:float=1e-5) -> None:
        super().__init__()
        self.layerStart1 = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(global_consts.halfTripleBatchSize + global_consts.nHarmonics + 3, int(global_consts.halfTripleBatchSize / 2 + recSize / 2), device = device))
        self.ReLuStart1 = nn.Sigmoid()
        self.layerStart2 = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(int(global_consts.halfTripleBatchSize / 2 + recSize / 2), recSize, device = device))
        self.ReLuStart2 = nn.Sigmoid()
        #self.recurrentLayers = torch.nn.utils.parametrizations.spectral_norm(nn.LSTM(input_size = recSize, hidden_size = recSize, num_layers = recLayerCount, batch_first = True, dropout = 0.05, device = device), name = "all_weights")
        self.recurrentLayers = nn.LSTM(input_size = recSize, hidden_size = recSize, num_layers = recLayerCount, batch_first = True, dropout = 0.05, device = device)
        for i in self.recurrentLayers._all_weights:
            for j in i:
                self.recurrentLayers = torch.nn.utils.parametrizations.spectral_norm(self.recurrentLayers, name = j)
        self.layerEnd1 = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(recSize, int(recSize / 2), device = device))
        self.ReLuEnd1 = nn.Sigmoid()
        self.layerEnd2 = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(int(recSize / 2), 1, device = device))
        self.ReLuEnd2 = nn.Sigmoid()

        self.device = device
        self.learningRate = learningRate
        self.recLayerCount = recLayerCount
        self.recSize = recSize
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0

        self.state = (torch.zeros(recLayerCount, 1, recSize, device = self.device), torch.zeros(recLayerCount, 1, recSize, device = self.device))
        

    def forward(self, spectrum:torch.Tensor) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next spectrum in a sequence"""

        x = spectrum.float().to(self.device)
        x = self.layerStart1(x)
        x = self.ReLuStart1(x)
        x = self.layerStart2(x)
        x = self.ReLuStart2(x)
        x, self.state = self.recurrentLayers(x.unsqueeze(0), self.state)
        x = x.squeeze(dim = 0)
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)
        return torch.max(x).squeeze()

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        self.state = (torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device), torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device))

class DataGenerator:
    """generates synthetic data for the discriminator to train on"""
    
    def __init__(self, voicebank, crfAi:SpecCrfAi, mode:str = "reclist", requireVowels:bool = False) -> None:
        self.voicebank = voicebank
        self.crfAi = crfAi
        self.mode = mode
        self.requireVowels = requireVowels
        self.rebuildPool()

    def rebuildPool(self) -> None:
        if self.requireVowels:
            self.longPool = [key for key, value in self.voicebank.phonemeDict.keys() if value[0].isPlosive and value[0].isVoiced]
            self.shortPool = [key for key, value in self.voicebank.phonemeDict.items() if value[0].isPlosive or not value[0].isVoiced]
        else:
            self.longPool = [key for key, value in self.voicebank.phonemeDict.items() if not value[0].isPlosive]
            self.shortPool = [key for key, value in self.voicebank.phonemeDict.items() if value[0].isPlosive]
        if len(self.shortPool) == 0:
            self.shortAvg = 0
        else:
            self.shortAvg = sum([self.voicebank.phonemeDict[i][0].specharm.size()[0] for i in self.shortPool]) / len(self.shortPool)
        if len(self.longPool) == 0:
            self.longAvg = 0
        else:
            self.longAvg = sum([self.voicebank.phonemeDict[i][0].specharm.size()[0] for i in self.longPool]) / len(self.longPool)

    def makeSequence(self, noise:list, length:int = None, phonemeLength:int = None) -> torch.Tensor:
        if self.mode == "reclist":
            if phonemeLength is None:
                raise ValueError("Length must be specified for reclist mode")
            longPhoneme = random.choice(self.longPool)
            shortPhonemes = [random.choice(self.shortPool) for i in range(floor(phonemeLength / 2))]
            phonemeSequence = []
            i = phonemeLength - 1
            while i > 0:
                if i % 2 == 0:
                    phonemeSequence.append(shortPhonemes[floor(i / 2)])
                else:
                    phonemeSequence.append(longPhoneme)
                i -= 1
        elif self.mode == "natural":
            phonemeSequence = random.choice(self.voicebank.wordDict[0].values())
        else:
            raise ValueError("Invalid mode for Data Generator")
        embeddings = [self.voicebank.phonemeDict[i][0].embedding for i in phonemeSequence]
        numShortPhonemes = sum([1 for i in phonemeSequence if i in self.shortPool])
        numLongPhonemes = sum([1 for i in phonemeSequence if i in self.longPool])
        effectiveLength = length - numShortPhonemes * self.shortAvg
        if effectiveLength >= 0:
            shortMultiplier = 1
            longMultiplier = effectiveLength / (numLongPhonemes * self.longAvg)
        else:
            shortMultiplier = longMultiplier = length / (numLongPhonemes * self.longAvg + numShortPhonemes * self.shortAvg)
            
        borders = [0, int(self.voicebank.phonemeDict[phonemeSequence[0]][0].specharm.size()[0] / 4), int(self.voicebank.phonemeDict[phonemeSequence[0]][0].specharm.size()[0] / 2)]
        referencePoint = self.voicebank.phonemeDict[phonemeSequence[0]][0].specharm.size()[0]
        for i, phoneme in enumerate(phonemeSequence):
            transitionLength = int(min(self.voicebank.phonemeDict[phonemeSequence[i]][0].specharm.size()[0] / 4, self.voicebank.phonemeDict[phonemeSequence[i - 1]][0].specharm.size()[0] / 4))
            borders.append(int(referencePoint - (1 + random.uniform(-0.2, 0.2) * noise[0]) * transitionLength) - 1)
            borders.append(int(referencePoint + random.uniform(-0.2, 0.2) * noise[0] * transitionLength))
            borders.append(int(referencePoint + (1 + random.uniform(-0.2, 0.2) * noise[0]) * transitionLength) + 1)
            if phoneme in self.shortPool:
                referencePoint += self.voicebank.phonemeDict[phonemeSequence[i]][0].specharm.size()[0] * shortMultiplier
            else:
                referencePoint += self.voicebank.phonemeDict[phonemeSequence[i]][0].specharm.size()[0] * longMultiplier
        borderLength = borders[-1] - borders[0]
        sequence = VocalSequence(borderLength,
                                 borders,
                                 phonemeSequence,
                                 [0 for i in range(len(phonemeSequence))],
                                 [0 for i in range(len(phonemeSequence))],
                                 torch.tensor([random.uniform(0, 1) * noise[1] for i in range(len(phonemeSequence))], device = self.crfAi.device),
                                 torch.tensor([random.uniform(0, 1) * noise[1] for i in range(len(phonemeSequence))], device = self.crfAi.device),
                                 torch.full((borderLength,), 300.5, device = self.crfAi.device),#pitch
                                 torch.full((borderLength,), random.uniform(-1, 1) * noise[2], device = self.crfAi.device),#steadiness
                                 torch.full((borderLength,), random.uniform(-1, 1) * noise[3], device = self.crfAi.device),#breathiness
                                 torch.zeros((borderLength,), device = self.crfAi.device),#AI balance
                                 torch.zeros((borderLength,), device = self.crfAi.device),#vibrato speed
                                 torch.full((borderLength,), -1, device = self.crfAi.device),#vibrato strength
                                 True,
                                 True,
                                 False,
                                 False,
                                 True,
                                 [],
                                 None
        )
        return sequence, embeddings

    def synthesize(self, noise:list, length:int, phonemeLength:int = None) -> torch.Tensor:
        """noise mappings: [borders, offsets/spacing, steadiness, breathiness]"""
        sequence, embeddings = self.makeSequence(noise, length, phonemeLength)
        output = torch.zeros([sequence.length, global_consts.halfTripleBatchSize + global_consts.nHarmonics + 3], device = self.crfAi.device)
        output[sequence.borders[0]:sequence.borders[3]] = getSpecharm(VocalSegment(sequence, self.voicebank, 0, self.crfAi.device), self.crfAi.device)
        for i in range(1, sequence.phonemeLength - 1):
            output[sequence.borders[3*i+2]:sequence.borders[3*i+3]] = getSpecharm(VocalSegment(sequence, self.voicebank, i, self.crfAi.device), self.crfAi.device)
        output[sequence.borders[-4]:sequence.borders[-1]] = getSpecharm(VocalSegment(sequence, self.voicebank, sequence.phonemeLength - 1, self.crfAi.device), self.crfAi.device)
        for i in range(1, sequence.phonemeLength):
            output[sequence.borders[3*i]:sequence.borders[3*i+2]] = self.crfWrapper(output[sequence.borders[3*i] - 1],
                                                                               output[sequence.borders[3*i]],
                                                                               output[sequence.borders[3*i+2]],
                                                                               output[sequence.borders[3*i+2]+1],
                                                                               embeddings[i - 1],
                                                                               embeddings[i],
                                                                               sequence.borders[3*i+2] - sequence.borders[3*i],
                                                                               sequence.pitch[sequence.borders[3*i]:sequence.borders[3*i+2]],
                                                                               (sequence.borders[3*i+1] - sequence.borders[3*i])/(sequence.borders[3*i+2] - sequence.borders[3*i]))
            
        return output
    
    def crfWrapper(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, embedding1:torch.Tensor, embedding2:torch.Tensor, outputSize:int, pitchCurve:torch.Tensor, slopeFactor:int):
        self.crfAi.eval()
        self.crfAi.requires_grad_(False)
        phase1 = specharm1[halfHarms:2 * halfHarms]
        phase2 = specharm2[halfHarms:2 * halfHarms]
        phase3 = specharm3[halfHarms:2 * halfHarms]
        phase4 = specharm4[halfHarms:2 * halfHarms]
        spectrum1 = specharm1[2 * halfHarms:]
        spectrum2 = specharm2[2 * halfHarms:]
        spectrum3 = specharm3[2 * halfHarms:]
        spectrum4 = specharm4[2 * halfHarms:]
        harm1 = specharm1[:halfHarms]
        harm2 = specharm2[:halfHarms]
        harm3 = specharm3[:halfHarms]
        harm4 = specharm4[:halfHarms]
        factor = log(0.5, slopeFactor / outputSize)
        factor = torch.pow(torch.linspace(0, 1, outputSize, device = self.crfAi.device), factor)
        embedding1 = dec2bin(torch.tensor(embedding1, device = self.crfAi.device), 32)
        embedding2 = dec2bin(torch.tensor(embedding2, device = self.crfAi.device), 32)
        spectrum = torch.squeeze(self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, embedding1, embedding2, factor)).transpose(0, 1)
        #for i in self.defectiveCrfBins:
        #    spectrum[:, i] = torch.mean(torch.cat((spectrum[:, i - 1].unsqueeze(1), spectrum[:, i + 1].unsqueeze(1)), 1), 1)
        borderRange = torch.zeros((outputSize,), device = self.crfAi.device)
        borderLimit = min(global_consts.crfBorderAbs, ceil(outputSize * global_consts.crfBorderRel))
        borderRange[:borderLimit] = torch.linspace(1, 0, borderLimit, device = self.crfAi.device)
        spectrum *= (1. - borderRange.unsqueeze(1))
        spectrum += torch.matmul(borderRange.unsqueeze(1), ((spectrum1 + spectrum2) / 2).unsqueeze(0))
        borderRange = torch.flip(borderRange, (0,))
        spectrum *= (1. - borderRange.unsqueeze(1))
        spectrum += torch.matmul(borderRange.unsqueeze(1), ((spectrum3 + spectrum4) / 2).unsqueeze(0))
        phases = torch.empty(outputSize, phase1.size()[0], device = self.crfAi.device)
        nativePitch = ceil(global_consts.tripleBatchSize / pitchCurve[0])
        originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.crfAi.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.crfAi.device))
        factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.crfAi.device), (spectrum1 + spectrum2) / 2, originSpace)
        harmsStart = (harm1 + harm2) * 0.5 / factors
        nativePitch = ceil(global_consts.tripleBatchSize / pitchCurve[-1])
        originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.crfAi.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.crfAi.device))
        factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.crfAi.device), (spectrum3 + spectrum4) / 2, originSpace)
        harmsEnd = (harm3 + harm4) * 0.5 / factors
        harms = torch.empty((outputSize, halfHarms), device = self.crfAi.device)
        harmLimit = torch.max(torch.cat((harm1.unsqueeze(1), harm2.unsqueeze(1), harm3.unsqueeze(1), harm4.unsqueeze(1)), 1))
        for i in range(outputSize):
            phases[i] = phaseInterp(phaseInterp(phase1, phase2, 0.5), phaseInterp(phase3, phase4, 0.5), i / (outputSize - 1))
            nativePitch = ceil(global_consts.tripleBatchSize / pitchCurve[i])
            originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.crfAi.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.crfAi.device))
            factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.crfAi.device), spectrum[i], originSpace)
            harms[i] = ((1. - (i + 1) / (outputSize + 1)) * harmsStart + (i + 1) / (outputSize + 1) * harmsEnd) * factors
            harms[i] = torch.min(harms[i], harmLimit)
            harms[i] = torch.max(harms[i], torch.tensor([0.,], device = self.crfAi.device))
        output = torch.cat((harms, phases, spectrum), 1)
        return output