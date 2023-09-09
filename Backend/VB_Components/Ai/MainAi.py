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
from Backend.VB_Components.Ai.TrAi import TrAi
from Backend.VB_Components.Ai.Util import HighwayLSTM, SpecNormHighwayLSTM, SpecNormLSTM, init_weights
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.DataHandler.VocalSegment import VocalSegment
from Backend.Resampler.Resamplers import getSpecharm
from Backend.Resampler.CubicSplineInter import interp
from Backend.Resampler.PhaseShift import phaseInterp
from Util import dec2bin

halfHarms = int(global_consts.nHarmonics / 2) + 1


class MainAi(nn.Module):
    """Class for the Ai postprocessing/spectral prediction component.
    
    Methods:
        forward: processes a spectrum tensor, updating the internal states and returning the predicted next spectrum
        
        resetState: resets the hidden states and cell states of the internal LSTM layers"""


    def __init__(self, dim:int, blockA:list, blockB:list, blockC:list, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05) -> None:
        """basic constructor accepting the learning rate and other hyperparameters as input"""

        super().__init__()
        
        #self.encoderA = SpecNormHighwayLSTM(input_size = dim, hidden_size = blockA[0], num_layers = blockA[1], proj_size = dim, batch_first = True, dropout = dropout, device = device)
        
        #self.decoderA = SpecNormHighwayLSTM(input_size = dim, hidden_size = blockA[0], num_layers = blockA[1], proj_size = dim, batch_first = True, dropout = dropout, device = device)
        
        self.encoderA = nn.Sequential(
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(dim, blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], dim, 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
        )
        
        self.decoderA = nn.Sequential(
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(dim, blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], dim, 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
        )
        
        self.encoderB = SpecNormHighwayLSTM(input_size = 10 * dim, hidden_size = blockB[0], proj_size = dim, num_layers = blockB[1], batch_first = True, dropout = dropout, device = device)
        
        self.decoderB = nn.Sequential(SpecNormHighwayLSTM(input_size = dim, hidden_size = blockB[0], num_layers = blockB[1], batch_first = True, dropout = dropout, device = device),
                                      nn.Linear(blockB[0], 10 * dim, device = device),
                                      nn.Sigmoid())
        
        self.blockC = nn.Transformer(d_model = 10 * dim + 5, nhead = blockC[2], num_encoder_layers = blockC[1], num_decoder_layers = blockC[1], dim_feedforward = blockC[0], dropout = dropout, activation = "gelu", batch_first = True, device = device)
        
        self.apply(init_weights)

        self.device = device
        self.learningRate = learningRate
        self.dim = dim
        self.blockAHParams = blockA
        self.blockBHParams = blockB
        self.blockCHParams = blockC
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0

        #self.encoderAState = (torch.zeros(blockA[1], 1, dim, device = self.device), torch.zeros(blockA[1], 1, blockA[0], device = self.device))
        #self.decoderAState = (torch.zeros(blockA[1], 1, 10 * dim, device = self.device), torch.zeros(blockA[1], 1, blockA[0], device = self.device))
        self.encoderBState = (torch.zeros(blockB[1], 1, dim, device = self.device), torch.zeros(blockB[1], 1, blockB[0], device = self.device))
        self.decoderBState = (torch.zeros(blockB[1], 1, 10 * dim, device = self.device), torch.zeros(blockB[1], 1, blockB[0], device = self.device))
        

    def forward(self, latent:torch.Tensor, level:int) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next spectrum in a sequence"""

        if latent.size()[0] % 100 != 0:
            padded = torch.cat((latent, torch.zeros((100 - latent.size()[0] % 100, *latent.size()[1:]), device = self.device)), 0)
        else:
            padded = latent

        if level > 0:
            skipA = self.encoderA(padded.transpose(0 , 1))
        if level > 1:
            skipB, self.encoderBState = self.encoderB(skipA.reshape(1, skipA.size()[1] / 10, skipA.size()[2] * 10), self.encoderBState)
        if level > 2:
            positionalEncoding = torch.zeros((5, skipB.size()[2]), device = self.device)
            positionalEncoding[0] = torch.sin(torch.arange(0, skipB.size()[2], device = self.device) * pi / 2)
            positionalEncoding[1] = torch.cos(torch.arange(0, skipB.size()[2], device = self.device) * pi / 4)
            positionalEncoding[2] = torch.sin(torch.arange(0, skipB.size()[2], device = self.device) * pi / 8)
            positionalEncoding[3] = torch.cos(torch.arange(0, skipB.size()[2], device = self.device) * pi / 16)
            positionalEncoding[4] = torch.sin(torch.arange(0, skipB.size()[2], device = self.device) * pi / 32)
            transformerInput = torch.cat((skipB, positionalEncoding), 1)
            skipB += self.blockC(transformerInput, transformerInput)[:, :skipB.size()[1]]
        if level > 1:
            decodedB = self.decoderB(skipB, self.decoderBState).reshape(1, skipA.size()[1], skipA.size()[2])
            skipA += decodedB[0]
            self.decoderBState = decodedB[1]
        if level > 0:
            output = latent + self.decoderA(skipA).transpose(0 , 1)[:latent.size()[0]]
        else:
            output = latent
        
        return output

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        #self.encoderAState = (torch.zeros(self.blockAHParams[1], 1, self.dim, device = self.device), torch.zeros(self.blockAHParams[1], 1, self.blockAHParams[0], device = self.device))
        #self.decoderAState = (torch.zeros(self.blockAHParams[1], 1, 10 * self.dim, device = self.device), torch.zeros(self.blockAHParams[1], 1, self.blockAHParams[0], device = self.device))
        self.encoderBState = (torch.zeros(self.blockBHParams[1], 1, self.dim, device = self.device), torch.zeros(self.blockBHParams[1], 1, self.blockBHParams[0], device = self.device))
        self.decoderBState = (torch.zeros(self.blockBHParams[1], 1, 10 * self.dim, device = self.device), torch.zeros(self.blockBHParams[1], 1, self.blockBHParams[0], device = self.device))

class MainCritic(nn.Module):
    
    def __init__(self, dim:int, blockA:list, blockB:list, blockC:list, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05) -> None:
        super().__init__()
        
        self.encoderA = nn.Sequential(
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(dim, blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], dim, 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
        )
        
        self.decoderA = nn.Sequential(
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(dim, blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], blockA[0], 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
            nn.ConstantPad1d((2, 2), 0),
            nn.utils.parametrizations.spectral_norm(nn.Conv1d(blockA[0], dim, 5, padding = 2, device = device)),
            nn.InstanceNorm1d(blockA[0], device = device),
            nn.Sigmoid(),
        )
        
        self.encoderB = SpecNormHighwayLSTM(input_size = 10 * dim, hidden_size = blockB[0], proj_size = dim, num_layers = blockB[1], batch_first = True, dropout = dropout, device = device)
        
        self.decoderB = nn.Sequential(SpecNormHighwayLSTM(input_size = dim, hidden_size = blockB[0], num_layers = blockB[1], batch_first = True, dropout = dropout, device = device),
                                      nn.Linear(blockB[0], 10 * dim, device = device),
                                      nn.Sigmoid())
        
        self.blockC = nn.Transformer(d_model = 10 * dim + 5, nhead = blockC[2], num_encoder_layers = blockC[1], num_decoder_layers = blockC[1], dim_feedforward = blockC[0], dropout = dropout, activation = "gelu", batch_first = True, device = device)
        
        self.final = nn.Linear(dim, 1, bias = False, device = device)
        
        self.apply(init_weights)

        self.device = device
        self.learningRate = learningRate
        self.dim = dim
        self.blockAHParams = blockA
        self.blockBHParams = blockB
        self.blockCHParams = blockC
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0

        #self.encoderAState = (torch.zeros(blockA[1], 1, dim, device = self.device), torch.zeros(blockA[1], 1, blockA[0], device = self.device))
        #self.decoderAState = (torch.zeros(blockA[1], 1, 10 * dim, device = self.device), torch.zeros(blockA[1], 1, blockA[0], device = self.device))
        self.encoderBState = (torch.zeros(blockB[1], 1, dim, device = self.device), torch.zeros(blockB[1], 1, blockB[0], device = self.device))
        self.decoderBState = (torch.zeros(blockB[1], 1, 10 * dim, device = self.device), torch.zeros(blockB[1], 1, blockB[0], device = self.device))
        

    def forward(self, latent:torch.Tensor, level:int) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next spectrum in a sequence"""

        if latent.size()[0] % 100 != 0:
            padded = torch.cat((latent, torch.zeros((100 - latent.size()[0] % 100, *latent.size()[1:]), device = self.device)), 0)
        else:
            padded = latent

        if level > 0:
            skipA = self.encoderA(padded.transpose(0 , 1))
        if level > 1:
            skipB, self.encoderBState = self.encoderB(skipA.reshape(1, skipA.size()[1] / 10, skipA.size()[2] * 10), self.encoderBState)
        if level > 2:
            positionalEncoding = torch.zeros((5, skipB.size()[2]), device = self.device)
            positionalEncoding[0] = torch.sin(torch.arange(0, skipB.size()[2], device = self.device) * pi / 2)
            positionalEncoding[1] = torch.cos(torch.arange(0, skipB.size()[2], device = self.device) * pi / 4)
            positionalEncoding[2] = torch.sin(torch.arange(0, skipB.size()[2], device = self.device) * pi / 8)
            positionalEncoding[3] = torch.cos(torch.arange(0, skipB.size()[2], device = self.device) * pi / 16)
            positionalEncoding[4] = torch.sin(torch.arange(0, skipB.size()[2], device = self.device) * pi / 32)
            transformerInput = torch.cat((skipB, positionalEncoding), 1)
            skipB += self.blockC(transformerInput, transformerInput)[:, :skipB.size()[1]]
        if level > 1:
            decodedB = self.decoderB(skipB, self.decoderBState).reshape(1, skipA.size()[1], skipA.size()[2])
            skipA += decodedB[0]
            self.decoderBState = decodedB[1]
        if level > 0:
            output = latent + self.decoderA(skipA).transpose(0 , 1)[:latent.size()[0]]
        else:
            output = latent
        
        return self.final(output[-1])

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        #self.encoderAState = (torch.zeros(self.blockAHParams[1], 1, self.dim, device = self.device), torch.zeros(self.blockAHParams[1], 1, self.blockAHParams[0], device = self.device))
        #self.decoderAState = (torch.zeros(self.blockAHParams[1], 1, 10 * self.dim, device = self.device), torch.zeros(self.blockAHParams[1], 1, self.blockAHParams[0], device = self.device))
        self.encoderBState = (torch.zeros(self.blockBHParams[1], 1, self.dim, device = self.device), torch.zeros(self.blockBHParams[1], 1, self.blockBHParams[0], device = self.device))
        self.decoderBState = (torch.zeros(self.blockBHParams[1], 1, 10 * self.dim, device = self.device), torch.zeros(self.blockBHParams[1], 1, self.blockBHParams[0], device = self.device))

class DataGenerator:
    """generates synthetic data for the discriminator to train on"""
    
    def __init__(self, voicebank, crfAi:TrAi, mode:str = "reclist", requireVowels:bool = False) -> None:
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