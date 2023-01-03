#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from typing import OrderedDict
import numpy as np
import math
from random import shuffle
from os import path, getenv
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import global_consts
from Backend.Resampler.PhaseShift import phaseInterp
from Backend.Resampler.CubicSplineInter import interp

halfHarms = int(global_consts.nHarmonics / 2) + 1

class SpecCrfAi(nn.Module):
    """class for generating crossfades between the spectra of different phonemes using AI.
    
    Attributes:
        layerStart1a, layerStart1b, ReLUStart1: leading RNN layer of the NN. Used for spectral processing.

        layerStart / End 1/2, ReLuStart / End 1/2: leading and trailing FC and Nonlinear layers of the NN. Used for spectral processing.

        hiddenLayers: torch.nn.Sequential object containing all layers between the leading and trailing ones
        
        threshold: final threshold layer applied to the output spectrum, to prevent overshooting

        device: the torch.device the AI is loaded on

        hiddenLayerCount: integer indicating the number of hidden layers of the NN

        hiddenLayerCount: integer indicating the size of the hidden layers of the NN

        learningRate: Learning Rate of the NN
        
        epoch: training epoch counter displayed in Metadata panels

        sampleCount: integer indicating the number of samples the AI has been trained with.
        
    Methods:
        __init__: Constructor initialising NN layers and prerequisite properties
        
        forward: Forward NN pass with unprocessed in-and outputs"""
        
        
    def __init__(self, device:torch.device = None, learningRate:float=5e-5, hiddenLayerCount:int = 3, hiddenLayerSize:int = 4 * (global_consts.halfTripleBatchSize + halfHarms), regularization:float=1e-5) -> None:
        """Constructor initialising NN layers and prerequisite attributes.
        
        Arguments:
            device: the device the AI is to be loaded on

            learningRate: desired learning rate of the NN as float. supports scientific format.
            
            hiddenLayerCount: number of hidden layers (between leading and trailing layers)

        Returns:
            None"""
            
            
        super(SpecCrfAi, self).__init__()
        self.layerStart1a = nn.RNN(input_size = 3 * global_consts.halfTripleBatchSize + 3, hidden_size = 2 * global_consts.halfTripleBatchSize + 2, num_layers = 1, batch_first = True, device = device)
        self.layerStart1b = nn.RNN(input_size = 3 * global_consts.halfTripleBatchSize + 3, hidden_size = 2 * global_consts.halfTripleBatchSize + 2, num_layers = 1, batch_first = True, device = device)
        self.ReLuStart1 = nn.ReLU()
        self.layerStart2 = torch.nn.Linear(4 * global_consts.halfTripleBatchSize + 4, hiddenLayerSize, device = device, bias = False)
        self.ReLuStart2 = nn.ReLU()
        hiddenLayerDict = OrderedDict([])
        for i in range(hiddenLayerCount):
            hiddenLayerDict["layer" + str(i)] = torch.nn.Linear(hiddenLayerSize, hiddenLayerSize, device = device)
            hiddenLayerDict["ReLu" + str(i)] = nn.ReLU()
        self.hiddenLayers = nn.Sequential(hiddenLayerDict)
        self.layerEnd1 = torch.nn.Linear(hiddenLayerSize, int(hiddenLayerSize / 2 + global_consts.halfTripleBatchSize / 2), device = device)
        self.ReLuEnd1 = nn.ReLU()
        self.layerEnd2 = torch.nn.Linear(int(hiddenLayerSize / 2 + global_consts.halfTripleBatchSize / 2), global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuEnd2 = nn.ReLU()
        
        self.threshold = torch.nn.Threshold(0.001, 0.001)

        self.device = device
        self.learningRate = learningRate
        self.hiddenLayerCount = hiddenLayerCount
        self.hiddenLayerSize = hiddenLayerSize
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0
        
    def forward(self, spectrum1:torch.Tensor, spectrum2:torch.Tensor, spectrum3:torch.Tensor, spectrum4:torch.Tensor, outputSize:float) -> torch.Tensor:
        """Forward NN pass.
        
        Arguments:
            spectrum1-4: The sets of two spectrum Tensors to perform the interpolation between, preceding and following the transition that is to be calculated, respectively.
            
            factor: Float between 0 and 1 determining the "position" within the interpolation. When using a value of 0 the output will be extremely similar to spectrum 1 and 2,
            while a values of 1 will result in output extremely similar to spectrum 3 and 4.
            
        Returns:
            Tensor object representing the NN output"""
        

        factor = torch.tile(torch.linspace(0, 1, outputSize, device = self.device).unsqueeze(-1).unsqueeze(-1), (1, 1, global_consts.halfTripleBatchSize + 1))
        spectrum1 = torch.unsqueeze(spectrum1.to(self.device), 0)
        spectrum2 = torch.unsqueeze(spectrum2.to(self.device), 0)
        spectrum3 = torch.unsqueeze(spectrum3.to(self.device), 0)
        spectrum4 = torch.unsqueeze(spectrum4.to(self.device), 0)
        spectra = torch.cat((spectrum1, spectrum2, spectrum3, spectrum4), dim = 0)
        limit = torch.max(spectra, dim = 0)[0]
        spectrum1tile = torch.tile(spectrum1.unsqueeze(0), (outputSize, 1, 1)) * (1. - factor)
        spectrum2tile = torch.tile(spectrum2.unsqueeze(0), (outputSize, 1, 1)) * (1. - factor)
        spectrum3tile = torch.tile(spectrum3.unsqueeze(0), (outputSize, 1, 1)) * factor
        spectrum4tile = torch.tile(spectrum4.unsqueeze(0), (outputSize, 1, 1)) * factor#outputSize, 5, hTBS
        x = torch.cat((spectrum3tile, spectrum4tile, factor), dim = 1)
        x = x.float()
        x = torch.flatten(x, 1)
        x = x.unsqueeze(0)
        state = torch.flatten(torch.cat((spectrum1, spectrum2), 1), 1).unsqueeze(0)
        x, state = self.layerStart1a(x, state)
        y = torch.cat((spectrum1tile, spectrum2tile, 1. - factor), dim = 1)
        y = y.float()
        y = torch.flatten(y, 1)
        y = y.unsqueeze(0)
        state = torch.flatten(torch.cat((spectrum3, spectrum4), 1), 1).unsqueeze(0)
        y, state = self.layerStart1b(y, state)
        x = torch.squeeze(torch.cat((x, y), 2))
        x = self.ReLuStart1(x)
        x = self.layerStart2(x)
        x = self.ReLuStart2(x)
        x = self.hiddenLayers(x)
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)
        x = torch.minimum(x, limit)

        """spectralFilterWidth = 3 * global_consts.filterTEEMult
        x = torch.fft.rfft(x, dim = 1)
        cutoffWindow = torch.zeros_like(x)
        cutoffWindow[:, 0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[:, int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        x = torch.fft.irfft(cutoffWindow * x, dim = 1, n = global_consts.halfTripleBatchSize + 1)"""
        x = self.threshold(x).transpose(0, 1)
        
        return x

class RelLoss(nn.Module):
    """function for calculating relative loss values between target and actual Tensor objects. Designed to be used with AI optimizers. Currently unused.
    
    Attributes:
        None
        
    Methods:
        __init__: basic class constructor
        
        forward: calculates relative loss based on input and target tensors after successful initialisation."""
    
    
    def __init__(self, weight=None, size_average=True):
        """basic class constructor.
        
        Arguments:
            weight: required by PyTorch in some situations. Unused.
            
            size_average: required by PyTorch in some situations. Unused.
            
        Returns:
            None"""
        
        
        super(RelLoss, self).__init__()
 
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> float:  
        """calculates relative loss based on input and target tensors after successful initialisation.
        
        Arguments:
            inputs: AI-generated input Tensor
            
            targets: target Tensor
            
        Returns:
            Relative error value calculated from the difference between input and target Tensor as Float"""
        
        differences = torch.abs(inputs - targets)
        correctors = targets / (inputs + 0.01) - 1
        out = torch.mean(torch.max(differences, correctors))
        return out

class SpecPredAi(nn.Module):
    """Class for the Ai postprocessing/spectral prediction component.
    
    Methods:
        forward: processes a spectrum tensor, updating the internal states and returning the predicted next spectrum
        
        resetState: resets the hidden states and cell states of the internal LSTM layers"""


    def __init__(self, device:torch.device = None, learningRate:float=5e-5, recLayerCount:int=3, recSize:int=halfHarms + global_consts.halfTripleBatchSize + 1, regularization:float=1e-5) -> None:
        """basic constructor accepting the learning rate and other hyperparameters as input"""

        super().__init__()
        
        self.layerStart1 = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, int(global_consts.halfTripleBatchSize / 2 + recSize / 2), device = device)
        self.ReLuStart1 = nn.ReLU()
        self.layerStart2 = torch.nn.Linear(int(global_consts.halfTripleBatchSize / 2 + recSize / 2), recSize, device = device)
        self.ReLuStart2 = nn.ReLU()
        self.recurrentLayers = nn.LSTM(input_size = recSize, hidden_size = recSize, num_layers = recLayerCount, batch_first = True, dropout = 0.05, device = device)
        self.layerEnd1 = torch.nn.Linear(recSize, int(recSize / 2 + global_consts.halfTripleBatchSize / 2), device = device)
        self.ReLuEnd1 = nn.ReLU()
        self.layerEnd2 = torch.nn.Linear(int(recSize / 2 + global_consts.halfTripleBatchSize / 2), global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuEnd2 = nn.ReLU()

        self.threshold = torch.nn.Threshold(0.001, 0.001)

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
        x = x.squeeze()
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)

        spectralFilterWidth = 2 * global_consts.filterTEEMult
        x = torch.fft.rfft(x, dim = 1)
        cutoffWindow = torch.zeros_like(x)
        cutoffWindow[:, 0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[:, int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        x = torch.fft.irfft(cutoffWindow * x, dim = 1, n = global_consts.halfTripleBatchSize + 1)
        x = self.threshold(x)

        return x

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        self.state = (torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device), torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device))


class HarmPredAi(nn.Module):
    """Class for the Ai postprocessing/spectral prediction component.
    
    Methods:
        forward: processes a harmonics tensor, updating the internal states and returning the predicted next harmonics batch
        
        resetState: resets the hidden states and cell states of the internal LSTM layers"""

    
    def __init__(self, device:torch.device = None, learningRate:float=5e-5, recLayerCount:int=3, recSize:int=halfHarms + global_consts.halfTripleBatchSize + 1, regularization:float=1e-5) -> None:
        """basic constructor accepting the learning rate and other hyperparameters as input"""

        super().__init__()

        self.layerStart1 = torch.nn.Linear(halfHarms, int(halfHarms / 2 + recSize / 2), device = device)
        self.ReLuStart1 = nn.ReLU()
        self.layerStart2 = torch.nn.Linear(int(halfHarms / 2 + recSize / 2), recSize, device = device)
        self.ReLuStart2 = nn.ReLU()
        self.recurrentLayers = nn.LSTM(input_size = recSize, hidden_size = recSize, num_layers = recLayerCount, batch_first = True, dropout = 0.05, device = device)
        self.layerEnd1 = torch.nn.Linear(recSize, int(recSize / 2 + halfHarms / 2), device = device)
        self.ReLuEnd1 = nn.ReLU()
        self.layerEnd2 = torch.nn.Linear(int(recSize / 2 + halfHarms / 2), halfHarms, device = device)
        self.ReLuEnd2 = nn.ReLU()

        self.device = device
        self.learningRate = learningRate
        self.recLayerCount = recLayerCount
        self.recSize = recSize
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0

        self.state = (torch.zeros(recLayerCount, 1, recSize, device = self.device), torch.zeros(recLayerCount, 1, recSize, device = self.device))

    def forward(self, harm:torch.Tensor) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next harmonics batch in a sequence"""
        
        x = harm.float().to(self.device)
        x = self.layerStart1(x)
        x = self.ReLuStart1(x)
        x = self.layerStart2(x)
        x = self.ReLuStart2(x)
        x, self.state = self.recurrentLayers(x.unsqueeze(0), self.state)
        x = x.squeeze()
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)

        return x

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        self.state = (torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device), torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device))


class AIWrapper():
    """Wrapper class for the mandatory AI components of a Voicebank. Controls data pre- and postprocessing, state loading and saving, Hyperparameters, and both training and inference."""

    def __init__(self, device = torch.device("cpu"), hparams:dict = None) -> None:
        """constructor taking a target device and dictionary of hyperparameters as input"""

        self.hparams = {
            "crf_lr": 0.000055,
            "crf_reg": 0.,
            "crf_hlc": 1,
            "crf_hls": 4000,
            "pred_lr": 1.,
            "pred_reg": 0.,
            "pred_rlc": 3,
            "pred_rs": 1024,
            "predh_lr": 1.,
            "predh_reg": 0.,
            "predh_rlc": 3,
            "predh_rs": 1024,
            "crf_def_thrh" : 0.05
        }
        if hparams:
            for i in hparams.keys():
                self.hparams[i] = hparams[i]
        self.crfAi = SpecCrfAi(device = device, learningRate=self.hparams["crf_lr"], regularization=self.hparams["crf_reg"], hiddenLayerCount=int(self.hparams["crf_hlc"]), hiddenLayerSize=int(self.hparams["crf_hls"]))
        self.predAi = SpecPredAi(device = device, learningRate=self.hparams["pred_lr"], regularization=self.hparams["pred_reg"], recSize=int(self.hparams["pred_rs"]), recLayerCount=int(self.hparams["pred_rlc"]))
        self.predAiHarm = HarmPredAi(device = device, learningRate=self.hparams["predh_lr"], regularization=self.hparams["predh_reg"], recSize=int(self.hparams["predh_rs"]), recLayerCount=int(self.hparams["predh_rlc"]))
        self.device = device
        self.final = False
        self.defectiveCrfBins = []
        #self.crfAiOptimizer = torch.optim.RMSprop(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=self.crfAi.regularization, momentum = 0.1)
        #self.predAiOptimizer = torch.optim.NAdam(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
        #self.predAiHarmOptimizer = torch.optim.NAdam(self.predAiHarm.parameters(), lr=self.predAiHarm.learningRate, weight_decay=self.predAiHarm.regularization)
        self.crfAiOptimizer = torch.optim.NAdam(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=self.crfAi.regularization)
        self.predAiOptimizer = torch.optim.Adadelta(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
        self.predAiHarmOptimizer = torch.optim.Adadelta(self.predAiHarm.parameters(), lr=self.predAiHarm.learningRate, weight_decay=self.predAiHarm.regularization)
        self.criterion = nn.L1Loss()
    
    @staticmethod
    def dataLoader(data) -> DataLoader:
        """helper method for shuffled data loading from an arbitrary dataset.
        
        Arguments:
            data: Tensor, List or Iterable representing a dataset with several elements
            
        Returns:
            Iterable representing the same dataset, with the order of its elements shuffled"""
        
        
        return DataLoader(dataset=data, shuffle=True)

    def getState(self) -> dict:
        """returns the state of the NN, its optimizer and their prerequisites as well as its epoch and sample count attributes in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's epoch attribute (epoch), weights (state dict), optimizer state (optimizer state dict) and sample count attribute (sampleCount)"""
            
        if self.final:
            aiState = {'crfAi_epoch': self.crfAi.epoch,
                'crfAi_model_state_dict': self.crfAi.state_dict(),
                'crfAi_sampleCount': self.crfAi.sampleCount,
                'predAi_epoch': self.predAi.epoch,
                'predAi_model_state_dict': self.predAi.state_dict(),
                'predAiHarm_model_state_dict': self.predAiHarm.state_dict(),
                'predAi_sampleCount': self.predAi.sampleCount,
                'final': True
            }
        else:
            aiState = {'crfAi_epoch': self.crfAi.epoch,
                'crfAi_model_state_dict': self.crfAi.state_dict(),
                'crfAi_optimizer_state_dict': self.crfAiOptimizer.state_dict(),
                'crfAi_sampleCount': self.crfAi.sampleCount,
                'predAi_epoch': self.predAi.epoch,
                'predAi_model_state_dict': self.predAi.state_dict(),
                'predAiHarm_model_state_dict': self.predAiHarm.state_dict(),
                'predAi_optimizer_state_dict': self.predAiOptimizer.state_dict(),
                'predAiHarm_optimizer_state_dict': self.predAiHarmOptimizer.state_dict(),
                'predAi_sampleCount': self.predAi.sampleCount,
                'final': False
            }
        return aiState

    def loadState(self, aiState:dict, mode:str = None, reset:bool=False) -> None:
        """loads the weights of the NNs managed by the wrapper from a dictionary, and reinitializes the NNs and/or their optimizers if required.
        
        Arguments:
            aiState: Dictionary in the same format as returned by getState(), containing all necessary information about the NNs
            
            mode: whether to load the weights for both NNs (None), only the phoneme crossfade Ai (crf), or only the prediction Ai (pred)
            
            reset: indicates whether the NNs and their optimizers should be reset before applying changed weights to them. Must be True when the dictionary contains weights
            for a NN using different hyperparameters than the currently active one."""

        
        if aiState["final"]:
            self.final = True
        else:
            self.final = False

        if (mode == None) or (mode == "crf"):
            if reset:
                self.crfAi = SpecCrfAi(device = self.device, learningRate=self.hparams["crf_lr"], regularization=self.hparams["crf_reg"], hiddenLayerCount=self.hparams["crf_hlc"], hiddenLayerSize=self.hparams["crf_hls"])
                if self.final:
                    self.crfAiOptimizer = torch.optim.NAdam(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=self.crfAi.regularization)
            self.crfAi.epoch = aiState['crfAi_epoch']
            self.crfAi.sampleCount = aiState["crfAi_sampleCount"]
            self.crfAi.load_state_dict(aiState['crfAi_model_state_dict'])
            if self.final:
                self.crfAiOptimizer.load_state_dict(aiState['crfAi_optimizer_state_dict'])
        if (mode == None) or (mode == "pred"):
            if reset:
                self.predAi = SpecPredAi(device = self.device, learningRate=self.hparams["pred_lr"], regularization=self.hparams["pred_reg"], recSize=self.hparams["pred_rs"])
                self.predAiHarm = HarmPredAi(device = self.device, learningRate=self.hparams["predh_lr"], regularization=self.hparams["predh_reg"], recSize=self.hparams["predh_rs"])
                if self.final:
                    self.predAiOptimizer = torch.optim.Adadelta(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
                    self.predAiHarmOptimizer = torch.optim.Adadelta(self.predAiHarm.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
            self.predAi.epoch = aiState["predAi_epoch"]
            self.predAi.sampleCount = aiState["predAi_sampleCount"]
            self.predAi.load_state_dict(aiState['predAi_model_state_dict'])
            self.predAiHarm.load_state_dict(aiState['predAiHarm_model_state_dict'])
            if self.final:
                self.predAiOptimizer.load_state_dict(aiState['predAi_optimizer_state_dict'])
                self.predAiHarmOptimizer.load_state_dict(aiState['predAiHarm_optimizer_state_dict'])
        self.crfAi.eval()
        self.predAi.eval()

    def interpolate(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, outputSize:int, pitchCurve:torch.Tensor) -> torch.Tensor:
        """forward pass of both NNs for generating a transition between two phonemes, with data pre- and postprocessing
        
        Arguments:
            specharm1-4: The four specharm Tensors to perform the interpolation between
            
            outputSize: length of the generated transition in engine ticks.

            pitchCurve: Tensor containing the pitch curve during the transition. Used to correctly calculate harmonic amplitudes during the transition.
            
        Returns:
            tuple of two Tensor objects, containing the interpolated audio spectrum without and with the prediction Ai applied to it, respectively."""

        
        self.crfAi.eval()
        self.predAi.eval()
        self.predAiHarm.eval()
        self.crfAi.requires_grad_(False)
        self.predAi.requires_grad_(False)
        self.predAiHarm.requires_grad_(False)
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
        spectrum = torch.squeeze(self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, outputSize)).transpose(0, 1)
        for i in self.defectiveCrfBins:
            spectrum[:, i] = torch.mean(torch.cat((spectrum[:, i - 1].unsqueeze(1), spectrum[:, i + 1].unsqueeze(1)), 1), 1)
        borderRange = torch.zeros((outputSize,), device = self.device)
        borderLimit = min(global_consts.crfBorderAbs, math.ceil(outputSize * global_consts.crfBorderRel))
        borderRange[:borderLimit] = torch.linspace(1, 0, borderLimit, device = self.device)
        spectrum *= (1. - borderRange.unsqueeze(1))
        spectrum += torch.matmul(borderRange.unsqueeze(1), ((spectrum1 + spectrum2) / 2).unsqueeze(0))
        borderRange = torch.flip(borderRange, (0,))
        spectrum *= (1. - borderRange.unsqueeze(1))
        spectrum += torch.matmul(borderRange.unsqueeze(1), ((spectrum3 + spectrum4) / 2).unsqueeze(0))
        phases = torch.empty(outputSize, phase1.size()[0], device = self.device)
        nativePitch = math.ceil(global_consts.tripleBatchSize / pitchCurve[0])
        originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.device))
        factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.device), (spectrum1 + spectrum2) / 2, originSpace)
        harmsStart = (harm1 + harm2) * 0.5 / factors
        nativePitch = math.ceil(global_consts.tripleBatchSize / pitchCurve[-1])
        originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.device))
        factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.device), (spectrum3 + spectrum4) / 2, originSpace)
        harmsEnd = (harm3 + harm4) * 0.5 / factors
        harms = torch.empty((outputSize, halfHarms), device = self.device)
        harmLimit = torch.max(torch.cat((harm1.unsqueeze(1), harm2.unsqueeze(1), harm3.unsqueeze(1), harm4.unsqueeze(1)), 1))
        for i in range(outputSize):
            phases[i] = phaseInterp(phaseInterp(phase1, phase2, 0.5), phaseInterp(phase3, phase4, 0.5), i / (outputSize - 1))
            nativePitch = math.ceil(global_consts.tripleBatchSize / pitchCurve[i])
            originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.device))
            factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.device), spectrum[i], originSpace)
            harms[i] = ((1. - (i + 1) / (outputSize + 1)) * harmsStart + (i + 1) / (outputSize + 1) * harmsEnd) * factors
            harms[i] = torch.min(harms[i], harmLimit)
            harms[i] = torch.max(harms[i], torch.tensor([0.,], device = self.device))
        output = torch.cat((harms, phases, spectrum), 1)
        predSpectrum = self.predAi(spectrum)
        predSpectrum = torch.max(predSpectrum, torch.tensor([0.0001,], device = self.device))
        predHarms = self.predAiHarm(harms)
        predHarms = torch.max(predHarms, torch.tensor([0.0001,], device = self.device))
        prediction = torch.cat((predHarms, phases, predSpectrum), 1)
        return output, torch.squeeze(prediction)

    def predict(self, specharm:torch.Tensor):
        """forward pass through the prediction Ai, taking a specharm as input and predicting the next one in a sequence. Includes data pre- and postprocessing."""

        self.predAi.eval()
        self.predAiHarm.eval()
        self.predAi.requires_grad_(False)
        self.predAiHarm.requires_grad_(False)
        if specharm.dim() == 1:
            specharm = specharm.unsqueeze(0)
        phases = specharm[:, halfHarms:2 * halfHarms]
        spectrum = specharm[:, 2 * halfHarms:]
        harms = specharm[:, :halfHarms]
        predSpectrum = self.predAi(spectrum)
        predSpectrum = torch.max(predSpectrum, torch.tensor([0.0001,], device = self.device))
        predHarms = self.predAiHarm(harms)
        predHarms = torch.max(predHarms, torch.tensor([0.0001,], device = self.device))
        prediction = torch.cat((predHarms, phases, predSpectrum), 1)
        return torch.squeeze(prediction)

    def reset(self) -> None:
        """resets the hidden states and cell states of the AI's LSTM layers."""

        self.predAi.resetState()
        self.predAiHarm.resetState()

    def finalize(self):
        self.final = True

    def trainCrf(self, indata, epochs:int=1, logging:bool = False) -> None:
        """NN training with forward and backward passes, loss criterion and optimizer runs based on a dataset of spectral transition samples.
        
        Arguments:
            indata: Tensor, List or other Iterable containing sets of specharm data. Each element should represent a phoneme transition.
            
            epochs: number of epochs to use for training as Integer.
            
        Returns:
            None"""
        

        self.crfAi.train()
        if logging:
            writer = SummaryWriter(path.join(getenv("APPDATA"), "Nova-Vox", "Logs"))
            #writer.add_graph(self, (indata[0][0], indata[0][1], indata[0][-2], indata[0][-1], torch.tensor([0.5])))
        else:
            writer = None

        if (self.crfAi.epoch == 0) or self.crfAi.epoch == epochs:
            self.crfAi.epoch = epochs
        else:
            self.crfAi.epoch = None
        reportedLoss = 0.
        for epoch in range(epochs):
            for data in self.dataLoader(indata):
                data = data.to(device = self.device)
                data = torch.squeeze(data)
                spectrum1 = data[2, 2 * halfHarms:]
                spectrum2 = data[3, 2 * halfHarms:]
                spectrum3 = data[-2, 2 * halfHarms:]
                spectrum4 = data[-1, 2 * halfHarms:]
                outputSize = data.size()[0] - 2
                output = self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, outputSize).transpose(0, 1)
                target = data[2:, 2 * halfHarms:]
                loss = self.criterion(output, target)
                self.crfAiOptimizer.zero_grad()
                loss.backward()
                self.crfAiOptimizer.step()
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.data))
            if writer != None:
                writer.add_scalar("loss", loss.data)
            self.crfAi.sampleCount += len(indata)
            reportedLoss = (reportedLoss * 99 + loss.data) / 100
        criterion = torch.zeros((global_consts.halfTripleBatchSize + 1,), device = self.device)
        criterionSteps = 0
        with torch.no_grad():
            for data in self.dataLoader(indata):
                data = data.to(device = self.device)
                data = torch.squeeze(data)
                spectrum1 = data[2, 2 * halfHarms:]
                spectrum2 = data[3, 2 * halfHarms:]
                spectrum3 = data[-2, 2 * halfHarms:]
                spectrum4 = data[-1, 2 * halfHarms:]
                outputSize = data.size()[0] - 2
                output = self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, outputSize).transpose(0, 1)
                criterionA = torch.cat((torch.ones((outputSize, 1), device = self.device), output[:, 1:] / output[:, :-1]), 1)
                criterionB = torch.cat((output[:, :-1] / output[:, 1:], torch.ones((outputSize, 1), device = self.device)), 1)
                criterion += torch.mean(criterionA + criterionB, dim = 0)
                criterionSteps += 1
            criterion /= criterionSteps
            criterion = torch.less(criterion, torch.tensor([self.hparams["crf_def_thrh"],], device = self.device))
        self.defectiveCrfBins = criterion.to_sparse().coalesce().indices()
        print("defective Crf frequency bins:", self.defectiveCrfBins)
        hparams = dict()
        hparams["epochs"] = epochs
        hparams["learning rate"] = self.crfAi.learningRate
        hparams["hidden layer count"] = self.crfAi.hiddenLayerCount
        metrics = dict()
        metrics["acc. sample count"] = self.crfAi.sampleCount
        metrics["wtd. train loss"] = reportedLoss
        if writer != None:
            writer.add_hparams(hparams, metrics)
            writer.close()
    
    def trainPred(self, indata, epochs:int=1, logging:bool = False) -> None:
        """trains the NN based on a dataset of specharm sequences"""

        self.predAi.train()
        self.predAiHarm.train()
        if logging:
            writer = SummaryWriter(path.join(getenv("APPDATA"), "Nova-Vox", "Logs"))
            #writer.add_graph(self, (indata[0][0], indata[0][1], indata[0][-2], indata[0][-1], torch.tensor([0.5])))
        else:
            writer = None
        if (self.predAi.epoch == 0) or self.predAi.epoch == epochs:
            self.predAi.epoch = epochs
        else:
            self.predAi.epoch = None
        reportedLoss = 0.
        for epoch in range(epochs):
            for data in self.dataLoader(indata):
                data = torch.squeeze(data)
                self.reset()
                input = data[:-1, 2 * halfHarms:]
                target = data[1:, 2 * halfHarms:]
                input = torch.squeeze(input)
                target = torch.squeeze(target)
                output = self.predAi(input)
                loss = self.criterion(output.squeeze(), target)
                self.predAiOptimizer.zero_grad()
                loss.backward()
                self.predAiOptimizer.step()
                reportedLoss = (reportedLoss * 99 + loss.data) / 100
                input = data[:-1, :halfHarms]
                target = data[1:, :halfHarms]
                input = torch.squeeze(input)
                target = torch.squeeze(target)
                output = self.predAiHarm(input)
                loss = self.criterion(output.squeeze(), target)
                self.predAiHarmOptimizer.zero_grad()
                loss.backward()
                self.predAiHarmOptimizer.step()
                reportedLoss = (reportedLoss * 99 + loss.data) / 100
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.data))
            if writer != None:
                writer.add_scalar("loss", loss.data)
            self.predAi.sampleCount += len(indata)
        hparams = dict()
        hparams["epochs"] = epochs
        metrics = dict()
        metrics["wtd. train loss"] = reportedLoss
        if writer != None:
            writer.add_hparams(hparams, metrics)
            writer.close()

    def trainCrfDebug(self, indata, testdata, epochs:int=1, logging:bool = False) -> None:
        """NN training with forward and backward passes, loss criterion and optimizer runs based on a dataset of spectral transition samples. Additionally performs a validation pass using a separate dataset, and plots the result.
        
        Arguments:
            indata: Tensor, List or other Iterable containing sets of specharm data. Each element should represent a phoneme transition.

            testdata: Tensor, List or other Iterable containing sets of specharm data. Each element should represent a phoneme transition. Used for the validation pass after each training epoch.
            
            epochs: number of epochs to use for training as Integer.
            
        Returns:
            None"""
        

        self.crfAi.train()
        if logging:
            writer = SummaryWriter(path.join(getenv("APPDATA"), "Nova-Vox", "Logs"))
            #writer.add_graph(self, (indata[0][0], indata[0][1], indata[0][-2], indata[0][-1], torch.tensor([0.5])))
        else:
            writer = None

        if (self.crfAi.epoch == 0) or self.crfAi.epoch == epochs:
            self.crfAi.epoch = epochs
        else:
            self.crfAi.epoch = None
        reportedLoss = 0.
        trainLossSpec = []
        testLossSpec = []
        trainLossSpecStd = []
        testLossSpecStd = []
        for epoch in range(epochs):
            trainLossSpecLocal = []
            testLossSpecLocal = []
            for data in self.dataLoader(indata):
                data = data.to(device = self.device)
                data = torch.squeeze(data)
                spectrum1 = data[2, 2 * halfHarms:]
                spectrum2 = data[3, 2 * halfHarms:]
                spectrum3 = data[-2, 2 * halfHarms:]
                spectrum4 = data[-1, 2 * halfHarms:]
                
                outputSize = data.size()[0] - 2
                output = self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, outputSize).transpose(0, 1)
                target = data[2:, 2 * halfHarms:]
                loss = self.criterion(output, target)
                trainLossSpecLocal.append(loss.data.item())
                self.crfAiOptimizer.zero_grad()
                loss.backward()
                self.crfAiOptimizer.step()
                print('epoch [{}/{}], train loss:{:.4f}'.format(epoch + 1, epochs, loss.data))
            for data in self.dataLoader(testdata):
                data = data.to(device = self.device)
                data = torch.squeeze(data)
                spectrum1 = data[2, 2 * halfHarms:]
                spectrum2 = data[3, 2 * halfHarms:]
                spectrum3 = data[-2, 2 * halfHarms:]
                spectrum4 = data[-1, 2 * halfHarms:]
                
                outputSize = data.size()[0] - 2
                output = self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, outputSize).transpose(0, 1)
                target = data[2:, 2 * halfHarms:]
                loss = self.criterion(output, target)
                testLossSpecLocal.append(loss.data.item())
                print('epoch [{}/{}], test loss:{:.4f}'.format(epoch + 1, epochs, loss.data))
            if writer != None:
                writer.add_scalar("loss", loss.data)
            self.crfAi.sampleCount += len(indata)
            reportedLoss = (reportedLoss * 99 + loss.data) / 100

            from numpy import array, std, mean
            trainLossSpecLocal = array(trainLossSpecLocal)
            testLossSpecLocal = array(testLossSpecLocal)
            trainLossSpec.append(mean(trainLossSpecLocal))
            testLossSpec.append(mean(testLossSpecLocal))
            trainLossSpecStd.append(std(trainLossSpecLocal))
            testLossSpecStd.append(std(testLossSpecLocal))

        hparams = dict()
        hparams["epochs"] = epochs
        hparams["learning rate"] = self.crfAi.learningRate
        hparams["hidden layer count"] = self.crfAi.hiddenLayerCount
        metrics = dict()
        metrics["acc. sample count"] = self.crfAi.sampleCount
        metrics["wtd. train loss"] = reportedLoss
        if writer != None:
            writer.add_hparams(hparams, metrics)
            writer.close()
        import matplotlib.pyplot as plt
        plt.plot(trainLossSpec)
        plt.plot(testLossSpec)
        plt.show()
        plt.plot(trainLossSpecStd)
        plt.plot(testLossSpecStd)
        plt.show()
