#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import global_consts

halfHarms = int(global_consts.nHarmonics / 2) + 1


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
        x = x.squeeze(dim = 0)
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
        x = x.squeeze(dim = 0)
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)

        return x

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        self.state = (torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device), torch.zeros(self.recLayerCount, 1, self.recSize, device = self.device))