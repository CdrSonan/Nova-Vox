#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from typing import OrderedDict
import torch
import torch.nn as nn
import global_consts

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
        
    def forward(self, spectrum1:torch.Tensor, spectrum2:torch.Tensor, spectrum3:torch.Tensor, spectrum4:torch.Tensor, embedding1:torch.Tensor, embedding2:torch.Tensor, factor:torch.Tensor) -> torch.Tensor:
        """Forward NN pass.
        
        Arguments:
            spectrum1-4: The sets of two spectrum Tensors to perform the interpolation between, preceding and following the transition that is to be calculated, respectively.
            
            factor: Float between 0 and 1 determining the "position" within the interpolation. When using a value of 0 the output will be extremely similar to spectrum 1 and 2,
            while a values of 1 will result in output extremely similar to spectrum 3 and 4.
            
        Returns:
            Tensor object representing the NN output"""
        
        outputSize = factor.size()[0]
        factor = torch.tile(factor.unsqueeze(-1).unsqueeze(-1), (1, 1, global_consts.halfTripleBatchSize + 1))
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
        embedding = torch.cat((factor[:, :, :global_consts.halfTripleBatchSize + 1 - 64], torch.tile(embedding1[None, :], (outputSize, 1, 1)), torch.tile(embedding2[None, :], (outputSize, 1, 1))), dim = 2)
        x = torch.cat((spectrum3tile, spectrum4tile, embedding), dim = 1)
        x = x.float()
        x = torch.flatten(x, 1)
        x = x.unsqueeze(0)
        state = torch.flatten(torch.cat((spectrum1, spectrum2), 1), 1).unsqueeze(0)
        x, state = self.layerStart1a(x, state)
        embedding = torch.cat((1. - factor[:, :, :global_consts.halfTripleBatchSize + 1 - 64], torch.tile(embedding1[None, :], (outputSize, 1, 1)), torch.tile(embedding2[None, :], (outputSize, 1, 1))), dim = 2)
        y = torch.cat((spectrum1tile, spectrum2tile, embedding), dim = 1)
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