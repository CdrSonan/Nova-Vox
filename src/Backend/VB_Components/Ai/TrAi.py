#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from typing import OrderedDict
from math import pi
import torch
import torch.nn as nn

from Backend.Resampler.PhaseShift import phaseInterp
import global_consts

halfHarms = int(global_consts.nHarmonics / 2) + 1


class TrAi(nn.Module):
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
        
        
    def __init__(self) -> None:
        """Constructor initialising NN layers and prerequisite attributes.
        
        Arguments:
            device: the device the AI is to be loaded on

            learningRate: desired learning rate of the NN as float. supports scientific format.
            
            hiddenLayerCount: number of hidden layers (between leading and trailing layers)

        Returns:
            None"""
            
            
        super(TrAi, self).__init__()
    
    def __new__(cls, device:torch.device = None, learningRate:float=5e-5, hiddenLayerCount:int = 3, hiddenLayerSize:int = 4 * (global_consts.halfTripleBatchSize + halfHarms), regularization:float=1e-5, compile:bool = False):
        instance = super().__new__(cls)
        if compile:
            instance = torch.compile(instance, dynamic = True, mode = "reduce-overhead")
        return instance
        
    def forward(self, spectrum1in:torch.Tensor, spectrum2in:torch.Tensor, spectrum3in:torch.Tensor, spectrum4in:torch.Tensor, embedding1:torch.Tensor, embedding2:torch.Tensor, factorIn:torch.Tensor) -> torch.Tensor:
        """Forward NN pass.
        
        Arguments:
            spectrum1-4: The sets of two spectrum Tensors to perform the interpolation between, preceding and following the transition that is to be calculated, respectively.
            
            factor: Float between 0 and 1 determining the "position" within the interpolation. When using a value of 0 the output will be extremely similar to spectrum 1 and 2,
            while a values of 1 will result in output extremely similar to spectrum 3 and 4.
            
        Returns:
            Tensor object representing the NN output"""
        
        #plain interpolation for debugging
        """length = factorIn.size()[0]
        space = torch.linspace(1, 0, length)
        result = torch.zeros((length, global_consts.frameSize))
        for i in range(length):
            result[i] = spectrum1in * space[i] + spectrum4in * (1 - space[i])
        return result"""
        
        
        outputSize = factorIn.size()[0]
        spectrum1 = torch.cat((spectrum1in[:global_consts.halfHarms], spectrum1in[global_consts.nHarmonics + 2:]), dim = 0)
        spectrum2 = torch.cat((spectrum2in[:global_consts.halfHarms], spectrum2in[global_consts.nHarmonics + 2:]), dim = 0)
        spectrum3 = torch.cat((spectrum3in[:global_consts.halfHarms], spectrum3in[global_consts.nHarmonics + 2:]), dim = 0)
        spectrum4 = torch.cat((spectrum4in[:global_consts.halfHarms], spectrum4in[global_consts.nHarmonics + 2:]), dim = 0)
        factor = torch.tile(factorIn.unsqueeze(-1).unsqueeze(-1), (1, 1, global_consts.reducedFrameSize))
        spectrum1 = torch.unsqueeze(spectrum1.to(self.device), 0)
        spectrum2 = torch.unsqueeze(spectrum2.to(self.device), 0)
        spectrum3 = torch.unsqueeze(spectrum3.to(self.device), 0)
        spectrum4 = torch.unsqueeze(spectrum4.to(self.device), 0)
        spectra = torch.cat((spectrum1, spectrum2, spectrum3, spectrum4), dim = 0)
        spectrum1tile = torch.tile(spectrum1.unsqueeze(0), (outputSize, 1, 1)) * (1. - factor)
        spectrum2tile = torch.tile(spectrum2.unsqueeze(0), (outputSize, 1, 1)) * (1. - factor)
        spectrum3tile = torch.tile(spectrum3.unsqueeze(0), (outputSize, 1, 1)) * factor
        spectrum4tile = torch.tile(spectrum4.unsqueeze(0), (outputSize, 1, 1)) * factor
        x = spectrum1tile + spectrum2tile + spectrum3tile + spectrum4tile
        x *= 0.5
        x = x.squeeze()
        phases1 = spectrum1in[global_consts.halfHarms:global_consts.nHarmonics + 2]
        phases2 = spectrum2in[global_consts.halfHarms:global_consts.nHarmonics + 2]
        phases3 = spectrum3in[global_consts.halfHarms:global_consts.nHarmonics + 2]
        phases4 = spectrum4in[global_consts.halfHarms:global_consts.nHarmonics + 2]
        phasesLeft = torch.tile((phases1 + phases2) / 2, (outputSize, 1))
        phasesRight = torch.tile((phases3 + phases4) / 2, (outputSize, 1))
        factorPhases = torch.tile(factorIn.unsqueeze(-1), (1, global_consts.halfHarms))
        phases = phaseInterp(phasesLeft, phasesRight, factorPhases)
        phases = torch.where(phases > pi, phases - 2 * pi, phases)
        
        return torch.cat((x[:, :global_consts.halfHarms], phases, x[:, global_consts.halfHarms:]), dim = 1)
