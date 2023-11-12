#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from math import sqrt, floor

import torch
import torch.nn as nn
import global_consts

halfHarms = int(global_consts.nHarmonics / 2) + 1


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

class GuideRelLoss(nn.Module):
    """function for calculating relative loss values between target and actual Tensor objects. Designed to be used with AI optimizers. Currently unused.
    
    Attributes:
        None
        
    Methods:
        __init__: basic class constructor
        
        forward: calculates relative loss based on input and target tensors after successful initialisation."""
    
    
    def __init__(self, weight=None, size_average=True, threshold = 0.05, device = 'cpu'):
        """basic class constructor.
        
        Arguments:
            weight: required by PyTorch in some situations. Unused.
            
            size_average: required by PyTorch in some situations. Unused.
            
        Returns:
            None"""
        
        
        super().__init__()
        if weight is None:
            self.weight = torch.tensor([1.,], device=device)
        else:
            self.weight = torch.tensor(weight, device=device)
        self.threshold = torch.tensor(threshold, device=device)
        self.exponent = torch.tensor(2, device=device)
 
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> float:  
        """calculates relative loss based on input and target tensors after successful initialisation.
        
        Arguments:
            inputs: AI-generated input Tensor
            
            targets: target Tensor
            
        Returns:
            Relative error value calculated from the difference between input and target Tensor as Float"""
        
        error = (torch.pow(inputs - targets, self.exponent) + 0.001) / (targets + 0.001)
        out = torch.mean(torch.max(error - self.threshold, torch.tensor([0,], device = self.threshold.device))) * self.weight
        return out

def gradientPenalty(model, real, fake, level, device):
    """calculates gradient penalty for a given model, real, and fake inputs.
    
    Arguments:
        model: model to calculate gradient penalty for
        
        real: real input to model
        
        fake: fake input to model
        
        device: device to run model on
        
    Returns:
        gradient penalty as Float"""
    
    limit = min(real.shape[0], fake.shape[0])
    alpha = torch.rand(1, 1, device=device)
    interpolates = alpha * real[:limit] + ((1 - alpha) * fake[:limit])
    interpolates.requires_grad_(True)
    disc_interpolates = model(interpolates, level)
    output = torch.ones_like(disc_interpolates, device=device)
    gradient = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=output, create_graph=True, retain_graph=True, only_inputs=True)[0]
    result = torch.pow(gradient.flatten().norm(2) - 1,  2).mean()
    return result

class HighwayLSTM(nn.Module):
    
    def __init__(self, input_size:int, hidden_size:int, dropout:float, device:torch.device, **kwargs) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, **kwargs, device = device)
        if "proj_size" in kwargs:
            highwayOut = kwargs["proj_size"]
        else:
            highwayOut = hidden_size
        self.highway = nn.Linear(input_size, highwayOut, bias = False, device = device)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch.Tensor, state:torch.Tensor) -> torch.Tensor:
        lstmOut = self.lstm(x.unsqueeze(0), state)
        return self.sigmoid(lstmOut[0].squeeze(0)) + self.dropout(self.highway(x)), lstmOut[1]

class SpecNormHighwayLSTM(HighwayLSTM):
    
    def __init__(self, input_size: int, hidden_size: int, dropout:float, device: torch.device, **kwargs) -> None:
        super().__init__(input_size, hidden_size, dropout, device, **kwargs)
        for i in self.lstm._all_weights:
            for j in i:
                self.lstm = nn.utils.parametrizations.spectral_norm(self.lstm, name = j)
        self.highway = nn.utils.parametrizations.spectral_norm(self.highway)

def init_weights_yilmaz_poli(weight:torch.Tensor) -> None:
    n_inputs = weight.size()[1]
    mean = max(-1, -8/n_inputs)
    var = 1 / sqrt(n_inputs)
    nn.init.normal_(weight, mean, var)

def init_weights_logistic(module:nn.Module) -> None:
    if isinstance(module, nn.Linear):
        #nn.init.xavier_uniform_(module.weight)
        init_weights_yilmaz_poli(module.weight)
    if isinstance(module, nn.Conv1d):
        #nn.init.xavier_uniform_(module.weight)
        init_weights_yilmaz_poli(module.weight)
    """if isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight != None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.out_proj.weight != None:
            nn.init.xavier_uniform_(module.out_proj.weight)
        nn.init.xavier_uniform_(module.q_proj_weight)
        nn.init.xavier_uniform_(module.k_proj_weight)
        nn.init.xavier_uniform_(module.v_proj_weight)
    if isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)"""

def init_weights_rectifier(module:nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

def init_weights_rectifier_leaky(module:nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight)
    

def norm_attention(attention:nn.MultiheadAttention) -> nn.MultiheadAttention:
    if attention.in_proj_weight != None:
        attention = nn.utils.parametrizations.spectral_norm(attention, name = "in_proj_weight")
    if attention.out_proj.weight != None:
        attention.out_proj = nn.utils.parametrizations.spectral_norm(attention.out_proj)
    attention = nn.utils.parametrizations.spectral_norm(attention, name = "q_proj_weight")
    attention = nn.utils.parametrizations.spectral_norm(attention, name = "k_proj_weight")
    attention = nn.utils.parametrizations.spectral_norm(attention, name = "v_proj_weight")
    return attention

class SpecNormLSTM(nn.Module):
    
    def __init__(self, input_size:int, hidden_size:int, dropout:float, device:torch.device, **kwargs) -> None:
        super().__init__()
        self.lstm = nn.RNN(input_size = input_size, hidden_size = hidden_size, **kwargs, device = device)
        for i in self.lstm._all_weights:
            for j in i:
                self.lstm = nn.utils.parametrizations.spectral_norm(self.lstm, name = j)
    
    def forward(self, x:torch.Tensor, state:torch.Tensor) -> torch.Tensor:
        return self.lstm(x, state)

def newEmbedding(currentEmbeddings:int, dim:int, device:torch.device) -> torch.Tensor:
    embedding = torch.zeros([dim,], device = device)
    power = 1. / (floor(currentEmbeddings / dim) + 1)
    embedding[currentEmbeddings % dim] = power
    return embedding
