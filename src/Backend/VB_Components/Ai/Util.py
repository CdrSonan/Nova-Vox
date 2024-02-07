#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from math import sqrt, floor

import torch
import torch.nn as nn
from s5 import S5Block
import global_consts

halfHarms = int(global_consts.nHarmonics / 2) + 1

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
        out = torch.sum(torch.max(error - self.threshold, torch.tensor([0,], device = self.threshold.device))) * self.weight
        return out

def gradientPenalty(model, real, fake, phase, embedding, device):
    """calculates gradient penalty for a given model, real, and fake inputs.
    
    Arguments:
        model: model to calculate gradient penalty for
        
        real: real input to model
        
        fake: fake input to model
        
        device: device to run model on
        
    Returns:
        gradient penalty as Float"""
    
    limit = min(real.shape[0], fake.shape[0])
    alpha = torch.rand_like(real[:limit], device=device)
    interpolates = alpha * real[:limit] + ((1 - alpha) * fake[:limit])
    interpolates.requires_grad_(True)
    disc_interpolates = model(interpolates, phase, embedding)
    output = torch.ones_like(disc_interpolates, device=device)
    gradient = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=output, create_graph=True, retain_graph=True, only_inputs=True)[0]
    result = torch.pow(gradient.flatten().norm(2),  2).mean() * 10
    return result

def newEmbedding(currentEmbeddings:int, dim:int, device:torch.device) -> torch.Tensor:
    embedding = torch.zeros([dim,], device = device)
    power = 1. / (floor(currentEmbeddings / dim) + 1)
    embedding[currentEmbeddings % dim] = power
    return embedding

def specNormS5(s5:S5Block)->S5Block:
    """normalizes S5Block weights using a spectral norm.
    
    Arguments:
        s5: S5Block to normalize
        
    Returns:
        normalized S5Block"""
    
    s5.attn_norm = nn.utils.parametrizations.spectral_norm(s5.attn_norm)
    s5.ff_enc = nn.utils.parametrizations.spectral_norm(s5.ff_enc)
    s5.ff_dec = nn.utils.parametrizations.spectral_norm(s5.ff_dec)
    s5.ff_norm = nn.utils.parametrizations.spectral_norm(s5.ff_norm)
    s5.s5.seq = nn.utils.parametrizations.spectral_norm(s5.s5.seq, "B")
    s5.s5.seq = nn.utils.parametrizations.spectral_norm(s5.s5.seq, "C")
    s5.s5.seq = nn.utils.parametrizations.spectral_norm(s5.s5.seq, "D")
    s5.s5.seq = nn.utils.parametrizations.spectral_norm(s5.s5.seq, "Lambda")
    return s5
