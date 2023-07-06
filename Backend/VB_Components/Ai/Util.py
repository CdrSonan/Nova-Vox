#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

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
    
    
    def __init__(self, weight=None, size_average=True, threshold = 0.5, device = 'cpu'):
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
 
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> float:  
        """calculates relative loss based on input and target tensors after successful initialisation.
        
        Arguments:
            inputs: AI-generated input Tensor
            
            targets: target Tensor
            
        Returns:
            Relative error value calculated from the difference between input and target Tensor as Float"""
        
        error = (torch.abs(inputs - targets) + 0.001)# / (targets + 0.001)
        out = torch.mean(torch.max(error - self.threshold, torch.tensor([0,], device = self.threshold.device))) * self.weight
        return out

def gradientPenalty(model, real, fake, device):
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
    with torch.backends.cudnn.flags(enabled=False):
        disc_interpolates = model(interpolates)
    output = torch.ones_like(disc_interpolates, device=device)
    gradient = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=output, create_graph=True, retain_graph=True, only_inputs=True)[0]
    result = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    return result
