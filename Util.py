#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from math import floor
import torch
import global_consts

def ensureTensorLength(tensor:torch.Tensor, length:int, fill:float) -> torch.Tensor:
    """helper function for ensuring a tensor has a certain length along dim 0, and padding/pruning it if it does not have the correct length"""

    lengthDelta = length - tensor.size()[0]
    if lengthDelta < 0:
        return tensor[:length]
    if lengthDelta > 0:
        return torch.cat((tensor, torch.full((lengthDelta, *tensor.size()[1:]), fill, dtype = tensor.dtype, device = tensor.device)), 0)
    return tensor

def noteToPitch(data:torch.Tensor) -> torch.Tensor:
    """Utility function for converting the y position of a note to its corresponding pitch, following the MIDI standard."""

    #return torch.full_like(data, global_consts.sampleRate) / (torch.pow(2, (data - torch.full_like(data, 69)) / torch.full_like(data, 12)) * 440)
    return torch.full_like(data, global_consts.sampleRate) / (torch.pow(2, (data - torch.full_like(data, 69 - 12)) / torch.full_like(data, 12)) * 440)

def binarySearch(array, expression, length) -> int:
    """performs a binary search across array, returning the index of the first element where expression evaluates to True.
    length is the highest index that will be checked, exclusive."""
    
    low = 0
    high = length
    while low < high - 1:
        mid = floor((low + high) / 2)
        if expression(array, mid):
            high = mid
        else:
            low = mid
    return high