#Copyright 2023, 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from math import floor
from numpy import ndarray, array as np_array
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

def freqBinToHarmonic(freqBin:torch.Tensor, pitch:float) -> torch.Tensor:
    """Utility function for converting a frequency bin to a harmonic number, given the pitch of the note."""

    #xScale = torch.linspace(0, global_consts.sampleRate / 2, global_consts.halfTripleBatchSize + 1)
    #harmScale = torch.linspace(0, global_consts.nHarmonics / 2 * global_consts.sampleRate / pitch, int(global_consts.nHarmonics / 2) + 1)
    #freqBinStep = global_consts.sampleRate / global_consts.tripleBatchSize
    #harmStep = global_consts.sampleRate / pitch
    return freqBin * global_consts.tripleBatchSize / pitch

def harmonicToFreqBin(harmonic:torch.Tensor, pitch:float) -> torch.Tensor:
    """Utility function for converting a harmonic number to a frequency bin, given the pitch of the note."""

    return harmonic * pitch / global_consts.tripleBatchSize

def freqToFreqBin(freq:torch.Tensor) -> torch.Tensor:
    """Utility function for converting a frequency to a frequency bin."""

    return freq * global_consts.tripleBatchSize / global_consts.sampleRate

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

def convertFormat(array, arrayFormat:str = "list") -> list:
    """converts a torch tensor, numpy array or ndarray, or list to a different format. Can be "list", "numpy" or "torch"."""
    
    if array.__class__ == list:
        if arrayFormat == "list":
            return array
        if arrayFormat == "numpy":
            return np_array(array)
        if arrayFormat == "torch":
            return torch.tensor(array)
    elif array.__class__ == ndarray:
        if arrayFormat == "list":
            return array.tolist()
        if arrayFormat == "numpy":
            return array
        if arrayFormat == "torch":
            return torch.from_numpy(array)
    else:
        if arrayFormat == "list":
            return array.cpu().tolist()
        if arrayFormat == "numpy":
            return array.cpu().numpy()
        if arrayFormat == "torch":
            return array

def dec2bin(x:torch.Tensor, bits:int) -> torch.Tensor:
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b:torch.Tensor, bits:int) -> torch.Tensor:
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

class SecureDict(dict):
    
    def __init__(self, *args, default = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = default
    
    def __getitem__(self, __key):
        return super().__getitem__(__key) if __key in self else (self.default if self.default is not None else __key)

def classesinmodule(module):
    """utility function for getting all classes in a Python module"""

    md = module.__dict__
    return [
        md[c] for c in md if (
            isinstance(md[c], type) and md[c].__module__ == module.__name__
        )
    ]
