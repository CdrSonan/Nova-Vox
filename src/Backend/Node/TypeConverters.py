#Copyright 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch

import global_consts

def convertClampedFloat(*args):
    """converts a value to a float, clamping it to the range between -1 and 1"""

    return max(-1, min(1, float(*args)))

def convertFloat(*args):
    """converts a value to a float"""

    return float(*args)

def convertInt(*args):
    """converts a value to an int"""

    return int(*args)

def convertBool(*args):
    """converts a value to a boolean"""

    return bool(*args)

def convertESPERAudio(*args):
    """converts a value to a Torch tensor following the ESPER audio format"""

    if isinstance(args[0], torch.Tensor):
        out = args[0].flatten()
    else:
        out = torch.tensor(*args)
    assert out.shape[0] == global_consts.frameSize, "Invalid ESPER audio tensor shape"
    return out

def convertPhoneme(*args):
    """converts input(s) to a tuple describing a phoneme, or the transition between several phonemes"""
    
    if len(args) == 1:
        return (str(args[0]), str(args[0]), 0.)
    elif len(args) == 2:
        return (str(args[0]), str(args[1]), 0.5)
    elif len(args) == 3:
        return (str(args[0]), str(args[1]), float(args[2]))
    raise TypeError("Invalid number of arguments for phoneme conversion")

converters = [convertClampedFloat, convertFloat, convertInt, convertBool, convertESPERAudio, convertPhoneme] #list of all available converters, can be extended through addons

def getConverter(name:str):
    """returns the converter function for a given node data type class"""

    for c in converters:
        if c.__name__ == "convert" + name:
            return c
    return None