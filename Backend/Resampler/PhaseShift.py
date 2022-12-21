#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
import global_consts
import math

def phaseShiftFourier(inputTensor:torch.Tensor, phase:float, device:torch.device) -> torch.Tensor:
    """function for phase shifting an arbitrary signal in fourier space.
    
    Arguments:
        inputTensor: complex tensor representing the input
        
        phase: the amount the phase is to be shifted by
        
        device: the device to perform the calculations on
        
    Retuens:
        tensor representing the phase shifted input. It is natively available on the device specified as argument"""


    absolutes = inputTensor.abs()
    phases = inputTensor.angle()
    phaseOffsets = torch.full(phases.size(), phase, device = device)
    phaseOffsets *= torch.linspace(0, int(global_consts.nHarmonics / 2), int(global_consts.nHarmonics / 2) + 1)
    phases += phaseOffsets
    return torch.polar(absolutes, phases)

def phaseShift(inputTensor:torch.Tensor, phase:float, device:torch.device) -> torch.Tensor:
    """function for shifting a tensor of only phases (in radians) by a configurable amount of time, relative to the lowest frequency.
    
    Arguments:
        inputTensor: float tensor representing the input
        
        phase: the amount the phase is to be shifted by, given relative to the lowest frequency (the first element) of the input
        
        device: the device to perform the calculations on
        
    Returns:
        tensor representing the phase shifted input. It is natively available on the device specified as argument"""


    phaseOffsets = torch.full(inputTensor.size(), phase, device = device)
    phaseOffsets *= torch.linspace(0, int(global_consts.nHarmonics / 2), int(global_consts.nHarmonics / 2) + 1)
    outputTensor = inputTensor + phaseOffsets
    outputTensor = torch.remainder(outputTensor, torch.tensor([2 * math.pi]))
    return outputTensor
    
def calculatePhaseDiff(batchRS:int, phases:torch.Tensor, pitch:torch.Tensor) -> int:
    """function for calculating the phase difference between the end of one and the beginning of another instance of the same sample. Used during voiced excitation looping.
    
    Arguments:
        batchRS: repetitition spacing used for looping, given as a number of audio batches/engine ticks
        
        phases: phase tensor of the sample being looped
        
        pitch: pitch tensor of the sample being looped
        
    Returns:
        integer representing the required shift of the latter sample  instance in 48kHz audio sample points"""


    batchRS = max(batchRS, 1.)
    position = (batchRS / 2) - 0.5
    lowerBin = math.floor(position)
    upperBin = math.ceil(position)
    if lowerBin == upperBin:
        alignPhase = phases[lowerBin]
        interpolatedPitch = pitch[lowerBin]
    else:
        lowerFactor = 1 - position + lowerBin
        upperFactor = 1 - upperBin + position
        alignPhase = lowerFactor * phases[lowerBin] + upperFactor * phases[upperBin]
        interpolatedPitch = lowerFactor * pitch[lowerBin] + upperFactor * pitch[upperBin]

    position = (-batchRS / 2) - 0.5
    lowerBin = math.floor(position)
    upperBin = math.ceil(position)
    if lowerBin == upperBin:
        finalPhase = phases[lowerBin]
    else:
        lowerFactor = 1 - position + lowerBin
        upperFactor = 1 - upperBin + position
        finalPhase = lowerFactor * phases[lowerBin] + upperFactor * phases[upperBin]

    phaseDiff = int(((finalPhase - alignPhase) % (2 * math.pi)) / (2 * math.pi) * interpolatedPitch)
    return phaseDiff

def phaseInterp(phaseA: torch.Tensor, phaseB: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    diff = phaseB - phaseA
    diff = torch.remainder(diff, 2 * math.pi)
    diffB = diff - 2 * math.pi
    mask = torch.ge(diff.abs(), diffB.abs())
    diff[mask] = diffB[mask]
    return torch.remainder(phaseA + factor * diff, 2 * math.pi)
