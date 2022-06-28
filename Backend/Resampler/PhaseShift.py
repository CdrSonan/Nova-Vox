import torch
import global_consts
import math

def phaseShiftFourier(inputTensor:torch.Tensor, pitch:torch.Tensor, phase:float, device:torch.device) -> torch.Tensor:
    """currently unused function for phase shifting an arbitrary signal in fourier space.
    
    Arguments:
        inputTensor: complex tensor representing the input
        
        pitch: the pitch of the f0 frequency, or other frequency the phase is to be taken from
        
        phase: the amount the phase is to be shifted by
        
        device: the device to perform the calculations on
        
    Retuens:
        tensor representing the phase shifted input. It is natively available on the device specified as argument"""


    absolutes = inputTensor.abs()
    phases = inputTensor.angle()
    phaseOffsets = torch.full(phases.size(), phase, device = device)
    phaseOffsets *= (torch.linspace(0, global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1) / pitch)
    phases += phaseOffsets
    return torch.polar(absolutes, phases)

def phaseShift(inputTensor:torch.Tensor, pitch:torch.Tensor, phase:float) -> torch.Tensor:
    """Function for phase shifting an arbitrary signal in time space.
    
    Arguments:
        inputTensor: complex tensor representing the input
        
        pitch: the pitch of the f0 frequency, or other frequency the phase is to be taken from
        
        phase: the amount the phase is to be shifted by
        
    Returns:
        tensor representing the phase shifted input. It is natively available on the device specified as argument"""


    phaseOffset = (global_consts.tripleBatchSize * 0.5 / math.pi) * (phase / pitch)
    return inputTensor[phaseOffset:]
    
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
