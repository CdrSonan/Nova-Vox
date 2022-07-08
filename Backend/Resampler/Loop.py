import math
import torch
import global_consts
from Backend.Resampler.PhaseShift import calculatePhaseDiff
    
def loopSamplerSpecharm(inputTensor:torch.Tensor, targetSize:int, repetititionSpacing:float, device:torch.device) -> torch.Tensor:
    """loops the spectra sequence of an AudioSample to match the target size with configurable overlap.
    
    Arguments:
        inputTensor: Tensor representing the voiced excitation of the input sample
        
        targetSize: the desired length of the output in engine ticks
        
        repetititionSpacing: integer between 0 and 1. Represents the desired amount of overlap between sample instances. 0 indicates no overlap, 1 indicates the entire sample length is part of the overlaps to the previous or next samples.
        
        device: the device the calculations are to be performed on.
        
    Returns:
        Tensor of length requiredLength representing the looped voiced excitation signal. It is natively available on the device specified in the parameters.
        
    It is also suited for looping any kind of data that can be represented as a vector for each engine tick. """
    phases = inputTensor[:, global_consts.nHarmonics:2 * global_consts.nHarmonics]
    
    inputTensor = torch.cat((inputTensor[:, :global_consts.nHarmonics], inputTensor[:, 2 * global_consts.nHarmonics:]), 1)
    repetititionSpacing = math.ceil(repetititionSpacing * inputTensor.size()[0] / 2)
    requiredTensors = math.ceil((targetSize - repetititionSpacing) / (inputTensor.size()[0] - repetititionSpacing))
    if requiredTensors <= 1:
        outputTensor = inputTensor.to(device = device, copy = True)
    else:
        outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[0] - repetititionSpacing) + repetititionSpacing, inputTensor.size()[1], device = device)
        workingTensor = inputTensor.to(device = device, copy = True)
        workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing, device = device), 1)
        outputTensor[0:inputTensor.size()[0]] += workingTensor
        del workingTensor

        for i in range(1, requiredTensors - 1):
            workingTensor = inputTensor.to(device = device, copy = True)
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing, device = device), 1)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing, device = device), 1)
            outputTensor[i * (inputTensor.size()[0] - repetititionSpacing):i * (inputTensor.size()[0] - repetititionSpacing) + inputTensor.size()[0]] += workingTensor
            del workingTensor

        workingTensor = inputTensor.to(device = device, copy = True)
        workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing, device = device), 1)
        outputTensor[(requiredTensors - 1) * (inputTensor.size()[0] - repetititionSpacing):] += workingTensor
        del workingTensor

    #phase calculations here

    return torch.cat((outputTensor[:, :global_consts.nHarmonics], phases, outputTensor[:, 2 * global_consts.nHarmonics:]), 1)
