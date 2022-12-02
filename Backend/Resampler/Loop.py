import math
import torch
import global_consts
from Backend.Resampler.PhaseShift import phaseInterp
    
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
    phases = inputTensor[:, int(global_consts.nHarmonics / 2) + 1:global_consts.nHarmonics + 2]
    
    inputTensor = torch.cat((inputTensor[:, :int(global_consts.nHarmonics / 2) + 1], inputTensor[:, global_consts.nHarmonics + 2:]), 1)
    repetititionSpacing = math.ceil(repetititionSpacing * inputTensor.size()[0] / 2)
    requiredTensors = math.ceil((targetSize - repetititionSpacing) / (inputTensor.size()[0] - repetititionSpacing))
    if requiredTensors <= 1:
        outputTensor = inputTensor.to(device = device, copy = True)
        outputPhases = phases.to(device = device, copy = True)
    else:
        outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[0] - repetititionSpacing) + repetititionSpacing, inputTensor.size()[1], device = device)
        outputPhases = torch.zeros(requiredTensors * (phases.size()[0] - repetititionSpacing) + repetititionSpacing, phases.size()[1], device = device)
        workingTensor = inputTensor.to(device = device, copy = True)
        workingPhases = phases.to(device = device, copy = True)
        workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing, device = device), 1)
        workingPhases[-repetititionSpacing:] = phaseInterp(workingPhases[-repetititionSpacing:], workingPhases[:repetititionSpacing], torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing, device = device), 1))
        outputTensor[0:inputTensor.size()[0]] += workingTensor
        outputPhases[0:inputTensor.size()[0]] += workingPhases
        del workingTensor

        for i in range(1, requiredTensors - 1):
            workingTensor = inputTensor.to(device = device, copy = True)
            workingPhases = phases.to(device = device, copy = True)
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing, device = device), 1)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing, device = device), 1)
            workingPhases[-repetititionSpacing:] = phaseInterp(workingPhases[-repetititionSpacing:], workingPhases[:repetititionSpacing], torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing, device = device), 1))
            workingPhases = workingPhases[repetititionSpacing:]
            outputTensor[i * (inputTensor.size()[0] - repetititionSpacing):i * (inputTensor.size()[0] - repetititionSpacing) + inputTensor.size()[0]] += workingTensor
            outputPhases[phases.size()[0] + (i - 1) * workingPhases.size()[0]:phases.size()[0] + i * workingPhases.size()[0]] += workingPhases
            del workingTensor

        workingTensor = inputTensor.to(device = device, copy = True)
        workingPhases = phases.to(device = device, copy = True)
        workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing, device = device), 1)
        workingPhases = workingPhases[repetititionSpacing:]
        outputTensor[(requiredTensors - 1) * (inputTensor.size()[0] - repetititionSpacing):] += workingTensor
        outputPhases[phases.size()[0] + (requiredTensors - 2) * workingPhases.size()[0]:] += workingPhases
        del workingTensor

    return torch.cat((outputTensor[:, :int(global_consts.nHarmonics / 2) + 1], outputPhases, outputTensor[:, int(global_consts.nHarmonics / 2) + 1:]), 1)
