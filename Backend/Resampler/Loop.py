import math
import torch
import global_consts
from Backend.Resampler.PhaseShift import calculatePhaseDiff

def loopSamplerVoicedExcitation(inputTensor:torch.Tensor, targetSize:int, repetititionSpacing:float, pitch:torch.Tensor, pitchAvg:float, phases:torch.Tensor, device:torch.device) -> torch.Tensor:
    """loops the voiced excitation signal of an AudioSample to match the target size with configurable overlap and phase synchronisation between instances.
    
    Arguments:
        inputTensor: Tensor representing the voiced excitation of the input sample
        
        targetSize: the desired length of the output in 48kHz audio sample points (not engine ticks)
        
        repetititionSpacing: integer between 0 and 1. Represents the desired amount of overlap between sample instances. 0 indicates no overlap, 1 indicates the entire sample length is part of the overlaps.
        
        pitch: pitch tensor of the input sample. used for phase synchronisation.
        
        pitchAvg: average of the pitch tensor. Is not calculated in this function since it is already pre-calculated for every sample.
        
        phases: phase tensor of the input sample
        
        device: the device the calculations are to be performed on.
        
    Returns:
        Tensor of length requiredLength representing the looped voiced excitation signal. It is natively available on the device specified in the parameters to the previous or next samples."""


    inputTensor = inputTensor.to(device = device)
    repetititionSpacing = repetititionSpacing.to(device = device)
    batchRS = repetititionSpacing * inputTensor.size()[0] / 2 / global_consts.batchSize
    repetititionSpacing = int(repetititionSpacing * math.ceil(inputTensor.size()[0] / 2) ) - math.ceil(global_consts.tripleBatchSize / pitchAvg / 2)
    phaseDiff = calculatePhaseDiff(batchRS, phases, pitch)
    requiredTensors = max(math.ceil((targetSize/global_consts.batchSize) / ((inputTensor.size()[0] - repetititionSpacing) / global_consts.batchSize)), 1)

    if requiredTensors <= 1:
        outputTensor = inputTensor.clone()
    else:
        outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[0] - repetititionSpacing) + repetititionSpacing, device = device)
        workingTensor = inputTensor.clone()
        workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing, device = device)
        outputTensor[0:workingTensor.size()[0]] += workingTensor
        cursor = workingTensor.size()[0] - repetititionSpacing

        for i in range(1, requiredTensors - 1):
            workingTensor = inputTensor.clone()
            workingTensor = workingTensor[phaseDiff:]
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing, device = device)
            outputTensor[cursor:cursor + workingTensor.size()[0]] += workingTensor
            cursor += workingTensor.size()[0] - repetititionSpacing
        workingTensor = inputTensor.clone()
        workingTensor = workingTensor[phaseDiff:]
        workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
        outputTensor[cursor:cursor + workingTensor.size()[0]] += workingTensor
    return outputTensor[0:targetSize * global_consts.batchSize]

def loopSamplerVoicedExcitation(inputTensor:torch.Tensor, targetSize:int, repetititionSpacing:float, pitch:torch.Tensor, pitchAvg:float, phases:torch.Tensor, device:torch.device) -> torch.Tensor:
    """loops the voiced excitation signal of an AudioSample to match the target size with configurable overlap and phase synchronisation between instances.
    
    Arguments:
        inputTensor: Tensor representing the voiced excitation of the input sample
        
        targetSize: the desired length of the output in 48kHz audio sample points (not engine ticks)
        
        repetititionSpacing: integer between 0 and 1. Represents the desired amount of overlap between sample instances. 0 indicates no overlap, 1 indicates the entire sample length is part of the overlaps.
        
        pitch: pitch tensor of the input sample. used for phase synchronisation.
        
        pitchAvg: average of the pitch tensor. Is not calculated in this function since it is already pre-calculated for every sample.
        
        phases: phase tensor of the input sample
        
        device: the device the calculations are to be performed on.
        
    Returns:
        Tensor of length requiredLength representing the looped voiced excitation signal. It is natively available on the device specified in the parameters to the previous or next samples."""


    inputTensor = inputTensor.to(device = device)
    inputTensor = torch.istft(inputTensor, 3 * global_consts.nHarmonics, global_consts.nHarmonics, 3 * global_consts.nHarmonics, onesided = True)
    repetititionSpacing = repetititionSpacing.to(device = device)
    batchRS = repetititionSpacing * inputTensor.size()[0] / 2 / global_consts.batchSize
    repetititionSpacing = int(repetititionSpacing * math.ceil(inputTensor.size()[0] / 2) ) - math.ceil(global_consts.tripleBatchSize / pitchAvg / 2)
    phaseDiff = calculatePhaseDiff(batchRS, phases, pitch)
    requiredTensors = max(math.ceil(targetSize / (inputTensor.size()[0] - repetititionSpacing) * global_consts.nHarmonics / global_consts.batchSize), 1)

    if requiredTensors <= 1:
        outputTensor = inputTensor.clone()
    else:
        outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[0] - repetititionSpacing) + repetititionSpacing, device = device)
        workingTensor = inputTensor.clone()
        workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing, device = device)
        outputTensor[0:workingTensor.size()[0]] += workingTensor
        cursor = workingTensor.size()[0] - repetititionSpacing

        for i in range(1, requiredTensors - 1):
            workingTensor = inputTensor.clone()
            workingTensor = workingTensor[phaseDiff:]
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing, device = device)
            outputTensor[cursor:cursor + workingTensor.size()[0]] += workingTensor
            cursor += workingTensor.size()[0] - repetititionSpacing
        workingTensor = inputTensor.clone()
        workingTensor = workingTensor[phaseDiff:]
        workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
        outputTensor[cursor:cursor + workingTensor.size()[0]] += workingTensor
        outputTensor = torch.stft(outputTensor, global_consts.nHarmonics, global_consts.nHarmonics, global_consts.nHarmonics, onesided = True)
    return outputTensor[0:targetSize]
    
def loopSamplerSpectrum(inputTensor:torch.Tensor, targetSize:int, repetititionSpacing:float, device:torch.device) -> torch.Tensor:
    """loops the spectra sequence of an AudioSample to match the target size with configurable overlap.
    
    Arguments:
        inputTensor: Tensor representing the voiced excitation of the input sample
        
        targetSize: the desired length of the output in engine ticks
        
        repetititionSpacing: integer between 0 and 1. Represents the desired amount of overlap between sample instances. 0 indicates no overlap, 1 indicates the entire sample length is part of the overlaps to the previous or next samples.
        
        device: the device the calculations are to be performed on.
        
    Returns:
        Tensor of length requiredLength representing the looped voiced excitation signal. It is natively available on the device specified in the parameters.
        
    This function is also used for It is also suited for looping any kind of data that can be represented as a vector for each engine tick. """


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
    return outputTensor[0:targetSize]
