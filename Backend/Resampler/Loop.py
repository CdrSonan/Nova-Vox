import math
import torch
import global_consts
import Backend.Resampler.PhaseShift
phaseShift = Backend.Resampler.PhaseShift.phaseShift

def loopSamplerVoicedExcitation(inputTensor, targetSize, repetititionSpacing, pitch, device):
    inputTensor = inputTensor.to(device = device)
    repetititionSpacing = repetititionSpacing.to(device = device)
    batchRS = math.ceil(repetititionSpacing * inputTensor.size()[0] / 2)
    repetititionSpacing = int(repetititionSpacing * math.ceil(inputTensor.size()[0] / 2))
    window = torch.hann_window(global_consts.tripleBatchSize, device = device)
    alignPhase = inputTensor[batchRS][pitch].angle()
    finalPhase = inputTensor[1][pitch].angle()
    phaseDiff = (finalPhase - alignPhase)
    requiredTensors = max(math.ceil((targetSize/global_consts.batchSize - batchRS) / (inputTensor.size()[1] - math.ceil(global_consts.tripleBatchSize / pitch) - batchRS)), 1)
    if requiredTensors <= 1:
        outputTensor = inputTensor.clone()
        #outputTensor = torch.istft(outputTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
    else:
        outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing) + repetititionSpacing, device = device)
            
        workingTensor = inputTensor.clone()
        #workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
        workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing, device = device)
        outputTensor[0:inputTensor.size()[1] * global_consts.batchSize] += workingTensor

        for i in range(1, requiredTensors - 1):
            workingTensor = inputTensor.clone()
            #workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
            #workingTensor = phaseShift(workingTensor, pitch, i * phaseDiff, device = device)
            phaseOffset = (global_consts.tripleBatchSize * 0.5 / math.pi) * (torch.remainder(i * phaseDiff, 2 * math.pi) / pitch)
            workingTensor = workingTensor[phaseOffset:]
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing, device = device)
            outputTensor[i * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing):i * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing) + inputTensor.size()[1] * global_consts.batchSize] += workingTensor #account for shrinking

        workingTensor = inputTensor.clone()
        #workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
        #workingTensor = phaseShift(workingTensor, pitch, (requiredTensors - 1) * phaseDiff, device = device)
        phaseOffset = (global_consts.tripleBatchSize * 0.5 / math.pi) * (torch.remainder((requiredTensors - 1) * phaseDiff, 2 * math.pi) / pitch)
        workingTensor = workingTensor[phaseOffset:]
        workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
        outputTensor[(requiredTensors - 1) * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing):] += workingTensor #account for shrinking
    return outputTensor[0:targetSize * global_consts.batchSize]
    
def loopSamplerSpectrum(inputTensor, targetSize, repetititionSpacing, device):
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