import math
import torch
import global_consts
import Backend.Resampler.PhaseShift
phaseShift = Backend.Resampler.PhaseShift.phaseShift

def loopSamplerVoicedExcitation(inputTensor, targetSize, repetititionSpacing, pitch):
    batchRS = math.ceil(repetititionSpacing * inputTensor.size()[1] / 2)
    repetititionSpacing = int(repetititionSpacing * global_consts.batchSize * math.ceil(inputTensor.size()[1] / 2))
    window = torch.hann_window(global_consts.tripleBatchSize)
    alignPhase = inputTensor[batchRS][pitch].angle()
    finalPhase = inputTensor[1][pitch].angle()
    phaseDiff = (finalPhase - alignPhase)
    requiredTensors = max(math.ceil((targetSize/global_consts.batchSize - batchRS) / (inputTensor.size()[1] - batchRS)), 1)
    if requiredTensors <= 1:
        outputTensor = inputTensor.clone()
        outputTensor = torch.istft(outputTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
    else:
        outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing) + repetititionSpacing)
            
        workingTensor = inputTensor.clone()
        workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
        workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing)
        outputTensor[0:inputTensor.size()[1] * global_consts.batchSize] += workingTensor

        for i in range(1, requiredTensors - 1):
            workingTensor = inputTensor.clone()
            workingTensor = phaseShift(workingTensor, pitch, i * phaseDiff)
            workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing)
            outputTensor[i * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing):i * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing) + inputTensor.size()[1] * global_consts.batchSize] += workingTensor

        workingTensor = inputTensor.clone()
        workingTensor = phaseShift(workingTensor, pitch, (requiredTensors - 1) * phaseDiff)
        workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
        workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing)
        outputTensor[(requiredTensors - 1) * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing):] += workingTensor
    return outputTensor[0:targetSize * global_consts.batchSize]
    
def loopSamplerSpectrum(inputTensor, targetSize, repetititionSpacing):
    repetititionSpacing = math.ceil(repetititionSpacing * inputTensor.size()[0] / 2)
    requiredTensors = math.ceil((targetSize - repetititionSpacing) / (inputTensor.size()[0] - repetititionSpacing))
    if requiredTensors <= 1:
        outputTensor = inputTensor.clone()
    else:
        outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[0] - repetititionSpacing) + repetititionSpacing, inputTensor.size()[1])
        workingTensor = inputTensor.clone()
        workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing), 1)
        outputTensor[0:inputTensor.size()[0]] += workingTensor
        del workingTensor

        for i in range(1, requiredTensors - 1):
            workingTensor = inputTensor.clone()
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing), 1)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing), 1)
            outputTensor[i * (inputTensor.size()[0] - repetititionSpacing):i * (inputTensor.size()[0] - repetititionSpacing) + inputTensor.size()[0]] += workingTensor
            del workingTensor

        workingTensor = inputTensor.clone()
        workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing), 1)
        outputTensor[(requiredTensors - 1) * (inputTensor.size()[0] - repetititionSpacing):] += workingTensor
        del workingTensor
    return outputTensor[0:targetSize]