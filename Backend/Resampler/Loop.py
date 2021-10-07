import math
import torch
import global_consts
import Backend.Resampler.PhaseShift
phaseShift = Backend.Resampler.PhaseShift.phaseShift

def loopSamplerVoicedExcitation(inputTensor, targetSize, repetititionSpacing, pitch, device):
    inputTensor = inputTensor.to(device = device)
    repetititionSpacing = repetititionSpacing.to(device = device)
    batchRS = math.ceil(repetititionSpacing * inputTensor.size()[1] / 2)
    repetititionSpacing = int(repetititionSpacing * global_consts.batchSize * math.ceil(inputTensor.size()[1] / 2) - math.ceil(global_consts.tripleBatchSize / pitch / 2))
    window = torch.hann_window(global_consts.tripleBatchSize, device = device)
    alignPhase = inputTensor[pitch, int(batchRS/2)].angle()
    finalPhase = inputTensor[pitch, -int(batchRS/2)].angle()
    phaseDiff = (alignPhase - finalPhase)
    requiredTensors = max(math.ceil((targetSize/global_consts.batchSize - batchRS) / (inputTensor.size()[1] - (3 / pitch) - batchRS)), 1)

    inputTensor = torch.istft(inputTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)

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
            phaseOffset = int(global_consts.tripleBatchSize / (2 * math.pi) * torch.remainder(i * phaseDiff, 2 * math.pi) / pitch) #BUGGED, related to z test input data size
            print(phaseOffset, math.ceil(global_consts.tripleBatchSize / pitch / 2), torch.remainder(i * phaseDiff, 2 * math.pi) / pitch, phaseDiff)
            workingTensor = workingTensor[phaseOffset:]
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing, device = device)
            outputTensor[cursor:cursor + workingTensor.size()[0]] += workingTensor
            cursor += workingTensor.size()[0] - repetititionSpacing

        workingTensor = inputTensor.clone()
        phaseOffset = int(global_consts.tripleBatchSize / (2 * math.pi) * torch.remainder(i * phaseDiff, 2 * math.pi) / pitch) #BUGGED, related to z test input data size
        print(phaseOffset, math.ceil(global_consts.tripleBatchSize / pitch / 2), torch.remainder(i * phaseDiff, 2 * math.pi) / pitch, phaseDiff)
        workingTensor = workingTensor[phaseOffset:]
        workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
        outputTensor[cursor:cursor + workingTensor.size()[0]] += workingTensor
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