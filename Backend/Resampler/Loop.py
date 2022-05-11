import math
import torch
import global_consts
import Backend.Resampler.PhaseShift
phaseShift = Backend.Resampler.PhaseShift.phaseShift

def loopSamplerVoicedExcitation(inputTensor, targetSize, repetititionSpacing, pitch, pitchavg, phases, device):
    inputTensor = inputTensor.to(device = device)
    repetititionSpacing = repetititionSpacing.to(device = device)
    batchRS = repetititionSpacing * inputTensor.size()[0] / 2 / global_consts.batchSize
    repetititionSpacing = int(repetititionSpacing * math.ceil(inputTensor.size()[0] / 2) )# - math.ceil(global_consts.tripleBatchSize / pitchavg / 2))
    #window = torch.hann_window(global_consts.tripleBatchSize, device = device)
    """counter = 0
    for i in pitch:
        counter += i
        if counter > 0.5 * repetititionSpacing:
            alignPhase = counter - (0.5 * repetititionSpacing)
            break
    for i in pitch:
        counter += i
        if counter > inputTensor.size()[0] - (0.5 * repetititionSpacing):
            finalPhase = counter - inputTensor.size()[0] + (0.5 * repetititionSpacing)
            break"""
    #refactor to util file; shared with ESPER/SpectralCalculator
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
    #alignPhase = phases[int(batchRS / 2)]
    #finalPhase = phases[int(-batchRS / 2) - 1]

    phaseDiff = int(((finalPhase - alignPhase) % (2 * math.pi)) / (2 * math.pi) * interpolatedPitch)
    requiredTensors = max(math.ceil((targetSize/global_consts.batchSize) / (inputTensor.size()[0] / global_consts.batchSize)), 1)
    #inputTensor = torch.istft(inputTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)

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
            #phaseOffset = int(global_consts.tripleBatchSize / (2 * math.pi) * torch.remainder(i * phaseDiff, 2 * math.pi) / pitch) #BUGGED, related to z test input data size
            #print(phaseOffset, math.ceil(global_consts.tripleBatchSize / pitch / 2), torch.remainder(i * phaseDiff, 2 * math.pi) / pitch, phaseDiff)
            workingTensor = workingTensor[phaseDiff:]
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing, device = device)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing, device = device)
            #import matplotlib.pyplot as plt
            #plt.plot(torch.arange(cursor, cursor + workingTensor.size()[0]), workingTensor)
            outputTensor[cursor:cursor + workingTensor.size()[0]] += workingTensor
            cursor += workingTensor.size()[0] - repetititionSpacing
        #plt.show()
        workingTensor = inputTensor.clone()
        #phaseOffset = int(global_consts.tripleBatchSize / (2 * math.pi) * torch.remainder(i * phaseDiff, 2 * math.pi) / pitch) #BUGGED, related to z test input data size
        #print(phaseOffset, math.ceil(global_consts.tripleBatchSize / pitch / 2), torch.remainder(i * phaseDiff, 2 * math.pi) / pitch, phaseDiff)
        workingTensor = workingTensor[phaseDiff:]
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
    print(outputTensor[0:targetSize].size(), targetSize)
    return outputTensor[0:targetSize]