import matplotlib.pyplot as plt
import torch
import global_consts
from Backend.Resampler.Loop import loopSamplerVoicedExcitation as shift
from math import sin, pi, floor

window = torch.hann_window(global_consts.tripleBatchSize)
freq = 100
size = 20
data = torch.empty(size * global_consts.batchSize)
for i in range(size * global_consts.batchSize):
    data[i] = sin(i * 2 * pi / freq)
    #data[i] += 0.1 * sin(1 + i * pi /25)
    #data[i] -= 0.1 * sin(1 + i * pi /10)

pitchDeltas = torch.full([size,], freq)
phaseAdvance = (2 * pi * global_consts.batchSize / freq)
phases = torch.linspace(0, (size - 1) * phaseAdvance, size)

newPhases = torch.empty([size,], dtype = torch.float64)
for i in range(pitchDeltas.size()[0]):
    func = data[i * global_consts.batchSize:(i + 1) * global_consts.batchSize]
    #arange = torch.arange(0, global_consts.batchSize, 2 * pi / pitchDeltas[i])
    arange = torch.linspace(0, (global_consts.batchSize - 1) * 2 * pi / pitchDeltas[i], global_consts.batchSize)
    sine = torch.sin(arange)
    cosine = torch.cos(arange)
    sine *= func
    cosine *= func
    sine = torch.sum(sine)# / pi
    cosine = torch.sum(cosine)# / pi
    newPhases[i] = torch.complex(sine, cosine).angle()
    if i > 0:
        newPhases[i] += floor(global_consts.batchSize / pitchDeltas[i - 1])
print(phases, newPhases)
