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
previousLimit = 0
previousPhase = 0
for i in range(pitchDeltas.size()[0]):
    limit = floor(global_consts.batchSize / pitchDeltas[i])
    func = data[i * global_consts.batchSize:i * global_consts.batchSize + limit * pitchDeltas[i]]
    arange = torch.linspace(0, (limit * pitchDeltas[i] - 1) * 2 * pi / pitchDeltas[i], limit * pitchDeltas[i])
    sine = torch.sin(arange)
    cosine = torch.cos(arange)
    sine *= func
    cosine *= func
    sine = torch.sum(sine)# / pi
    cosine = torch.sum(cosine)# / pi
    phase = torch.complex(sine, cosine).angle()
    if phase < 0:
        phase += 2 * pi
    phase += 2 * pi * previousLimit
    phase += previousPhase
    newPhases[i] = phase
    previousLimit = limit
    previousPhase = phase
plt.plot(phases)
plt.plot(newPhases)
plt.show()