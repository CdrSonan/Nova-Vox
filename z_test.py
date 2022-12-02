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
    data[i] = sin(4 + i * 2 * pi / freq)
    data[i] += 0.8 * sin(1 + i * 2 * pi / freq * 0.9)
    data[i] -= 0.5 * sin(1.3 + i * 2 * pi / freq * 5)

plt.plot(data)

pitchDeltas = torch.full([size,],freq)

#phaseAdvance = (2 * pi * global_consts.batchSize / freq)
#phases = torch.linspace(0, (20 - 1) * phaseAdvance, 20)

phases = torch.empty([size,], dtype = torch.float64)
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
    sine = torch.sum(sine)# / pi (would be required for normalization, but amplitude is irrelevant here, so normalization is not required)
    cosine = torch.sum(cosine)# / pi
    phase = torch.complex(sine, cosine).angle()
    #if i == 0 and torch.isclose(phase, torch.zeros([1,]), atol = 1e-7):#edge case occured with synthetic data and phase at i=0 being exactly 0. Might be able to be removed.
    #    phase = 0.
    if phase < 0:
        phase += 2 * pi
    offset = previousLimit
    if phase < previousPhase % (2 * pi):
        offset += 1
    phase += 2 * pi * offset
    phases[i] = phase
    previousLimit = limit + offset
    previousPhase = phase

dataShift = shift(data, 20 * global_consts.batchSize, torch.tensor([0.5]), torch.full([global_consts.batchSize], freq), freq, phases, None)

#plt.plot(dataShift)

#plt.show()
