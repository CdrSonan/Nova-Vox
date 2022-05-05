import matplotlib.pyplot as plt
import torch
import global_consts
from Backend.Resampler.Loop import loopSamplerVoicedExcitation as shift
from math import sin, pi

window = torch.hann_window(global_consts.tripleBatchSize)

data = torch.empty(3 * global_consts.batchSize)
for i in range(3 * global_consts.batchSize):
    data[i] = sin(i * 2 * pi / 150)
    #data[i] += 0.1 * sin(1 + i * pi /25)
    #data[i] -= 0.1 * sin(1 + i * pi /10)

plt.plot(data)

phaseAdvance = (2 * pi * global_consts.batchSize / 150) % (2 * pi)
phases = torch.linspace(0, (3 - 1) * phaseAdvance, 3)
phases = torch.remainder(phases, torch.tensor(2 * pi))
#data = torch.stft(data, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)

#plt.plot(data)

dataShift = shift(data, 6 * global_consts.batchSize, torch.tensor([0.5]), torch.full([global_consts.batchSize], 150), 150, phases, None)

plt.plot(dataShift)

plt.show()

#!!!use real pitch instead of fourier index-based pitch!!!