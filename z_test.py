import matplotlib.pyplot as plt
import torch
import global_consts
from Backend.Resampler.Loop import loopSamplerVoicedExcitation as shift
from math import sin, pi

window = torch.hann_window(global_consts.tripleBatchSize)

data = torch.empty(10 * global_consts.tripleBatchSize)
for i in range(10 * global_consts.tripleBatchSize):
    data[i] = sin(i * pi / 128)
    data[i] += 0.1 * sin(i * pi /32)

print(data.size())
plt.plot(data)

data = torch.stft(data, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)

dataShift = shift(data, 40 * global_consts.tripleBatchSize, torch.tensor([0.5]), 1, None)

print(dataShift.size())
plt.plot(dataShift)

plt.show()