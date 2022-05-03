import matplotlib.pyplot as plt
import torch
import global_consts
from Backend.Resampler.Loop import loopSamplerVoicedExcitation as shift
from math import sin, pi

window = torch.hann_window(global_consts.tripleBatchSize)

data = torch.empty(10 * global_consts.tripleBatchSize)
for i in range(10 * global_consts.tripleBatchSize):
    data[i] = sin(1 + i * pi / 100)
    data[i] += 0.3 * sin(i * pi /25)

plt.plot(data)

#data = torch.stft(data, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)

#plt.plot(data)

dataShift = shift(data, 80 * global_consts.tripleBatchSize, torch.tensor([0.5]), torch.full([global_consts.tripleBatchSize], 200), 200, None)

plt.plot(dataShift)

plt.show()

#!!!use real pitch instead of fourier index-based pitch!!!