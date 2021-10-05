import matplotlib.pyplot as plt
import torch
import global_consts
from Backend.Resampler.Loop import loopSamplerVoicedExcitation as shift
from math import sin, pi

print(torch.linspace(0, global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1))

data = torch.empty(global_consts.tripleBatchSize)
for i in range(global_consts.tripleBatchSize):
    data[i] = sin(i * pi / 128)
    data[i] += 0.1 * sin(i * pi /32)

dataShift = shift(data, 2 * global_consts.tripleBatchSize, torch.tensor([0.5]), 1, None)
print(data.size())
print(dataShift.size())
plt.plot(data)
plt.plot(dataShift)
plt.show()