import matplotlib.pyplot as plt
import torch
import global_consts
from Backend.Resampler.PhaseShift import phaseShift as shift
from math import sin, pi

print(torch.linspace(0, global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1))

data = torch.empty(global_consts.tripleBatchSize + 1)
for i in range(global_consts.tripleBatchSize + 1):
    data[i] = sin(i * pi / 128)
    data[i] += 0.1 * sin(i * pi /32)
plt.plot(data)
#plt.show()
fourier = torch.fft.rfft(data)
print(fourier)
#plt.plot(fourier)
#plt.show()
fourierShift = shift(fourier, 1, 0.5 * pi, None)
print(fourierShift)
#plt.plot(fourierShift)
#plt.show()
dataShift = torch.fft.irfft(fourierShift)
plt.plot(dataShift)
plt.show()