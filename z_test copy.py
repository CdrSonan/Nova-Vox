import matplotlib.pyplot as plt
from Backend.Resampler.PhaseShift import phaseShift
import torch
from math import pi
func = torch.sin(torch.linspace(0, 6, 128))
plt.plot(func)
func = torch.fft.rfft(func)
absolutes = func.abs()
phases = func.angle()
phases = phaseShift(phases, pi / 7, torch.device("cpu"))
#phases = phaseShift(phases, pi / 2, torch.device("cpu"))
func2 = torch.fft.irfft(torch.polar(absolutes, phases))
plt.plot(func2)
plt.show()