import matplotlib.pyplot as plt
from Backend.Resampler.PhaseShift import phaseInterp
import torch
from math import pi
funca = torch.full((10, 20), 1.)
funcb = torch.full((10, 20), 5.)
print(phaseInterp(funca, funcb, 0.3))
print(phaseInterp(funca, funcb, 0.5))
print(phaseInterp(funca, funcb, 0.7))