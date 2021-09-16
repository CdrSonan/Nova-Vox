import torch

def phaseShift(self, inputTensor, pitch, phase):
    absolutes = inputTensor.abs()
    phases = inputTensor.angle()
    phaseOffsets = torch.full(phases.size(), phase / pitch)
    phases += phaseOffsets
    return torch.polar(absolutes, phases)