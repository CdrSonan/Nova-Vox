import torch

def phaseShift(inputTensor, pitch, phase, device):
    absolutes = inputTensor.abs()
    phases = inputTensor.angle()
    phaseOffsets = torch.full(phases.size(), phase / pitch, device = device)
    phases += phaseOffsets
    return torch.polar(absolutes, phases)