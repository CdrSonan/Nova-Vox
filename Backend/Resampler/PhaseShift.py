import torch
import global_consts
def phaseShift(inputTensor, pitch, phase, device):
    absolutes = inputTensor.abs()
    phases = inputTensor.angle()
    phaseOffsets = torch.full(phases.size(), phase, device = device)
    phaseOffsets *= (torch.linspace(0, global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1) / pitch)
    phases += phaseOffsets
    return torch.polar(absolutes, phases)