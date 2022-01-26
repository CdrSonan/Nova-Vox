import torch

class SequenceStatusControl:
    """
    def __init__(self):
        self.ai = torch.zeros(0)
        self.rs = torch.zeros(0)
    """
    def __init__(self, sequence):
        phonemeLength = sequence.phonemeLength
        self.ai = torch.zeros(phonemeLength)
        self.rs = torch.zeros(phonemeLength)

class Inputs:
    """
    def __init__(self):
        self.borders = torch.Tensor([0, 1, 2])
        self.startCaps = torch.zeros(0, dtype = torch.bool)
        self.endCaps = torch.zeros(0, dtype = torch.bool)
        self.phonemes = []
        self.offsets = torch.zeros(0)
        self.repetititionSpacing = torch.ones(0)
        self.pitch = torch.ones(0)
        self.steadiness = torch.zeros(0)
        self.breathiness = torch.zeros(0)
        self.aiParamInputs = []
    """
    def __init__(self, sequence):
        self.borders = sequence.borders
        self.phonemes = sequence.phonemes
        self.startCaps = sequence.startCaps
        self.endCaps = sequence.endCaps
        self.offsets = sequence.offsets
        self.repetititionSpacing = sequence.repetititionSpacing
        self.pitch = sequence.pitch
        self.steadiness = sequence.steadiness
        self.breathiness = sequence.breathiness
        self.aiParamInputs = sequence.aiParamInputs