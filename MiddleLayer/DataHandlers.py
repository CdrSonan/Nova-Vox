from Backend.VB_Components.Voicebank import LiteVoicebank
import torch

class Parameter:
    def __init__(self, path):
        #self.nn = LiteParameter(path)
        self.curve = torch.tensor()
        self.name = ""
        self.enabled = True

class Track:
    def __init__(self, path):
        self.voicebank = LiteVoicebank(path)
        self.notes = []
        self.pitch = torch.tensor([])
        self.breathiness = torch.tensor([])
        self.steadiness = torch.tensor([])
        self.paramStack = []