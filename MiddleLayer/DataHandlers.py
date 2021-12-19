from Backend.VB_Components.Voicebank import LiteVoicebank
import torch

class Parameter:
    def __init__(self, path):
        #self.nn = LiteParameter(path)
        self.curve = torch.full([1000], 0)
        self.name = ""
        self.enabled = True

class Track:
    def __init__(self, path):
        self.voicebank = LiteVoicebank(path)
        self.notes = []
        self.sequence = []
        self.pitch = torch.full((1000,), 0, dtype = torch.half)
        self.breathiness = torch.full((1000,), 0, dtype = torch.half)
        self.steadiness = torch.full((1000,), 0, dtype = torch.half)
        self.loopOverlap = torch.full((1000,), 0, dtype = torch.half)
        self.loopOffset = torch.full((1000,), 0, dtype = torch.half)
        self.vibratoSpeed = torch.full((1000,), 0, dtype = torch.half)
        self.vibratoStrength = torch.full((1000,), 0, dtype = torch.half)
        self.usePitch = False
        self.useBreathiness = False
        self.useSteadiness = False
        self.useVibratoSpeed = False
        self.useVibratoStrength = False
        self.paramStack = []

class Note:
    def __init__(self, xPos, yPos):
        self.length = 1
        self.xPos = xPos
        self.yPos = yPos
        self.phonemeMode = False
        self.content = ""