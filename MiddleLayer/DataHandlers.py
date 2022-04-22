from kivy.properties import ObjectProperty
import torch
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.Param_Components.AiParams import AiParam
import global_consts

class Parameter:
    def __init__(self, path):
        self.paramPath = path
        self.curve = torch.full([1000], 0)
        self.name = ""
        self.enabled = True
    def to_param(self, length):
        return AiParam(self.path)

class Track:
    def __init__(self, path):
        self.volume = 1.
        self.vbPath = path
        self.notes = []
        self.phonemes = []
        self.pitch = torch.full((5000,), -1., dtype = torch.half)
        self.basePitch = torch.full((5000,), -1., dtype = torch.half)
        self.breathiness = torch.full((5000,), 0, dtype = torch.half)
        self.steadiness = torch.full((5000,), 0, dtype = torch.half)
        self.loopOverlap = torch.tensor([], dtype = torch.half)
        self.loopOffset = torch.tensor([], dtype = torch.half)
        self.vibratoSpeed = torch.full((5000,), 0, dtype = torch.half)
        self.vibratoStrength = torch.full((5000,), 0, dtype = torch.half)
        self.usePitch = True
        self.useBreathiness = True
        self.useSteadiness = True
        self.useVibratoSpeed = True
        self.useVibratoStrength = True
        self.pauseThreshold = 100
        self.mixinVB = None
        self.paramStack = []
        self.borders = [0, 1, 2]
        self.length = 5000
    def generateCaps(self):
        noneList = []
        for i in self.phonemes:
            noneList.append(False)
        return [noneList, noneList]
    def to_sequence(self):
        caps = self.generateCaps()
        pitch = torch.full_like(self.pitch, global_consts.sampleRate) / (torch.pow(2, (self.pitch - torch.full_like(self.pitch, 69)) / torch.full_like(self.pitch, 12)) * 440)
        return VocalSequence(self.length, self.borders, self.phonemes, caps[0], caps[1], self.loopOffset, self.loopOverlap, pitch, self.steadiness, self.breathiness)

class Note:
    def __init__(self, xPos, yPos, start = 0, end = 1, reference = None):
        self.reference = ObjectProperty()
        self.reference = reference
        self.length = 100
        self.xPos = xPos
        self.yPos = yPos
        self.phonemeMode = True
        self.content = ""
        self.phonemeStart = start
        self.phonemeEnd = end