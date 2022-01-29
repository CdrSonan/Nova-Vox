from kivy.properties import ObjectProperty
import torch
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.Param_Components.AiParams import AiParam

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
        self.vbPath = path
        self.notes = []
        self.sequence = []
        self.pitch = torch.full((1000,), 1., dtype = torch.half)
        self.breathiness = torch.full((1000,), 0, dtype = torch.half)
        self.steadiness = torch.full((1000,), 0, dtype = torch.half)
        self.loopOverlap = torch.tensor([], dtype = torch.half)
        self.loopOffset = torch.tensor([], dtype = torch.half)
        self.vibratoSpeed = torch.full((1000,), 0, dtype = torch.half)
        self.vibratoStrength = torch.full((1000,), 0, dtype = torch.half)
        self.usePitch = False
        self.useBreathiness = False
        self.useSteadiness = False
        self.useVibratoSpeed = False
        self.useVibratoStrength = False
        self.paramStack = []
        self.borders = [0, 1, 2]
    def generateCaps(self):
        noneList = []
        for i in self.phonemes:
            noneList.append(False)
        return [noneList, noneList]
    def to_sequence(self, length):
        caps = self.generateCaps(self)
        return VocalSequence(length, self.borders, self.phonemes, caps[0], caps[1], self.loopOffset, self.loopOverlap, self.pitch, self.steadiness, self.breathiness)

class Note:
    def __init__(self, xPos, yPos, start = 0, end = 1, reference = None):
        self.reference = ObjectProperty()
        self.reference = reference
        self.length = 100
        self.xPos = xPos
        self.yPos = yPos
        self.phonemeMode = False
        self.content = ""
        self.phonemeStart = start
        self.phonemeEnd = end