#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.properties import ObjectProperty
import torch
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.Param_Components.AiParams import AiParam
from Backend.VB_Components.Voicebank import LiteVoicebank
import global_consts

class Parameter():
    """class for holding and managing a parameter curve as seen by the main process. Exact layout will change with node tree implementation."""

    def __init__(self, path:str) -> None:
        self.paramPath = path
        self.curve = torch.full([1000], 0)
        self.name = ""
        self.enabled = True
    def to_param(self, length:int) -> AiParam:
        return AiParam(self.paramPath)

class Track():
    """class for holding and managing a vocal track as seen by the main process. Contains all settings required for processing on the main process.
    
    Methods:
        __init__: constructor for a track with default settings and a Voicebank specified by a filepath
        
        generateCaps: utility function for generating startCap and endCap attributes for the track.
        
        toSequence: converts the track to a VOcalSequence object for sending it to the rendering thread"""


    def __init__(self, path:str) -> None:
        """constructor function for a track with default settings.
        
        Arguments:
            path: string or path object pointing to the voicebank file that should be used by the track."""


        self.volume = 1.
        self.vbPath = path
        self.notes = []
        self.phonemes = []
        self.pitch = torch.full((5000,), -1., dtype = torch.half)
        self.basePitch = torch.full((5000,), -1., dtype = torch.half)
        self.breathiness = torch.full((5000,), 0, dtype = torch.half)
        self.steadiness = torch.full((5000,), 0, dtype = torch.half)
        self.aiBalance = torch.full((5000,), 0, dtype = torch.half)
        self.loopOverlap = torch.tensor([], dtype = torch.half)
        self.loopOffset = torch.tensor([], dtype = torch.half)
        self.vibratoSpeed = torch.full((5000,), 0, dtype = torch.half)
        self.vibratoStrength = torch.full((5000,), 0, dtype = torch.half)
        self.usePitch = True
        self.useBreathiness = True
        self.useSteadiness = True
        self.useAIBalance = True
        self.useVibratoSpeed = True
        self.useVibratoStrength = True
        self.pauseThreshold = 100
        self.mixinVB = None
        self.paramStack = []
        self.borders = [0, 1, 2]
        self.length = 5000
        self.phonemeLengths = dict()
        tmpVb = LiteVoicebank(self.vbPath)
        for i in tmpVb.phonemeDict.keys():
            if tmpVb.phonemeDict[i].isPlosive:
                self.phonemeLengths[i] = tmpVb.phonemeDict[i].specharm.size()[0]
            else:
                self.phonemeLengths[i] = None
                
    #TODO: update to _autopause system
    def generateCaps(self) -> tuple([list, list]):
        """utility function for generating startCap and endCap attributes for the track. These are not required by the main process, and are instead sent to the rendering process with to_sequence"""

        noneList = []
        for i in self.phonemes:
            noneList.append(False)
        return [noneList, noneList]

    def toSequence(self) -> VocalSequence:
        """converts the track to a VocalSequence object for sending it to the rendering thread. Also handles conversion of MIDI pitch to frequency."""

        caps = self.generateCaps()
        pitch = torch.full_like(self.pitch, global_consts.sampleRate) / (torch.pow(2, (self.pitch - torch.full_like(self.pitch, 69)) / torch.full_like(self.pitch, 12)) * 440)
        return VocalSequence(self.length, self.borders, self.phonemes, caps[0], caps[1], self.loopOffset, self.loopOverlap, pitch, self.steadiness, self.breathiness, self.aiBalance, self.vibratoSpeed, self.vibratoStrength, self.useBreathiness, self.useSteadiness, self.useAIBalance, self.useVibratoSpeed, self.useVibratoStrength)

class Note():
    """Container class for a note as handled by the main process. Contains a reference property pointing at its UI representation."""

    def __init__(self, xPos, yPos, start = 0, end = 1, reference = None) -> None:
        self.reference = ObjectProperty()
        self.reference = reference
        self.length = 100
        self.xPos = xPos
        self.yPos = yPos
        self.phonemeMode = True
        self.content = ""
        self.phonemeStart = start
        self.phonemeEnd = end
