#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.VB_Components.Voicebank import LiteVoicebank
from Backend.Util import ensureTensorLength, noteToPitch

class Nodegraph():
    def __init__(self) -> None:
        self.nodes = []
        #self.nodes[1].connect()#TODO:finish
        self.params = dict()

    def pack(self):
        for i in self.nodes:
            i.pack()

    def unpack(self):
        for i in self.nodes:
            i.unpack()

class Parameter():
    """class for holding and managing a parameter curve as seen by the main process. Exact layout will change with node tree implementation."""

    def __init__(self, path:str) -> None:
        self.curve = torch.full([1000], 0)
        self.enabled = True

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
        self.nodegraph = Nodegraph()
        self.borders = [0, 1, 2]
        self.length = 5000
        self.phonemeLengths = dict()
        tmpVb = LiteVoicebank(self.vbPath)
        for i in tmpVb.phonemeDict.keys():
            if tmpVb.phonemeDict[i][0].isPlosive:
                self.phonemeLengths[i] = tmpVb.phonemeDict[i][0].specharm.size()[0]
            else:
                self.phonemeLengths[i] = None

    def validate(self) -> None:
        """validates the data of the track, ensuring there are no inconsistencies capable of causing crashes"""

        self.length = max(
            self.length,
            5000,
            self.pitch.size()[0],
            self.basePitch.size()[0],
            self.breathiness.size()[0],
            self.steadiness.size()[0],
            self.aiBalance.size()[0],
            self.vibratoSpeed.size()[0],
            self.vibratoStrength.size()[0]
        )
        self.volume = max(self.volume, 0.)
        self.volume = min(self.volume, 2.)
        self.pauseThreshold = max(self.pauseThreshold, 0)
        self.pitch = ensureTensorLength(self.pitch, self.length, -1)
        self.basePitch = ensureTensorLength(self.basePitch, self.length, -1)
        self.breathiness = ensureTensorLength(self.breathiness, self.length, 0)
        self.steadiness = ensureTensorLength(self.steadiness, self.length, 0)
        self.aiBalance = ensureTensorLength(self.aiBalance, self.length, 0)
        self.vibratoSpeed = ensureTensorLength(self.vibratoSpeed, self.length, 0)
        self.vibratoStrength = ensureTensorLength(self.vibratoStrength, self.length, 0)
        if len(self.borders) > 3 * len(self.phonemes) + 3:
            self.borders = self.borders[:3 * len(self.phonemes) + 3]
        elif len(self.borders) < 3 * len(self.phonemes) + 3:
            self.borders.extend(range(int(self.borders[-1]) + 1, int(self.borders[-1]) + 4 + 3 * len(self.phonemes) - len(self.borders)))
        if len(self.notes) > 0:
            maxBorder = len(self.borders) - 1
            end = self.notes[-1].xPos + self.notes[-1].length
            for i in range(len(self.borders)):
                if self.borders[maxBorder - i] > end:
                    self.borders[maxBorder - i] = end - 2 * i
                else:
                    break
        self.borders.sort()
        for i in range(1, len(self.borders)):
            if self.borders[i] <= self.borders[i - 1] + 1:
                self.borders[i] = self.borders[i - 1] + 2
        for i, phoneme in enumerate(self.phonemes):
            if (phoneme not in self.phonemeLengths.keys()) and (phoneme not in ["_X", "_0", "_autopause"]):
                self.phonemes[i] = "_X"
        if self.loopOverlap.size()[0] > len(self.phonemes):
            self.loopOverlap = self.loopOverlap[:len(self.phonemes)]
        elif self.loopOverlap.size()[0] < len(self.phonemes):
            self.loopOverlap = torch.cat((self.loopOverlap, torch.zeros((len(self.phonemes) - self.loopOverlap.size()[0],), device = self.loopOverlap.device, dtype = torch.half)), 0)
        if self.loopOffset.size()[0] > len(self.phonemes):
            self.loopOffset = self.loopOffset[:len(self.phonemes)]
        elif self.loopOffset.size()[0] < len(self.phonemes):
            self.loopOffset = torch.cat((self.loopOffset, torch.zeros((len(self.phonemes) - self.loopOffset.size()[0],), device = self.loopOffset.device, dtype = torch.half)), 0)
        currentxPos = 0
        currentPhoneme = 0
        for i in self.notes:
            i.length = max(i.length, 1)
            if i.xPos <= currentxPos:
                i.xPos = currentxPos + 1
            currentxPos = i.xPos
            i.phonemeStart = currentPhoneme
            i.phonemeEnd = max(i.phonemeEnd, currentPhoneme)
            currentPhoneme = i.phonemeEnd
        #audio cache
        #vbPath
        #mixinVB
        
                
    def generateCaps(self) -> tuple([list, list]):
        """utility function for generating startCap and endCap attributes for the track. These are not required by the main process, and are instead sent to the rendering process with to_sequence"""

        startCaps = [False] * len(self.phonemes)
        endCaps = [False] * len(self.phonemes)
        for index, phoneme in enumerate(self.phonemes):
            if phoneme == "_autopause":
                if index < len(self.phonemes) - 1:
                    startCaps[index + 1] = True
                if index > 0:
                    endCaps[index - 1] = True
        return startCaps, endCaps

    def convert(self) -> tuple:
        """converts the track to a tuple for sending it to the rendering thread. Also handles conversion of MIDI pitch to frequency."""

        caps = self.generateCaps()
        
        pitch = noteToPitch(self.pitch)
        borders = []
        for i in self.borders:
            borders.append(int(i))#TODO: switch to integer tensor representation
        sequence = VocalSequence(self.length, borders, self.phonemes, caps[0], caps[1], self.loopOffset, self.loopOverlap, pitch, self.steadiness, self.breathiness, self.aiBalance, self.vibratoSpeed, self.vibratoStrength, self.useBreathiness, self.useSteadiness, self.useAIBalance, self.useVibratoSpeed, self.useVibratoStrength)
        return self.vbPath, None, sequence#None object is placeholder for wrapped NodeGraph
        #TODO: add node wrapping

class Note():
    """Container class for a note as handled by the main process. Contains a reference property pointing at its UI representation."""

    def __init__(self, xPos, yPos, start = 0, end = 1, reference = None) -> None:
        self.reference = reference
        self.length = 100
        self.xPos = xPos
        self.yPos = yPos
        self.phonemeMode = True
        self.content = ""
        self.phonemeStart = start
        self.phonemeEnd = end
