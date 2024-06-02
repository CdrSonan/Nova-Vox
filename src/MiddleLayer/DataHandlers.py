#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from copy import copy
from bisect import bisect_left
import torch
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.VB_Components.Voicebank import LiteVoicebank
from Util import ensureTensorLength, noteToPitch
from API.Node import PackedNode

class Nodegraph():
    def __init__(self) -> None:
        self.nodes = []
        self.params = dict()
    
    def addNode(self, node):
        self.nodes.append(node)
    
    def removeNode(self, node):
        self.nodes.remove(node)

    def pack(self):
        packedNodes = []
        for i in self.nodes:
            packedNodes.append(PackedNode(i))
        for idx, i in enumerate(self.nodes):
            for j in i.inputs.keys():
                if i.inputs[j].attachedTo is not None:
                    packedNodes[idx].inputs[j].attachedTo = packedNodes[self.nodes.index(i.inputs[j].attachedTo.node)].outputs[i.inputs[j].attachedTo.name]
        return packedNodes, self.params

class Parameter():
    """class for holding and managing a parameter curve as seen by the main process. Exact layout will change with node tree implementation."""

    def __init__(self, path:str) -> None:
        self.curve = torch.full([5000], 0)
        self.enabled = True

class Track():
    """class for holding and managing a vocal track as seen by the main process. Contains all settings required for processing on the main process.
    
    Methods:
        __init__: constructor for a track with default settings and a Voicebank specified by a filepath
        
        toSequence: converts the track to a VOcalSequence object for sending it to the rendering thread"""


    def __init__(self, path:str) -> None:
        """constructor function for a track with default settings.
        
        Arguments:
            path: string or path object pointing to the voicebank file that should be used by the track."""


        self.volume = 1.
        self.vbPath = path
        self.wordDict = (dict(), [])
        self.notes = []
        self.phonemeIndices = []
        self.phonemes = PhonemeProxy(self)
        self.pitch = torch.full((5000,), -1., dtype = torch.half)
        self.basePitch = torch.full((5000,), -1., dtype = torch.half)
        self.breathiness = torch.full((5000,), 0, dtype = torch.half)
        self.steadiness = torch.full((5000,), 0, dtype = torch.half)
        self.aiBalance = torch.full((5000,), 0, dtype = torch.half)
        self.loopOverlap = LoopProxy(self, "overlap")
        self.loopOffset = LoopProxy(self, "offset")
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
        self.borders = BorderProxy(self)
        self.offsets = []
        self.length = 5000
        self.phonemeLengths = dict()
        tmpVb = LiteVoicebank(None)
        tmpVb.loadPhonemeDict(self.vbPath, False)
        for i in tmpVb.phonemeDict.keys:
            if tmpVb.phonemeDict.fetch(i, True)[0].isPlosive:
                self.phonemeLengths[i] = tmpVb.phonemeDict.fetch(i, True)[0].specharm.size()[0]
            else:
                self.phonemeLengths[i] = None
            self.wordDict = tmpVb.wordDict

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
        self.volume = min(self.volume, 1.2)
        self.pauseThreshold = max(self.pauseThreshold, 0)
        self.pitch = ensureTensorLength(self.pitch, self.length, -1)
        self.basePitch = ensureTensorLength(self.basePitch, self.length, -1)
        self.breathiness = ensureTensorLength(self.breathiness, self.length, 0)
        self.steadiness = ensureTensorLength(self.steadiness, self.length, 0)
        self.aiBalance = ensureTensorLength(self.aiBalance, self.length, 0)
        self.vibratoSpeed = ensureTensorLength(self.vibratoSpeed, self.length, 0)
        self.vibratoStrength = ensureTensorLength(self.vibratoStrength, self.length, 0)
        for i in range(1, len(self.borders)):
            if self.borders[i] <= self.borders[i - 1] + 1:
                self.borders[i] = self.borders[i - 1] + 2
        for i, phoneme in enumerate(self.phonemes):
            if (phoneme not in self.phonemeLengths.keys()) and (phoneme not in ["_autopause", "pau", "-"]):
                self.phonemes[i] = "pau"
        currentxPos = 0
        for i in self.notes:
            i.length = max(i.length, 1)
            if i.xPos <= currentxPos:
                i.xPos = currentxPos + 1
            currentxPos = i.xPos
            if len(i.loopOverlap) != len(i.phonemes):
                i.loopOverlap += [0.5] * (len(i.phonemes) - len(i.loopOverlap))
            if len(i.loopOffset) != len(i.phonemes):
                i.loopOffset += [0.5] * (len(i.phonemes) - len(i.loopOffset))
        self.buildPhonemeIndices()
        #audio cache
        #vbPath
        #mixinVB
    
    def buildPhonemeIndices(self) -> None:
        self.phonemeIndices = []
        for i in self.notes:
            if len(self.phonemeIndices) == 0:
                self.phonemeIndices.append(len(i))
            else:
                self.phonemeIndices.append(self.phonemeIndices[-1] + len(i))

    def convert(self) -> tuple:
        """converts the track to a tuple for sending it to the rendering thread. Also handles conversion of MIDI pitch to frequency."""
        
        pitch = noteToPitch(self.pitch)
        borders = []
        for i in self.borders:
            borders.append(int(i))
        sequence = VocalSequence(self.length, borders, self.phonemes(), self.loopOffset(), self.loopOverlap(), pitch, self.steadiness, self.breathiness, self.aiBalance, self.vibratoSpeed, self.vibratoStrength, self.useBreathiness, self.useSteadiness, self.useAIBalance, self.useVibratoSpeed, self.useVibratoStrength, [], None)
        return self.vbPath, self.nodegraph.pack(), sequence

class NoteContext():
    
    def __init__(self, start, preutterance) -> None:
        self.start = start
        self.end = None
        self.preutterance = preutterance
        self.trailingAutopause = None

class Note():
    """Container class for a note as handled by the main process. Contains a reference property pointing at its UI representation."""

    def __init__(self, xPos:int, yPos:int, track:Track, reference = None) -> None:
        self.reference = reference
        self.length = 100
        self.xPos = xPos
        self.yPos = yPos
        self.track = track
        self.phonemeMode = True
        self.content = ""
        self.phonemes = []
        self.borders = []
        self.loopOverlap = []
        self.loopOffset = []
        self.pronuncIndex = None
        self.autopause = False
        self.carryOver = False
    
    def __len__(self) -> int:
         return len(self.phonemes) + self.autopause
    
    def __getitem__(self, val):
        if val.__class__ == slice:
            for i in range(val.start, val.stop, val.step):
                yield self[i]
        else:
            if val < len(self.phonemes):
                return self.phonemes[val], self.loopOverlap[val], self.loopOffset[val]
            elif self.autopause and (val == len(self.phonemes)):
                return "_autopause", 0, 0
            else:
                raise IndexError("index out of range")
            
    def __setitem__(self, val, newVal):
        if val.__class__ == slice:
            if len(newVal) != len(range(val.start, val.stop, val.step)):
                raise ValueError("new value must have same length as slice")
            for i in range(val.start, val.stop, val.step):
                self[i] = newVal[i]
        else:
            if val < len(self.phonemes):
                self.phonemes[val] = newVal
            else:
                raise IndexError("index out of range")
            
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def makeContext(self, keepStart:bool = False):
        if keepStart and len(self.phonemes) > 0:
            if self.track.phonemeLengths[self.phonemes[0]] == None or len(self.phonemes) == 1:
                start = self.borders[0]
            else:
                start = self.borders[3]
        else:
            start = self.xPos
        if len(self.phonemes) == 0:
            context = NoteContext(start, 0)
            print("context created for empty note.")
        elif self.carryOver:
            context = NoteContext(start, -self.length / (len(self.phonemes) + 1))
        elif self.track.phonemeLengths[self.phonemes[0]] == None:
            context = NoteContext(start, 0)
        else:
            context = NoteContext(start, self.track.phonemeLengths[self.phonemes[0]])
        return self.propagateContext(context)
    
    def propagateContext(self, context:NoteContext):
        if len(self) == 0 and not self.carryOver:
            if self.track.notes.index(self) < len(self.track.notes) - 1:
                return self.track.notes[self.track.notes.index(self) + 1].propagateContext(context)
            context.end = self.xPos + self.length
            context.trailingAutopause = None
            return context
        if self.autopause:
            context.trailingAutopause = self.track.borders[self.track.phonemeIndices[self.track.notes.index(self)] * 3 + 1]
            context.end = self.xPos + self.length
        else:
            context.trailingAutopause = None
            context.end = self.track.borders[self.track.phonemeIndices[self.track.notes.index(self)] * 3 + 1]
        return context
    
    def delegatePhonemes(self, notes:list):
        if len(self) == 0 and self.track.notes.index(self) < len(self.track.notes) - 1:
            return self.track.notes[self.track.notes.index(self) + 1].delegatePhonemes(notes)
        if len(self.phonemes) == 0 and self.content[0] == "-":
            notes.append(self)
            if len(self) == 1 and self.track.notes.index(self) < len(self.track.notes) - 1:
                return self.track.notes[self.track.notes.index(self) + 1].delegatePhonemes(notes)
        return notes
    
    def determineAutopause(self):
        previousAutopause = copy(self.autopause)
        if self.track.notes.index(self) == len(self.track.notes) - 1:
            self.autopause = False
            return 0
        self.autopause = self.track.notes[self.track.notes.index(self) + 1].xPos - self.xPos - self.length > self.track.pauseThreshold
        if self.autopause and not previousAutopause:
            for _ in range(3):
                self.borders.append(0)
            self.track.buildPhonemeIndices()
            return 1
        elif not self.autopause and previousAutopause:
            for _ in range(3):
                self.borders.pop()
            self.track.buildPhonemeIndices()
            return -1
            

class PhonemeProxy():
    
    def __init__(self, track:Track) -> None:
        self.track = track
    
    def __getitem__(self, val):
        if val.__class__ == slice:
            if val.start == None:
                start = 0
            elif val.start < 0:
                start = len(self) + val.start
            else:
                start = val.start
            if val.stop == None:
                stop = len(self)
            elif val.stop < 0:
                stop = len(self) + val.stop
            else:
                stop = val.stop
            if val.step == None:
                step = 1
            else:
                step = val.step
            return [self[i] for i in range(start, stop, step)]
        else:
            if val < 0:
                phonIndex = len(self) + val
            else:
                phonIndex = val
            noteIndex = bisect_left(self.track.phonemeIndices, phonIndex)
            if phonIndex == self.track.phonemeIndices[noteIndex]:
                noteIndex += 1
            while (self.track.phonemeIndices[noteIndex] == self.track.phonemeIndices[noteIndex - 1]) and (noteIndex > 0):
                noteIndex -= 1
            if noteIndex > 0:
                if (phonIndex - self.track.phonemeIndices[noteIndex - 1]) >= len(self.track.notes[noteIndex].phonemes):
                    return "_autopause"
                return self.track.notes[noteIndex].phonemes[phonIndex - self.track.phonemeIndices[noteIndex - 1]]
            if phonIndex >= len(self.track.notes[noteIndex].phonemes):
                return "_autopause"
            return self.track.notes[noteIndex].phonemes[phonIndex]
    
    def __setitem__(self, val, newVal):
        if val.__class__ == slice:
            if val.start == None:
                start = 0
            elif val.start < 0:
                start = len(self) + val.start
            else:
                start = val.start
            if val.stop == None:
                stop = len(self)
            elif val.stop < 0:
                stop = len(self) + val.stop
            else:
                stop = val.stop
            if val.step == None:
                step = 1
            else:
                step = val.step
            if len(newVal) != len(range(start, stop, step)):
                raise ValueError("new value must have same length as slice")
            for i in range(start, stop, step):
                self[i] = newVal[i - start]
        else:
            if val < 0:
                phonIndex = len(self) + val
            else:
                phonIndex = val
            noteIndex = bisect_left(self.track.phonemeIndices, phonIndex)
            if phonIndex == self.track.phonemeIndices[noteIndex]:
                noteIndex += 1
            while (self.track.phonemeIndices[noteIndex] == self.track.phonemeIndices[noteIndex - 1]) and (noteIndex > 0):
                noteIndex -= 1
            if noteIndex > 0:
                if (phonIndex - self.track.phonemeIndices[noteIndex - 1]) >= len(self.track.notes[noteIndex].phonemes):
                    raise ValueError("cannot replace automatically inserted pause phoneme")
                self.track.notes[noteIndex].phonemes[phonIndex - self.track.phonemeIndices[noteIndex - 1]] = newVal
            else:
                if phonIndex >= len(self.track.notes[noteIndex].phonemes):
                    raise ValueError("cannot replace automatically inserted pause phoneme")
                self.track.notes[noteIndex].phonemes[phonIndex] = newVal
            
    def __len__(self):
        if len(self.track.phonemeIndices) == 0:
            return 0
        return self.track.phonemeIndices[-1]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __call__(self) -> list:
        return list(self[:])
    
    def __str__(self) -> str:
        return str(self())
            
class BorderProxy():
    
    def __init__(self, track:Track) -> None:
        self.track = track
        self.wrappingBorders = [0, 1, 2]
        
    def __getitem__(self, val):
        if val.__class__ == slice:
            if val.start == None:
                start = 0
            elif val.start < 0:
                start = len(self) + val.start
            else:
                start = val.start
            if val.stop == None:
                stop = len(self)
            elif val.stop < 0:
                stop = len(self) + val.stop
            else:
                stop = val.stop
            if val.step == None:
                step = 1
            else:
                step = val.step
            return [self[i] for i in range(start, stop, step)]
        else:
            if val < 0:
                brdIndex = len(self) + val
            else:
                brdIndex = val
            if brdIndex == 0 or len(self.track.phonemeIndices) == 0:
                return self.wrappingBorders[brdIndex]
            noteIndex = bisect_left(self.track.phonemeIndices, brdIndex / 3.)
            if noteIndex == len(self.track.phonemeIndices):
                return self.wrappingBorders[brdIndex - self.track.phonemeIndices[noteIndex - 1] * 3]
            while (self.track.phonemeIndices[noteIndex] == self.track.phonemeIndices[noteIndex - 1]) and (noteIndex > 0):
                noteIndex -= 1
            if noteIndex > 0:
                return self.track.notes[noteIndex].borders[brdIndex - self.track.phonemeIndices[noteIndex - 1] * 3 - 1]#TODO: Index out of range error occurs here
            return self.track.notes[noteIndex].borders[brdIndex - 1]
    
    def __setitem__(self, val, newVal):
        if val.__class__ == slice:
            if val.start == None:
                start = 0
            elif val.start < 0:
                start = len(self) + val.start
            else:
                start = val.start
            if val.stop == None:
                stop = len(self)
            elif val.stop < 0:
                stop = len(self) + val.stop
            else:
                stop = val.stop
            if val.step == None:
                step = 1
            else:
                step = val.step
            if len(newVal) != len(range(start, stop, step)):
                raise ValueError("new value must have same length as slice")
            for i in range(start, stop, step):
                self[i] = newVal[i - start]
        else:
            if val < 0:
                brdIndex = len(self) + val
            else:
                brdIndex = val
            if brdIndex == 0 or len(self.track.phonemeIndices) == 0:
                self.wrappingBorders[brdIndex] = newVal
                return
            noteIndex = bisect_left(self.track.phonemeIndices, brdIndex / 3.)
            if noteIndex == len(self.track.phonemeIndices):
                self.wrappingBorders[brdIndex - self.track.phonemeIndices[noteIndex - 1] * 3] = newVal
                return
            while (self.track.phonemeIndices[noteIndex] == self.track.phonemeIndices[noteIndex - 1]) and (noteIndex > 0):
                noteIndex -= 1
            if noteIndex > 0:
                self.track.notes[noteIndex].borders[brdIndex - self.track.phonemeIndices[noteIndex - 1] * 3 - 1] = newVal
            else:
                self.track.notes[noteIndex].borders[brdIndex - 1] = newVal
        
    def __len__(self):
        if len(self.track.phonemeIndices) == 0:
            return 3
        return self.track.phonemeIndices[-1] * 3 + 3
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def __call__(self) -> list:
        return list(self[:])
    
    def __str__(self) -> str:
        return str(self())

class LoopProxy():
    
    def __init__(self, track:Track, type:str) -> None:
        self.track = track
        self.type = type
    
    def __getitem__(self, val):
        if val.__class__ == slice:
            if val.start == None:
                start = 0
            elif val.start < 0:
                start = len(self) + val.start
            else:
                start = val.start
            if val.stop == None:
                stop = len(self)
            elif val.stop < 0:
                stop = len(self) + val.stop
            else:
                stop = val.stop
            if val.step == None:
                step = 1
            else:
                step = val.step
            return [self[i] for i in range(start, stop, step)]
        else:
            if val < 0:
                phonIndex = len(self) + val
            else:
                phonIndex = val
            noteIndex = bisect_left(self.track.phonemeIndices, phonIndex)
            if phonIndex == self.track.phonemeIndices[noteIndex]:
                noteIndex += 1
            while (self.track.phonemeIndices[noteIndex] == self.track.phonemeIndices[noteIndex - 1]) and (noteIndex > 0):
                noteIndex -= 1
            if self.type == "offset":
                if noteIndex > 0:
                    if (phonIndex - self.track.phonemeIndices[noteIndex - 1]) >= len(self.track.notes[noteIndex].loopOffset):
                        return 0
                    return self.track.notes[noteIndex].loopOffset[phonIndex - self.track.phonemeIndices[noteIndex - 1]]
                if phonIndex >= len(self.track.notes[noteIndex].loopOffset):
                    return 0
                return self.track.notes[noteIndex].loopOffset[phonIndex]
            elif self.type == "overlap":
                if noteIndex > 0:
                    if (phonIndex - self.track.phonemeIndices[noteIndex - 1]) >= len(self.track.notes[noteIndex].loopOverlap):
                        return 0
                    return self.track.notes[noteIndex].loopOverlap[phonIndex - self.track.phonemeIndices[noteIndex - 1]]
                if phonIndex >= len(self.track.notes[noteIndex].loopOverlap):
                    return 0
                return self.track.notes[noteIndex].loopOverlap[phonIndex]
    
    def __setitem__(self, val, newVal):
        if val.__class__ == slice:
            if val.start == None:
                start = 0
            elif val.start < 0:
                start = len(self) + val.start
            else:
                start = val.start
            if val.stop == None:
                stop = len(self)
            elif val.stop < 0:
                stop = len(self) + val.stop
            else:
                stop = val.stop
            if val.step == None:
                step = 1
            else:
                step = val.step
            if len(newVal) != len(range(start, stop, step)):
                raise ValueError("new value must have same length as slice")
            for i in range(start, stop, step):
                self[i] = newVal[i - start]
        else:
            if val < 0:
                phonIndex = len(self) + val
            else:
                phonIndex = val
            noteIndex = bisect_left(self.track.phonemeIndices, phonIndex)
            if phonIndex == self.track.phonemeIndices[noteIndex]:
                noteIndex += 1
            while (self.track.phonemeIndices[noteIndex] == self.track.phonemeIndices[noteIndex - 1]) and (noteIndex > 0):
                noteIndex -= 1
            if self.type == "offset":
                if noteIndex > 0:
                    if (phonIndex - self.track.phonemeIndices[noteIndex - 1]) >= len(self.track.notes[noteIndex].phonemes):
                        print("WARNING: changing loop settings of automatically inserted pause phoneme has no effect")
                        return
                    self.track.notes[noteIndex].loopOffset[phonIndex - self.track.phonemeIndices[noteIndex - 1]] = newVal
                else:
                    if phonIndex >= len(self.track.notes[noteIndex].phonemes):
                        print("WARNING: changing loop settings of automatically inserted pause phoneme has no effect")
                        return
                    self.track.notes[noteIndex].loopOffset[phonIndex] = newVal
            elif self.type == "overlap":
                if noteIndex > 0:
                    if (phonIndex - self.track.phonemeIndices[noteIndex - 1]) >= len(self.track.notes[noteIndex].phonemes):
                        print("WARNING: changing loop settings of automatically inserted pause phoneme has no effect")
                        return
                    self.track.notes[noteIndex].loopOverlap[phonIndex - self.track.phonemeIndices[noteIndex - 1]] = newVal
                else:
                    if phonIndex >= len(self.track.notes[noteIndex].phonemes):
                        print("WARNING: changing loop settings of automatically inserted pause phoneme has no effect")
                        return
                    self.track.notes[noteIndex].loopOverlap[phonIndex] = newVal
            
    def __len__(self):
        if len(self.track.phonemeIndices) == 0:
            return 0
        return self.track.phonemeIndices[-1]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __call__(self) -> list:
        return list(self[:])
    
    def __str__(self) -> str:
        return str(self())
