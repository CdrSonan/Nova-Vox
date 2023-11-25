#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, OptionProperty
from kivy.clock import mainthread

import torch
import math
from bisect import bisect_left

import sounddevice

import global_consts

from MiddleLayer.IniParser import readSettings
from MiddleLayer.BorderSystem import calculateBorders

from Util import ensureTensorLength, noteToPitch

from Localization.editor_localization import getLanguage
loc = getLanguage()

class MiddleLayer(Widget):
    """Central class of the program. Contains data handlers for all data on the main process, and callbacks for modifying it, which are triggered from the UI.
    Through its manager attribute, it indirectly also handles communication between the main and rendering process.
    Contains some functions for updating the UI when mode changes occur, but these functions may be moved to a dedicated class in the future."""

    def __init__(self, **kwargs) -> None:
        """Constructor called once during program startup. uses the id list of the main UI for referencing various UI elements and updating them. Functionality related to such UI updates may be moved to a dedicated class in the future, deprecating this argument."""
        
        super().__init__(**kwargs)
        self.undoStack = []
        self.redoStack = []
        self.undoActive = True
        self.singleUndoActive = False
        self.ui = None
        self.trackList = []
        from Backend.NV_Multiprocessing.Manager import RenderManager
        self.manager = RenderManager(self.trackList)
        self.unsavedChanges = False
        self.activeTrack = None
        self.activeParam = "steadiness"
        self.mode = OptionProperty("notes", options = ["notes", "timing", "pitch"])
        self.mode = "notes"
        self.tool = OptionProperty("draw", options = ["draw", "line", "arch", "reset"])
        self.tool = "draw"
        self.shift = BooleanProperty()
        self.shift = False
        self.alt = BooleanProperty()
        self.alt = False
        self.ctrl = BooleanProperty()
        self.ctrl = False
        self.scrollValue = 0.
        self.xScale = 1.
        self.audioBuffer = []
        self.mainAudioBufferPos = 0
        self.deletions = []
        self.playing = False
        settings = readSettings()
        self.undoLimit = int(settings["undolimit"])
        device = None
        devices = sounddevice.query_devices()
        for i in devices:
            if i["name"] == settings["audiodevice"]:
                device = i["name"] + ", " + settings["audioapi"]
        self.audioStream = sounddevice.OutputStream(global_consts.sampleRate, global_consts.audioBufferSize, device, callback = self.playCallback, latency = float(settings["audiolatency"]))
        self.scriptCache = ""
        
    def setUI(self, ui) -> None:
        """sets the ui and id list references of the main UI. Functionality related to such UI updates may be moved to a dedicated class in the future, deprecating this function."""

        self.ui = ui
        self.ids = ui.ids

    def addParam(self, param, name) -> None:
        pass #TODO: implement

    def updatePianoRoll(self) -> None:
        """updates the piano roll UI after a track or mode change"""

        self.ids["pianoRoll"].updateTrack()

    def changePianoRollMode(self) -> None:
        """helper function for piano roll UI updates when changing modes"""
        self.ids["pianoRoll"].changeMode()

    def applyScroll(self) -> None:
        """helper function for synchromizing scrolling between the piano roll and adaptive space"""

        self.ids["pianoRoll"].applyScroll(self.scrollValue)
        self.ids["adaptiveSpace"].applyScroll(self.scrollValue)

    def applyZoom(self) -> None:
        """helper function for synchromizing zoom between the piano roll and adaptive space"""

        self.ids["pianoRoll"].applyZoom(self.xScale)
        self.ids["adaptiveSpace"].applyZoom(self.xScale)

    def offsetPhonemes(self, index:int, offset:int) -> None:
        """adds or deletes phonemes from the active track, and recalculates timing markers to fit the new sequence.

        Arguments:
            index: the index to add phonemes at, or remove them from

            offset: if positive, number of phonemes added. If negative, number of phonemes removed.
            
            pause: Flag indicating whether an _autopause control phoneme should be inserted
            
            futurePhonemes: when using positive offset, a "preview" of the phonemes that are to be inserted. Used for timing marker calculations.
            
        When using a positive offset, this function adds the placeholder phoneme _X (or several of them). These should be overwritten with normal
        phonemes before submitting the change to the rendering process with the final flag set."""

        if index == 0:
            phonIndex = 0
        else:
            phonIndex = self.trackList[self.activeTrack].phonemeIndices[index - 1]
        #update phoneme list and other phoneme-space lists
        if offset > 0:
            for _ in range(offset):
                for _ in range(3):
                    self.trackList[self.activeTrack].notes[index].borders.append(0)
                self.trackList[self.activeTrack].notes[index].loopOverlap.append(0.5)
                self.trackList[self.activeTrack].notes[index].loopOffset.append(0.5)
        elif offset < 0:
            for _ in range(-offset):
                for _ in range(3):
                    self.trackList[self.activeTrack].notes[index].borders.pop()
                self.trackList[self.activeTrack].notes[index].loopOverlap.pop()
                self.trackList[self.activeTrack].notes[index].loopOffset.pop()
        self.trackList[self.activeTrack].offsets.append((phonIndex, offset))
        self.submitOffset(False, phonIndex, offset)
    
    def switchNote(self, index:int) -> None:
        """Switches the places of the notes at positions index and index + 1 or the active track. Does currently not clean up timing markers afterwards, so doing so manually or by prompting a call of offsetPhonemes is currently required."""

        note = self.trackList[self.activeTrack].notes.pop(index + 1)
        self.trackList[self.activeTrack].notes.insert(index, note)
        self.adjustNote(index + 1, None, None, False, False)
        self.adjustNote(index, None, None, False, True)
        self.trackList[self.activeTrack].notes[index].reference.index = index
        self.trackList[self.activeTrack].notes[index + 1].reference.index = index + 1
        if index == 0:
            start = 0
            borderStart = 0
        else:
            start = self.trackList[self.activeTrack].phonemeIndices[index - 1]
            borderStart = 3 * start + 1
        if index == len(self.trackList[self.activeTrack].phonemeIndices) - 2:
            end = len(self.trackList[self.activeTrack].phonemes)
            borderEnd = len(self.trackList[self.activeTrack].borders)
        else:
            end = self.trackList[self.activeTrack].phonemeIndices[index + 1]
            borderEnd = 3 * end + 1
        for i in range(start, end):
            self.repairBorders(i)
        self.submitNamedPhonParamChange(False, "phonemes", start, self.trackList[self.activeTrack].phonemes[start:end])
        self.submitNamedPhonParamChange(False, "borders", start, self.trackList[self.activeTrack].borders[borderStart:borderEnd])
    
    def adjustNote(self, index:int, oldLength:int = None, oldPos:int = None, keepStart:bool = False, adjustPrevious:bool = False) -> None:
        """Adjusts a note's attributes after it has been moved or scaled with respect to its surrounding notes. The current position and length are read from its UI representation, the old one must be given as arguments."""

        if oldLength == None:
            oldLength = self.trackList[self.activeTrack].notes[index].length
        if oldPos == None:
            oldPos = self.trackList[self.activeTrack].notes[index].xPos
        switch = None
        if index > 0 and self.trackList[self.activeTrack].notes[index - 1].xPos > self.trackList[self.activeTrack].notes[index].xPos:
            self.switchNote(index - 1)
            index -= 1
            switch = False
        elif index < len(self.trackList[self.activeTrack].notes) - 1 and self.trackList[self.activeTrack].notes[index + 1].xPos < self.trackList[self.activeTrack].notes[index].xPos:
            self.switchNote(index)
            index += 1
            switch = True
        autopauseOffset = self.trackList[self.activeTrack].notes[index].determineAutopause()
        self.trackList[self.activeTrack].buildPhonemeIndices()
        if autopauseOffset not in (0. , None):
            if autopauseOffset > 0:
                self.submitOffset(False, self.trackList[self.activeTrack].phonemeIndices[index] - 1, autopauseOffset)
                self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].phonemeIndices[index] - 1, ["_autopause",])
                self.trackList[self.activeTrack].offsets.append((self.trackList[self.activeTrack].phonemeIndices[index] - 1, autopauseOffset))
            else:
                self.submitOffset(False, self.trackList[self.activeTrack].phonemeIndices[index], autopauseOffset)
                self.trackList[self.activeTrack].offsets.append((self.trackList[self.activeTrack].phonemeIndices[index], autopauseOffset))
        if index == len(self.trackList[self.activeTrack].notes) - 1:
            self.trackList[self.activeTrack].borders[-2] = self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length
            self.trackList[self.activeTrack].borders[-1] = self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length + global_consts.refTransitionLength
        if self.trackList[self.activeTrack].notes[index].content == "-" and self.trackList[self.activeTrack].notes[index].phonemeMode == False:
            if index > 0:
                return self.adjustNote(index - 1, None, None, keepStart, adjustPrevious)
        context = self.trackList[self.activeTrack].notes[index].makeContext(keepStart)
        calculateBorders(self.trackList[self.activeTrack].notes[index], context)
        if index == 0:
            start = 0
        else:
            start = self.trackList[self.activeTrack].phonemeIndices[index - 1] * 3 + 1
        if index == len(self.trackList[self.activeTrack].phonemeIndices) - 1:
            end = self.trackList[self.activeTrack].phonemeIndices[-1] * 3 + 3
        else:
            end = self.trackList[self.activeTrack].phonemeIndices[index] * 3 + 1
        self.submitNamedPhonParamChange(False, "borders", start, list(self.trackList[self.activeTrack].borders[start:end]))
        if index > 0:
            self.recalculateBasePitch(index - 1, self.trackList[self.activeTrack].notes[index - 1].xPos, max(min(oldPos, self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length), 1))
        self.recalculateBasePitch(index, oldPos, oldPos + oldLength)
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            self.recalculateBasePitch(index + 1, oldPos + oldLength, oldPos + oldLength + self.trackList[self.activeTrack].notes[index - 1].length)
        if (self.trackList[self.activeTrack].notes[index].xPos != oldPos or adjustPrevious) and index > 0:
            self.adjustNote(index - 1, None, None, True, False)
        for i in range(start, end):
            self.repairBorders(i)
        return switch
    
    def syllableSplit(self, word:str) -> list:
        """splits a word into syllables using the wordDict of the loaded Voicebank. Returns a list of syllables, or None if the word cannot be split in a valid way."""
        
        for i in range(len(self.trackList[self.activeTrack].wordDict[1])):
            for j in self.trackList[self.activeTrack].wordDict[1][len(self.trackList[self.activeTrack].wordDict[1]) - i - 1].keys():
                if word.startswith(j):
                    if len(word) == len(j):
                        return [j]
                    append = self.syllableSplit(word[len(j):])
                    if append == None:
                        return None
                    else:
                        return [j] + append
        return None

    def repairNotes(self, index:int) -> None:
        """checks if the note at position index of the active track has exactly the same position as the previous note. If so, it is moved forward by one tick, ensuring that no note gets assigned a length of 0."""
        if index == 0 or index == len(self.trackList[self.activeTrack].notes):
            return
        if self.trackList[self.activeTrack].notes[index].xPos == self.trackList[self.activeTrack].notes[index - 1].xPos:
            self.trackList[self.activeTrack].notes[index].xPos += 1
            self.repairNotes(index + 1)

    def repairBorders(self, index:int) -> None:
        """checks if the border at position index of the active track has exactly the same position as the previous note. If so, it is moved forward by one tick, ensuring that no note gets assigned a length of 0."""

        if index == 0 or index == len(self.trackList[self.activeTrack].borders):
            return
        if self.trackList[self.activeTrack].borders[index] < self.trackList[self.activeTrack].borders[index - 1] + 1.:
            self.trackList[self.activeTrack].borders[index] = self.trackList[self.activeTrack].borders[index - 1] + 1.
            self.submitNamedPhonParamChange(False, "borders", index, [self.trackList[self.activeTrack].borders[index],])
            self.repairBorders(index + 1)

    def changeLength(self, length:int) -> None:
        """changes the length of the active track, and submits the change to the renderer"""

        self.trackList[self.activeTrack].length = length
        self.trackList[self.activeTrack].pitch = ensureTensorLength(self.trackList[self.activeTrack].pitch, length, -1)
        self.trackList[self.activeTrack].basePitch = ensureTensorLength(self.trackList[self.activeTrack].basePitch, length, -1)
        self.trackList[self.activeTrack].breathiness = ensureTensorLength(self.trackList[self.activeTrack].breathiness, length, 0)
        self.trackList[self.activeTrack].steadiness = ensureTensorLength(self.trackList[self.activeTrack].steadiness, length, 0)
        self.trackList[self.activeTrack].aiBalance = ensureTensorLength(self.trackList[self.activeTrack].aiBalance, length, 0)
        self.trackList[self.activeTrack].vibratoSpeed = ensureTensorLength(self.trackList[self.activeTrack].vibratoSpeed, length, 0)
        self.trackList[self.activeTrack].vibratoStrength = ensureTensorLength(self.trackList[self.activeTrack].vibratoStrength, length, 0)
        self.audioBuffer[self.activeTrack] = ensureTensorLength(self.audioBuffer[self.activeTrack], length * global_consts.batchSize, 0)
        self.ids["adaptiveSpace"].applyLength(length)
        self.submitChangeLength(True, length)

    def validate(self) -> None:
        """validates the data held by the middle layer, fixes any errors encountered, and restarts the renderer"""

        for index, track in enumerate(self.trackList):
            track.validate()
            self.audioBuffer[index] = ensureTensorLength(self.audioBuffer[index], track.length * global_consts.batchSize, 0)
        del self.deletions[:]
        self.manager.restart(self.trackList)
        self.updatePianoRoll()
        self.ui.updateParamPanel()

    def submitTerminate(self) -> None:
        self.manager.sendChange("terminate", True)

    def submitAddTrack(self, track) -> None:
        self.manager.sendChange("addTrack", True, *track.convert())

    def submitRemoveTrack(self, index:int) -> None:
        self.manager.sendChange("removeTrack", True, index)
    
    def submitDuplicateTrack(self, index:int) -> None:
        self.manager.sendChange("duplicateTrack", True, index)
    
    def submitChangeVB(self, index:int, path:str) -> None:
        self.manager.sendChange("changeVB", False, index, path)
    
    def submitNodegraphUpdate(self) -> None:
        #placeholder until NodeGraph is fully implemented
        self.manager.sendChange("nodeUpdate", True, self.activeTrack, None)
    
    def submitEnableParam(self, name:str) -> None:
        self.manager.sendChange("enableParam", True, self.activeTrack, name)
    
    def submitDisableParam(self, name:str) -> None:
        self.manager.sendChange("disableParam", True, self.activeTrack, name)
    
    def submitBorderChange(self, final:bool, index:int, data) -> None:
        self.manager.sendChange("changeInput", final, self.activeTrack, "borders", index, data)
    
    def submitNamedParamChange(self, final:bool, param, index:int, data) -> None:
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    
    def submitNamedPhonParamChange(self, final:bool, param, index:int, data) -> None:
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    
    def submitParamChange(self, final:bool, param, index:int, data) -> None:
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    
    def submitOffset(self, final:bool, index:int, offset:int) -> None:
        self.manager.sendChange("offset", final, self.activeTrack, index, offset)

    def submitChangeLength(self, final:bool, length:int) -> None:
        self.manager.sendChange("changeLength", final, self.activeTrack, length)
    
    def submitFinalize(self) -> None:
        self.manager.sendChange("finalize", True)
    
    def updateRenderStatus(self, track:int, index:int, value:int) -> None:
        """updates the visual representation of the rendering progress of the note at index index of track track"""

        for i in self.deletions:
            if i == track:
                return None
            elif i < track:
                track -= 1
        for i in self.trackList[track].offsets:
            if i[0] <= index:
                index += i[1]
        noteIndex = bisect_left(self.trackList[track].phonemeIndices, index)
        while (self.trackList[track].phonemeIndices[noteIndex] == self.trackList[track].phonemeIndices[noteIndex - 1]) and (noteIndex > 0):
            noteIndex -= 1
        if noteIndex > 0:
            if (index - self.trackList[track].phonemeIndices[noteIndex]) >= len(self.trackList[track].notes[noteIndex].phonemes):
                return
            index -= self.trackList[track].phonemeIndices[noteIndex - 1]
        if index >= len(self.trackList[track].notes[noteIndex].phonemes):
            return
        if self.trackList[track].notes[noteIndex].reference:
            self.trackList[track].notes[noteIndex].reference.updateStatus(index, value)
    
    def updateAudioBuffer(self, track:int, index:int, data:torch.Tensor) -> None:
        """updates the audio buffer. The data of the track at position track of the trackList is updated with the tensor Data, starting from position index"""

        for i in self.deletions:
            if i == track:
                return None
            elif i < track:
                track -= 1
        self.audioBuffer[track][index:index + len(data)] = data
    
    @mainthread
    def movePlayhead(self, position:int) -> None:
        """sets the position of the playback head in the piano roll UI"""

        self.ids["pianoRoll"].changePlaybackPos(position)
    
    def play(self, state:bool = None) -> None:
        """starts or stops audio playback.
        If state is true, starts playback, if state is false, stops playback. If state is None or not given, starts playback if it is not already in progress, and stops playback if it is."""

        if state == None:
            state = not(self.playing)
        if state == True:
            self.ids["playButton"].state = "down"
            self.audioStream.start()
        if state == False:
            self.ids["playButton"].state = "normal"
            self.audioStream.stop()
        self.playing = state
    
    def playCallback(self, outdata, frames, time, status) -> None:
        """callback function used for updating the driver audio stream with new data from the audio buffer during playback"""

        if self.playing:
            newBufferPos = self.mainAudioBufferPos + global_consts.audioBufferSize
            mainAudioBuffer = torch.zeros((int(newBufferPos - self.mainAudioBufferPos),))
            volumes = []
            for i in range(len(self.audioBuffer)):
                buffer = self.audioBuffer[i][self.mainAudioBufferPos:newBufferPos] * self.trackList[i].volume
                mainAudioBuffer += buffer
                volumes.append(buffer.abs().max())
            for i in self.ids["singerList"].children:
                i.children[0].children[0].children[0].children[0].value = volumes[i.index]
            buffer = mainAudioBuffer.expand(2, -1).transpose(0, 1).numpy()
            self.mainAudioBufferPos = newBufferPos
            self.movePlayhead(int((self.mainAudioBufferPos - self.audioStream.latency * self.audioStream.samplerate) / global_consts.batchSize))
        else:
            buffer = torch.zeros([global_consts.audioBufferSize, 2], dtype = torch.float32).expand(-1, 2).numpy()
            self.movePlayhead(int(self.mainAudioBufferPos / global_consts.batchSize))
        outdata[:] = buffer.copy()

    def posToNote(self, pos1:int, pos2:int) -> tuple:
        pos1Out = 0
        for i in range(len(self.trackList[self.activeTrack].notes)):
            if self.trackList[self.activeTrack].notes[i].xPos > pos1:
                pos1Out = max(i - 1, 0)
                break
        pos2Out = len(self.trackList[self.activeTrack].notes)
        for i in range(pos1Out, len(self.trackList[self.activeTrack].notes)):
            if self.trackList[self.activeTrack].notes[i].xPos > pos2:
                pos2Out = max(i, 0)
                break
        return (pos1Out, pos2Out)

    def recalculateBasePitch(self, index:int, oldStart:float, oldEnd:float) -> None:
        """recalculates the base pitch curve after the note at position index of the active track has been modified. oldStart and oldEnd are the start and end of the note before the transformation leading to the function call."""

        dipWidth = global_consts.pitchDipWidth
        dipHeight = global_consts.pitchDipHeight
        transitionLength1 = min(global_consts.pitchTransitionLength, int(self.trackList[self.activeTrack].notes[index].length))
        transitionLength2 = min(global_consts.pitchTransitionLength, int(self.trackList[self.activeTrack].notes[index].length))
        currentHeight = self.trackList[self.activeTrack].notes[index].yPos
        transitionPoint1 = self.trackList[self.activeTrack].notes[index].xPos
        transitionPoint2 = self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length
        if index == 0:
            previousHeight = None
        elif len(self.trackList[self.activeTrack].notes[index]) == 0:
            previousHeight = None
        elif self.trackList[self.activeTrack].notes[index].autopause:
            previousHeight = None
            if index == 0:
                oldStart = min(oldStart, int(self.trackList[self.activeTrack].borders[2]))
            else:
                oldStart = min(oldStart, int(self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].phonemeIndices[index - 1] + 2]))
        else:
            previousHeight = self.trackList[self.activeTrack].notes[index - 1].yPos
            transitionLength1 = min(transitionLength1, self.trackList[self.activeTrack].notes[index - 1].length)
            if self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length < transitionPoint1:
                transitionLength1 += transitionPoint1 - self.trackList[self.activeTrack].notes[index - 1].xPos - self.trackList[self.activeTrack].notes[index - 1].length
                transitionPoint1 = (transitionPoint1 + self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length) / 2
        if index == len(self.trackList[self.activeTrack].notes) - 1:
            nextHeight = None
        elif self.trackList[self.activeTrack].phonemeIndices[index - 1] == self.trackList[self.activeTrack].phonemeIndices[index]:
            nextHeight = None
        elif self.trackList[self.activeTrack].notes[index].autopause:
            nextHeight = None
            oldEnd = max(oldEnd, int(self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].phonemeIndices[index]]))
        else:
            transitionPoint2 = self.trackList[self.activeTrack].notes[index + 1].xPos
            nextHeight = self.trackList[self.activeTrack].notes[index + 1].yPos
            transitionLength2 = min(transitionLength2, self.trackList[self.activeTrack].notes[index + 1].length)
            if self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length < transitionPoint1:
                transitionLength2 += transitionPoint2 - self.trackList[self.activeTrack].notes[index].xPos - self.trackList[self.activeTrack].notes[index].length
                transitionPoint2 = (transitionPoint2 + self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length) / 2
        scalingFactor = min(self.trackList[self.activeTrack].notes[index].length / 2 / dipWidth, 1.)
        dipWidth *= scalingFactor
        dipHeight *= scalingFactor
        start = int(transitionPoint1 - math.ceil(transitionLength1 / 2))
        end = int(transitionPoint2 + math.ceil(transitionLength2 / 2))
        transitionPoint1 = int(transitionPoint1)
        transitionPoint2 = int(transitionPoint2)
        if previousHeight == None:
            start =  self.trackList[self.activeTrack].notes[index].xPos
        if nextHeight == None:
            end = self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length
        pitchDelta = (self.trackList[self.activeTrack].pitch[start:end] - self.trackList[self.activeTrack].basePitch[start:end]) * torch.heaviside(self.trackList[self.activeTrack].basePitch[start:end], torch.ones_like(self.trackList[self.activeTrack].basePitch[start:end]))
        self.trackList[self.activeTrack].basePitch[oldStart:oldEnd] = torch.full_like(self.trackList[self.activeTrack].basePitch[oldStart:oldEnd], -1.)
        self.trackList[self.activeTrack].pitch[oldStart:oldEnd] = torch.full_like(self.trackList[self.activeTrack].basePitch[oldStart:oldEnd], -1.)
        self.trackList[self.activeTrack].basePitch[start:end] = torch.full_like(self.trackList[self.activeTrack].basePitch[start:end], currentHeight)
        if previousHeight != None:
            self.trackList[self.activeTrack].basePitch[transitionPoint1 - math.ceil(transitionLength1 / 2):transitionPoint1 + math.ceil(transitionLength1 / 2)] = torch.pow(torch.cos(torch.linspace(0, math.pi / 2, 2 * math.ceil(transitionLength1 / 2))), 2) * (previousHeight - currentHeight) + torch.full([2 * math.ceil(transitionLength1 / 2),], currentHeight)
        if nextHeight != None:
            self.trackList[self.activeTrack].basePitch[transitionPoint2 - math.ceil(transitionLength2 / 2):transitionPoint2 + math.ceil(transitionLength2 / 2)] = torch.pow(torch.cos(torch.linspace(0, math.pi / 2, 2 * math.ceil(transitionLength2 / 2))), 2) * (currentHeight - nextHeight) + torch.full([2 * math.ceil(transitionLength2 / 2),], nextHeight)
        self.trackList[self.activeTrack].basePitch[self.trackList[self.activeTrack].notes[index].xPos:self.trackList[self.activeTrack].notes[index].xPos + int(dipWidth)] -= torch.pow(torch.sin(torch.linspace(0, math.pi, int(dipWidth))), 2) * dipHeight
        vibratoSpeed = (self.trackList[self.activeTrack].vibratoSpeed[start:end] / 3. + 0.66) * global_consts.maxVibratoSpeed
        vibratoStrength = (self.trackList[self.activeTrack].vibratoStrength[start:end] / 2. + 0.5) * global_consts.maxVibratoStrength
        vibratoCurve = torch.cumsum(vibratoSpeed.to(torch.float32), 0)
        if previousHeight != None:
            vibratoCurve -= torch.sum(vibratoSpeed[:transitionPoint1 + math.ceil(transitionLength1 / 2) - start])
        vibratoCurve = vibratoStrength * torch.sin(vibratoCurve)
        vibratoCurve *= torch.pow(torch.sin(torch.linspace(0, math.pi, end - start + 1)[:-1]), torch.tensor([global_consts.vibratoEnvelopeExp,], device = self.trackList[self.activeTrack].basePitch.device))
        self.trackList[self.activeTrack].basePitch[start:end] += vibratoCurve
        self.trackList[self.activeTrack].pitch[start:end] = self.trackList[self.activeTrack].basePitch[start:end] + torch.heaviside(self.trackList[self.activeTrack].basePitch[start:end], torch.ones_like(self.trackList[self.activeTrack].basePitch[start:end])) * pitchDelta
        data = self.trackList[self.activeTrack].pitch[start:end]
        if type(data) == list:
            data = torch.tensor(data, dtype = torch.half)
        self.trackList[self.activeTrack].pitch[start:start + data.size()[0]] = data
        data = noteToPitch(data)
        self.submitNamedPhonParamChange(False, "pitch", start, data)
