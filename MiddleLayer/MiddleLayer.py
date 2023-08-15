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

import sounddevice

import global_consts

from MiddleLayer.IniParser import readSettings
import MiddleLayer.DataHandlers as dh
from MiddleLayer.BorderSystem import recalculateBorders

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
    

    def addTrack(self, track):
        """Adds a new track to the track list. Used for loading projects from disk."""
        #TODO: actually use for loading from disk
        self.trackList.append(track)
        self.audioBuffer.append(torch.zeros([track.length * global_consts.batchSize,]))

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

    def offsetPhonemes(self, index:int, offset:int, pause:bool = False, futurePhonemes:list = None) -> None:
        """adds or deletes phonemes from the active track, and recalculates timing markers to fit the new sequence.

        Arguments:
            index: the index to add phonemes at, or remove them from

            offset: if positive, number of phonemes added. If negative, number of phonemes removed.
            
            pause: Flag indicating whether an _autopause control phoneme should be inserted
            
            futurePhonemes: when using positive offset, a "preview" of the phonemes that are to be inserted. Used for timing marker calculations.
            
        When using a positive offset, this function adds the placeholder phoneme _X (or several of them). These should be overwritten with normal
        phonemes before submitting the change to the rendering process with the final flag set."""

        #TODO: refactor timing calculations to dedicated function and add callback in switchNote
        phonIndex = self.trackList[self.activeTrack].notes[index].phonemeStart
        #update phoneme list and other phoneme-space lists
        addition = 0
        if offset > 0:
            if pause:
                if len(self.trackList[self.activeTrack].phonemes) == phonIndex:
                    addition = 3
                else:
                    addition = 4
            else:
                addition = 1
            if len(self.trackList[self.activeTrack].phonemes) > phonIndex and self.trackList[self.activeTrack].notes[index].phonemeEnd > phonIndex:
                if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] == "_autopause":
                    phonIndex += 1
            for i in range(offset):
                self.trackList[self.activeTrack].phonemes.insert(phonIndex + i, "_X")
                self.trackList[self.activeTrack].loopOverlap = torch.cat([self.trackList[self.activeTrack].loopOverlap[0:phonIndex], torch.tensor([0.5], dtype = torch.half), self.trackList[self.activeTrack].loopOverlap[phonIndex:]], dim = 0)
                self.trackList[self.activeTrack].loopOffset = torch.cat([self.trackList[self.activeTrack].loopOffset[0:phonIndex], torch.tensor([0.5], dtype = torch.half), self.trackList[self.activeTrack].loopOffset[phonIndex:]], dim = 0)
                for j in range(3):
                    self.trackList[self.activeTrack].borders.insert(phonIndex * 3 + addition, 0)
        elif offset < 0:
            for i in range(-offset):
                self.trackList[self.activeTrack].phonemes.pop(phonIndex)
                self.trackList[self.activeTrack].loopOverlap = torch.cat([self.trackList[self.activeTrack].loopOverlap[0:phonIndex], self.trackList[self.activeTrack].loopOverlap[phonIndex + 1:]], dim = 0)
                self.trackList[self.activeTrack].loopOffset = torch.cat([self.trackList[self.activeTrack].loopOffset[0:phonIndex], self.trackList[self.activeTrack].loopOffset[phonIndex + 1:]], dim = 0)
                for j in range(3):
                    self.trackList[self.activeTrack].borders.pop(phonIndex * 3)
        if futurePhonemes:
            self.trackList[self.activeTrack].phonemes[phonIndex:phonIndex + len(futurePhonemes)] = futurePhonemes
        for i in self.trackList[self.activeTrack].notes[index + 1:]:
            i.phonemeStart += offset
            i.phonemeEnd += offset
        self.trackList[self.activeTrack].notes[index].phonemeEnd += offset
        if pause and offset > 0:
            self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] = "_autopause"
        self.trackList[self.activeTrack].offsets.append((phonIndex, offset))
        self.submitOffset(False, phonIndex, offset, addition)
                
        start, end = recalculateBorders(index, self.trackList[self.activeTrack], None)
        if index > 0 and self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] != "_autopause":
            start = min(start, recalculateBorders(index - 1, self.trackList[self.activeTrack], None)[0])
        start = min(start, 3 * phonIndex)

        for i in range(start, end):
            self.repairBorders(i)
        self.submitNamedPhonParamChange(False, "borders", start, self.trackList[self.activeTrack].borders[start:end])

    def makeAutoPauses(self, index:int) -> None:
        """helper function for calculating the _autopause phonemes required by the note at position index of the active track"""

        if index < len(self.trackList[self.activeTrack].notes) - 1 and self.trackList[self.activeTrack].notes[index].phonemeEnd < len(self.trackList[self.activeTrack].phonemes):
            offset = 0
            if self.trackList[self.activeTrack].notes[index + 1].phonemeStart < self.trackList[self.activeTrack].notes[index + 1].phonemeEnd:
                if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart] == "_autopause":
                    offset -= 1
            if self.trackList[self.activeTrack].notes[index + 1].xPos - self.trackList[self.activeTrack].notes[index].xPos - self.trackList[self.activeTrack].notes[index].length > self.trackList[self.activeTrack].pauseThreshold:
                offset += 1
            if offset != 0:
                self.offsetPhonemes(index + 1, offset, True)
            if offset == 1:
                self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart] = "_autopause"
                self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index + 1].phonemeStart, self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart:self.trackList[self.activeTrack].notes[index + 1].phonemeEnd])
        if index > 0:
            offset = 0
            if self.trackList[self.activeTrack].notes[index].phonemeStart < self.trackList[self.activeTrack].notes[index].phonemeEnd:
                if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] == "_autopause":
                    offset -= 1
            if self.trackList[self.activeTrack].notes[index].xPos - self.trackList[self.activeTrack].notes[index - 1].xPos - self.trackList[self.activeTrack].notes[index - 1].length > self.trackList[self.activeTrack].pauseThreshold:
                offset += 1
            if offset != 0:
                self.offsetPhonemes(index, offset, True)
            if offset == 1:
                self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] = "_autopause"
                self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index].phonemeStart, self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd])

    def recalculatePauses(self, index:int) -> None:
        """recalculates all _autopause phonemes for the track at position index of the track list. Used during startup and repair operations."""

        for i in range(len(self.trackList[index].notes)):
            self.makeAutoPauses(i)
    
    def switchNote(self, index:int) -> None:
        """Switches the places of the notes at positions index and index + 1 or the active track. Does currently not clean up timing markers afterwards, so doing so manually or by prompting a call of offsetPhonemes is currently required."""

        note = self.trackList[self.activeTrack].notes.pop(index + 1)
        seq1 = self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd]
        seq2 = self.trackList[self.activeTrack].phonemes[note.phonemeStart:note.phonemeEnd]
        brd1 = self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeStart:3 * self.trackList[self.activeTrack].notes[index].phonemeEnd]
        brd2 = self.trackList[self.activeTrack].borders[3 * note.phonemeStart:3 * note.phonemeEnd]
        if index == len(self.trackList[self.activeTrack].notes) - 1:#notes are 1 element shorter because of pop
            scalingFactor = (self.trackList[self.activeTrack].notes[index].phonemeEnd - self.trackList[self.activeTrack].notes[index].phonemeStart + 1) / (note.phonemeEnd - note.phonemeStart + 1)
            for i in range(len(self.trackList[self.activeTrack].borders) - 3, len(self.trackList[self.activeTrack].borders)):
                self.trackList[self.activeTrack].borders[i] = (self.trackList[self.activeTrack].borders[i] - note.xPos) * scalingFactor / note.length * (self.trackList[self.activeTrack].notes[index].xPos - note.xPos) + self.trackList[self.activeTrack].notes[index].xPos
            for i in range(3 * note.phonemeEnd - 3 * note.phonemeStart):
                brd2[i] = (brd2[i] - note.xPos) * scalingFactor + note.xPos
        self.trackList[self.activeTrack].notes.insert(index, note)
        phonemeMid = note.phonemeEnd - note.phonemeStart
        self.trackList[self.activeTrack].notes[index].phonemeStart = self.trackList[self.activeTrack].notes[index + 1].phonemeStart
        self.trackList[self.activeTrack].notes[index + 1].phonemeEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        self.trackList[self.activeTrack].notes[index].phonemeEnd = self.trackList[self.activeTrack].notes[index].phonemeStart + phonemeMid
        self.trackList[self.activeTrack].notes[index + 1].phonemeStart = self.trackList[self.activeTrack].notes[index].phonemeStart + phonemeMid
        self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd] = seq2
        self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart:self.trackList[self.activeTrack].notes[index + 1].phonemeEnd] = seq1
        self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeStart:3 * self.trackList[self.activeTrack].notes[index].phonemeEnd] = brd2
        self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index + 1].phonemeStart:3 * self.trackList[self.activeTrack].notes[index + 1].phonemeEnd] = brd1
        self.makeAutoPauses(index + 1)
        self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index].phonemeStart, seq2)
        self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index + 1].phonemeStart, seq1)
        end = recalculateBorders(index + 1, self.trackList[self.activeTrack], None)[1]
        start = recalculateBorders(index, self.trackList[self.activeTrack], None)[0]
        for i in range(start, end):
            self.repairBorders(i)
        self.submitNamedPhonParamChange(False, "borders", start, self.trackList[self.activeTrack].borders[start:end])

    def scaleNote(self, index:int, oldLength:int) -> None:
        """Changes the length of the note at position index of the active track. The new length is read from its UI representation, the old length must be given as an argument.
        It does not perform any checks of surrounding notes or other conditions. Therefore, it is recommended to call changeNoteLength instead whenever possible."""

        start, end = recalculateBorders(index, self.trackList[self.activeTrack], oldLength)
        for i in range(start, end):
            self.repairBorders(i)
        self.submitNamedPhonParamChange(False, "borders", start, self.trackList[self.activeTrack].borders[start:end])
    
    def adjustNote(self, index:int, oldLength:int, oldPos:int) -> None:
        """Adjusts a note's attributes after it has been moved or scaled with respect to its surrounding notes. The current position and length are read from its UI representation, the old one must be given as arguments."""

        result = None
        nextLength = oldLength
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            if self.trackList[self.activeTrack].notes[index].xPos > self.trackList[self.activeTrack].notes[index + 1].xPos:
                nextLength = self.trackList[self.activeTrack].notes[index + 1].length
                if index + 2 < len(self.trackList[self.activeTrack].notes):
                    nextLength = max(min(nextLength, self.trackList[self.activeTrack].notes[index + 2].xPos - self.trackList[self.activeTrack].notes[index + 1].xPos), 1)
                self.switchNote(index)
                result = True
                self.scaleNote(index + 1, oldLength)
        self.scaleNote(index, nextLength)
        if index > 0:
            if self.trackList[self.activeTrack].notes[index - 1].xPos > self.trackList[self.activeTrack].notes[index].xPos:
                self.switchNote(index - 1)
                result = False
                self.scaleNote(index, max(min(oldPos - self.trackList[self.activeTrack].notes[index - 1].xPos, self.trackList[self.activeTrack].notes[index - 1].length), 1))
                self.scaleNote(index - 1, oldLength)
            else:
                self.scaleNote(index - 1, max(min(oldPos - self.trackList[self.activeTrack].notes[index - 1].xPos, self.trackList[self.activeTrack].notes[index - 1].length), 1))
        self.repairNotes(index)
        self.makeAutoPauses(index)
        if index > 0:
            self.recalculateBasePitch(index - 1, self.trackList[self.activeTrack].notes[index - 1].xPos, max(min(oldPos, self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length), 1))
        self.recalculateBasePitch(index, oldPos, oldPos + oldLength)
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            self.recalculateBasePitch(index + 1, oldPos + oldLength, oldPos + oldLength + nextLength)
        return result
    
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

    def submitAddTrack(self, track:dh.Track) -> None:
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
    
    def submitOffset(self, final:bool, index:int, offset:int, addition:int) -> None:
        self.manager.sendChange("offset", final, self.activeTrack, index, offset, addition)

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
        for i in self.trackList[track].notes:
            if i.phonemeEnd > index:
                break
        else:
            return
        if i.reference:
            i.reference.updateStatus(index, value)
    
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
        elif self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] == "_autopause":
            previousHeight = None
            oldStart = min(oldStart, int(self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeStart + 2]))
        else:
            previousHeight = self.trackList[self.activeTrack].notes[index - 1].yPos
            transitionLength1 = min(transitionLength1, self.trackList[self.activeTrack].notes[index - 1].length)
            if self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length < transitionPoint1:
                transitionLength1 += transitionPoint1 - self.trackList[self.activeTrack].notes[index - 1].xPos - self.trackList[self.activeTrack].notes[index - 1].length
                transitionPoint1 = (transitionPoint1 + self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length) / 2
        if index == len(self.trackList[self.activeTrack].notes) - 1:
            nextHeight = None
        elif self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart] == "_autopause":
            nextHeight = None
            oldEnd = max(oldEnd, int(self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeEnd]))
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
        self.submitNamedPhonParamChange(True, "pitch", start, data)
