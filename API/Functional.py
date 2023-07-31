# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

global middleLayer

from copy import copy
import os

import torch

from UI.code.editor.Main import middleLayer
from MiddleLayer.IniParser import readSettings
from MiddleLayer.FileIO import loadNVX, validateTrackData
from MiddleLayer.DataHandlers import Note
from MiddleLayer.UndoRedo import enqueueUndo, enqueueRedo, clearRedoStack, enqueueUiCallback

class UnifiedAction:
    def __init__(self, action, undo = False, redo = False, useUiCallback = False, immediate = False, *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.undo = undo
        self.redo = redo
        self.useUiCallback = useUiCallback
        if immediate:
            self.__call__()

    def inverseAction(self):
        return UnifiedAction(self.action, undo = self.undo, redo = self.redo, immediate = False, *self.args, **self.kwargs)

    def __call__(self):
        inverse = self.inverseAction()
        self.action(*self.args, **self.kwargs)
        if self.useUiCallback:
            enqueueUiCallback(self.uiCallback())
        if self.undo:
            inverse.undo = False
            inverse.redo = True
            enqueueRedo(inverse)
        elif self.redo:
            inverse.undo = True
            inverse.redo = False
            enqueueUndo(inverse)
        else:
            inverse.undo = True
            inverse.redo = False
            enqueueUndo(inverse)
            clearRedoStack()

    def __repr__(self):
        return f"UnifiedAction(action={self.action}, undo = {self.undo}, redo = {self.redo}, UI Callback = {self.uiCallback}, args={self.args}, kwargs={self.kwargs})"

class UiCallback:
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        self.callback(*self.args, **self.kwargs)

    def __eq__(self, other):
        if other.__class__ == self.__class__:
            return self.callback == other.callback and self.args == other.args and self.kwargs == other.kwargs
        else:
            return False

    def __repr__(self):
        return f"UiCallback(callback={self.callback}, args={self.args}, kwargs={self.kwargs})"

class UnifiedActionGroup:
    def __init__(self, *actions, undo = False, redo = False, immediate = True):
        self.actions = actions
        self.undo = undo
        self.redo = redo
        self.immediate = immediate
        for i in range(len(self.actions)):
            self.actions[i].undo = self.undo
            self.actions[i].redo = self.redo
        if immediate:
            self.__call__()

    def inverseAction(self):
        actions = [action.inverseAction() for action in self.actions].reverse()
        if self.undo:
            return UnifiedActionGroup(*actions, undo = False, redo = True, immediate = False)
        else:
            return UnifiedActionGroup(*actions, undo = True, redo = False, immediate = False)

    def __call__(self):
        for action in self.actions:
            action.action(*action.args, **action.kwargs)
            if action.uiCallback:
                enqueueUiCallback(action.uiCallback)
        inverse = self.inverseAction()
        if self.undo:
            enqueueRedo(inverse)
        elif self.redo:
            enqueueUndo(inverse)
        else:
            enqueueUndo(inverse)
            clearRedoStack()

def runUiCallbacks():
    middleLayer.runUiCallbacks()

class ImportVoicebank(UnifiedAction):
    def __init__(self, file, *args, **kwargs):
        data = torch.load(os.path.join(readSettings()["datadir"], "Voices", file), map_location = torch.device("cpu"))["metadata"]
        super().__init__(middleLayer.importVoicebank, file, data.name, data.image, *args, **kwargs)
        self.index = len(middleLayer.trackList) - 1

    def inverseAction(self):
        return DeleteTrack(self.index, *self.args, **self.kwargs)

class ChangeTrack(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        super().__init__(middleLayer.changeTrack, index, *args, **kwargs)
        
    def inverseAction(self):
        return ChangeTrack(copy(middleLayer.activeTrack), *self.args, **self.kwargs)

class CopyTrack(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        data = torch.load(os.path.join(readSettings()["datadir"], "Voices", middleLayer.trackList[index].vbPath), map_location = torch.device("cpu"))["metadata"]
        super().__init__(middleLayer.copyTrack, index, data.name, data.image, *args, **kwargs)

    def inverseAction(self):
        return DeleteTrack(len(middleLayer.trackList) - 1, *self.args, **self.kwargs)

class DeleteTrack(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        super().__init__(middleLayer.deleteTrack, index, *args, **kwargs)
        self.index = index

    def inverseAction(self):
        return UnifiedAction(middleLayer.addTrack, copy(middleLayer.trackList[self.index]), *self.args, **self.kwargs)

class AddParam(UnifiedAction):
    def __init__(self, param, name, *args, **kwargs):
        super().__init__(middleLayer.importParam, param, name, *args, **kwargs)

    def inverseAction(self):
        return RemoveParam(len(middleLayer.activeTrack.paramList) - 1, *self.args, **self.kwargs)

class RemoveParam(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        super().__init__(middleLayer.deleteParam, index, *args, **kwargs)
        self.index = index

    def inverseAction(self):
        return UnifiedAction(middleLayer.addParam, copy(middleLayer.activeTrack.paramList[self.index]), *self.args, **self.kwargs)

class EnableParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.enableParam, param, *args, **kwargs)
        self.param = param

    def inverseAction(self):
        return DisableParam(self.param, *self.args, **self.kwargs)
    
    def uiCallback(self):
        for i in middleLayer.ids["paramList"].children:
            if i.name == self.param:
                i.children[0].state = "down"

class DisableParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.disableParam, param, *args, **kwargs)
        self.param = param

    def inverseAction(self):
        return EnableParam(self.param, *self.args, **self.kwargs)
    
    def uiCallback(self):
        for i in middleLayer.ids["paramList"].children:
            if i.name == self.param:
                i.children[0].state = "normal"

class MoveParam(UnifiedAction):
    def __init__(self, index, delta, *args, **kwargs):
        super().__init__(middleLayer.moveParam, index, delta, *args, **kwargs)
        self.index = index
        self.delta = delta

    def inverseAction(self):
        return MoveParam(self.index + self.delta, -self.delta, *self.args, **self.kwargs)

class SwitchParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.changeParam, param, *args, **kwargs)

    def inverseAction(self):
        return SwitchParam(copy(middleLayer.activeParam), *self.args, **self.kwargs)
    
    def uiCallback(self):
        for i in middleLayer.ids["paramList"].children:
            if i.name == middleLayer.activeParam:
                i.children[0].state = "down"

class ChangeParam(UnifiedAction):
    def __init__(self, data, start, section = None, *args, **kwargs):
        super().__init__(middleLayer.applyParamChanges, data, start, section, *args, **kwargs)
        self.start = start
        self.section = section
        self.length = data.size()[0]

    def inverseAction(self):
        if middleLayer.activeParam == "steadiness":
            oldData = middleLayer.trackList[middleLayer.activeTrack].steadiness[self.start:self.start + self.length]
        elif middleLayer.activeParam == "breathiness":
            oldData = middleLayer.trackList[middleLayer.activeTrack].breathiness[self.start:self.start + self.length]
        elif middleLayer.activeParam == "AI Balance":
            oldData = middleLayer.trackList[middleLayer.activeTrack].aiBalance[self.start:self.start + self.length]
        elif middleLayer.activeParam == "loop":
            if self.section:
                oldData = middleLayer.trackList[middleLayer.activeTrack].loopOverlap[self.start:self.start + self.length]
            else:
                oldData = middleLayer.trackList[middleLayer.activeTrack].loopOffset[self.start:self.start + self.length]
        elif middleLayer.activeParam == "vibrato":
            if self.section:
                oldData = middleLayer.trackList[middleLayer.activeTrack].vibratoSpeed[self.start:self.start + self.length]
            else:
                oldData = middleLayer.trackList[middleLayer.activeTrack].vibratoStrength[self.start:self.start + self.length]
        else:
            oldData = middleLayer.trackList[middleLayer.activeTrack].nodegraph.params[self.activeParam].curve[self.start:self.start + self.length]
        return ChangeParam(oldData, self.start, self.section, *self.args, **self.kwargs)

    def uiCallback(self):
        middleLayer.ids["adaptiveSpace"].redraw()

class ChangePitch(UnifiedAction):
    def __init__(self, data, start, *args, **kwargs):
        super().__init__(middleLayer.applyPitchChanges, data, start, *args, **kwargs)
        self.start = start
        self.size = data.size()[0]

    def inverseAction(self):
        return ChangePitch(middleLayer.trackList[middleLayer.activeTrack].pitch[self.start:self.start + self.size], self.start, *self.args, **self.kwargs)
    
    def uiCallback(self):
        if middleLayer.mode == "pitch":
            middleLayer.ids["pianoRoll"].redrawPitch()

class ChangeMode(UnifiedAction):
    def __init__(self, mode, *args, **kwargs):
        super().__init__(middleLayer.changePianoRollMode, mode, *args, **kwargs)
        
    def inverseAction(self):
        return ChangeMode(middleLayer.mode, *self.args, **self.kwargs)

class Scroll(UnifiedAction):
    def __init__(self, scroll, *args, **kwargs):
        super().__init__(middleLayer.applyScroll, scroll, *args, **kwargs)

    def inverseAction(self):
        return Scroll(middleLayer.scrollValue, *self.args, **self.kwargs)

class Zoom(UnifiedAction):
    def __init__(self, zoom, *args, **kwargs):
        super().__init__(middleLayer.applyZoom, zoom, *args, **kwargs)

    def inverseAction(self):
        return Zoom(middleLayer.xScale, *self.args, **self.kwargs)

class AddNote(UnifiedAction):
    def __init__(self, index, x, y, *args, **kwargs):
        super().__init__(middleLayer.addNote, index, x, y, None, *args, **kwargs)
        self.index = index
        self.x = x
        self.y = y
        
    def inverseAction(self):
        return RemoveNote(self.index, *self.args, **self.kwargs)
    
    def UiCallback(self):
        middleLayer.ids["pianoRoll"].add_widget(Note(index = self.index, xPos = self.x, yPos = self.y, length = 100, height = middleLayer.ids["pianoRoll"].yScale))

class RemoveNote(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        super().__init__(middleLayer.removeNote, index, *args, **kwargs)
        self.index = index
    
    def inverseAction(self):
        note = middleLayer.trackList[middleLayer.activeTrack].notes[self.index]
        return AddNote(note.index, note.x, note.y, note.reference, *self.args, **self.kwargs) #TODO: process remaining note data
    
    def UiCallback(self):
        middleLayer.ids["pianoRoll"].remove_widget(middleLayer.ids["pianoRoll"].notes[self.index])
        del middleLayer.ids["pianoRoll"].notes[self.index]

class ChangeNoteLength(UnifiedAction):
    def __init__(self, index, x, length, *args, **kwargs):
        self.index = index
        super().__init__(middleLayer.changeNoteLength, index, x, length, *args, **kwargs)
    
    def inverseAction(self):
        return ChangeNoteLength(self.index, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].xPos, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].length, *self.args, **self.kwargs)
    
    def UiCallback(self):
        middleLayer.ids["pianoRoll"].notes[self.index].length = self.length
        middleLayer.ids["pianoRoll"].notes[self.index].redraw()

class MoveNote(UnifiedAction):
    def __init__(self, index, x, y, *args, **kwargs):
        self.index = index
        super().__init__(middleLayer.moveNote, index, x, y, *args, **kwargs)
    
    def inverseAction(self):
        return MoveNote(self.index, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].xPos, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].yPos, *self.args, **self.kwargs)
    
    def UiCallback(self):
        middleLayer.ids["pianoRoll"].notes[self.index].x = middleLayer.trackList[middleLayer.activeTrack].notes[self.index].xPos
        middleLayer.ids["pianoRoll"].notes[self.index].y = middleLayer.trackList[middleLayer.activeTrack].notes[self.index].yPos
        middleLayer.ids["pianoRoll"].notes[self.index].redraw()

class ChangeLyrics(UnifiedAction):
    def __init__(self, index, inputText, pronuncIndex, *args, **kwargs):
        super().__init__(middleLayer.changeLyrics, index, inputText, pronuncIndex, *args, **kwargs)
        self.index = index
    
    def inverseAction(self):
        return ChangeLyrics(self.index, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].content, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].pronuncIndex, *self.args, **self.kwargs)
    
    def UiCallback(self):
        middleLayer.ids["pianoRoll"].notes[self.index].redrawStatusBars()

class MoveBorder(UnifiedAction):
    def __init__(self, border, pos, *args, **kwargs):
        super().__init__(middleLayer.changeBorder, border, pos, *args, **kwargs)
        self.border = border
    
    def inverseAction(self):
        return MoveBorder(self.border, middleLayer.trackList[middleLayer.activeTrack].borders[self.border], *self.args, **self.kwargs)
    
    def UiCallback(self):
        if middleLayer.mode == "timing":
            middleLayer.ids["pianoRoll"].removeTiming()
            middleLayer.ids["pianoRoll"].drawTiming()

class ChangeVoicebank(UnifiedAction):
    def __init__(self, index, path, *args, **kwargs):
        super().__init__(middleLayer.changeVB, *args, **kwargs)
        self.index = index
    
    def inverseAction(self):
        return ChangeVoicebank(self.index, middleLayer.trackList[middleLayer.activeTrack].vbPath, *self.args, **self.kwargs)

class ForceChangeTrackLength(UnifiedAction):
    def __init__(self, length, *args, **kwargs):
        super().__init__(middleLayer.changeLength, length, *args, **kwargs)
    
    def inverseAction(self):
        return ForceChangeTrackLength(middleLayer.trackList[middleLayer.activeTrack].length, *self.args, **self.kwargs)

class ChangeVolume(UnifiedAction):
    def __init__(self, index, volume, *args, **kwargs):
        super().__init__(middleLayer.updateVolume, index, volume, *args, **kwargs)
        self.index = index
    
    def inverseAction(self):
        return ChangeVolume(self.index, middleLayer.trackList[middleLayer.activeTrack].volume, *self.args, **self.kwargs)

class LoadNVX(UnifiedAction):
    def __init__(self, path, *args, **kwargs):
        super().__init__(loadNVX, path, middleLayer, *args, **kwargs)
    
    def inverseAction(self):
        def restoreTrackList(tracks:list):
            for i in range(len(middleLayer.trackList)):
                middleLayer.deleteTrack(0)
            for trackData in tracks:
                track = validateTrackData(trackData)
                vbData = torch.load(track["vbPath"], map_location = torch.device("cpu"))["metadata"]
                middleLayer.importVoicebankNoSubmit(track["vbPath"], vbData.name, vbData.image)
                middleLayer.trackList[-1].volume = track["volume"]
                for note in track["notes"]:
                    middleLayer.trackList[-1].notes.append(Note(note["xPos"], note["yPos"], note["phonemeStart"], note["phonemeEnd"]))
                    middleLayer.trackList[-1].notes[-1].length = note["length"]
                    middleLayer.trackList[-1].notes[-1].phonemeMode = note["phonemeMode"]
                    middleLayer.trackList[-1].notes[-1].content = note["content"]
                middleLayer.trackList[-1].phonemes = track["phonemes"]
                middleLayer.trackList[-1].pitch = track["pitch"]
                middleLayer.trackList[-1].basePitch = track["basePitch"]
                middleLayer.trackList[-1].breathiness = track["breathiness"]
                middleLayer.trackList[-1].steadiness = track["steadiness"]
                middleLayer.trackList[-1].aiBalance = track["aiBalance"]
                middleLayer.trackList[-1].loopOverlap = track["loopOverlap"]
                middleLayer.trackList[-1].loopOffset = track["loopOffset"]
                middleLayer.trackList[-1].vibratoSpeed = track["vibratoSpeed"]
                middleLayer.trackList[-1].vibratoStrength = track["vibratoStrength"]
                middleLayer.trackList[-1].usePitch = track["usePitch"]
                middleLayer.trackList[-1].useBreathiness = track["useBreathiness"]
                middleLayer.trackList[-1].useSteadiness = track["useSteadiness"]
                middleLayer.trackList[-1].useAIBalance = track["useAIBalance"]
                middleLayer.trackList[-1].useVibratoSpeed = track["useVibratoSpeed"]
                middleLayer.trackList[-1].useVibratoStrength = track["useVibratoStrength"]
                middleLayer.trackList[-1].nodegraph = track["nodegraph"]
                middleLayer.trackList[-1].borders = track["borders"]
                middleLayer.trackList[-1].length = track["length"]
                middleLayer.trackList[-1].mixinVB = track["mixinVB"]
                middleLayer.trackList[-1].pauseThreshold = track["pauseThreshold"]
        return UnifiedAction(restoreTrackList, middleLayer.trackList, *self.args, **self.kwargs)
