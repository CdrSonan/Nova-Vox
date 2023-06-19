# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

global middleLayer

from copy import copy
from UI.code.editor.Main import middleLayer
from MiddleLayer.UndoRedo import enqueueUndo, enqueueRedo, clearRedoStack, enqueueUiCallback

class UnifiedAction:
    def __init__(self, action, undo = False, redo = False, uiCallback = None, immediate = True, *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.undo = undo
        self.redo = redo
        self.uiCallback = uiCallback
        if immediate:
            self.__call__()

    def inverseAction(self):
        return UnifiedAction(self.action, undo = self.undo, redo = self.redo, immediate = False, *self.args, **self.kwargs)

    def __call__(self):
        self.action(*self.args, **self.kwargs)
        if self.uiCallback:
            enqueueUiCallback(self.uiCallback)
        inverse = self.inverseAction()
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
    def __init__(self, voicebank, *args, **kwargs):
        # get name and inImage based on path
        
        super().__init__(middleLayer.importVoicebank, *args, **kwargs)
        self.voicebank = voicebank

class ChangeTrack(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        super().__init__(middleLayer.changeTrack, index, *args, **kwargs)
        self.previousTrack = copy(middleLayer.activeTrack)
        
    def inverseAction(self):
        return ChangeTrack(self.previousTrack, *self.args, **self.kwargs)

class CopyTrack(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        # get name and inImage based on existing track
        super().__init__(middleLayer.copyTrack, index, *args, **kwargs)

    def inverseAction(self):
        return DeleteTrack(len(middleLayer.trackList) - 1, *self.args, **self.kwargs)

class DeleteTrack(UnifiedAction):
    def __init__(self, track, *args, **kwargs):
        super().__init__(middleLayer.deleteTrack, *args, **kwargs)
        self.track = track

class AddParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.addParam, *args, **kwargs)
        self.param = param

class RemoveParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.removeParam, *args, **kwargs)
        self.param = param

class EnableParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.enableParam, *args, **kwargs)
        self.param = param

class DisableParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.disableParam, *args, **kwargs)
        self.param = param

class MoveParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.moveParam, *args, **kwargs)
        self.param = param

class SwitchParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.changeParam, *args, **kwargs)
        self.param = param

class ChangeParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        super().__init__(middleLayer.applyParamChanges, *args, **kwargs)
        self.param = param

class ChangePitch(UnifiedAction):
    def __init__(self, pitch, *args, **kwargs):
        super().__init__(middleLayer.applyPitchChanges, *args, **kwargs)
        self.pitch = pitch

class ChangeMode(UnifiedAction):
    def __init__(self, mode, *args, **kwargs):
        super().__init__(middleLayer.changePianoRollMode, *args, **kwargs)
        self.mode = mode

class Scroll(UnifiedAction):
    def __init__(self, scroll, *args, **kwargs):
        super().__init__(middleLayer.applyScroll, *args, **kwargs)
        self.scroll = scroll

class Zoom(UnifiedAction):
    def __init__(self, zoom, *args, **kwargs):
        super().__init__(middleLayer.applyZoom, *args, **kwargs)
        self.zoom = zoom

class AddNote(UnifiedAction):
    def __init__(self, note, *args, **kwargs):
        super().__init__(middleLayer.addNote, *args, **kwargs)
        self.note = note

class RemoveNote(UnifiedAction):
    def __init__(self, note, *args, **kwargs):
        super().__init__(middleLayer.removeNote, *args, **kwargs)
        self.note = note

class ChangeNoteLength(UnifiedAction):
    def __init__(self, note, *args, **kwargs):
        super().__init__(middleLayer.changeNoteLength, *args, **kwargs)
        self.note = note

class MoveNote(UnifiedAction):
    def __init__(self, note, *args, **kwargs):
        super().__init__(middleLayer.moveNote, *args, **kwargs)
        self.note = note

class ChangeNoteStart(UnifiedAction):
    def __init__(self, note, *args, **kwargs):
        super().__init__(middleLayer.changeNoteLength, *args, **kwargs)
        self.note = note

class ChangeLyrics(UnifiedAction):
    def __init__(self, note, *args, **kwargs):
        super().__init__(middleLayer.changeLyrics, *args, **kwargs)
        self.note = note

class MoveBorder(UnifiedAction):
    def __init__(self, border, *args, **kwargs):
        super().__init__(middleLayer.changeBorder, *args, **kwargs)
        self.border = border

class ChangeVoicebank(UnifiedAction):
    def __init__(self, voicebank, *args, **kwargs):
        super().__init__(middleLayer.changeVB, *args, **kwargs)
        self.voicebank = voicebank

class RepairNotes(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.repairNotes, *args, **kwargs)

class RepairBorders(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.repairBorders, *args, **kwargs)

class ForceChangeTrackLength(UnifiedAction):
    def __init__(self, track, *args, **kwargs):
        super().__init__(middleLayer.changeLength, *args, **kwargs)
        self.track = track

class Validate(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.validate, *args, **kwargs)

class ChangeVolume(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.updateVolume, *args, **kwargs)

class MovePlayhead(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.movePlayhead, *args, **kwargs)

class Play(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.play, *args, **kwargs)

class RestartRenderer(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.manager.restart, *args, **kwargs)

class SaveNVX(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.saveNVX, *args, **kwargs)

class LoadNVX(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.loadNVX, *args, **kwargs)

class ExportFile(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.exportFile, *args, **kwargs)

class ImportFile(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.importFile, *args, **kwargs)

class LoadNVXPartial(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.loadNVXPartial, *args, **kwargs)

class SaveNVXPartial(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.saveNVXPartial, *args, **kwargs)

class ChangeSettings(UnifiedAction):
    def __init__(self, *args, **kwargs):
        super().__init__(middleLayer.changeSettings, *args, **kwargs)