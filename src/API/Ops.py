# Copyright 2023, 2024 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

global middleLayer

from copy import copy, deepcopy
import os

import torch

import h5py

from io import BytesIO
from kivy.core.image import Image as CoreImage

from UI.editor.Main import middleLayer
from MiddleLayer.IniParser import readSettings
from MiddleLayer.FileIO import validateTrackData
from MiddleLayer.DataHandlers import Note, Track
from MiddleLayer.UndoRedo import enqueueUndo, enqueueRedo, clearRedoStack

from Backend.VB_Components.Voicebank import LiteVoicebank
from Backend.DataHandler.HDF5 import MetadataStorage

from Util import noteToPitch, convertFormat

import global_consts

from UI.editor.Headers import SingerPanel, ParamPanel
from UI.editor.PianoRoll import PhonemeSelector, Note as UiNote

from API.Addon import override

class UnifiedAction:
    def __init__(self, action, *args, undo = False, redo = False, immediate = False, uiCallback = True, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.undo = undo
        self.redo = redo
        self.useUiCallback = uiCallback
        if immediate:
            self.__call__()

    def inverseAction(self):
        return UnifiedAction(self.action, undo = self.undo, redo = self.redo, immediate = False, *self.args, **self.kwargs)
    
    def uiCallback(self):
        return
    
    def merge(self, other):
        return None

    def __call__(self, *args, **kwargs):
        if middleLayer.undoActive:
            with NoUndo():
                inverse = self.inverseAction()
                returnValue = self.action(*self.args, **self.kwargs)
                if self.useUiCallback:
                    self.uiCallback()
        else:
            inverse = self.inverseAction()
            returnValue = self.action(*self.args, **self.kwargs)
            if self.useUiCallback:
                self.uiCallback()
        if not middleLayer.undoActive:
            return
        if self.undo:
            inverse.undo = False
            inverse.redo = True
            enqueueRedo(inverse)
        elif self.redo:
            inverse.undo = True
            inverse.redo = False
            enqueueUndo(inverse)
        elif self.undo != None:
            inverse.undo = True
            inverse.redo = False
            enqueueUndo(inverse)
            clearRedoStack()
        return returnValue

    def __repr__(self):
        return f"UnifiedAction(action={self.action}, undo = {self.undo}, redo = {self.redo}, args={self.args}, kwargs={self.kwargs})"

class UnifiedActionGroup:
    def __init__(self, *actions, undo = False, redo = False, immediate = False):
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
    
    def merge(self, other):
        return None

    def __call__(self):
        with NoUndo():
            inverse = self.inverseAction()
            for action in self.actions:
                action.action(*action.args, **action.kwargs)
                if action.uiCallback:
                    action.uiCallback()
        if not middleLayer.undoActive:
            return
        if self.undo:
            enqueueRedo(inverse)
        elif self.redo:
            enqueueUndo(inverse)
        else:
            enqueueUndo(inverse)
            clearRedoStack()
    
    def append(self, action):
        self.actions.append(action)
        self.actions[-1].undo = self.undo
        self.actions[-1].redo = self.redo
    
    def __len__(self):
        return len(self.actions)

class NoUndo():
    def __enter__(self):
        middleLayer.undoActive = False
    def __exit__(self, exc_type, exc_value, traceback):
        middleLayer.undoActive = True

class SingleUndo():
    def __enter__(self):
        middleLayer.singleUndoActive = True
        middleLayer.undoStack.append(UnifiedActionGroup())
    def __exit__(self, exc_type, exc_value, traceback):
        middleLayer.singleUndoActive = False
        if len(middleLayer.undoStack[-1]) == 0:
            middleLayer.undoStack.pop(-1)

def singleUndo(func):
    def wrapper(*args, **kwargs):
        with SingleUndo():
            func(*args, **kwargs)
    return wrapper

def _importVoicebankNoSubmit(path:str, name:str, inImage) -> None:
        """Creates a new vocal track with a Voicebank loaded from disk, but does not submit it to the rendering process.
        The rendering process needs to be restarted, or submitAddTrack needs to be called separately for the new track to be recognized

        Arguments:
            path: filepath of the .nvvb Voicebank file used for the track

            name: display name of the Voicebank/track

            inImage: image displayed in the track header"""
        

        track = Track(path)
        middleLayer.trackList.append(track)
        canvas_img = inImage
        data = BytesIO()
        canvas_img.save(data, format='png')
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext='png')
        image = im.texture
        middleLayer.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(middleLayer.trackList) - 1))
        middleLayer.audioBuffer.append(torch.zeros([track.length * global_consts.batchSize,]))

class ImportVoicebank(UnifiedAction):
    def __init__(self, file, *args, **kwargs):
        @override
        def action(file, name, image):
            track = Track(file)
            middleLayer.trackList.append(track)
            data = BytesIO()
            image.save(data, format='png')
            data.seek(0)
            im = CoreImage(BytesIO(data.read()), ext='png')
            image = im.texture
            middleLayer.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(middleLayer.trackList) - 1))
            middleLayer.audioBuffer.append(torch.zeros([track.length * global_consts.batchSize,]))
            middleLayer.ids["singerList"].children[0].children[0].trigger_action(duration = 0)
            middleLayer.submitAddTrack(middleLayer.trackList[-1])
        with h5py.File(os.path.join(readSettings()["datadir"], "Voices", file), "r") as f:
            data = MetadataStorage(f).toMetadata()
        super().__init__(action, file, data.name, data.image, *args, **kwargs)

    def inverseAction(self):
        return DeleteTrack(-1)

class ChangeTrack(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        @override
        def action(index):
            middleLayer.activeTrack = index
            middleLayer.ui.updateParamPanel()
            middleLayer.updatePianoRoll()
        super().__init__(action, index, *args, **kwargs)
        
    def inverseAction(self):
        return ChangeTrack(copy(middleLayer.activeTrack))

class CopyTrack(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        @override
        def action(index, name, image):
            data = BytesIO()
            image.save(data, format='png')
            data.seek(0)
            im = CoreImage(BytesIO(data.read()), ext='png')
            image = im.texture
            reference = middleLayer.trackList[index]
            middleLayer.trackList.append(Track(reference.vbPath))
            middleLayer.trackList[-1].volume = copy(reference.volume)
            for i in reference.notes:
                middleLayer.trackList[-1].notes.append(Note(i.xPos, i.yPos, middleLayer.trackList[-1], None))
                middleLayer.trackList[-1].notes[-1].length = copy(i.length)
                middleLayer.trackList[-1].notes[-1].phonemeMode = copy(i.phonemeMode)
                middleLayer.trackList[-1].notes[-1].content = copy(i.content)
                middleLayer.trackList[-1].notes[-1].phonemes = deepcopy(i.phonemes)
                middleLayer.trackList[-1].notes[-1].borders = deepcopy(i.borders)
                middleLayer.trackList[-1].notes[-1].pronuncIndex = copy(i.pronuncIndex)
                middleLayer.trackList[-1].notes[-1].autopause = copy(i.autopause)
                middleLayer.trackList[-1].notes[-1].carryOver = copy(i.carryOver)
            middleLayer.trackList[-1].pitch = reference.pitch.clone()
            middleLayer.trackList[-1].basePitch = reference.basePitch.clone()
            middleLayer.trackList[-1].breathiness = reference.breathiness.clone()
            middleLayer.trackList[-1].steadiness = reference.steadiness.clone()
            middleLayer.trackList[-1].aiBalance = reference.aiBalance.clone()
            middleLayer.trackList[-1].loopOverlap = reference.loopOverlap.clone()
            middleLayer.trackList[-1].loopOffset = reference.loopOffset.clone()
            middleLayer.trackList[-1].vibratoSpeed = reference.vibratoSpeed.clone()
            middleLayer.trackList[-1].vibratoStrength = reference.vibratoStrength.clone()
            middleLayer.trackList[-1].usePitch = copy(reference.usePitch)
            middleLayer.trackList[-1].useBreathiness = copy(reference.useBreathiness)
            middleLayer.trackList[-1].useSteadiness = copy(reference.useSteadiness)
            middleLayer.trackList[-1].useAIBalance = copy(reference.useAIBalance)
            middleLayer.trackList[-1].useVibratoSpeed = copy(reference.useVibratoSpeed)
            middleLayer.trackList[-1].useVibratoStrength = copy(reference.useVibratoStrength)
            middleLayer.trackList[-1].nodegraph = copy(reference.nodegraph)
            middleLayer.trackList[-1].length = copy(reference.length)
            middleLayer.trackList[-1].mixinVB = copy(reference.mixinVB)
            middleLayer.trackList[-1].pauseThreshold = copy(reference.pauseThreshold)
            middleLayer.trackList[-1].borders.wrappingBorders = deepcopy(reference.borders.wrappingBorders)
            middleLayer.trackList[-1].buildPhonemeIndices()
            middleLayer.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(middleLayer.trackList) - 1))
            middleLayer.audioBuffer.append(deepcopy(middleLayer.audioBuffer[index]))
            middleLayer.ids["singerList"].children[0].children[0].trigger_action(duration = 0)
            middleLayer.submitDuplicateTrack(index)
        with h5py.File(os.path.join(readSettings()["datadir"], "Voices", middleLayer.trackList[index].vbPath), "r") as f:
            data = MetadataStorage(f).toMetadata()
        super().__init__(action, index, data.name, data.image, *args, **kwargs)

    def inverseAction(self):
        return DeleteTrack(-1)

class DeleteTrack(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        @override
        def action(index):
            if index < 0:
                index += len(middleLayer.trackList)
            middleLayer.trackList.pop(index)
            middleLayer.audioBuffer.pop(index)
            toRemove = None
            for i in middleLayer.ids["singerList"].children:
                if i.index > index:
                    i.index -= 1
                elif i.index == index:
                    toRemove = i
                elif i.index == index - 1:
                    i.children[0].trigger_action(duration = 0)
            if toRemove != None:
                middleLayer.ids["singerList"].remove_widget(toRemove)
            middleLayer.deletions.append(index)
            middleLayer.submitRemoveTrack(index)
            if len(middleLayer.trackList) == 0:
                middleLayer.ids["paramList"].clear_widgets()
                middleLayer.ids["adaptiveSpace"].clear_widgets()
                middleLayer.activeTrack = None
                middleLayer.updatePianoRoll()
        super().__init__(action, index, *args, **kwargs)
        self.index = index
    
    def inverseAction(self):
        
        return _ReinsertTrack(copy(middleLayer.trackList[self.index]), self.index)

class _ReinsertTrack(UnifiedAction):
    def __init__(self, track, index, *args, **kwargs):
        @override
        def action(track, index):
            with h5py.File(os.path.join(readSettings()["datadir"], "Voices", track.vbPath), "r") as f:
                metadata = MetadataStorage(f).toMetadata()
            middleLayer.trackList.insert(index, track)
            data = BytesIO()
            metadata.image.save(data, format='png')
            data.seek(0)
            im = CoreImage(BytesIO(data.read()), ext='png')
            image = im.texture
            middleLayer.ids["singerList"].add_widget(SingerPanel(name = metadata.name, image = image, index = index), index = len(middleLayer.trackList) - 1 - index)
            for i in middleLayer.ids["singerList"].children:
                if i.index > index:
                    i.index += 1
                elif i.index == index:
                    i.children[0].trigger_action(duration = 0)
            middleLayer.audioBuffer.insert(index, torch.zeros([track.length * global_consts.batchSize,]))
            middleLayer.ids["singerList"].children[0].children[0].trigger_action(duration = 0)
            middleLayer.submitAddTrack(middleLayer.trackList[-1])
        super().__init__(action, track, index, *args, **kwargs)
        self.index = index

    def inverseAction(self):
        return DeleteTrack(self.index)

class AddParam(UnifiedAction):
    def __init__(self, param, name, *args, **kwargs):
        @override
        def action(param, name):
            pass
        super().__init__(action, param, name, *args, **kwargs)

    def inverseAction(self):
        return RemoveParam(len(middleLayer.activeTrack.paramList) - 1)

class RemoveParam(UnifiedAction):
    def __init__(self, name, *args, **kwargs):
        @override
        def action(name):
            middleLayer.trackList[self.activeTrack].nodegraph.delete(name)
            if name == self.activeParam:
                middleLayer.changeParam(self.activeParam - 1)
            for i in middleLayer.ids["paramList"].children:
                if i.name == name:
                    i.parent.remove_widget(i)
        middleLayer.submitNodegraphUpdate
        super().__init__(action, name, *args, **kwargs)
        self.name = name

    def inverseAction(self):
        return UnifiedAction(middleLayer.addParam, copy(middleLayer.activeTrack.paramList[self.name]))

class EnableParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        @override
        def action(name):
            if name == "steadiness":
                middleLayer.trackList[middleLayer.activeTrack].useSteadiness = True
            elif name == "breathiness":
                middleLayer.trackList[middleLayer.activeTrack].useBreathiness = True
            elif name == "AI balance":
                middleLayer.trackList[middleLayer.activeTrack].useAIBalance = True
            elif name == "vibrato speed":
                middleLayer.trackList[middleLayer.activeTrack].useVibratoSpeed = True
            elif name == "vibrato strength":
                middleLayer.trackList[middleLayer.activeTrack].useVibratoStrength = True
            else:
                middleLayer.trackList[middleLayer.activeTrack].nodegraph.params[name].enabled = True
            middleLayer.submitEnableParam(name)
        super().__init__(action, param)
        self.param = param

    def inverseAction(self):
        return DisableParam(self.param, *self.args, **self.kwargs)
    
    def uiCallback(self):
        for i in middleLayer.ids["paramList"].children:
            if i.name == self.param:
                i.children[0].state = "down"

class DisableParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        @override
        def action(name):
            if name == "steadiness":
                middleLayer.trackList[middleLayer.activeTrack].useSteadiness = False
            elif name == "breathiness":
                middleLayer.trackList[middleLayer.activeTrack].useBreathiness = False
            elif name == "AI balance":
                middleLayer.trackList[middleLayer.activeTrack].useAIBalance = False
            elif name == "vibrato speed":
                middleLayer.trackList[middleLayer.activeTrack].useVibratoSpeed = False
            elif name == "vibrato strength":
                middleLayer.trackList[middleLayer.activeTrack].useVibratoStrength = False
            else:
                middleLayer.trackList[middleLayer.activeTrack].nodegraph.params[name].enabled = False
            middleLayer.submitDisableParam(name)
        super().__init__(action, param)
        self.param = param

    def inverseAction(self):
        return EnableParam(self.param, *self.args, **self.kwargs)
    
    def uiCallback(self):
        for i in middleLayer.ids["paramList"].children:
            if i.name == self.param:
                i.children[0].state = "normal"

class MoveParam(UnifiedAction):
    def __init__(self, index, delta, *args, **kwargs):
        @override
        def action(index, delta):
            for i in middleLayer.ids["paramList"].children:
                if i.index == index:
                    param = middleLayer.trackList[middleLayer.activeTrack].nodegraph.params[i.name]
                    i.parent.remove_widget(i)
                    break
            middleLayer.ids["paramList"].add_widget(ParamPanel(name = param.name, switchable = param.switchable, sortable = param.sortable, deletable = param.deletable, index = index), index = index + delta, switchState = param.switchState)
            middleLayer.changeParam(index + delta)
        super().__init__(action, index, delta, *args, **kwargs)
        self.index = index
        self.delta = delta

    def inverseAction(self):
        return MoveParam(self.index + self.delta, -self.delta)

class SwitchParam(UnifiedAction):
    def __init__(self, param, *args, **kwargs):
        @override
        def action(name):
            if name == "loop overlap" or name == "loop offset":
                middleLayer.activeParam = "loop"
            elif name == "vibrato speed" or name == "vibrato strength":
                middleLayer.activeParam = "vibrato"
            else:
                middleLayer.activeParam = name
            middleLayer.ids["adaptiveSpace"].children[0].redraw()
        super().__init__(action, param, *args, **kwargs)

    def inverseAction(self):
        #TODO: fix UI disappearing when middleLayer.activeParam is None
        return SwitchParam(copy(middleLayer.activeParam))
    
    def uiCallback(self):
        if middleLayer.activeParam == "loop" and middleLayer.mode != "timing":
            middleLayer.ui.setMode("timing")
        elif middleLayer.activeParam == "vibrato" and middleLayer.mode != "pitch":
            middleLayer.ui.setMode("pitch")
        elif middleLayer.activeParam != "loop" and middleLayer.activeParam != "vibrato" and middleLayer.mode != "notes":
            middleLayer.ui.setMode("notes")
        for i in middleLayer.ids["paramList"].children:
            if i.name == middleLayer.activeParam:
                i.state = "down"
            else:
                i.state = "normal"

class ChangeParam(UnifiedAction):
    def __init__(self, data, start, section = None, *args, **kwargs):
        @override
        def action(data, start, section):
            if middleLayer.activeParam == "steadiness":
                middleLayer.trackList[middleLayer.activeTrack].steadiness[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                middleLayer.submitNamedParamChange(True, "steadiness", start, torch.tensor(data, dtype = torch.half))
            elif middleLayer.activeParam == "breathiness":
                middleLayer.trackList[middleLayer.activeTrack].breathiness[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                middleLayer.submitNamedParamChange(True, "breathiness", start, torch.tensor(data, dtype = torch.half))
            elif middleLayer.activeParam == "AI balance":
                middleLayer.trackList[middleLayer.activeTrack].aiBalance[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                middleLayer.submitNamedParamChange(True, "aiBalance", start, torch.tensor(data, dtype = torch.half))
            elif middleLayer.activeParam == "loop":
                if section:
                    middleLayer.trackList[middleLayer.activeTrack].loopOverlap[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                    middleLayer.submitNamedPhonParamChange(True, "repetititionSpacing", start, torch.tensor(data, dtype = torch.half))
                else:
                    middleLayer.trackList[middleLayer.activeTrack].loopOffset[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                    middleLayer.submitNamedPhonParamChange(True, "offsets", start, torch.tensor(data, dtype = torch.half))
            elif middleLayer.activeParam == "vibrato":
                if section:
                    middleLayer.trackList[middleLayer.activeTrack].vibratoSpeed[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                    middleLayer.submitNamedParamChange(True, "vibratoSpeed", start, torch.tensor(data, dtype = torch.half))
                else:
                    middleLayer.trackList[middleLayer.activeTrack].vibratoStrength[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                    middleLayer.submitNamedParamChange(True, "vibratoStrength", start, torch.tensor(data, dtype = torch.half))
                for i in range(*middleLayer.posToNote(start, start + len(data))):
                    middleLayer.recalculateBasePitch(i, middleLayer.trackList[middleLayer.activeTrack].notes[i].xPos, middleLayer.trackList[middleLayer.activeTrack].notes[i].xPos + middleLayer.trackList[middleLayer.activeTrack].notes[i].length)
                middleLayer.ids["pianoRoll"].redrawPitch()
                middleLayer.submitFinalize()
            else:
                middleLayer.trackList[middleLayer.activeTrack].nodegraph.params[middleLayer.activeParam].curve[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                middleLayer.submitParamChange(True, middleLayer.activeParam, start, torch.tensor(data, dtype = torch.half))
        super().__init__(action, data, start, section, *args, **kwargs)
        self.start = start
        self.section = section
        self.length = len(data)

    def inverseAction(self):
        if middleLayer.activeParam == "steadiness":
            oldData = middleLayer.trackList[middleLayer.activeTrack].steadiness[self.start:self.start + self.length]
        elif middleLayer.activeParam == "breathiness":
            oldData = middleLayer.trackList[middleLayer.activeTrack].breathiness[self.start:self.start + self.length]
        elif middleLayer.activeParam == "AI balance":
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
            oldData = middleLayer.trackList[middleLayer.activeTrack].nodegraph.params[middleLayer.activeParam].curve[self.start:self.start + self.length]
        return ChangeParam(convertFormat(oldData, "list"), self.start, self.section)

    def uiCallback(self):
        middleLayer.ids["adaptiveSpace"].redraw()

class ChangePitch(UnifiedAction):
    def __init__(self, data, start, *args, **kwargs):
        @override
        def action(data, start):
            if type(data) == list:
                data = torch.tensor(data, dtype = torch.half)
            middleLayer.trackList[middleLayer.activeTrack].pitch[start:start + data.size()[0]] = data
            data = noteToPitch(data)
            middleLayer.submitNamedPhonParamChange(True, "pitch", start, data)
        super().__init__(action, data, start, *args, **kwargs)
        self.start = start
        if type(data) == list:
            self.size = len(data)
        else:
            self.size = data.size()[0]

    def inverseAction(self):
        return ChangePitch(middleLayer.trackList[middleLayer.activeTrack].pitch[self.start:self.start + self.size], self.start)
    
    def uiCallback(self):
        if middleLayer.mode == "pitch":
            middleLayer.ids["pianoRoll"].redrawPitch()

class AddNote(UnifiedAction):
    def __init__(self, index, x, y, reference, *args, **kwargs):
        @override
        def action(index, x, y, reference):
            if middleLayer.activeTrack == None:
                return
            if index == len(middleLayer.trackList[middleLayer.activeTrack].notes):
                middleLayer.trackList[middleLayer.activeTrack].notes.append(Note(x, y, middleLayer.trackList[middleLayer.activeTrack], reference))
            else:
                middleLayer.trackList[middleLayer.activeTrack].notes.insert(index, Note(x, y, middleLayer.trackList[middleLayer.activeTrack], reference))
            middleLayer.adjustNote(index, 100, None, False, True)
            middleLayer.submitFinalize()
        super().__init__(action, index, x, y, reference, *args, **kwargs)
        self.index = index
        self.x = x
        self.y = y
        
    def inverseAction(self):
        return RemoveNote(self.index)

class _ReinsertNote(UnifiedAction):
    def __init__(self, index, note, *args, **kwargs):
        def action(index, note):
            newNote = UiNote(index = index, xPos = note.xPos, yPos = note.yPos, length = 100, height = middleLayer.ids["pianoRoll"].yScale)
            returnValue = AddNote(index, note.xPos, note.yPos, newNote)()
            middleLayer.ids["pianoRoll"].children[0].add_widget(newNote, index = 5)
            middleLayer.ids["pianoRoll"].notes.append(newNote)
            ChangeNoteLength(index, note.xPos, note.length)()
            ChangeLyrics(index, note.content, None)()
            return returnValue
        super().__init__(action, index, note, *args, **kwargs)
        self.index = index
        self.note = note
    
    def inverseAction(self):
        return RemoveNote(self.index)

class RemoveNote(UnifiedAction):
    def __init__(self, index, *args, **kwargs):
        @override
        def action(index):
            if index == 0:
                middleLayer.offsetPhonemes(index, -middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index])
            else:
                middleLayer.offsetPhonemes(index, middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index - 1] - middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index])
            if middleLayer.trackList[middleLayer.activeTrack].notes[index].reference:
                middleLayer.ids["pianoRoll"].children[0].remove_widget(middleLayer.trackList[middleLayer.activeTrack].notes[index].reference)
                del middleLayer.trackList[middleLayer.activeTrack].notes[index].reference
            middleLayer.trackList[middleLayer.activeTrack].notes.pop(index)
            if index < len(middleLayer.trackList[middleLayer.activeTrack].notes):
                middleLayer.adjustNote(index, None, None, False, True)
                middleLayer.submitFinalize()
        self.index = index
        super().__init__(action, index, *args, **kwargs)
    
    def inverseAction(self):
        return _ReinsertNote(self.index, middleLayer.trackList[middleLayer.activeTrack].notes[self.index])

class ChangeNoteLength(UnifiedAction):
    def __init__(self, index, x, length, *args, **kwargs):
        @override
        def action(index, x, length):
            if index == 0:
                oldStart = int(middleLayer.trackList[middleLayer.activeTrack].borders[0])
            else:
                oldStart = int(middleLayer.trackList[middleLayer.activeTrack].borders[3 * middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index - 1]])
            if index == len(middleLayer.trackList[middleLayer.activeTrack].notes) - 1 or middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index - 1] == middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index]:
                oldEnd = int(middleLayer.trackList[middleLayer.activeTrack].borders[-1])
            else:
                oldEnd = int(middleLayer.trackList[middleLayer.activeTrack].borders[3 * middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index] + 2])
            middleLayer.trackList[middleLayer.activeTrack].notes[index].length = max(length, 1)
            middleLayer.trackList[middleLayer.activeTrack].notes[index].xPos = max(x, 0)
            switch = middleLayer.adjustNote(index, oldStart, oldEnd, adjustPrevious = True)
            middleLayer.submitFinalize()
            return switch
        super().__init__(action, index, x, length, *args, **kwargs)
        self.index = index
        self.length = max(length, 1)
        self.x = max(x, 0)
    
    def inverseAction(self):
        return ChangeNoteLength(self.index, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].xPos, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].length)
    
    def uiCallback(self):
        if middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference:
            middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference.length = self.length
            middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference.x = self.x
            middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference.redraw()
    
    def merge(self, other):
        if type(other) == ChangeNoteLength and self.index == other.index:
            return self
        return None

class MoveNote(UnifiedAction):
    def __init__(self, index, x, y, *args, **kwargs):
        @override
        def action(index, x, y):
            if index == 0:
                oldStart = int(middleLayer.trackList[middleLayer.activeTrack].borders[0])
            else:
                oldStart = int(middleLayer.trackList[middleLayer.activeTrack].borders[3 * middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index - 1]])
            middleLayer.trackList[middleLayer.activeTrack].notes[index].xPos = max(x, 0)
            middleLayer.trackList[middleLayer.activeTrack].notes[index].yPos = y
            switch = middleLayer.adjustNote(index, oldStart, adjustPrevious = True)
            middleLayer.submitFinalize()
            return switch
        super().__init__(action, index, x, y, *args, **kwargs)
        self.index = index
        self.x = max(x, 0)
        self.y = y
    
    def inverseAction(self):
        return MoveNote(self.index, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].xPos, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].yPos)
    
    def uiCallback(self):
        if middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference:
            middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference.xPos = middleLayer.trackList[middleLayer.activeTrack].notes[self.index].xPos
            middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference.yPos = middleLayer.trackList[middleLayer.activeTrack].notes[self.index].yPos
            middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference.redraw()
    
    def merge(self, other):
        if type(other) == MoveNote and self.index == other.index:
            return self
        return None

class ChangeLyrics(UnifiedAction):
    def __init__(self, index, inputText, pronuncIndex = None, *args, **kwargs):
        @override
        def action(index, inputText, pronuncIndex):
            middleLayer.trackList[middleLayer.activeTrack].notes[index].content = inputText
            middleLayer.trackList[middleLayer.activeTrack].notes[index].pronuncIndex = pronuncIndex
            
            if inputText.startswith("- "):
                inputText = inputText[2:]
                if middleLayer.trackList[middleLayer.activeTrack].notes[index].phonemeMode:
                    middleLayer.trackList[middleLayer.activeTrack].notes[index].carryOver = True
                else:
                    middleLayer.trackList[middleLayer.activeTrack].notes[index].carryOver = False
            else:
                middleLayer.trackList[middleLayer.activeTrack].notes[index].carryOver = False
            
            if middleLayer.trackList[middleLayer.activeTrack].notes[index].phonemeMode:
                text = inputText.split(" ")
            else:
                expressionKey = None
                if "_" in inputText:
                    inputText = inputText.split("_")
                    if len(inputText) > 1:
                        inputText = "".join(inputText[:-1])
                        expressionKey = inputText[-1]
                if inputText in middleLayer.trackList[middleLayer.activeTrack].wordDict[0]:
                    if len(middleLayer.trackList[middleLayer.activeTrack].wordDict[0][inputText]) == 0:
                        text = middleLayer.syllableSplit(inputText)
                    elif len(middleLayer.trackList[middleLayer.activeTrack].wordDict[0][inputText]) == 1:
                        text = middleLayer.trackList[middleLayer.activeTrack].wordDict[0][inputText][0].split(" ")
                    elif pronuncIndex != None:
                        text = middleLayer.trackList[middleLayer.activeTrack].wordDict[0][inputText][pronuncIndex].split(" ")
                    else:
                        text = middleLayer.trackList[middleLayer.activeTrack].wordDict[0][inputText][0].split(" ")
                        middleLayer.trackList[middleLayer.activeTrack].notes[index].reference.add_widget(PhonemeSelector(middleLayer.trackList[middleLayer.activeTrack].wordDict[0][inputText], index, inputText, reference = middleLayer.trackList[middleLayer.activeTrack].notes[index].reference))
                else:
                    text = middleLayer.syllableSplit(inputText)
                    if text == None:
                        text = ["pau"]
            phonemes = []
            for i in text:
                if  not middleLayer.trackList[middleLayer.activeTrack].notes[index].phonemeMode and (expressionKey != None):
                    phoneme = i + "_" + expressionKey
                else:
                    phoneme = i
                if phoneme in middleLayer.trackList[middleLayer.activeTrack].phonemeLengths:
                    phonemes.append(phoneme)
                elif i in middleLayer.trackList[middleLayer.activeTrack].phonemeLengths:
                    phonemes.append(i)
            offset = len(phonemes) - len(middleLayer.trackList[middleLayer.activeTrack].notes[index].phonemes)
            middleLayer.offsetPhonemes(index, offset)
            middleLayer.trackList[middleLayer.activeTrack].notes[index].phonemes = phonemes
            middleLayer.adjustNote(index, None, None, False, True)
            if index == 0:
                phonemeIndex = 0
            else:
                phonemeIndex = middleLayer.trackList[middleLayer.activeTrack].phonemeIndices[index - 1]
            middleLayer.submitNamedPhonParamChange(False, "phonemes", phonemeIndex, phonemes)
            offsets = []
            for i in phonemes:
                if i in ("pau", "_autopause"):
                    offsets += [0.]
                elif middleLayer.trackList[middleLayer.activeTrack].phonemeLengths[i] == None:
                    offsets += [0.5]
                else:
                    offsets += [0.05]
            middleLayer.trackList[middleLayer.activeTrack].loopOffset[phonemeIndex:phonemeIndex + len(offsets)] = offsets#index out of range (last note?)
            middleLayer.submitNamedPhonParamChange(False, "offsets", phonemeIndex, offsets)
            middleLayer.submitFinalize()
        super().__init__(action, index, inputText, pronuncIndex, *args, **kwargs)
        self.index = index
    
    def inverseAction(self):
        return ChangeLyrics(self.index, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].content, middleLayer.trackList[middleLayer.activeTrack].notes[self.index].pronuncIndex)
    
    def uiCallback(self):
        middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference.children[0].text = middleLayer.trackList[middleLayer.activeTrack].notes[self.index].content
        middleLayer.trackList[middleLayer.activeTrack].notes[self.index].reference.redrawStatusBars()

class MoveBorder(UnifiedAction):
    def __init__(self, border, pos, *args, **kwargs):
        @override
        def action(border, pos):
            middleLayer.trackList[middleLayer.activeTrack].borders[border] = pos
            if middleLayer.mode == "timing":
                middleLayer.ids["adaptiveSpace"].updateFromBorder(border, pos)
            middleLayer.submitBorderChange(False, border, [pos,])
        super().__init__(action, border, pos, *args, **kwargs)
        self.border = border
        self.pos = pos
    
    def inverseAction(self):
        return MoveBorder(self.border, middleLayer.trackList[middleLayer.activeTrack].borders[self.border])
    
    def uiCallback(self):
        if middleLayer.mode == "timing":
            middleLayer.ids["pianoRoll"].removeTiming()
            middleLayer.ids["pianoRoll"].drawTiming()
    
    def merge(self, other):
        if type(other) == MoveBorder and self.border == other.border:
            return self
        return None

class ChangeVoicebank(UnifiedAction):
    def __init__(self, index, path, *args, **kwargs):
        @override
        def action(index, path):
            middleLayer.trackList[middleLayer.activeTrack].phonemeLengths = dict()
            tmpVb = LiteVoicebank(path)
            for i in tmpVb.phonemeDict.keys():
                if tmpVb.phonemeDict[i][0].isPlosive:
                    middleLayer.trackList[middleLayer.activeTrack].phonemeLengths[i] = tmpVb.phonemeDict[i][0].specharm.size()[0]
                else:
                    middleLayer.trackList[middleLayer.activeTrack].phonemeLengths[i] = None
            middleLayer.submitChangeVB(index, path)
            for i, note in enumerate(middleLayer.trackList[middleLayer.activeTrack].notes):
                ChangeLyrics(i, note.content)()
        super().__init__(action, index, path, *args, **kwargs)
        self.index = index
    
    def inverseAction(self):
        return ChangeVoicebank(self.index, middleLayer.trackList[middleLayer.activeTrack].vbPath)
    
    def uiCallback(self):
        with h5py.File(os.path.join(readSettings()["datadir"], "Voices", middleLayer.trackList[self.index].vbPath), "r") as f:
            data = MetadataStorage(f).toMetadata()
        for i in middleLayer.ids["singerList"].children:
            if i.index == self.index:
                i.name = data.name
                canvas_img = data.image
                data = BytesIO()
                canvas_img.save(data, format='png')
                data.seek(0)
                im = CoreImage(BytesIO(data.read()), ext='png')
                i.image = im.texture
                break

class ChangeVolume(UnifiedAction):
    def __init__(self, index, volume, *args, **kwargs):
        @override
        def action(index, volume):
            middleLayer.trackList[index].volume = volume
        super().__init__(action, index, volume, *args, **kwargs)
        self.index = index
        self.volume = volume
    
    def inverseAction(self):
        return ChangeVolume(self.index, middleLayer.trackList[middleLayer.activeTrack].volume)
    
    def uiCallback(self):
        for i in middleLayer.ids["singerList"].children:
            if i.index == self.index:
                i.children[0].children[0].children[1].value = middleLayer.trackList[middleLayer.activeTrack].volume
                break
    
    def merge(self, other):
        if type(other) == ChangeVolume and self.index == other.index:
            return self
        return None

class LoadNVX(UnifiedAction):
    def __init__(self, path, *args, **kwargs):
        @override
        def action(path):
            if path == "":
                return
            data = torch.load(path, map_location = torch.device("cpu"))
            tracks = data["tracks"]
            for i in range(len(middleLayer.trackList)):
                DeleteTrack(0)()
            for trackData in tracks:
                track = validateTrackData(trackData)
                with h5py.File(track["vbPath"], "r") as f:
                    vbData = MetadataStorage(f).toMetadata()
                _importVoicebankNoSubmit(track["vbPath"], vbData.name, vbData.image)
                middleLayer.trackList[-1].volume = track["volume"]
                for note in track["notes"]:
                    middleLayer.trackList[-1].notes.append(Note(note["xPos"], note["yPos"], middleLayer.trackList[-1], None))
                    middleLayer.trackList[-1].notes[-1].length = note["length"]
                    middleLayer.trackList[-1].notes[-1].phonemeMode = note["phonemeMode"]
                    middleLayer.trackList[-1].notes[-1].content = note["content"]
                    middleLayer.trackList[-1].notes[-1].pronuncIndex = note["pronuncIndex"]
                    middleLayer.trackList[-1].notes[-1].phonemes = note["phonemes"]
                    middleLayer.trackList[-1].notes[-1].autopause = note["autopause"]
                    middleLayer.trackList[-1].notes[-1].borders = note["borders"]
                    middleLayer.trackList[-1].notes[-1].carryOver = note["carryOver"]
                    middleLayer.trackList[-1].notes[-1].loopOverlap = note["loopOverlap"]
                    middleLayer.trackList[-1].notes[-1].loopOffset = note["loopOffset"]
                middleLayer.trackList[-1].pitch = track["pitch"]
                middleLayer.trackList[-1].basePitch = track["basePitch"]
                middleLayer.trackList[-1].breathiness = track["breathiness"]
                middleLayer.trackList[-1].steadiness = track["steadiness"]
                middleLayer.trackList[-1].aiBalance = track["aiBalance"]
                middleLayer.trackList[-1].vibratoSpeed = track["vibratoSpeed"]
                middleLayer.trackList[-1].vibratoStrength = track["vibratoStrength"]
                middleLayer.trackList[-1].usePitch = track["usePitch"]
                middleLayer.trackList[-1].useBreathiness = track["useBreathiness"]
                middleLayer.trackList[-1].useSteadiness = track["useSteadiness"]
                middleLayer.trackList[-1].useAIBalance = track["useAIBalance"]
                middleLayer.trackList[-1].useVibratoSpeed = track["useVibratoSpeed"]
                middleLayer.trackList[-1].useVibratoStrength = track["useVibratoStrength"]
                middleLayer.trackList[-1].nodegraph = track["nodegraph"]
                middleLayer.trackList[-1].length = track["length"]
                middleLayer.trackList[-1].mixinVB = track["mixinVB"]
                middleLayer.trackList[-1].pauseThreshold = track["pauseThreshold"]
                middleLayer.trackList[-1].borders.wrappingBorders = track["wrappingBorders"]
                middleLayer.trackList[-1].buildPhonemeIndices()
            middleLayer.validate()
        super().__init__(action, path, *args, **kwargs)
    
    def inverseAction(self):
        def restoreTrackList(tracks:list):
            for i in range(len(middleLayer.trackList)):
                middleLayer.deleteTrack(0)
            for trackData in tracks:
                track = validateTrackData(trackData)
                with h5py.File(track["vbPath"], "r") as f:
                    vbData = MetadataStorage(f).toMetadata()
                _importVoicebankNoSubmit(track["vbPath"], vbData.name, vbData.image)
                middleLayer.trackList[-1].volume = track["volume"]
                for note in track["notes"]:
                    middleLayer.trackList[-1].notes.append(Note(note["xPos"], note["yPos"], middleLayer.trackList[-1], None))
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
        return UnifiedAction(restoreTrackList, middleLayer.trackList)
