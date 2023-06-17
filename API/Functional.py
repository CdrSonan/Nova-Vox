# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

global middleLayer

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