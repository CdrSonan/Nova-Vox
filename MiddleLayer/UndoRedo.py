# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

global middleLayer

from UI.code.editor.Main import middleLayer

def enqueueUndo(action):
    if middleLayer.singleUndoActive:
        middleLayer.undoStack[-1].append(action)
    else:
        middleLayer.undoStack.append(action)
    if len(middleLayer.undoStack) > middleLayer.undoLimit:
        middleLayer.undoStack.pop(0)

def enqueueRedo(action):
    middleLayer.redoStack.append(action)

def clearRedoStack():
    middleLayer.redoStack = []

def undo():
    inverse = middleLayer.undoStack[-1].inverseAction()
    middleLayer.undoStack.pop(-1)()
    enqueueRedo(inverse)

def redo():
    inverse = middleLayer.redoStack[-1].inverseAction()
    middleLayer.redoStack.pop(-1)()
    enqueueUndo(inverse)
