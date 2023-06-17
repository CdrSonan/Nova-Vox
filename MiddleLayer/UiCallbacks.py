# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

global middleLayer

from UI.code.editor.Main import middleLayer
from API.Functional import UiCallback

#Callback subclasses should define an __eq__ method that returns True even if the statements are not equivalent, but overwrite each other's changes.

class CBMoveNote(UiCallback):
    def __init__(self, index, x, y):
        super.__init__(self, callback = middleLayer.moveNote, index = index, x = x, y = y)
    
    def __eq__(self, other):
        if other.__class__ == self.__class__:
            return self.index == other.index
        else:
            return False
