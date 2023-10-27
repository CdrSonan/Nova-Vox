# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from sys import getrefcount
from Backend.VB_Components.Voicebank import LiteVoicebank

class VoicebankManager():
    """Class that acts as a central storage for all voicebanks currently used for rendering"""
    
    def __init__(self) -> None:
        self.voicebanks = dict()
        
    def getVoicebank(self, path: str, device) -> LiteVoicebank:
        """retrieves a reference to a voicebank from the voicebank manager, and adds it to the manager if it is not already present."""

        if path not in self.voicebanks.keys():
            self.voicebanks[path] = LiteVoicebank(path, device)
        return self.voicebanks[path]
    
    def clean(self) -> None:
        """Cleans up all voicebanks that are not referenced by any other object"""

        toDelete = []
        for i in self.voicebanks:
            if getrefcount(self.voicebanks[i]) == 3:
                toDelete.append(i)
        while len(toDelete) > 0:
            self.voicebanks.pop(toDelete.pop())
