#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
from Backend.DataHandler.VocalSequence import VocalSequence

class SequenceStatusControl():
    """Container for the control flags of a VocalSequence object. Used by the rendering process for tracking which segments of the VocalSequence need to be re-rendered or loaded from cache."""

    def __init__(self, sequence:VocalSequence = None) -> None:
        if sequence == None:
            self.ai = torch.zeros(0)
            self.rs = torch.zeros(0)
        else:
            phonemeLength = sequence.phonemeLength
            self.ai = torch.zeros(phonemeLength)
            self.rs = torch.zeros(phonemeLength)
            
    def duplicate(self):
        duplicate = SequenceStatusControl()
        duplicate.ai = self.ai.clone()
        duplicate.rs = self.rs.clone()
        return duplicate

class StatusChange():
    """Container for messages sent from the rendering process to the main process. Can represent a change of the rendering process of a phoneme, update for the audio buffer or track index offset (after track deletion)"""

    def __init__(self, track:int, index:int, value:torch.Tensor, type:str = "status") -> None:
        self.track = track
        self.index = index
        self.value = value
        self.type = type

class InputChange():
    """Container for messages sent from the main process to the rendering process"""

    def __init__(self, type:str, final:bool, *data) -> None:
        self.type = type
        self.final = final
        self.data = data