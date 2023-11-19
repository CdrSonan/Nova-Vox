#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from torch import device, float32, zeros
from Backend.DataHandler.VocalSequence import VocalSequence

class VocalSegment():
    """Class representing the segment covered by a single phoneme within a VocalSequence. Designed to be modified by ESPER.
    
    Attributes:
        start1-3, end1-3: timing borders of the segment
        
        startCap, endCap: Whether there is a transition from the previous, and to the next phoneme
        
        phonemeKey: The key of the phoneme of the segment
        
        vb: The Voicebank to use data from
        
        offset: The offset applied to the audio before sampling. Non-zero values discard the beginning of the audio.
        
        repetititionSpacing: The amout of overlap applied when looping the sample
        
        pitch: relevant part of the pitch parameter curve
        
        steadiness: relevant part of the steadiness parameter curve
        
    Methods:
        __init__: Constructor function based on a VocalSequence object
    """


    def __init__(self, inputs:VocalSequence, vb, index:int, device:device) -> None:
        self.start1 = inputs.borders[3*index]
        self.start2 = inputs.borders[3*index+1]
        self.start3 = inputs.borders[3*index+2]
        self.end1 = inputs.borders[3*index+3]
        self.end2 = inputs.borders[3*index+4]
        self.end3 = inputs.borders[3*index+5]
        self.startCap = inputs.startCaps[index]
        self.endCap = inputs.endCaps[index]
        self.phonemeKey = inputs.phonemes[index]
        self.vb = vb
        self.offset = inputs.offsets[index]
        self.repetititionSpacing = inputs.repetititionSpacing[index]
        self.pitch = inputs.pitch[self.start1:self.end3].to(dtype = float32, device = device)
        if inputs.useSteadiness:
            self.steadiness = inputs.steadiness[self.start1:self.end3].to(device = device)
        else:
            self.steadiness = zeros([self.end3 - self.start1,], device = device)
