#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from torch import Tensor

class VocalSequence():
    """Class holding information about a vocal track as required by the rendering process"""
    
    def __init__(self, length:int, borders:Tensor, phonemes:list, startCaps:list, endCaps:list, offsets:Tensor, repetititionSpacing:Tensor, pitch:Tensor, steadiness:Tensor, breathiness:Tensor, aiBalance:Tensor, vibratoSpeed:Tensor, vibratoStrength:Tensor, useBreathiness:bool, useSteadiness:bool, useAIBalance:bool, useVibratoSpeed:bool, useVibratoStrength:bool) -> None:
        self.length = length
        self.phonemeLength = len(phonemes)
        self.borders = borders
        self.phonemes = phonemes
        self.startCaps = startCaps
        self.endCaps = endCaps
        if self.phonemeLength > 0:
            startCaps[0] = True
            endCaps[-1] = True
        self.offsets = offsets
        self.repetititionSpacing = repetititionSpacing
        self.pitch = pitch
        self.steadiness = steadiness
        self.breathiness = breathiness
        self.aiBalance = aiBalance
        self.vibratoSpeed = vibratoSpeed
        self.vibratoStrength = vibratoStrength
        self.useBreathiness = useBreathiness
        self.useSteadiness = useSteadiness
        self.useAIBalance = useAIBalance
        self.useVibratoSpeed = useVibratoSpeed
        self.useVibratoStrength = useVibratoStrength
        self.parameters = []
        