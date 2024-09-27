#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from copy import copy, deepcopy
from torch import Tensor

class VocalSequence():
    """Class holding information about a vocal track as required by the rendering process"""
    
    def __init__(self, length:int, borders:list, phonemes:list, offsets:Tensor, repetititionSpacing:Tensor, pitch:Tensor, steadiness:Tensor, breathiness:Tensor, aiBalance:Tensor, genderFactor:Tensor, vibratoSpeed:Tensor, vibratoStrength:Tensor, useBreathiness:bool, useSteadiness:bool, useAIBalance:bool, useGenderFactor:bool, useVibratoSpeed:bool, useVibratoStrength:bool, customCurves:list, nodeGraphFunction) -> None:
        self.length = length
        self.phonemeLength = len(phonemes)
        self.borders = borders
        self.phonemes = phonemes
        self.offsets = offsets
        self.repetititionSpacing = repetititionSpacing
        self.pitch = pitch
        self.steadiness = steadiness
        self.breathiness = breathiness
        self.aiBalance = aiBalance
        self.genderFactor = genderFactor
        self.vibratoSpeed = vibratoSpeed
        self.vibratoStrength = vibratoStrength
        self.useBreathiness = useBreathiness
        self.useSteadiness = useSteadiness
        self.useAIBalance = useAIBalance
        self.useGenderFactor = useGenderFactor
        self.useVibratoSpeed = useVibratoSpeed
        self.useVibratoStrength = useVibratoStrength
        self.customCurves = customCurves
        self.nodeGraphFunction = nodeGraphFunction
        
    def duplicate(self):
        return VocalSequence(copy(self.length),
                             deepcopy(self.borders),
                             deepcopy(self.phonemes),
                             self.offsets.clone(),
                             self.repetititionSpacing.clone(),
                             self.pitch.clone(),
                             self.steadiness.clone(),
                             self.breathiness.clone(),
                             self.aiBalance.clone(),
                             self.genderFactor.clone(),
                             self.vibratoSpeed.clone(),
                             self.vibratoStrength.clone(),
                             copy(self.useBreathiness),
                             copy(self.useSteadiness),
                             copy(self.useAIBalance),
                             copy(self.useGenderFactor),
                             copy(self.useVibratoSpeed),
                             copy(self.useVibratoStrength),
                             deepcopy(self.customCurves),
                             copy(self.nodeGraphFunction))
