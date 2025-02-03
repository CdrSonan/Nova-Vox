#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
import math

def phaseInterp(phaseA: torch.Tensor, phaseB: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """performs linear interpolation between two phases, which are expected to be values between 0 and 2*pi, based on a factor between 0 and 1.
    This function always chooses the shorter of the two possible paths between the two phases.
    (For example, if phaseA is 0.1*pi and phaseB is 1.9*pi, the interpolation will take the shorter path, crossing 0, rather than the longer path crossing pi.)"""


    diff = phaseB - phaseA
    diff = torch.remainder(diff, 2 * math.pi)
    diffB = diff - 2 * math.pi
    mask = torch.ge(diff.abs(), diffB.abs())
    diff[mask] = diffB[mask]
    return torch.remainder(phaseA + factor * diff, 2 * math.pi)
