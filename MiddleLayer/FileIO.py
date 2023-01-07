#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
from MiddleLayer.MiddleLayer import MiddleLayer

def saveNVX(path:str, middleLayer:MiddleLayer) -> None:
    """backend function for saving a .nvx file"""

    data = {}
    torch.save(data, path)

def loadNVX(path:str, middleLayer:MiddleLayer) -> None:
    """backend function for loading a .nvx file"""

    data = torch.load(path)
    middleLayer.manager.restart(middleLayer.trackList)