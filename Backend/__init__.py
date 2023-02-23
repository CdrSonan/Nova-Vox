#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
from MiddleLayer.IniParser import readSettings

global mainDevice
global aiDevice
settings = readSettings()
accel = settings["accelerator"]
if accel == "CPU":
    mainDevice = torch.device('cpu')
    aiDevice = torch.device('cpu')
if accel == "Hybrid":
    mainDevice = torch.device('cpu')
    aiDevice = torch.device('cuda')
    tc = settings["tensorcores"]
    if tc == "enabled":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
if accel == "GPU":
    mainDevice = torch.device('cuda')
    aiDevice = torch.device('cuda')
    tc = settings["tensorcores"]
    if tc == "disabled":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False