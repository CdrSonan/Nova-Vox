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
    tc = settings["tensorCores"]
    if tc == "enabled":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
if accel == "GPU":
    mainDevice = torch.device('cuda')
    aiDevice = torch.device('cuda')
    tc = settings["tensorCores"]
    if tc == "disabled":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False