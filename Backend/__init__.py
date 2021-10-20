import torch

global mainDevice
global aiDevice
settings = {}
with open("settings.ini", 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(" ")
        settings[line[0]] = line[1]
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