import torch
import tkinter.filedialog

hiddenLayerNumberMax = 10

data = torch.load("data.dat")
filepath = tkinter.filedialog.askopenfilename(filetypes = ((".smpl", ".smpl"), ("all files", "*")))
newSamples = torch.open(filepath)
types = newSamples[1]
newSamples = newSamples[0]
newBatch = input("start new batch? >>>")
if newBatch == "yes":
    data.append([])
print(len(data))
for i in range(newSamples.size()[0]):
    if types[i] == "VV":
        _type = 0
    elif types[i] == "VC":
        _type = 1
    elif types[i] == "Vc":
        _type = 2
    elif types[i] == "CV":
        _type = 3
    elif types[i] == "CC":
        _type = 4
    elif types[i] == "Cc":
        _type = 5
    elif types[i] == "cV":
        _type = 6
    elif types[i] == "cC":
        _type = 7
    elif types[i] == "cc":
        _type = 8
    sample = [newSamples[i], _type]
    data[len(data) - 1].append(sample)