import torch
import tkinter.filedialog

hiddenLayerNumberMax = 10

data = torch.load("data.dat")
filepath = tkinter.filedialog.askopenfilename(filetypes = ((".smpl", ".smpl"), ("all files", "*")))
newSamples = torch.open(filepath)
newBatch = input("start new batch? >>>")
if newBatch == "yes":
    data.append([])
print(len(data))
for i in range(newSamples.size()[0]):
    sample = newSamples[i]
    data[len(data) - 1].append(sample)