import torch
import tkinter.filedialog
import Backend.ESPER.PitchCalculator
calculatePitch = Backend.ESPER.PitchCalculator.calculatePitch
import Backend.ESPER.SpectralCalculator
calculateSpectra = Backend.ESPER.SpectralCalculator.calculateSpectra

unvoicedIterations = 20

hiddenLayerNumberMax = 10

data = torch.load("data.cval")
filepath = tkinter.filedialog.askopenfilename(filetypes = ((".dat", ".dat"), ("all files", "*")))
newSamples = torch.open(filepath)
newBatch = input("start new batch? >>>")
if newBatch == "yes":
    data.append([])
print(len(data))
for i in range(newSamples.size()[0]):
    sample = newSamples[i]
    sample.voicedFilter = 1
    sample.unvoicedIterations = unvoicedIterations
    calculatePitch(sample)
    calculateSpectra(sample)
    sample = sample.spectrum + sample.spectra
    data[len(data) - 1].append(sample)