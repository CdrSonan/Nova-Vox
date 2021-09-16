import torch
import tkinter.filedialog
import Backend.VB_Components.Voicebank
Voicebank = Backend.VB_Components.Voicebank.LiteVoicebank
import Backend.DataHandler.VocalSequence
VocalSequence = Backend.DataHandler.VocalSequence.VocalSequence

filepath = tkinter.filedialog.askopenfilename(filetypes = ((".nvvb Voicebanks", ".nvvb"), ("all_files", "*")))
if filepath != "":
    vb = Voicebank(filepath)
    filepath = tkinter.filedialog.askopenfilename(filetypes = (("text files", ".txt"), ("all_files", "*")))
    if filepath != "":
        with open(filepath, 'r') as f:
            exec(f.read())