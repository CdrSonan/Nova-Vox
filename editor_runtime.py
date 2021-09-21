import torch
import torch.multiprocessing as mp
import torchaudio
import tkinter.filedialog
import global_consts
import Backend.VB_Components.Voicebank
Voicebank = Backend.VB_Components.Voicebank.LiteVoicebank
import Backend.DataHandler.VocalSequence
VocalSequence = Backend.DataHandler.VocalSequence.VocalSequence
import Backend.DataHandler.Manager
RenderManager = Backend.DataHandler.Manager.RenderManager

if __name__ == '__main__':
    mp.freeze_support()
    filepath = tkinter.filedialog.askopenfilename(filetypes = ((".nvvb Voicebanks", ".nvvb"), ("all_files", "*")))
    if filepath != "":
        vb = Voicebank(filepath)
        filepath = tkinter.filedialog.askopenfilename(filetypes = (("text files", ".txt"), ("all_files", "*")))
        if filepath != "":
            with open(filepath, 'r') as f:
                sequence = None
                exec(f.read())
                sequenceList = [sequence]
                voicebankList = [vb]
                aiParamStackList = [None]
            
                manager = RenderManager(sequenceList, voicebankList, aiParamStackList)
                while True:
                    userInput = input("command?")
                    if userInput == "quit":
                        manager.stop()
                        break
                    elif userInput == "status":
                        print(manager.statusControl[0].rs)
                        print(manager.statusControl[0].ai)
                        print(manager.outputList[0].status)
                    elif userInput == "render":
                        manager.statusControl[0].rs *= 0
                        manager.statusControl[0].ai *= 0
                        manager.outputList[0].status *= 0
                        manager.rerenderFlag.set()
                    elif userInput == "save":
                        torchaudio.save("output.wav", torch.unsqueeze(manager.outputList[0].waveform.detach(), 0), global_consts.sampleRate, format="wav", encoding="PCM_S", bits_per_sample=32)