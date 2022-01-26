from tracemalloc import stop
import torch
import torch.multiprocessing as mp
import global_consts
import Backend.NV_Multiprocessing.RenderProcess
"""
class Outputs:
    
    def __init__(self):
        self.waveform = torch.zeros(0)
        self.status = torch.zeros(0)
    
    def __init__(self, sequence):
        self.waveform = torch.zeros(sequence.length * global_consts.batchSize)
        self.status = torch.zeros(sequence.phonemeLength)
"""

class RenderManager:
    def __init__(self, sequenceList, voicebankList, aiParamStackList):
        self.statusControl = []
        self.inputList = []
        #self.outputList = []
        self.rerenderFlag = mp.Event()
        self.connection, remoteConnection = mp.Pipe()
        for i in sequenceList:
            self.statusControl.append(SequenceStatusControl(i))
            self.inputList.append(Inputs(i))
            #self.outputList.append(Outputs(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.outputList, self.rerenderFlag, remoteConnection), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def addTrack(self, sequence, voicebankList, aiParamStackList):
        self.statusControl.append(SequenceStatusControl(sequence))
        self.inputList.append(Inputs(sequence))
        self.outputList.append(Outputs(sequence))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.outputList, self.rerenderFlag), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def removeTrack(self, index, voicebankList, aiParamStackList):
        del self.statusControl[index]
        del self.inputList[index]
        del self.outputList[index]
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.outputList, self.rerenderFlag), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def addParam(self, sequenceList, voicebankList, aiParamStackList):
        for i in range(len(self.statusControl)):
            self.statusControl[i].ai *= 0
        del self.inputList[:]
        for i in sequenceList:
            self.inputList.append(Inputs(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.outputList, self.rerenderFlag), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def removeParam(self, index, sequence, voicebankList, aiParamStackList):
        self.statusControl[index].ai *= 0
        del self.inputList[index]
        self.inputList.insert(index, Inputs(sequence))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.outputList, self.rerenderFlag), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def restart(self, sequenceList, voicebankList, aiParamStackList):
        stop()
        del self.statusControl[:]
        del self.inputList[:]
        del self.outputList[:]
        for i in sequenceList:
            self.statusControl.append(SequenceStatusControl(i))
            self.inputList.append(Inputs(i))
            self.outputList.append(Outputs(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.outputList, self.rerenderFlag), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def stop(self):
        self.renderProcess.terminate()
        