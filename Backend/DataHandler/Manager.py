from tracemalloc import stop
import torch
import torch.multiprocessing as mp
import global_consts
import Backend.NV_Multiprocessing.RenderProcess

class SequenceStatusControl:
    """
    def __init__(self):
        self.ai = torch.zeros(0)
        self.rs = torch.zeros(0)
    """
    def __init__(self, sequence):
        phonemeLength = sequence.phonemeLength
        self.ai = torch.zeros(phonemeLength)
        self.rs = torch.zeros(phonemeLength)

class Inputs:
    """
    def __init__(self):
        self.borders = torch.Tensor([0, 1, 2])
        self.startCaps = torch.zeros(0, dtype = torch.bool)
        self.endCaps = torch.zeros(0, dtype = torch.bool)
        self.phonemes = []
        self.offsets = torch.zeros(0)
        self.repetititionSpacing = torch.ones(0)
        self.pitch = torch.ones(0)
        self.steadiness = torch.zeros(0)
        self.breathiness = torch.zeros(0)
        self.aiParamInputs = []
    """
    def __init__(self, sequence):
        self.borders = sequence.borders
        self.phonemes = sequence.phonemes
        self.startCaps = sequence.startCaps
        self.endCaps = sequence.endCaps
        self.offsets = sequence.offsets
        self.repetititionSpacing = sequence.repetititionSpacing
        self.pitch = sequence.pitch
        self.steadiness = sequence.steadiness
        self.breathiness = sequence.breathiness
        self.aiParamInputs = sequence.aiParamInputs

class Outputs:
    """
    def __init__(self):
        self.waveform = torch.zeros(0)
        self.status = torch.zeros(0)
    """
    def __init__(self, sequence):
        self.waveform = torch.zeros(sequence.length * global_consts.batchSize)
        self.status = torch.zeros(sequence.phonemeLength)

class RenderManager:
    def __init__(self, sequenceList, voicebankList, aiParamStackList):
        self.manager = mp.Manager()
        self.statusControl = self.manager.list()
        self.inputList = self.manager.list()
        self.outputList = self.manager.list()
        self.rerenderFlag = self.manager.Event()
        for i in sequenceList:
            self.statusControl.append(SequenceStatusControl(i))
            self.inputList.append(Inputs(i))
            self.outputList.append(Outputs(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.outputList, self.rerenderFlag), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def addTrack(self, sequence, voicebankList, aiParamStackList):
        self.statusControl.append(SequenceStatusControl(i))
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