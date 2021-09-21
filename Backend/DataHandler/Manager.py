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
        self.status = torch.zeros(sequence.length)

class RenderManager:
    def __init__(self, sequenceList, voicebankList, aiParamStackList):
        if __name__ == '__main__':
            mp.freeze_support()
            with mp.Manager() as manager:
                self.statusControl = manager.list()
                self.inputList = manager.list()
                self.outputList = manager.list()
                self.rerenderFlag = manager.Event()
                for i in sequenceList:
                    self.statusControl.append(SequenceStatusControl(i))
                    self.inputList.append(Inputs(i))
                    self.outputList.append(Outputs(i.timeLength))

                renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, self.inputList, voicebankList, aiParamStackList, self.outputList, self.rerenderFlag), name = "Nova-Vox rendering backend", daemon = True)
                renderProcess.start()