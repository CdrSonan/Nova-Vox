from multiprocessing import Manager
import torch
import torch.multiprocessing as mp
import global_consts
import NV_Multiprocessing.RenderProcess
import VocalSequence

class SequenceStatusControl:
    def __init__(self):
        self.ai = torch.zeros(0)
        self.rs = torch.zeros(0)
    def __init__(self, sequence):
        phonemeLength = sequence.phonemeLength
        self.ai = torch.zeros(phonemeLength)
        self.rs = torch.zeros(phonemeLength)

class Inputs:
    def __init__(self):
        self.voicebank = None
        self.borders = torch.Tensor([0, 1, 2])
        self.startCaps = torch.zeros(0, dtype = torch.bool)
        self.endCaps = torch.zeros(0, dtype = torch.bool)
        self.phonemes = []
        self.offsets = torch.zeros(0)
        self.repetititionSpacing = torch.ones(0)
        self.pitch = torch.ones(0)
        self.steadiness = torch.zeros(0)
        self.breathiness = torch.zeros(0)
        self.aiParamStack = []
    def __init__(self, sequence):
        self.voicebank = sequence.Voicebank
        self.borders = sequence.borders
        self.phonemes = sequence.phonemes
        self.offsets = sequence.offsets
        self.repetititionSpacing = sequence.repetititionSpacing
        self.pitch = sequence.pitch
        self.steadiness = sequence.steadiness
        self.breathiness = sequence.breathiness
        self.aiParamStack = sequence.aiParamStack

class Outputs:
    def __init__(self):
        self.waveform = torch.zeros(0)
    def __init__(self, sequence):
        self.waveform = torch.zeros(sequence.length * global_consts.batchSize)


if __name__ == '__main__':
    mp.freeze_support()

    sequenceList = []
    rerenderFlag = mp.Event
    with mp.Manager() as manager:
        statusControl = manager.list()
        inputList = manager.list()
        outputList = manager.list()
        for i in sequenceList:
            statusControl.append(SequenceStatusControl(i))
            inputList.append(Inputs(i))
            outputList.append(Outputs(i.timeLength))


        renderProcess = mp.Process(target=NV_Multiprocessing.RenderProcess.renderProcess, args=(statusControl, inputList, outputList, rerenderFlag), daemon = True)
        renderProcess.start()