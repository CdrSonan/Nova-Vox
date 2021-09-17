from multiprocessing import Manager
import torch
import torch.multiprocessing as mp
import global_consts
import NV_Multiprocessing.ResamplerProcess
import NV_Multiprocessing.AiParamProcess
import NV_Multiprocessing.SynthProcess
import VocalSequence

class SequenceStatusControl:
    def __init__(self):
        self.resamplerVersions = torch.zeros(0)
        self.aiParamVersions = torch.zeros(0)
        self.synthVersions = torch.zeros(0)
        self.identities = torch.arange(0)
    def __init__(self, sequence):
        phonemeLength = sequence.phonemeLength
        self.resamplerVersions = torch.zeros(phonemeLength)
        self.aiParamVersions = torch.zeros(phonemeLength)
        self.synthVersions = torch.zeros(phonemeLength)
        self.identities = torch.arange(phonemeLength)

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

class FirstIntermediates:
    def __init__(self):
        self.spectrum = torch.Tensor([[]])
        self.excitation = torch.Tensor([])
        self.voicedExcitation = torch.Tensor([])
    def __init__(self, sequence):
        self.spectrum = torch.zeros((sequence.length, global_consts.halfTripleBatchSize + 1))
        self.excitation = torch.zeros(sequence.length * global_consts.batchSize)
        self.voicedExcitation = torch.zeros(sequence.length * global_consts.batchSize)

class SecondIntermediates:
    def __init__(self):
        self.spectrum = torch.Tensor([[]])
    def __init__(self, sequence):
        self.spectrum = torch.zeros((sequence.length, global_consts.halfTripleBatchSize + 1))

class Outputs:
    def __init__(self):
        self.waveform = torch.zeros(0)
    def __init__(self, sequence):
        self.waveform = torch.zeros(sequence.length * global_consts.batchSize)


if __name__ == '__main__':
    mp.freeze_support()

    sequenceList = []

    with mp.Manager() as manager:
        statusControl = manager.list()
        inputList = manager.list()
        firstInterList = manager.list()
        secondInterList = manager.list()
        outputList = manager.list()
        for i in sequenceList:
            statusControl.append(SequenceStatusControl(i))
            inputList.append(Inputs(i))
            outputList.append(Outputs(i.timeLength))


        resamplerProcess = mp.Process(target=NV_Multiprocessing.ResamplerProcess.resamplerProcess, args=(statusControl, inputList, firstInterList), daemon = True)
        resamplerProcess.start()
        aiParamProcess = mp.Process(target=NV_Multiprocessing.AiParamProcess.aiParamProcess, args=(statusControl, inputList, firstInterList, secondInterList), daemon = True)
        aiParamProcess.start()
        synthProcess = mp.Process(target=NV_Multiprocessing.SynthProcess.synthProcess, args=(statusControl, firstInterList, secondInterList, outputList), daemon = True)
        synthProcess.start()