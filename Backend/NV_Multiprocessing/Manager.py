import torch.multiprocessing as mp
import Backend.NV_Multiprocessing.RenderProcess
from Backend.NV_Multiprocessing.Interface import SequenceStatusControl
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
        self.rerenderFlag = mp.Event()
        self.connection, remoteConnection = mp.Pipe()
        for i in sequenceList:
            self.statusControl.append(SequenceStatusControl(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, sequenceList, self.rerenderFlag, remoteConnection), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()
    """
    def addTrack(self, sequence, voicebankList, aiParamStackList):
    def removeTrack(self, index, voicebankList, aiParamStackList):
    def addParam(self, sequenceList, voicebankList, aiParamStackList):
    def removeParam(self, index, sequence, voicebankList, aiParamStackList):
    """
    def restart(self, sequenceList, voicebankList, aiParamStackList):
        #stop()
        del self.statusControl[:]
        for i in sequenceList:
            self.statusControl.append(SequenceStatusControl(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.rerenderFlag), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def stop(self):
        self.renderProcess.terminate()
        