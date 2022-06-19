import torch.multiprocessing as mp
import Backend.NV_Multiprocessing.RenderProcess
from Backend.NV_Multiprocessing.Interface import SequenceStatusControl, InputChange, StatusChange

class RenderManager:
    def __init__(self, sequenceList, voicebankList, aiParamStackList):
        self.statusControl = []
        self.inputList = []
        self.rerenderFlag = mp.Event()
        #self.connection, remoteConnection = mp.Pipe()
        self.connection = mp.Queue(0)
        self.remoteConnection = mp.Queue(0)
        for i in sequenceList:
            self.statusControl.append(SequenceStatusControl(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, sequenceList, self.rerenderFlag, self.connection, self.remoteConnection), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()
    def receiveChange(self):
        try:
            return self.remoteConnection.get_nowait()
        except:
        #if self.connection.poll():
        #    return self.connection.recv()
            return None
    def sendChange(self, type, final = True, data1 = None, data2 = None, data3 = None, data4 = None, data5 = None):
        self.connection.put(InputChange(type, final, data1, data2, data3, data4, data5), True)
        if final:
            self.rerenderFlag.set()
    def restart(self, sequenceList, voicebankList, aiParamStackList):
        self.stop()
        del self.statusControl[:]
        for i in sequenceList:
            self.statusControl.append(SequenceStatusControl(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(self.statusControl, voicebankList, aiParamStackList, self.inputList, self.rerenderFlag, self.connection, self.remoteConnection), name = "Nova-Vox rendering process", daemon = True)
        self.renderProcess.start()

    def stop(self):
        self.renderProcess.terminate()
        