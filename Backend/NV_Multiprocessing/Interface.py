import torch

class SequenceStatusControl:
    def __init__(self, sequence = None):
        if sequence == None:
            self.ai = torch.zeros(0)
            self.rs = torch.zeros(0)
        else:
            phonemeLength = sequence.phonemeLength
            self.ai = torch.zeros(phonemeLength)
            self.rs = torch.zeros(phonemeLength)

class StatusChange:
    def __init__(self, track, index, value, type = False):
        self.track = track
        self.index = index
        self.value = value
        self.type = type

class InputChange:
    def __init__(self, type, final, data1, data2, data3, data4, data5):
        self.type = type
        self.final = final
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.data5 = data5