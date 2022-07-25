import torch
from Backend.DataHandler.VocalSequence import VocalSequence

class SequenceStatusControl():
    """Container for the control flags of a VocalSequence object. Used by the rendering process for tracking which segments of the VocalSequence need to be re-rendered or loaded from cache."""

    def __init__(self, sequence:VocalSequence = None) -> None:
        if sequence == None:
            self.ai = torch.zeros(0)
            self.rs = torch.zeros(0)
        else:
            phonemeLength = sequence.phonemeLength
            self.ai = torch.zeros(phonemeLength)
            self.rs = torch.zeros(phonemeLength)

class StatusChange():
    """Container for messages sent from the rendering process to the main process. Can represent a change of the rendering process of a phoneme, update for the audio buffer or track index offset (after track deletion)"""

    def __init__(self, track:int, index:int, value:torch.Tensor, type:bool = False) -> None:
        self.track = track
        self.index = index
        self.value = value
        self.type = type

class InputChange():
    """Container for messages sent from the main process to the rendering process"""
    #TODO: rework to use *data opt parameter
    def __init__(self, type:str, final:bool, *data) -> None:
        self.type = type
        self.final = final
        self.data = data