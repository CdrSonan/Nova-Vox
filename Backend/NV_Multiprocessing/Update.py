import torch
from Backend.NV_Multiprocessing.Interface import SequenceStatusControl

def trimSequence(index:int, position:int, delta:int, inputList:list, internalStatusControl:SequenceStatusControl) -> tuple[list, list]:
    phonemes = inputList[index].phonemes
    offsets = inputList[index].offsets
    repetititionSpacing = inputList[index].repetititionSpacing
    borders = inputList[index].borders
    startCaps = inputList[index].startCaps
    endCaps = inputList[index].endCaps
    if delta > 0:
        phonemes = phonemes[0:position] + ["_0"] * delta + phonemes[position:]
        offsets = torch.cat([offsets[0:position], torch.zeros([delta,]), offsets[position:]], 0)
        repetititionSpacing = torch.cat([repetititionSpacing[0:position], torch.full([delta,], 0.5), repetititionSpacing[position:]], 0)
        borders = borders[0:3 * position] + [0] * (3 * delta) + borders[3 * position:]
        startCaps = startCaps[0:position] + [False] * delta + startCaps[position:]
        endCaps = endCaps[0:position] + [False] * delta + endCaps[position:]
        if position == 0:
            startCaps[0] = True
            if len(startCaps) > position + delta:
                startCaps[position + delta] = False
        if position + delta >= len(endCaps) - 1:
            endCaps[position] = False
            endCaps[-1] = True
        internalStatusControl.rs = torch.cat([internalStatusControl.rs[0:position], torch.zeros([delta,]), internalStatusControl.rs[position:]], 0)
        internalStatusControl.ai = torch.cat([internalStatusControl.ai[0:position], torch.zeros([delta,]), internalStatusControl.ai[position:]], 0)
    elif delta < 0:
        phonemes = phonemes[0:position] + phonemes[position - delta:]
        offsets = torch.cat([offsets[0:position], offsets[position - delta:]], 0)
        repetititionSpacing = torch.cat([repetititionSpacing[0:position], repetititionSpacing[position - delta:]], 0)
        borders = borders[0:3 * position] + borders[3 * position - 3 * delta:]
        startCaps = startCaps[0:position] + startCaps[position - delta:]
        endCaps = endCaps[0:position] + endCaps[position - delta:]
        if len(startCaps) > 0:
            if position == 0:
                startCaps[0] = True
            if position >= len(endCaps) - 1:
                endCaps[-1] = True
        internalStatusControl.rs = torch.cat([internalStatusControl.rs[0:position], internalStatusControl.rs[position - delta:]], 0)
        internalStatusControl.ai = torch.cat([internalStatusControl.ai[0:position], internalStatusControl.ai[position - delta:]], 0)
    inputList[index].phonemes = phonemes
    inputList[index].offsets = offsets
    inputList[index].repetititionSpacing = repetititionSpacing
    inputList[index].borders = borders
    inputList[index].startCaps = startCaps
    inputList[index].endCaps = endCaps
    return inputList, internalStatusControl

def posToSegment(index:int, pos1:float, pos2:float, inputList:list) -> tuple(int, int):
    pos1Out = None
    for i in range(int(len(inputList[index].borders) / 3)):
        if inputList[index].borders[3 * i + 2] > pos1:
            pos1Out = max(i - 1, 0)
            break
    if pos1Out == None or pos1Out == int(len(inputList[index].borders) / 3) - 1:
        return (0, 0)
    pos2Out = int(len(inputList[index].borders) / 3) - 1
    for i in range(pos1Out, int(len(inputList[index].borders) / 3)):
        if inputList[index].borders[3 * i] > pos2:
            pos2Out = max(i, 0)
            break
    return (pos1Out, pos2Out)
