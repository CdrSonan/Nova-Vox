#Copyright 2022, 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
from Backend.Node.NodeBaseLib import getNodeCls, OutputNode

def trimSequence(index:int, position:int, delta:int, inputList:list, statusControl:list) -> tuple([list, list]):
    """Function used for adding to or removing phonemes from a VocalSequence. Automatically updates control structures and schedules updates as required
    
    Parameters:
        index: the index of the track (in the InputList list) that phonemes are to be added to/removed from.
        
        position: When adding phonemes, the index of the phoneme (within the VocalSequence) they are added behind. When removing phonemes, the first index of the portion that is to be removed
        
        delta: The number of phonemes to add/remove. Use positive values for adding phonemes, and negative ones for removing them.
        
        addition: offset for the position at which borders are added. Used to compensate for _autopause phonemes present in the sequence.
        
        inputList: list of VocalSequence objects representing all data held by the rendering process
        
        internalStatusControl: SequenceStatusControl object corresponding to inputList at position index
        
    Returns:
        Tuple(inputList, internalStatusControl): inputList and internalStatusControl are the objects of the same name passed to the function as arguments, with the necessary adjustments to reflect the new phonemes.
        
    When adding phonemes, the function adds the placeholder phoneme "_0" in the position for the new phonemes. Therefore, it must be ensured that the phonemes are updated between the call of this function and the next
    rendering iteration. The intended way for this is by setting the final flag of the InputChange object triggering updateFromMain, and subsequently this function, to False. A second InputChange can then be used to
    update the phonemes as required. MiddleLayer.offsetPhonemes, the equivalent operation to this function on the main process, uses the "_X" placeholder phoneme instead of "_0". This is to make it easier to pinpoint
    issues during debugging. Neither phoneme should be implemented by any Voicebank"""


    phonemes = inputList[index].phonemes
    offsets = inputList[index].offsets
    repetititionSpacing = inputList[index].repetititionSpacing
    borders = inputList[index].borders
    if delta > 0:
        phonemes = phonemes[0:position] + ["_0"] * delta + phonemes[position:]
        offsets = offsets[0:position] + [0.5] * delta + offsets[position:]
        repetititionSpacing = repetititionSpacing[0:position] + [0.5] * delta + repetititionSpacing[position:]
        borders = borders[0:3 * position] + [borders[3 * position - 1] + 1] * (3 * delta) + borders[3 * position:]
        statusControl[index].rs = torch.cat([statusControl[index].rs[0:position], torch.zeros([delta,]), statusControl[index].rs[position:]], 0)
        statusControl[index].ai = torch.cat([statusControl[index].ai[0:position], torch.zeros([delta,]), statusControl[index].ai[position:]], 0)
    elif delta < 0:
        phonemes = phonemes[0:position] + phonemes[position - delta:]
        offsets = offsets[0:position] + offsets[position - delta:]
        repetititionSpacing = repetititionSpacing[0:position] + repetititionSpacing[position - delta:]
        borders = borders[0:3 * position] + borders[3 * position - 3 * delta:]
        statusControl[index].rs = torch.cat([statusControl[index].rs[0:position], statusControl[index].rs[position - delta:]], 0)
        statusControl[index].ai = torch.cat([statusControl[index].ai[0:position], statusControl[index].ai[position - delta:]], 0)
    inputList[index].phonemes = phonemes
    inputList[index].offsets = offsets
    inputList[index].repetititionSpacing = repetititionSpacing
    inputList[index].borders = borders
    return inputList, statusControl

def posToSegment(index:int, pos1:float, pos2:float, inputList:list) -> tuple([int, int]):
    """helper function converting the range between two time positions in a VocalSequence to a range between two phoneme indices.
    
    Arguments:
        index: the index of the track (in the InputList list) that the conversion is to be performed for.

        pos1, pos2: start and end of the interval to be converted. Both should be positive numbers and represent a time since the song startin engine ticks.
        
        inputList: list of VocalSequence objects representing all data held by the rendering process
        
    Returns:
        Tuple(pos1Out, pos2Out): two phoneme indices chosen so that the range between the phonemes fully encompasses the space between pos1 and pos2, while still being as small as possible.
        
    The main purpose of this function is to convert a time interval that is to be updated, e.g. because of a parameter curve change by the user, to the range of phonemes that needs to be updated to achieve this.
    Therefore, this step is only required for parameters that use a curve, and not for changes of per-phoneme properties."""


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

class NodeParam:
    def __init__(self, value, enabled = True):
        self._value = value
        self.enabled = enabled

def unpackNodes(nodeList:list, paramList:list) -> list:
    """Function used to unpack a list of Node objects into a list of dictionaries, for serialization and transmission to the rendering process.
    
    Arguments:
        nodeList: list of Node objects to be unpacked.
    
    Returns:
        a List of NodeBase objects, representing the graph structure of the nodes in nodeList.
    """
    
    nodes = []
    for nodeInfo in nodeList:
        nodes.append(getNodeCls(nodeInfo["type"])())
    for node, nodeInfo in zip(nodes, nodeList):
        for key, i in nodeInfo["inputs"].items():
            node.inputs[key]._value = i[1]
            if i[0]:
                node.inputs[key].attachedTo = nodes[i[2]].outputs[i[3]]
        for key, i in nodeInfo["outputs"].items():
            node.outputs[key]._value = i
        for key, i in nodeInfo["auxData"].items():
            node.auxData[key] = i
    for node in nodes:
        if isinstance(node, OutputNode):
            node.checkStatic()
    return nodes, {key: NodeParam(*value) for key, value in paramList.items()}
