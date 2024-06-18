#Copyright 2022, 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch

import global_consts

from Localization.editor_localization import getLanguage
loc = getLanguage()

from Backend.Node.NodeBase import NodeBase

class InputNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {}
        outputs = {"Audio": "ESPERAudio",
                   "Phoneme": "Phoneme",
                   "Pitch": "Float",
                   "Transition": "ClampedFloat",
                   "Breathiness": "ClampedFloat",
                   "Steadiness": "ClampedFloat",
                   "AI_Balance": "ClampedFloat",
                   "Loop_Offset": "ClampedFloat",
                   "Loop_Overlap": "ClampedFloat",
                   "Vibrato_Strengh": "ClampedFloat",
                   "Vibrato_Speed": "ClampedFloat",}
        def func(self):
            return {"Audio": self.audio,
                   "Phoneme": self.phoneme,
                   "Pitch": self.pitch,
                   "Transition": self.transition,
                   "Breathiness": self.breathiness,
                   "AI_Balance": self.AIBalance,
                   "Steadiness": self.steadiness,
                   "Loop_Offset": self.loopOffset,
                   "Loop_Overlap": self.loopOverlap,
                   "Vibrato_Strengh": self.vibratoStrengh,
                   "Vibrato_Speed": self.vibratoSpeed,}
        super().__init__(inputs, outputs, func, True, **kwargs)
        self.audio = torch.empty([0, global_consts.frameSize])
        self.phoneme = ("_0", "_0", 0.5)
        self.pitch = 100.
        self.transition = 0.
        self.breathiness = 0.
        self.steadiness = 0.
        self.AIBalance = 0.
        self.loopOffset = 0.
        self.loopOverlap = 0.5
        self.vibratoStrengh = 0.
        self.vibratoSpeed = 0.

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Input"
        else:
            name = "Input"
        return [loc["n_io"], name]

class CurveInputNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {}
        outputs = {"Curve": "ClampedFloat"}
        def func(self):
            return {"Curve": self.curve}
        super().__init__(inputs, outputs, func, True, **kwargs)
        self.curve = 0.
        self.auxData = {"name": "custom curve"}
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Curve Input"
        else:
            name = "Curve Input"
        return [loc["n_io"], name]

class OutputNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio",}
        outputs = {}
        def func(self, Audio):
            self.audio = Audio
            return {}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.audio = torch.empty([0, global_consts.frameSize])
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Output"
        else:
            name = "Output"
        return [loc["n_io"], name]

class addFloatNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float"}
        outputs = {"Result": "Float"}
        def func(self, A, B):
            return {"Result": A + B}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Add"
        else:
            name = "Add"
        return [loc["n_math"], name]

class subtractFloatNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float"}
        outputs = {"Result": "Float"}
        def func(self, A, B):
            return {"Result": A - B}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Subtract"
        else:
            name = "Subtract"
        return [loc["n_math"], name]

class multiplyFloatNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float"}
        outputs = {"Result": "Float"}
        def func(self, A, B):
            return {"Result": A * B}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Multiply"
        else:
            name = "Multiply"
        return [loc["n_math"], name]

class divideFloatNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float", "_0": "Float"}
        outputs = {"Result": "Float"}
        def func(self, A, B, _0):
            if B == 0:
                return {"Result": _0}
            return {"Result": A / B}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Divide"
        else:
            name = "Divide"
        return [loc["n_math"], name]

additionalNodes = []

def getNodeCls(name:str) -> type:
    md = globals()
    NodeClasses = [
        md[c] for c in md if (
            isinstance(md[c], type) and md[c].__module__ == __name__
        )
    ]
    if len(additionalNodes) > 0:
        NodeClasses.append(*additionalNodes)
    for cls in NodeClasses:
        if cls.__name__ == name:
            return cls
    return None
