#Copyright 2022, 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import math
from random import random

import torch
import torchaudio

import global_consts
from Util import freqToFreqBin, freqBinToHarmonic, harmonicToFreqBin, amplitudeToDecibels, decibelsToAmplitude, rebaseHarmonics
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
        self.audio = torch.zeros([global_consts.frameSize + global_consts.halfTripleBatchSize + 2,])
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

class SinNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.sin(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Sin"
        else:
            name = "Sin"
        return [loc["n_math"], loc["n_math_trig"], name]

class CosNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.cos(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Cos"
        else:
            name = "Cos"
        return [loc["n_math"], loc["n_math_trig"], name]

class TanNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.tan(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Tan"
        else:
            name = "Tan"
        return [loc["n_math"], loc["n_math_trig"], name]

class ASinNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.asin(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "ASin"
        else:
            name = "ASin"
        return [loc["n_math"], loc["n_math_trig"], name]
    
class ACosNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.acos(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "ACos"
        else:
            name = "ACos"
        return [loc["n_math"], loc["n_math_trig"], name]

class ATanNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.atan(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "ATan"
        else:
            name = "ATan"
        return [loc["n_math"], loc["n_math_trig"], name]
    
class SinhNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.sinh(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Sinh"
        else:
            name = "Sinh"
        return [loc["n_math"], loc["n_math_trig"], name]

class CoshNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.cosh(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Cosh"
        else:
            name = "Cosh"
        return [loc["n_math"], loc["n_math_trig"], name]

class TanhNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": math.tanh(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Tanh"
        else:
            name = "Tanh"
        return [loc["n_math"], loc["n_math_trig"], name]

class PowerNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Base": "Float", "Exponent": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Base, Exponent):
            return {"Result": math.pow(Base, Exponent)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Power"
        else:
            name = "Power"
        return [loc["n_math"], name]

class MaxNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float"}
        outputs = {"Result": "Float"}
        def func(self, A, B):
            return {"Result": max(A, B)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Max"
        else:
            name = "Max"
        return [loc["n_math"], name]

class MinNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float"}
        outputs = {"Result": "Float"}
        def func(self, A, B):
            return {"Result": min(A, B)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Min"
        else:
            name = "Min"
        return [loc["n_math"], name]

class AbsNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": abs(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Abs"
        else:
            name = "Abs"
        return [loc["n_math"], name]

class FloorNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Int"}
        def func(self, Input):
            return {"Result": math.floor(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Floor"
        else:
            name = "Floor"
        return [loc["n_math"], name]

class CeilNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Int"}
        def func(self, Input):
            return {"Result": math.ceil(Input)}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Ceil"
        else:
            name = "Ceil"
        return [loc["n_math"], name]

class EqualNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float"}
        outputs = {"Result": "Bool"}
        def func(self, A, B):
            return {"Result": A == B}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Equal"
        else:
            name = "Equal"
        return [loc["n_math"], name]

class NearNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float", "Threshold": "Float"}
        outputs = {"Result": "Bool"}
        def func(self, A, B, Threshold):
            return {"Result": abs(A - B) <= Threshold}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Near"
        else:
            name = "Near"
        return [loc["n_math"], name]

class GreaterNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float", "Include_Equal": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, A, B, Include_Equal):
            return {"Result": A > B if Include_Equal else A >= B}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Greater"
        else:
            name = "Greater"
        return [loc["n_math"], name]
    
class LessNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Float", "B": "Float", "Include_Equal": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, A, B, Include_Equal):
            return {"Result": A < B if Include_Equal else A <= B}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Less"
        else:
            name = "Less"
        return [loc["n_math"], name]

class DerivativeNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Input):
            return {"Result": Input - self.prevInput}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.prevInput = 0.
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Derivative"
        else:
            name = "Derivative"
        return [loc["n_math"], name]

class IntegralNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float", "Reset": "Bool", "Decay": "ClampedFloat"}
        outputs = {"Result": "Float"}
        def func(self, Input, Reset, Decay):
            if Reset:
                self.integral = 0.
            else:
                self.integral += Input
                self.integral *= 0.5 * Decay + 0.5
            return {"Result": self.integral}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.integral = 0.
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Integral"
        else:
            name = "Integral"
        return [loc["n_math"], name]

class FloatSmoothingNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float", "Exponent": "ClampedFloat"}
        outputs = {"Result": "Float"}
        def func(self, Input, Exponent):
            effExponent = 0.5 * Exponent + 0.5
            self.smoothed = self.smoothed * effExponent + Input * (1. - effExponent)
            return {"Result": self.smoothed}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.smoothed = 0.

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Smoothing"
        else:
            name = "Smoothing"
        return [loc["n_math"], name]

class RangeMapNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float", "In_Min": "Float", "In_Max": "Float", "Out_Min": "Float", "Out_Max": "Float", "Clamp": "Bool"}
        outputs = {"Result": "Float"}
        def func(self, Input, In_Min, In_Max, Out_Min, Out_Max, Clamp):
            result = (Input - In_Min) / (In_Max - In_Min)
            if Clamp:
                result = max(0., min(1., result))
            return {"Result": result * (Out_Max - Out_Min) + Out_Min}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Range Map"
        else:
            name = "Range Map"
        return [loc["n_math"], name]

class FloatDelayNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Float", "Delay": "Int"}
        outputs = {"Result": "Float"}
        def func(self, Input, Delay):
            if len(self.delayBuffer) < Delay:
                if len(self.delayBuffer) == 0:
                    self.delayBuffer = [Input] * Delay
                self.delayBuffer += [Input] * (Delay - len(self.delayBuffer))
            elif len(self.delayBuffer) > Delay:
                self.delayBuffer = self.delayBuffer[:Delay]
            self.delayBuffer.append(Input)
            return {"Result": self.delayBuffer.pop(0)}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.delayBuffer = []
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Delay"
        else:
            name = "Delay"
        return [loc["n_math"], name]

class RNGNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Trigger": "Bool"}
        outputs = {"Result": "ClampedFloat"}
        def func(self, Trigger):
            if Trigger:
                self.stored = random() * 2. - 1.
            return {"Result": self.stored}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.stored = random() * 2. - 1.
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "random number generator"
        else:
            name = "random number generator"
        return [loc["n_math"], name]

class SwitchBoolNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Condition": "Bool", "if_True": "Bool", "if_False": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, Condition, if_True, if_False):
            return {"Result": if_True if Condition else if_False}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Switch Bool"
        else:
            name = "Switch Bool"
        return [loc["n_logic"], name]

class SwitchNumericNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Condition": "Bool", "if_True": "Float", "if_False": "Float"}
        outputs = {"Result": "Float"}
        def func(self, Condition, if_True, if_False):
            return {"Result": if_True if Condition else if_False}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Switch Numeric"
        else:
            name = "Switch Numeric"
        return [loc["n_logic"], name]

class SwitchAudioNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Condition": "Bool", "if_True": "ESPERAudio", "if_False": "ESPERAudio"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Condition, if_True, if_False):
            return {"Result": if_True if Condition else if_False}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Switch Audio"
        else:
            name = "Switch Audio"
        return [loc["n_logic"], name]

class SwitchPhonemeNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Condition": "Bool", "if_True": "Phoneme", "if_False": "Phoneme"}
        outputs = {"Result": "Phoneme"}
        def func(self, Condition, if_True, if_False):
            return {"Result": if_True if Condition else if_False}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Switch Phoneme"
        else:
            name = "Switch Phoneme"
        return [loc["n_logic"], name]

class LogicAndNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Bool", "B": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, A, B):
            return {"Result": A and B}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "AND"
        else:
            name = "AND"
        return [loc["n_logic"], name]

class LogicOrNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Bool", "B": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, A, B):
            return {"Result": A or B}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "OR"
        else:
            name = "OR"
        return [loc["n_logic"], name]

class LogicNotNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, Input):
            return {"Result": not Input}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "NOT"
        else:
            name = "NOT"
        return [loc["n_logic"], name]

class LogicXorNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Bool", "B": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, A, B):
            return {"Result": A != B}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "XOR"
        else:
            name = "XOR"
        return [loc["n_logic"], name]

class LogicNandNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Bool", "B": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, A, B):
            return {"Result": not (A and B)}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "NAND"
        else:
            name = "NAND"
        return [loc["n_logic"], name]

class LogicNorNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Bool", "B": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, A, B):
            return {"Result": not (A or B)}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "NOR"
        else:
            name = "NOR"
        return [loc["n_logic"], name]

class LogicXnorNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "Bool", "B": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, A, B):
            return {"Result": A == B}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "XNOR"
        else:
            name = "XNOR"
        return [loc["n_logic"], name]

class PosFlankNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, Input):
            return {"Result": Input and not self.prevInput}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.prevInput = False
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Positive Flank"
        else:
            name = "Positive Flank"
        return [loc["n_logic"], name]

class NegFlankNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, Input):
            return {"Result": not Input and self.prevInput}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.prevInput = False
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Negative Flank"
        else:
            name = "Negative Flank"
        return [loc["n_logic"], name]

class FlipflopNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, Input):
            self.state = not self.state if Input else self.state
            return {"Result": self.state}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.state = False
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Flip-Flop"
        else:
            name = "Flip-Flop"
        return [loc["n_logic"], name]

class BoolDelayNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Input": "Bool", "Delay": "Int"}
        outputs = {"Result": "Bool"}
        def func(self, Input, Delay):
            if len(self.delayBuffer) < Delay:
                if len(self.delayBuffer) == 0:
                    self.delayBuffer = [Input] * Delay
                self.delayBuffer += [Input] * (Delay - len(self.delayBuffer))
            elif len(self.delayBuffer) > Delay:
                self.delayBuffer = self.delayBuffer[:Delay]
            self.delayBuffer.append(Input)
            return {"Result": self.delayBuffer.pop(0)}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.delayBuffer = []
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Delay"
        else:
            name = "Delay"
        return [loc["n_logic"], name]

class PhonemeInListNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Phoneme": "Phoneme"}
        outputs = {"Result": "Bool"}
        def func(self, Phoneme):
            if Phoneme[2] > 0.5:
                phoneme = Phoneme[1]
            else:
                phoneme = Phoneme[0]
            return {"Result": phoneme in self.auxData["list"]}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.auxData = {"list": []}
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Phoneme in List"
        else:
            name = "Phoneme in List"
        return [loc["n_phonetics"], name]

class SoftPhonemeInListNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Phoneme": "Phoneme"}
        outputs = {"Result": "Float"}
        def func(self, Phoneme):
            result = 0.
            if Phoneme[0] in self.auxData["list"]:
                result += 1. - Phoneme[2]
            if Phoneme[1] in self.auxData["list"]:
                result += Phoneme[2]
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.auxData = {"list": []}
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Soft Phoneme in List"
        else:
            name = "Soft Phoneme in List"
        return [loc["n_phonetics"], name]

class SplitPhonemeNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Phoneme": "Phoneme"}
        outputs = {"Phoneme": "Phoneme", "Expression": "Phoneme"}
        def func(self, Phoneme):
            combA = Phoneme[0]
            combB = Phoneme[1]
            transition = Phoneme[2]
            if combA.startswith("_"):
                phonA = combA
                exprA = ""
            elif "_" in combA:
                phonA, exprA = combA.split("_", 1)
            else:
                phonA = combA
                exprA = ""
            if combB.startswith("_"):
                phonB = combB
                exprB = ""
            elif "_" in combB:
                phonB, exprB = combB.split("_", 1)
            else:
                phonB = combB
                exprB = ""
            phoneme = (phonA, phonB, transition)
            expression = (exprA, exprB, transition)
            return {"Phoneme": phoneme, "Expression": expression}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Split Phoneme"
        else:
            name = "Split Phoneme"
        return [loc["n_phonetics"], name]

class PhonemeDelayNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Phoneme": "Phoneme", "Delay": "Int"}
        outputs = {"Result": "Phoneme"}
        def func(self, Phoneme, Delay):
            if len(self.delayBuffer) < Delay:
                if len(self.delayBuffer) == 0:
                    self.delayBuffer = [Phoneme] * Delay
                self.delayBuffer += [Phoneme] * (Delay - len(self.delayBuffer))
            elif len(self.delayBuffer) > Delay:
                self.delayBuffer = self.delayBuffer[:Delay]
            self.delayBuffer.append(Phoneme)
            return {"Result": self.delayBuffer.pop(0)}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.delayBuffer = []
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Phoneme Delay"
        else:
            name = "Phoneme Delay"
        return [loc["n_phonetics"], name]

class AudioSmoothingNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Exponent": "ClampedFloat"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Exponent):
            effExponent = 0.5 * Exponent + 0.5
            self.smoothed = self.smoothed * effExponent + Audio * (1. - effExponent)
            return {"Result": self.smoothed}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.smoothed = torch.zeros([global_consts.frameSize,])
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Smoothing"
        else:
            name = "Smoothing"
        return [loc["n_audio"], name]

class AudioDelayNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Delay": "Int"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Delay):
            if len(self.delayBuffer) < Delay:
                if len(self.delayBuffer) == 0:
                    self.delayBuffer = [Audio] * Delay
                self.delayBuffer += [Audio] * (Delay - len(self.delayBuffer))
            elif len(self.delayBuffer) > Delay:
                self.delayBuffer = self.delayBuffer[:Delay]
            self.delayBuffer.append(Audio)
            return {"Result": self.delayBuffer.pop(0)}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.delayBuffer = []
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Delay"
        else:
            name = "Delay"
        return [loc["n_audio"], name]

class AudioVolumeNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio"}
        outputs = {"Result": "Float"}
        def func(self, Audio):
            voicedVolume = torch.mean(torch.sqrt(Audio[:global_consts.halfHarms]))
            unvoicedVolume = torch.mean(torch.sqrt(Audio[global_consts.nHarmonics + 2:global_consts.frameSize]))
            return {"Result": (voicedVolume + unvoicedVolume).item()}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "get audio volume"
        else:
            name = "get audio volume"
        return [loc["n_audio"], name]

class AudioAmplitudeNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio"}
        outputs = {"Result": "Float"}
        def func(self, Audio):
            return {"Result": Audio[1]}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "get audio main amplitude"
        else:
            name = "get audio main amplitude"
        return [loc["n_audio"], name]

class AddAudioNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "ESPERAudio", "B": "ESPERAudio"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, A, B):
            #TODO: implement SLERP for phase portion of ESPERAudio
            result = A + B
            result[global_consts.halfHarms:global_consts.nHarmonics + 2] = A[global_consts.halfHarms:global_consts.nHarmonics + 2]
            result[global_consts.frameSize:] = A[global_consts.frameSize:]
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Add Audio"
        else:
            name = "Add Audio"
        return [loc["n_audio"], name]

class MixAudioNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "ESPERAudio", "B": "ESPERAudio", "Mix": "ClampedFloat"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, A, B, Mix):
            mix = 0.5 * Mix + 0.5
            #TODO: implement SLERP for phase portion of ESPERAudio
            return {"Result": A * (1. - mix) + B * mix}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Mix Aduio"
        else:
            name = "Mix Audio"
        return [loc["n_audio"], name]

class SubtractAudioNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": "ESPERAudio", "B": "ESPERAudio"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, A, B):
            result = torch.max(A - B, torch.zeros_like(A))
            result[global_consts.halfHarms:global_consts.nHarmonics + 2] = A[global_consts.halfHarms:global_consts.nHarmonics + 2]
            result[global_consts.frameSize:] = A[global_consts.frameSize:]
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Subtract Audio"
        else:
            name = "Subtract Audio"
        return [loc["n_audio"], name]

class AdjustVolumeNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Volume": "ClampedFloat"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Volume):
            result = Audio.clone()
            result[:global_consts.frameSize] *= (1. + Volume)
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Adjust Volume"
        else:
            name = "Adjust Volume"
        return [loc["n_audio"], name]

class SeparateVoicedUnvoicedNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio"}
        outputs = {"Voiced": "ESPERAudio", "Unvoiced": "ESPERAudio"}
        def func(self, Audio):
            voiced = torch.zeros_like(Audio)
            unvoiced = torch.zeros_like(Audio)
            voiced[:global_consts.nHarmonics + 2] = Audio[:global_consts.nHarmonics + 2]
            voiced[-1] = Audio[-1]
            unvoiced[global_consts.nHarmonics + 2:] = Audio[global_consts.nHarmonics + 2:]
            return {"Voiced": voiced, "Unvoiced": unvoiced}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Separate Voiced/Unvoiced"
        else:
            name = "Separate Voiced/Unvoiced"
        return [loc["n_audio"], name]

class SamplerNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Record": "Bool", "Play": "Bool"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Record, Play):
            if Record:
                if not self.recording:
                    self.recording = True
                    self.recorded = []
                self.recorded.append(Audio)
            else:
                self.recording = False
            if Play:
                if not self.playing:
                    self.playing = True
                    self.playIndex = 0
                if self.playIndex >= len(self.recorded):
                    self.playIndex = 0
                result = self.recorded[self.playIndex]
                self.playIndex += 1
            else:
                self.playing = False
                result = torch.zeros_like(Audio)
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.recording = False
        self.recorded = []
        self.playing = False
        self.playIndex = 0
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Sampler"
        else:
            name = "Sampler"
        return [loc["n_audio_adv"], name]

class ADSREnvelopeNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Trigger": "Bool", "Attack": "Float", "Decay": "Float", "Sustain": "ClampedFloat", "Release": "Float"}
        outputs = {"Result": "ClampedFloat"}
        def func(self, Trigger, Attack, Decay, Sustain, Release):
            if Trigger:
                self.state = 0
            if self.state == 0:
                self.state += Attack
            elif self.state == 1:
                self.state += Decay
            elif self.state == 2:
                self.state = Sustain
            elif self.state == 3:
                self.state += Release
            return {"Result": self.state}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "ADSR Envelope"
        else:
            name = "ADSR Envelope"
        return [loc["n_audio_adv"], name]

class SineLFONode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Frequency": "Float", "Phase": "ClampedFloat", "Trigger": "Bool"}
        outputs = {"Result": "ClampedFloat"}
        def func(self, Frequency, Phase, Trigger):
            if Trigger:
                self.phase = Phase
            self.phase += 2. * global_consts.tickRate / Frequency
            if self.phase >= 1.:
                self.phase -= 2.
            return {"Result": math.sin(math.pi + math.pi * self.phase)}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.phase = -1.
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Sine LFO"
        else:
            name = "Sine LFO"
        return [loc["n_LFO"], name]

class SquareLFONode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Frequency": "Float", "Phase": "ClampedFloat", "Trigger": "Bool"}
        outputs = {"Result": "ClampedFloat"}
        def func(self, Frequency, Phase, Trigger):
            if Trigger:
                self.phase = Phase
            self.phase += 2. * global_consts.tickRate / Frequency
            if self.phase >= 1.:
                self.phase -= 2.
            return {"Result": 1. if self.phase > 0. else -1.}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.phase = -1.
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Square LFO"
        else:
            name = "Square LFO"
        return [loc["n_LFO"], name]

class SawLFONode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Frequency": "Float", "Phase": "ClampedFloat", "Trigger": "Bool"}
        outputs = {"Result": "ClampedFloat"}
        def func(self, Frequency, Phase, Trigger):
            if Trigger:
                self.phase = Phase
            self.phase += 2. * global_consts.tickRate / Frequency
            if self.phase >= 1.:
                self.phase -= 2.
            return {"Result": self.phase}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.phase = -1.
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Saw LFO"
        else:
            name = "Saw LFO"
        return [loc["n_LFO"], name]

class InvSawLFONode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Frequency": "Float", "Phase": "ClampedFloat", "Trigger": "Bool"}
        outputs = {"Result": "ClampedFloat"}
        def func(self, Frequency, Phase, Trigger):
            if Trigger:
                self.phase = Phase
            self.phase += 2. * global_consts.tickRate / Frequency
            if self.phase >= 1.:
                self.phase -= 2.
            return {"Result": -self.phase}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.phase = -1.
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Inverted Saw LFO"
        else:
            name = "Inverted Saw LFO"
        return [loc["n_LFO"], name]

class TriangleLFONode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Frequency": "Float", "Phase": "ClampedFloat", "Trigger": "Bool"}
        outputs = {"Result": "ClampedFloat"}
        def func(self, Frequency, Phase, Trigger):
            if Trigger:
                self.phase = Phase
            self.phase += 2. * global_consts.tickRate / Frequency
            if self.phase >= 1.:
                self.phase -= 2.
            if self.phase < -0.5:
                result = 2. * self.phase + 2.
            elif self.phase < 0.5:
                result = -2. * self.phase
            else:
                result = 2. * self.phase - 2.
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.phase = -1.
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Triangle LFO"
        else:
            name = "Triangle LFO"
        return [loc["n_LFO"], name]

class PulseLFONode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Frequency": "Float", "Phase": "ClampedFloat", "Trigger": "Bool"}
        outputs = {"Result": "Bool"}
        def func(self, Frequency, Phase, Trigger):
            if Trigger:
                self.phase = Phase
            self.phase += 2. * global_consts.tickRate / Frequency
            if self.phase >= 1.:
                self.phase -= 2.
                result = True
            else:
                result = False
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Pulse LFO"
        else:
            name = "Pulse LFO"
        return [loc["n_LFO"], name]

class HighpassNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Cutoff": "Float", "Slope": "ClampedFloat"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Cutoff):
            centerBin = freqToFreqBin(Cutoff)
            slope = 0.5 * self.slope + 0.5
            slopeRange = min(centerBin, global_consts.halfTripleBatchSize - centerBin)
            lowBin = max(0, centerBin - slopeRange * slope)
            highBin = min(global_consts.halfTripleBatchSize, centerBin + slopeRange * slope)
            result = Audio.clone()
            for i in range(lowBin):
                result[global_consts.nHarmonics + 2 + i] = 0.
            for i in range(lowBin, highBin):
                result[global_consts.nHarmonics + 2 + i] *= (i - lowBin) / (highBin - lowBin)
            for i in range(global_consts.halfHarms):
                if harmonicToFreqBin(i, result[-1]) < lowBin:
                    result[i] = 0.
                elif harmonicToFreqBin(i, result[-1]) < highBin:
                    result[i] *= (harmonicToFreqBin(i, result[-1]) - lowBin) / (highBin - lowBin)
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Highpass Filter"
        else:
            name = "Highpass Filter"
        return [loc["n_eq"], name]

class LowpassNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Cutoff": "Float", "Slope": "ClampedFloat"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Cutoff):
            centerBin = freqToFreqBin(Cutoff)
            slope = 0.5 * self.slope + 0.5
            slopeRange = min(centerBin, global_consts.halfTripleBatchSize - centerBin)
            lowBin = max(0, centerBin - slopeRange * slope)
            highBin = min(global_consts.halfTripleBatchSize, centerBin + slopeRange * slope)
            result = Audio.clone()
            for i in range(lowBin, highBin):
                result[global_consts.nHarmonics + 2 + i] *= 1. - ((i - lowBin) / (highBin - lowBin))
            for i in range(highBin, global_consts.halfTripleBatchSize):
                result[global_consts.nHarmonics + 2 + i] = 0.
            for i in range(global_consts.halfHarms):
                if harmonicToFreqBin(i, result[-1]) > highBin:
                    result[i] = 0.
                elif harmonicToFreqBin(i, result[-1])  > lowBin:
                    result[i] *= 1. - ((harmonicToFreqBin(i, result[-1]) - lowBin) / (highBin - lowBin))
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Lowpass Filter"
        else:
            name = "Lowpass Filter"
        return [loc["n_eq"], name]

class BandpassNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Cutoff": "Float", "Width": "Float", "Slope": "ClampedFloat"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Cutoff, Width):
            centerBin = freqToFreqBin(Cutoff)
            widthBins = freqToFreqBin(Width)
            slope = 0.5 * self.slope + 0.5
            slopeRange = min(centerBin - widthBins / 2., global_consts.halfTripleBatchSize - centerBin - widthBins / 2., widthBins / 2.)
            outerLowBin = max(0, centerBin - widthBins / 2. - slopeRange * slope)
            innerLowBin = max(0, centerBin - widthBins / 2. + slopeRange * slope)
            innerHighBin = min(global_consts.halfTripleBatchSize, centerBin + widthBins / 2. - slopeRange * slope)
            outerHighBin = min(global_consts.halfTripleBatchSize, centerBin + widthBins / 2. + slopeRange * slope)
            result = Audio.clone()
            for i in range(outerLowBin):
                result[global_consts.nHarmonics + 2 + i] = 0.
            for i in range(outerLowBin, innerLowBin):
                result[global_consts.nHarmonics + 2 + i] *= (i - outerLowBin) / (innerLowBin - outerLowBin)
            for i in range(innerHighBin, outerHighBin):
                result[global_consts.nHarmonics + 2 + i] *= 1. - ((i - innerHighBin) / (outerHighBin - innerHighBin))
            for i in range(outerHighBin, global_consts.halfTripleBatchSize):
                result[global_consts.nHarmonics + 2 + i] = 0.
            for i in range(global_consts.halfHarms):
                if harmonicToFreqBin(i, result[-1]) < outerLowBin or harmonicToFreqBin(i, result[-1]) > outerHighBin:
                    result[i] = 0.
                elif harmonicToFreqBin(i, result[-1]) < innerLowBin:
                    result[i] *= (harmonicToFreqBin(i, result[-1]) - outerLowBin) / (innerLowBin - outerLowBin)
                elif harmonicToFreqBin(i, result[-1]) > innerHighBin:
                    result[i] *= 1. - ((harmonicToFreqBin(i, result[-1]) - innerHighBin) / (outerHighBin - innerHighBin))
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Bandpass Filter"
        else:
            name = "Bandpass Filter"
        return [loc["n_eq"], name]

class BandrejectNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Cutoff": "Float", "Width": "Float", "Slope": "ClampedFloat"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Cutoff, Width):
            centerBin = freqToFreqBin(Cutoff)
            widthBins = freqToFreqBin(Width)
            slope = 0.5 * self.slope + 0.5
            slopeRange = min(centerBin - widthBins / 2., global_consts.halfTripleBatchSize - centerBin - widthBins / 2., widthBins / 2.)
            innerLowBin = max(0, centerBin - widthBins / 2. - slopeRange * slope)
            outerLowBin = max(0, centerBin - widthBins / 2. + slopeRange * slope)
            outerHighBin = min(global_consts.halfTripleBatchSize, centerBin + widthBins / 2. - slopeRange * slope)
            innerHighBin = min(global_consts.halfTripleBatchSize, centerBin + widthBins / 2. + slopeRange * slope)
            result = Audio.clone()
            for i in range(outerLowBin, innerLowBin):
                result[global_consts.nHarmonics + 2 + i] *= 1. - ((i - innerLowBin) / (outerLowBin - innerLowBin))
            for i in range(innerLowBin, innerHighBin):
                result[global_consts.nHarmonics + 2 + i] *= 0.
            for i in range(innerHighBin, outerHighBin):
                result[global_consts.nHarmonics + 2 + i] *= (i - innerHighBin) / (outerHighBin - innerHighBin)
            for i in range(global_consts.halfHarms):
                if harmonicToFreqBin(i, result[-1]) < outerLowBin:
                    continue
                elif harmonicToFreqBin(i, result[-1]) < innerLowBin:
                    result[i] *= 1. - ((harmonicToFreqBin(i, result[-1]) - innerLowBin) / (outerLowBin - innerLowBin))
                elif harmonicToFreqBin(i, result[-1]) < innerHighBin:
                    result[i] = 0.
                elif harmonicToFreqBin(i, result[-1]) < outerHighBin:
                    result[i] *= (harmonicToFreqBin(i, result[-1]) - innerHighBin) / (outerHighBin - innerHighBin)
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Bandreject Filter"
        else:
            name = "Bandreject Filter"
        return [loc["n_eq"], name]

class ThreeBandEQNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio",
                  "Low_Gain": "ClampedFloat", "Low_Freq": "Float", "Low_Width": "Float",
                  "Mid_Gain": "ClampedFloat", "Mid_Freq": "Float", "Mid_Width": "Float",
                  "High_Gain": "ClampedFloat", "High_Freq": "Float", "High_Width": "Float"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Low_Gain, Low_Freq, Low_Width, Mid_Gain, Mid_Freq, Mid_Width, High_Gain, High_Freq, High_Width):
            lowFreq = freqToFreqBin(Low_Freq)
            lowWidth = freqToFreqBin(Low_Width)
            midFreq = freqToFreqBin(Mid_Freq)
            midWidth = freqToFreqBin(Mid_Width)
            highFreq = freqToFreqBin(High_Freq)
            highWidth = freqToFreqBin(High_Width)
            result = Audio.clone()
            for i in range(global_consts.halfTripleBatchSize + 1):
                result[global_consts.nHarmonics + 2 + i] *= 1. + self.normalDistribution(lowFreq, lowWidth, i) * Low_Gain
                result[global_consts.nHarmonics + 2 + i] *= 1. + self.normalDistribution(midFreq, midWidth, i) * Mid_Gain
                result[global_consts.nHarmonics + 2 + i] *= 1. + self.normalDistribution(highFreq, highWidth, i) * High_Gain
            for i in range(global_consts.halfHarms):
                result[i] *= 1. + self.normalDistribution(lowFreq, lowWidth, harmonicToFreqBin(i, result[-1])) * Low_Gain
                result[i] *= 1. + self.normalDistribution(midFreq, midWidth, harmonicToFreqBin(i, result[-1])) * Mid_Gain
                result[i] *= 1. + self.normalDistribution(highFreq, highWidth, harmonicToFreqBin(i, result[-1])) * High_Gain
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
    
    @staticmethod
    def normalDistribution(mean:float, std:float, x:float) -> float:
        return math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2. * math.pi))
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "3-Band EQ"
        else:
            name = "3-Band EQ"
        return [loc["n_eq"], name]

class CompressorNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Threshold": "Float", "Ratio": "ClampedFloat", "Attack": "Float", "Release": "Float"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Threshold, Ratio, Attack, Release):
            threshold = decibelsToAmplitude(Threshold)
            ratio = -0.5 * Ratio + 0.5
            amplitude = (torch.sum(Audio[:global_consts.halfHarms]) + torch.sum(Audio[global_consts.nHarmonics + 2:global_consts.frameSize])) / 2.
            result = Audio.clone()
            if amplitude > threshold and not self.active:
                self.active = True
                self.target = (threshold + (amplitude - threshold) * ratio) / amplitude
                self.stepsLeft = int(250. * Attack)
            elif amplitude < threshold and self.active:
                self.active = False
                self.target = 1.
                self.stepsLeft = int(250. * Release)
            if self.active and amplitude > self.target:
                self.target = amplitude
            if self.stepsLeft > 0:
                self.current += (self.target - self.current) / self.stepsLeft
                self.stepsLeft -= 1
            result[:global_consts.halfHarms] *= self.current
            result[global_consts.nHarmonics + 2:global_consts.frameSize] *= self.current
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.active = False
        self.current = 1.
        self.target = 1.
        self.stepsLeft = 0
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Compressor"
        else:
            name = "Compressor"
        return [loc["n_eq"], name]

class LimiterNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Threshold": "Float", "Attack": "Float", "Release": "Float"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Threshold, Attack, Release):
            threshold = decibelsToAmplitude(Threshold)
            amplitude = (torch.sum(Audio[:global_consts.halfHarms]) + torch.sum(Audio[global_consts.nHarmonics + 2:global_consts.frameSize])) / 2.
            result = Audio.clone()
            if amplitude > threshold and not self.active:
                self.active = True
                self.target = threshold / amplitude
                self.stepsLeft = int(250. * Attack)
            elif amplitude < threshold and self.active:
                self.active = False
                self.target = 1.
                self.stepsLeft = int(250. * Release)
            if self.active and amplitude > self.target:
                self.target = amplitude
            if self.stepsLeft > 0:
                self.current += (self.target - self.current) / self.stepsLeft
                self.stepsLeft -= 1
            result[:global_consts.halfHarms] *= self.current
            result[global_consts.nHarmonics + 2:global_consts.frameSize] *= self.current
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.active = False
        self.current = 1.
        self.target = 1.
        self.stepsLeft = 0
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Limiter"
        else:
            name = "Limiter"
        return [loc["n_eq"], name]

class ExpanderNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Threshold": "Float", "Ratio": "ClampedFloat", "Attack": "Float", "Release": "Float"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Threshold, Ratio, Attack, Release):
            threshold = decibelsToAmplitude(Threshold)
            ratio = Ratio + 1.
            amplitude = (torch.sum(Audio[:global_consts.halfHarms]) + torch.sum(Audio[global_consts.nHarmonics + 2:global_consts.frameSize])) / 2.
            result = Audio.clone()
            if amplitude < threshold and not self.active:
                self.active = True
                self.target = (threshold + (amplitude - threshold) * ratio) / amplitude
                self.stepsLeft = int(250. * Attack)
            elif amplitude > threshold and self.active:
                self.active = False
                self.target = 1.
                self.stepsLeft = int(250. * Release)
            if self.active and amplitude < self.target:
                self.target = amplitude
            if self.stepsLeft > 0:
                self.current += (self.target - self.current) / self.stepsLeft
                self.stepsLeft -= 1
            result[:global_consts.halfHarms] *= self.current
            result[global_consts.nHarmonics + 2:global_consts.frameSize] *= self.current
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.active = False
        self.current = 1.
        self.target = 1.
        self.stepsLeft = 0
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Expander"
        else:
            name = "Expander"
        return [loc["n_eq"], name]

class GateNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Threshold": "Float", "Attack": "Float", "Release": "Float"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Threshold, Attack, Release):
            threshold = decibelsToAmplitude(Threshold)
            amplitude = (torch.sum(Audio[:global_consts.halfHarms]) + torch.sum(Audio[global_consts.nHarmonics + 2:global_consts.frameSize])) / 2.
            result = Audio.clone()
            if amplitude < threshold and not self.active:
                self.active = True
                self.target = 0.
                self.stepsLeft = int(250. * Attack)
            elif amplitude > threshold and self.active:
                self.active = False
                self.target = 1.
                self.stepsLeft = int(250. * Release)
            if self.stepsLeft > 0:
                self.current += (self.target - self.current) / self.stepsLeft
                self.stepsLeft -= 1
            result[:global_consts.halfHarms] *= self.current
            result[global_consts.nHarmonics + 2:global_consts.frameSize] *= self.current
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.active = False
        self.current = 1.
        self.target = 1.
        self.stepsLeft = 0
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Gate"
        else:
            name = "Gate"
        return [loc["n_eq"], name]

class IRConvolutionNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio):
            result = torch.zeros_like(Audio)
            self.buffer = self.buffer.roll(0, 1)
            self.buffer[0] = Audio
            for i in range(self.buffer.size()[0]):
                result[global_consts.nHarmonics + 2:global_consts.frameSize] += self.buffer[i] * self.ir[i].abs()
                for j in range(global_consts.halfHarms):
                    bin = harmonicToFreqBin(j, Audio[-1])
                    bin = min(bin, global_consts.halfTripleBatchSize)
                    ir_interp = self.ir[i][bin] + (self.ir[i][bin + 1] - self.ir[i][bin]) * (bin - int(bin))
                    component = torch.polar(self.buffer[i][j], self.buffer[i][j + global_consts.halfHarms]) * ir_interp
                    result[j] = torch.abs(component)
                    result[j + global_consts.halfHarms] = torch.angle(component)
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.ir = torch.zeros((1, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64)
        self.buffer = torch.zeros((1, global_consts.halfTripleBatchSize + 1))
    
    def updateIRFromFile(self, path:str) -> None:
        loadedData = torchaudio.load(path)
        sampleRate = loadedData[1]
        transform = torchaudio.transforms.Resample(sampleRate, global_consts.sampleRate)
        waveform = transform(loadedData[0][0])
        if waveform.size(0) < global_consts.tripleBatchSize:
            waveform = torch.cat((waveform, torch.zeros(global_consts.tripleBatchSize - waveform.size(0))), 0)
        self.ir = torch.stft(waveform, global_consts.tripleBatchSize, global_consts.batchSize, global_consts.tripleBatchSize, window=torch.hann_window(global_consts.tripleBatchSize), return_complex=True).transpose(0, 1)
        self.buffer = torch.zeros_like(self.ir)

class IRReverbNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Wetness": "ClampedFloat", "Pre_Delay": "Float"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Wetness, Pre_Delay):
            result = torch.zeros_like(Audio)
            if self.buffer.size()[0] > self.ir.size()[0] + int(Pre_Delay * 250.):
                self.buffer = self.buffer[:self.ir.size()[0] + int(Pre_Delay * 250.)]
            elif self.buffer.size()[0] < self.ir.size()[0] + int(Pre_Delay * 250.):
                self.buffer = torch.cat((self.buffer, torch.zeros(self.ir.size()[0] + int(Pre_Delay * 250.) - self.buffer.size()[0])), 0)
            self.buffer = self.buffer.roll(0, 1)
            self.buffer[0] = Audio
            for i in range(int(Pre_Delay * 250.), self.buffer.size()[0]):
                result[global_consts.nHarmonics + 2:global_consts.frameSize] += self.buffer[i][global_consts.nHarmonics + 2:global_consts.frameSize] * self.ir[i - int(Pre_Delay * 250.)].abs()
                for j in range(global_consts.halfHarms):
                    bin = harmonicToFreqBin(j, Audio[-1])
                    bin = min(bin, global_consts.halfTripleBatchSize)
                    ir_interp = self.ir[i - int(Pre_Delay * 250.)][bin] + (self.ir[i - int(Pre_Delay * 250.)][bin + 1] - self.ir[i - int(Pre_Delay * 250.)][bin]) * (bin - int(bin))
                    component = torch.polar(self.buffer[i][j], self.buffer[i][j + global_consts.halfHarms]) * ir_interp
                    result[j] = torch.abs(component)
                    result[j + global_consts.halfHarms] = torch.angle(component)
            result = result * (0.5 * Wetness + 0.5) + Audio * (0.5 - 0.5 * Wetness)
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.ir = torch.zeros((1, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64)
        self.buffer = torch.zeros((1, global_consts.halfTripleBatchSize + 1))
        self.predelay = 0
    
    def updateIRFromFile(self, path:str) -> None:
        loadedData = torchaudio.load(path)
        sampleRate = loadedData[1]
        transform = torchaudio.transforms.Resample(sampleRate, global_consts.sampleRate)
        waveform = transform(loadedData[0][0])
        if waveform.size(0) < global_consts.tripleBatchSize:
            waveform = torch.cat((waveform, torch.zeros(global_consts.tripleBatchSize - waveform.size(0))), 0)
        self.ir = torch.stft(waveform, global_consts.tripleBatchSize, global_consts.batchSize, global_consts.tripleBatchSize, window=torch.hann_window(global_consts.tripleBatchSize), return_complex=True).transpose(0, 1)
        self.buffer = torch.zeros_like(self.ir)
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "IR Convolution"
        else:
            name = "IR Convolution"
        return [loc["n_fx"], name]

class ChorusNode(NodeBase):
    def __init__(self, **kwargs) -> None:
        inputs = {"Audio": "ESPERAudio", "Wetness": "ClampedFloat", "Voices": "Int", "Depth": "ClampedFloat", "Rate": "Float"}
        outputs = {"Result": "ESPERAudio"}
        def func(self, Audio, Wetness, Voices, Depth, Rate):
            result = Audio.clone()
            for i in range(Voices):
                lfo = math.sin(self.phase + 2. * math.pi * i / Voices)
                voice = self.buffer.clone()
                voice[:global_consts.nHarmonics + 2] = rebaseHarmonics(Audio[:global_consts.nHarmonics + 2], 1. + lfo * (0.5 + Depth * 0.4))
                voice[global_consts.nHarmonics + 2:global_consts.frameSize] += torch.roll(Audio[global_consts.nHarmonics + 2:global_consts.frameSize], int(10. * lfo * Depth), 0)
                voice[global_consts.nHarmonics + 2:global_consts.frameSize] += torch.roll(self.buffer[global_consts.nHarmonics + 2:global_consts.frameSize], int(-10. * lfo * Depth), 0)
                result += voice
            result *= (0.5 * Wetness + 0.5) / Voices
            self.phase += 2. * math.pi * Rate / 250.
            if self.phase >= 2. * math.pi:
                self.phase -= 2. * math.pi
            self.buffer = Audio.clone()
            return {"Result": result}
        super().__init__(inputs, outputs, func, False, **kwargs)
        self.phase = 0.
        self.buffer = torch.zeros((global_consts.frameSize,))
    
    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Chorus"
        else:
            name = "Chorus"
        return [loc["n_fx"], name]
    
    

# TODO: Implement the following nodes
# - reverb, chorus, phaser, flanger
# - noise generator
# - VST3 host
# - multiband compressor and/or dynamic EQ
# - V-synth params: formant shift, growl, brightness, strength
# - distortion, quantization, maybe bitcrusher?
# - impulse response reverb/convolution

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
