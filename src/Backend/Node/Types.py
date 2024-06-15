#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from torch import Tensor

from UI.editor.Util import FloatInput, IntInput

class ClampedFloat():
    """node data type class for a Float confined within the [-1, 1] interval"""

    def __init__(self) -> None:
        self.UIColor = (1., 0.1, 0.1)
        self.defaultValue = 0.
        self.hasWidget = True

    @staticmethod
    def convert(*args):
        result = float(*args)
        result = max(-1., result)
        result = min(1., result)
        return result
    
    def make_widget(self, parent, setter = None):
        widget = Slider(min = -1., max = 1., value = self.defaultValue)
        parent.add_widget(widget)
        widget.bind(value = setter)


class Float():
    """node data type class for a standard Float"""

    def __init__(self) -> None:
        self.UIColor = (1., 0.7, 0.7)
        self.defaultValue = 0.5
        self.hasWidget = True

    @staticmethod
    def convert(*args):
        return float(*args)
    
    def make_widget(self, parent, setter = None):
        widget = FloatInput(text = str(self.defaultValue))
        parent.add_widget(widget)
        widget.bind(text = setter)


class Int():
    """node data type class for a standard integer"""

    def __init__(self) -> None:
        self.UIColor = (0.1, 1., 0.1)
        self.defaultValue = 1
        self.hasWidget = True

    @staticmethod
    def convert(*args):
        return int(*args)
    
    def make_widget(self, parent, setter = None):
        widget = IntInput(text = str(self.defaultValue))
        parent.add_widget(widget)
        widget.bind(text = setter)


class Bool():
    """node data type class for a standard boolean"""

    def __init__(self) -> None:
        self.UIColor = (0.5, 0.5, 0.5)
        self.defaultValue = False
        self.hasWidget = True

    @staticmethod
    def convert(*args):
        return bool(*args)
    
    def make_widget(self, parent, setter = None):
        widget = CheckBox(active = self.defaultValue)
        parent.add_widget(widget)
        widget.bind(active = setter)


class ESPERAudio():
    """node data type class for a PyTorch tensor representing a "Specharm", a point in an audio signal encoded using ESPER."""

    #TODO: finish this class
    def __init__(self) -> None:
        self.UIColor = (1., 1., 1.)
        self.defaultValue = 0.5
        self.hasWidget = False

    @staticmethod
    def convert(*args):
        return Tensor(*args)


class Phoneme():
    """node data type class for a "phoneme state" of a track. Consists of one or two phonemes, and a value between 0 and 1 representing their relative strength in the case of two phonemes"""

    #TODO: finish this class
    def __init__(self) -> None:
        self.UIColor = (0.1, 0.1, 1.)
        self.defaultValue = 0.5
        self.hasWidget = False

    @staticmethod
    def convert(*args):
        return None

types = [ClampedFloat, Float, Int, Bool, ESPERAudio, Phoneme] #list of all available node data types, can be extended through addons

def getType(name:str):
    """returns the node data type class for a given node data type name"""

    for t in types:
        if t.__name__ == name:
            return t
    return None