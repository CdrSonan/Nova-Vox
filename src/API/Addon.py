# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from typing import Any
from bisect import bisect_left, bisect_right
from kivy.lang import Builder
global middleLayer
from UI.code.editor.Main import middleLayer

class Override():
    def __init__(self, type:str, callback:callable, priority:int = 0) -> None:
        self._type = type
        self.callback = callback
        self.priority = priority
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.callback(*args, **kwargs)
    
    def __eq__(self, __value: object) -> bool:
        return self.priority == __value
    
class UIExtension():
    def __init__(self, instance:Any, priority:int = 0) -> None:
        self.instance = instance
        self.priority = priority
        
    def __eq__(self, __value: object) -> bool:
        return self.priority == __value

def override(func:callable):
    def wrapper(*args, **kwargs):
        funcID = "".join(func.__module__, ".", func.__name__)
        if funcID in middleLayer.overrides:
            for override in middleLayer.overrides[funcID]["prepends"]:
                override(*args, **kwargs)
            if "override" in middleLayer.overrides[funcID].keys():
                returnVal = middleLayer.overrides[funcID]["override"](*args, **kwargs)
            else:
                returnVal = func(*args, **kwargs)
            for override in middleLayer.overrides[funcID]["appends"]:
                override(*args, **kwargs)
            return returnVal
    return wrapper

def registerOverride(target:callable, type:str, callback:callable, priority:int = 0) -> None:
    if type not in ("prepend", "append", "override"):
        raise ValueError("invalid override type")
    if target not in middleLayer.overrides:
        middleLayer.overrides[target] = {"prepends":[], "appends": []}
    override = Override(type, callback, priority)
    if type == "override":
        middleLayer.overrides[target]["override"] = callback
    elif type == "prepend":
        index = bisect_left(middleLayer.overrides[target]["prepends"], override)
        middleLayer.overrides[target]["prepends"].insert(index, override)
    elif type == "append":
        index = bisect_right(middleLayer.overrides[target]["appends"], override)
        middleLayer.overrides[target]["appends"].insert(index, override)

def registerKV(rule:str) -> None:
    Builder.load_string(rule)

def registerUIElement(instance:Any, location:str, priority:int = 0):
    if location not in middleLayer.UIExtensions.keys():
        raise ValueError("invalid UI element location")
    extension = UIExtension(instance, priority)
    index = bisect_left(middleLayer.UIExtensions[location], priority)
    middleLayer.UIExtensions[location].insert(index, extension)

def registerNode(nodeClass:Any):
    middleLayer.nodeClasses.append(nodeClass)
