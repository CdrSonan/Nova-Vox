# Copyright 2023, 2024 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from typing import Any
from inspect import getframeinfo
from sys import _getframe
from bisect import bisect_left, bisect_right
from os import path
from kivy.lang import Builder
from MiddleLayer.IniParser import readSettings
import Backend.NodeLib

global overrides, UIExtensions
overrides = dict()
UIExtensions = {"addonPanel": [], "filePanel": [], "noteContextMenu": []}

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
    def __init__(self, instance:Any, addon: str, priority:int = 0) -> None:
        self.instance = instance
        self.addon = addon
        self.priority = priority
        
    def __eq__(self, __value: object) -> bool:
        return self.priority == __value

def override(func:callable):
    def wrapper(*args, **kwargs):
        funcID = ".".join([func.__module__, func.__qualname__])
        if funcID in overrides.keys():
            for override in overrides[funcID]["prepends"]:
                args, kwargs = override(*args, **kwargs)
            if "override" in overrides[funcID].keys():
                returnVal = overrides[funcID]["override"](*args, **kwargs)
            else:
                returnVal = func(*args, **kwargs)
            for override in overrides[funcID]["appends"]:
                returnVal, args, kwargs = override(returnVal, *args, **kwargs)
            return returnVal
    return wrapper

def registerOverride(target:callable, type:str, callback:callable, priority:int = 0) -> None:
    if type not in ("prepend", "append", "override"):
        raise ValueError("invalid override type")
    if target not in overrides:
        overrides[target] = {"prepends":[], "appends": []}
    override = Override(type, callback, priority)
    if type == "override":
        overrides[target]["override"] = callback
    elif type == "prepend":
        index = bisect_left(overrides[target]["prepends"], override)
        overrides[target]["prepends"].insert(index, override)
    elif type == "append":
        index = bisect_right(overrides[target]["appends"], override)
        overrides[target]["appends"].insert(index, override)

def registerKV(rule:str) -> None:
    Builder.load_string(rule)

def getAddon(level:int = 2) -> str:
    file = getframeinfo(_getframe(level))[0]
    addonPath = path.join(readSettings()["datadir"], "addons")
    addon = None
    while not path.samefile(addonPath, file):
        file, addon = path.split(file)
        if addon == "":
            return None
    return str(addon)

def registerUIElement(instance:Any, location:str, priority:int = 0, addon:str = None):
    if location not in UIExtensions.keys():
        raise ValueError("invalid UI element location")
    if addon == None:
        addon = getAddon()
    extension = UIExtension(instance, addon, priority)
    index = bisect_left(UIExtensions[location], priority)
    UIExtensions[location].insert(index, extension)

def registerNode(nodeClass:Any):
    Backend.NodeLib.additionalNodes.append(nodeClass)
