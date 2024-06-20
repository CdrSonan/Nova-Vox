# Copyright 2024 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch

from kivy.uix.textinput import TextInput

from Backend.Node.Node import Node, Param
from Backend.Node.NodeBase import NodeBase

additionalNodes = []

class CurveInputNode(Node):
    def __init__(self, base: NodeBase, **kwargs) -> None:
        super().__init__(base, **kwargs)
        self.setCurveName("curve")
        self.add_widget(TextInput(text = self.base.auxData["name"], multiline = False, size_hint = (1, 1), on_text_validate = self.onTextValidate))
    
    def remove(self):
        del self.parent.parent.track.nodegraph.params[self.base.auxData["name"]]
        return super().remove()
    
    def setCurveName(self, name:str):
        if name in self.parent.parent.track.nodegraph.params.keys():
            name = name + "_new"
        if self.base.auxData["name"] in self.parent.parent.track.nodegraph.params.keys():
            self.parent.parent.track.nodegraph.params[name] = self.parent.parent.track.nodegraph.params.pop(self.base.auxData["name"])
        else:
            self.parent.parent.track.nodegraph.params[name] = Param(torch.full((self.parent.parent.track.length,), 0, dtype = torch.half), True)
        self.base.auxData["name"] = name
        return name
    
    def onTextValidate(self, instance:TextInput):
        name = self.setCurveName(instance.text)
        return name

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