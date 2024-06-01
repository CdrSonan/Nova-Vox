#Copyright 2022, 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from Localization.editor_localization import getLanguage
loc = getLanguage()

from Util import classesinmodule

from API.Node import *

class addFloatNode(Node):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": Float, "B": Float}
        outputs = {"Result": Float}
        def func(A, B):
            return {"Result": A + B}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        if loc["lang"] == "en":
            name = "Add"
        else:
            name = "Add"
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
