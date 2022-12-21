#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

class AiParam:
    """currently unused container class for an AI-driven parameter"""
    
    def __init__(self, filepath = None):
        pass

class AiParamStack:
    """currently unused class for holding and managing an execution stack of AI-driven parameters. Will be replaced by appropriate container for node tree."""

    def __init__(self, params):
        self.params = params
        self.enabled = []
        for i in self.params:
            self.enabled.append(True)
    def addParam(self, filepath):
        self.params.append(AiParam(filepath))
        self.enabled.append(True)
    def removeParam(self, index):
        del self.params[index]
        del self.enabled[index]
    def enableParam(self, index):
        self.enabled[index] = True
    def disableParam(self, index):
        self.enabled[index] = False