# Copyright 2022-2024 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from copy import copy
import os

from kivy.uix.popup import Popup
from kivy.uix.button import ButtonBehavior
from kivy.uix.treeview import TreeViewLabel
from kivy.core.image import Image as CoreImage
from kivy.properties import ObjectProperty

import h5py

from Backend.Node import NodeBaseLib, NodeLib, Node
from Backend.DataHandler.HDF5 import MetadataStorage
from MiddleLayer.IniParser import readSettings
from UI.editor.Util import ManagedPopup

from Util import classesinmodule

import API.Ops

from Localization.editor_localization import getLanguage
loc = getLanguage()

class TreeViewButton(ButtonBehavior, TreeViewLabel):
    """basic class implementing a tree view node with button behavior"""
    nodeBase = ObjectProperty(None)
    node = ObjectProperty(None)
    editor = ObjectProperty(None)

    def on_press(self):
        if self.node == None:
            self.editor.addNode(Node.Node(self.nodeBase()), True)
        else:
            self.editor.addNode(self.node(self.nodeBase()), True)
        return super().on_press()

class SingerSettingsPanel(Popup):
    """Popup displaying per-track settings and the node editor"""

    def __init__(self, index, **kwargs) -> None:
        global middleLayer
        from UI.editor.Main import middleLayer
        super().__init__(**kwargs)
        self.index = index
        self.vbData = []
        self.voicebanks = []
        self.modVoicebanks = None
        self.filepaths = []
        self.listVoicebanks()
        self.pauseThreshold = middleLayer.trackList[self.index].pauseThreshold
        self.unvoicedShift = middleLayer.trackList[self.index].unvoicedShift
        if middleLayer.trackList[index].mixinVB == None:
            self.mixinVB = "None"
        else:
            self.mixinVB = self.voicebanks[self.filepaths.index(middleLayer.trackList[self.index].mixinVB)]
        self.mainVB = self.voicebanks[self.filepaths.index(middleLayer.trackList[self.index].vbPath)]
        self.children[0].children[0].children[0].children[0].children[2].children[6].text = self.mainVB
        self.children[0].children[0].children[0].children[0].children[2].children[6].values = self.voicebanks
        self.children[0].children[0].children[0].children[0].children[2].children[4].text = self.mixinVB
        self.children[0].children[0].children[0].children[0].children[2].children[4].values = self.modVoicebanks
        self.children[0].children[0].children[0].children[0].children[2].children[2].text = str(self.unvoicedShift)
        self.children[0].children[0].children[0].children[0].children[2].children[0].text = str(self.pauseThreshold)
        self.makeNodeTree()
        middleLayer.activePanel = self

    def listVoicebanks(self) -> None:
        """creates a list of installed Voicebanks for switching the main or mix-in Voicebank used by the track"""

        global middleLayer
        from UI.editor.Main import middleLayer
        voicePath = os.path.join(readSettings()["datadir"], "Voices")
        if os.path.isdir(voicePath) == False:
            popup = ManagedPopup(title = loc["error"], message = loc["dataDir_err"])
            popup.open()
            return
        files = os.listdir(voicePath)
        for file in files:
            if file.endswith(".nvvb"):
                with h5py.File(os.path.join(voicePath, file), "r") as f:
                    metadata = MetadataStorage(f).toMetadata()
                self.vbData.append(metadata)
                self.voicebanks.append(metadata.name)
                self.filepaths.append(os.path.join(voicePath, file))
        self.modVoicebanks = copy(self.voicebanks)
        self.modVoicebanks.append("None")

    def makeNodeTree(self):
        """builds the hierarchical view of all available nodes by searching the appropriate 1st-party and Addon Python modules"""

        
        NodeBaseClasses = classesinmodule(NodeBaseLib)
        if len(NodeBaseLib.additionalNodes) > 0:
            NodeBaseClasses.append(*NodeBaseLib.additionalNodes)
        for i in NodeBaseClasses:
            branch = i.name()
            parent = self.children[0].children[0].children[0].children[0].children[1].children[0].root
            while len(branch) > 1:
                for j in parent.nodes:
                    if j.text == branch[0]:
                        parent = j
                        break
                else:
                    widget = TreeViewLabel(text = branch[0])
                    self.children[0].children[0].children[0].children[0].children[1].children[0].add_node(widget, parent)
                    parent = widget
                branch = branch[1:]
            nodeUI = NodeLib.getNodeCls(i.__name__)
            if nodeUI == None:
                self.children[0].children[0].children[0].children[0].children[1].children[0].add_node(TreeViewButton(text = branch[0],
                                                                                                                     nodeBase = i,
                                                                                                                     always_release = True,
                                                                                                                     editor = self.children[0].children[0].children[0].children[1]),parent)
            else:
                self.children[0].children[0].children[0].children[0].children[1].children[0].add_node(TreeViewButton(text = branch[0],
                                                                                                                     nodeBase = i,
                                                                                                                     node = nodeUI,
                                                                                                                     always_release = True,
                                                                                                                     editor = self.children[0].children[0].children[0].children[1]), parent)
        

    def processDelete(self):
        """deletes the currently selected node(s) from the nodegraph"""
        
        print("deleting nodes")
        for node in self.children[0].children[0].children[0].children[1].children[0].children:
            if isinstance(node, Node.Node) and node.selected:
                node.remove()
                

    def on_pre_dismiss(self) -> None:
        """applies all changed settings before closing the popup"""

        global middleLayer
        from UI.editor.Main import middleLayer
        self.children[0].children[0].children[0].children[1].prepareClose()
        if self.children[0].children[0].children[0].children[0].children[2].children[4].text == "None":
            effMixinVB = None
        else:
            effMixinVB = self.filepaths[self.voicebanks.index(self.children[0].children[0].children[0].children[0].children[2].children[4].text)]
        if middleLayer.trackList[self.index].mixinVB != effMixinVB:
            middleLayer.trackList[self.index].mixinVB = effMixinVB
            API.Ops.ChangeTrackSettings(self.index, "mixinVB", effMixinVB)()
        if middleLayer.trackList[self.index].pauseThreshold != int(self.children[0].children[0].children[0].children[0].children[2].children[0].text):
            middleLayer.trackList[self.index].pauseThreshold = int(self.children[0].children[0].children[0].children[0].children[2].children[0].text)
            middleLayer.recalculatePauses(self.index)
        effUnvoicedShift = min(max(float(self.children[0].children[0].children[0].children[0].children[2].children[2].text), 0), 1)
        if middleLayer.trackList[self.index].unvoicedShift != effUnvoicedShift:
            middleLayer.trackList[self.index].unvoicedShift = effUnvoicedShift
            API.Ops.ChangeTrackSettings(self.index, "unvoicedShift", effUnvoicedShift)()
        if middleLayer.trackList[self.index].vbPath != self.filepaths[self.voicebanks.index(self.children[0].children[0].children[0].children[0].children[2].children[6].text)]:
            middleLayer.trackList[self.index].vbPath = self.filepaths[self.voicebanks.index(self.children[0].children[0].children[0].children[0].children[2].children[6].text)]
            API.Ops.ChangeVoicebank(self.index, middleLayer.trackList[self.index].vbPath)()
        middleLayer.submitNodegraphUpdate(middleLayer.trackList[self.index].nodegraph)
        middleLayer.ui.updateParamPanel()
        middleLayer.activePanel = None

class LicensePanel(Popup):
    """panel displaying the Nova-Vox license, contributors and other info"""
    
    pass
