from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.treeview import TreeViewLabel, TreeViewNode
from kivy.properties import ObjectProperty

from copy import copy

import os
from Backend import NodeLib
import torch

from MiddleLayer.IniParser import readSettings

class TreeViewButton(Button, TreeViewNode):
    """basic class implementing a tree view node with button behavior"""
    node = ObjectProperty(None)
    editor = ObjectProperty(None)

    def on_press(self):
        self.editor.add_widget(self.node())
        return super().on_press()

class SingerSettingsPanel(Popup):
    """Popup displaying per-track settings"""

    def __init__(self, index, **kwargs) -> None:
        global middleLayer
        from UI.code.editor.Main import middleLayer
        super().__init__(**kwargs)
        self.index = index
        self.vbData = []
        self.voicebanks = []
        self.modVoicebanks = None
        self.filepaths = []
        self.listVoicebanks()
        self.pauseThreshold = middleLayer.trackList[self.index].pauseThreshold
        if middleLayer.trackList[index].mixinVB == None:
            self.mixinVB = "None"
        else:
            self.mixinVB = self.voicebanks[self.filepaths.index(middleLayer.trackList[self.index].mixinVB)]
        self.mainVB = self.voicebanks[self.filepaths.index(middleLayer.trackList[self.index].vbPath)]
        self.children[0].children[0].children[0].children[0].children[2].children[4].text = self.mainVB
        self.children[0].children[0].children[0].children[0].children[2].children[4].values = self.voicebanks
        self.children[0].children[0].children[0].children[0].children[2].children[2].text = self.mixinVB
        self.children[0].children[0].children[0].children[0].children[2].children[2].values = self.modVoicebanks
        self.children[0].children[0].children[0].children[0].children[2].children[0].text = str(self.pauseThreshold)
        self.makeNodeTree()

    def listVoicebanks(self) -> None:
        """creates a list of installed Voicebanks for switching the one used by the track"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        voicePath = os.path.join(readSettings()["dataDir"], "Voices")
        if os.path.isdir(voicePath) == False:
            popup = Popup(title = "error", content = Label(text = "no valid data directory"), size_hint = (None, None), size = (400, 400))
            popup.open()
            return
        files = os.listdir(voicePath)
        for file in files:
            if file.endswith(".nvvb"):
                data = torch.load(os.path.join(voicePath, file))
                self.vbData.append(data["metadata"])
                self.voicebanks.append(data["metadata"].name)
                self.filepaths.append(os.path.join(voicePath, file))
        self.modVoicebanks = copy(self.voicebanks)
        self.modVoicebanks.append("None")

    def makeNodeTree(self):
        def classesinmodule(module):
            md = module.__dict__
            return [
                md[c] for c in md if (
                    isinstance(md[c], type) and md[c].__module__ == module.__name__
                )
            ]
        NodeClasses = classesinmodule(NodeLib)
        for i in NodeClasses:
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
            self.children[0].children[0].children[0].children[0].children[1].children[0].add_node(TreeViewButton(text = branch[0], node = i, always_release = True, editor = self.children[0].children[0].children[0].children[1].children[0]), parent)

    def on_pre_dismiss(self) -> None:
        """applies all changed settings before closing the popup"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if self.children[0].children[0].children[0].children[0].children[2].children[2].text == "None":
            middleLayer.trackList[self.index].mixinVB = None
        else:
            middleLayer.trackList[self.index].mixinVB = self.filepaths[self.voicebanks.index(self.children[0].children[0].children[0].children[0].children[2].children[2].text)]
        if middleLayer.trackList[self.index].pauseThreshold != int(self.children[0].children[0].children[0].children[0].children[2].children[0].text):
            middleLayer.trackList[self.index].pauseThreshold = int(self.children[0].children[0].children[0].children[0].children[2].children[0].text)
            middleLayer.recalculatePauses(self.index)
        if middleLayer.trackList[self.index].vbPath != self.filepaths[self.voicebanks.index(self.children[0].children[0].children[0].children[0].children[2].children[4].text)]:
            middleLayer.trackList[self.index].vbPath = self.filepaths[self.voicebanks.index(self.children[0].children[0].children[0].children[0].children[2].children[4].text)]
            middleLayer.submitChangeVB(self.index, middleLayer.trackList[self.index].vbPath)
            for i in range(len(middleLayer.ids["singerList"])):
                if middleLayer.ids["singerList"][i].index == self.index:
                    middleLayer.ids["singerList"][i].name = self.children[0].children[0].children[0].children[0].children[2].children[4].text
                    middleLayer.ids["singerList"][i].image = self.vbData[self.voicebanks.index(self.children[0].children[0].children[0].children[0].children[2].children[4].text)].image
                    break
        
class LicensePanel(Popup):
    """panel displaying the Nova-Vox license, contributors and other info"""
    
    pass
