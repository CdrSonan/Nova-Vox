from kivy.uix.popup import Popup
from kivy.uix.label import Label

from copy import copy

import os
import torch

from MiddleLayer.IniParser import readSettings

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
