#Copyright 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import tkinter
from ttkthemes import ThemedTk
import sys
import os

import torch

from Backend.VB_Components.Voicebank import Voicebank
from MiddleLayer.IniParser import readSettings
from UI.devkit.Widgets import Frame, Label, Button, Checkbutton
from Localization.devkit_localization import getLanguage
loc = getLanguage()

class NewVBUi(Frame):
    def __init__(self, reference, master=None) -> None:
        logging.info("initializing new VB UI")
        Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["new_vb"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("assets/icon/nova-vox-logo-black.ico")
        else:
            logo = tkinter.PhotoImage(file="assets/icon/nova-vox-logo-black.gif")
            self.master.call('wm', 'iconphoto', self.master._w, logo)
        self.reference = reference
        accelerator = readSettings()["accelerator"]
        if accelerator == "CPU":
            self.device = torch.device("cpu")
        elif accelerator == "GPU":
            self.device = torch.device("cuda")
        else:
            print("could not read accelerator setting. Accelerator has been set to CPU by default.")
            self.device = torch.device("cpu")
    
    def createWidgets(self):
        dictionaries, trAis, mainAis = self.fetchPresets()
        
        self.dictionary = Frame(self)
        self.dictionaryVariable = tkinter.StringVar()
        self.dictionaryLabel = Label(self.dictionary)
        self.dictionaryLabel["text"] = loc["new_dictionary"]
        self.dictionaryLabel.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.dictionaryChoice = tkinter.ttk.OptionMenu(self.dictionary, self.dictionaryVariable, "None", *dictionaries)
        self.dictionaryChoice.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.dictionary.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.trAi = Frame(self)
        self.trAiVariable = tkinter.StringVar()
        self.trAiLabel = Label(self.trAi)
        self.trAiLabel["text"] = loc["new_tr_ai"]
        self.trAiLabel.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.trAiChoice = tkinter.ttk.OptionMenu(self.trAi, self.trAiVariable, "None", *trAis)
        self.trAiChoice.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.trAi.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.mainAi = Frame(self)
        self.mainAiVariable = tkinter.StringVar()
        self.mainAiLabel = Label(self.mainAi)
        self.mainAiLabel["text"] = loc["new_main_ai"]
        self.mainAiLabel.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.mainAiChoice = tkinter.ttk.OptionMenu(self.mainAi, self.mainAiVariable, "None", *mainAis)
        self.mainAiChoice.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.mainAi.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.phonemePH = Frame(self)
        self.phonemePHVariable = tkinter.BooleanVar()
        self.phonemePHSwitch = Checkbutton(self.phonemePH)
        self.phonemePHSwitch["variable"] = self.phonemePHVariable
        self.phonemePHSwitch.pack(side = "right", fill = "x")
        self.phonemePHLabel = Label(self.phonemePH)
        self.phonemePHLabel["text"] = loc["new_phoneme_placeholder"]
        self.phonemePHLabel.pack(side = "right", fill = "x")
        
        self.okButton = Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.create
        self.okButton.pack(side = "right", expand = True)
        
        self.cancelButton = Button(self)
        self.cancelButton["text"] = loc["cancel"]
        self.cancelButton["command"] = self.close
        self.cancelButton.pack(side = "right", expand = True)

    @staticmethod
    def fetchPresets():
        dataDir = readSettings()["datadir"]
        dictionaries = []
        trAis = []
        mainAis = []
        for i in os.listdir(os.path.join(dataDir, "Devkit_Presets", "Dictionaries")):
            dictionaries.append(i.split(".")[0])
        for i in os.listdir(os.path.join(dataDir, "Devkit_Presets", "TrAis")):
            trAis.append(i.split(".")[0])
        for i in os.listdir(os.path.join(dataDir, "Devkit_Presets", "MainAis")):
            mainAis.append(i.split(".")[0])
        return dictionaries, trAis, mainAis
    
    def close(self):
        logging.info("closing new VB UI")
        self.master.destroy()

    def create(self):
        global loadedVB
        from UI.devkit.Main import loadedVB
        logging.info("creating new VB")
        loadedVB = Voicebank(None, self.device)
        if self.dictionaryVariable.get() != "None":
            loadedVB.loadWordDict(os.path.join(readSettings()["dataDir"], "Devkit_Presets", "Dictionaries", self.dictionaryVariable.get() + ".hdf5"), False)
        if self.trAiVariable.get() != "None":
            loadedVB.loadTrWeights(os.path.join(readSettings()["dataDir"], "Devkit_Presets", "TrAis", self.trAiVariable.get() + ".hdf5"))
        if self.mainAiVariable.get() != "None":
            loadedVB.loadMainWeights(os.path.join(readSettings()["dataDir"], "Devkit_Presets", "MainAis", self.mainAiVariable.get() + ".hdf5"))
        if self.phonemePHVariable.get() and self.dictionaryVariable.get() != "None":
            with open(os.path.join(readSettings()["dataDir"], "Devkit_Phonetics", "Lists", self.dictionaryVariable.get()), "r") as f:
                for line in f:
                    items = line.split(" ")
                    if items[0] in loadedVB.phonemeDict.keys():
                        continue
                    loadedVB.addPhoneme(items[0], None)
                    loadedVB.phonemeDict[items[0]][0].isVoiced = (items[1] in ("V", "T"))
                    loadedVB.phonemeDict[items[0]][0].isTransition = (items[1] in ("P", "T"))
        self.reference.onVBLoaded(loc["unsaved_vb"])
        self.master.destroy()
