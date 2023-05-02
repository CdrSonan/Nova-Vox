#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import tkinter
from ttkthemes import ThemedTk
import torch
import sys

from Backend.VB_Components.Voicebank import Voicebank
import global_consts
from MiddleLayer.IniParser import readSettings
from Localization.devkit_localization import getLanguage
loc = getLanguage()

from UI.code.devkit.Metadata import MetadataUi
from UI.code.devkit.PhonemeDict import PhonemedictUi
from UI.code.devkit.CrfAi import CrfaiUi
from UI.code.devkit.PredAi import PredaiUi
from UI.code.devkit.UtauImport import UtauImportUi
from UI.code.devkit.AdvSettings import AdvSettingsUi
from UI.code.devkit.Widgets import Frame, Label, Button

global loadedVB
loadedVB = None

class RootUi(Frame):
    """Class of the Devkit main window"""

    def __init__(self, master=ThemedTk(theme = "black")) -> None:
        """Initialize a new main window. Called once during devkit startup"""

        logging.info("initializing Root UI")
        Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["no_vb"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        else:
            logo = tkinter.PhotoImage(file="icon/nova-vox-logo-black.gif")
            self.master.call('wm', 'iconphoto', self.master._w, logo)

        accelerator = readSettings()["accelerator"]
        if accelerator == "CPU":
            self.device = torch.device("cpu")
        elif accelerator == "hybrid":
            self.device = torch.device("cuda")
        elif accelerator == "GPU":
            self.device = torch.device("cuda")
        else:
            print("could not read accelerator setting. Accelerator has been set to CPU by default.")
            self.device = torch.device("cpu")
        
    def createWidgets(self) -> None:
        """Initialize all widgets of the main window. Called once during main window initialization."""

        self.infoDisplay = Label(self)
        self.infoDisplay["text"] = loc["version_label"] + global_consts.version
        self.infoDisplay.pack(side = "top", fill = "x", padx = 20, pady = 20)
        
        self.metadataButton = Button(self)
        self.metadataButton["text"] = loc["metadat_btn"]
        self.metadataButton["command"] = self.onMetadataPress
        self.metadataButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.metadataButton["state"] = "disabled"
        
        self.phonemedictButton = Button(self)
        self.phonemedictButton["text"] = loc["phon_btn"]
        self.phonemedictButton["command"] = self.onPhonemedictPress
        self.phonemedictButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.phonemedictButton["state"] = "disabled"
        
        self.crfaiButton = Button(self)
        self.crfaiButton["text"] = loc["crfai_btn"]
        self.crfaiButton["command"] = self.onCrfaiPress
        self.crfaiButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.crfaiButton["state"] = "disabled"
        
        self.predaiButton = Button(self)
        self.predaiButton["text"] = loc["predai_btn"]
        self.predaiButton["command"] = self.onPredaiPress
        self.predaiButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.predaiButton["state"] = "disabled"
        
        self.worddictButton = Button(self)
        self.worddictButton["text"] = loc["dict_btn"]
        self.worddictButton["command"] = self.onWorddictPress
        self.worddictButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.worddictButton["state"] = "disabled"

        self.utauimportButton = Button(self)
        self.utauimportButton["text"] = loc["utau_btn"]
        self.utauimportButton["command"] = self.onUtauimportPress
        self.utauimportButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.utauimportButton["state"] = "disabled"

        self.advSettingsButton = Button(self)
        self.advSettingsButton["text"] = loc["advsettings"]
        self.advSettingsButton["command"] = self.onAdvSettingsPress
        self.advSettingsButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.advSettingsButton["state"] = "disabled"
        
        self.saveButton = Button(self)
        self.saveButton["text"] = loc["save_as"]
        self.saveButton["command"] = self.onSavePress
        self.saveButton.pack(side = "right", expand = True)
        self.saveButton["state"] = "disabled"
        
        self.openButton = Button(self)
        self.openButton["text"] = loc["open"]
        self.openButton["command"] = self.onOpenPress
        self.openButton.pack(side = "right", expand = True)
        
        self.newButton = Button(self)
        self.newButton["text"] = loc["new"]
        self.newButton["command"] = self.onNewPress
        self.newButton.pack(side = "right", expand = True)

        self.bind("<Destroy>", self.onDestroy)
        
    def onMetadataPress(self) -> None:
        """opens Metadata UI window when Metadata button in the main window is pressed"""

        logging.info("Metadata button callback")
        self.metadataUi = MetadataUi(tkinter.Toplevel())
        self.metadataUi.mainloop()
    
    def onPhonemedictPress(self) -> None:
        """opens Phoneme Dict UI window when Phoneme Dict button in the main window is pressed"""

        logging.info("PhonemeDict button callback")
        self.phonemedictUi = PhonemedictUi(tkinter.Toplevel())
        self.phonemedictUi.mainloop()
    
    def onCrfaiPress(self) -> None:
        """opens Phoneme Crossfade AI UI window when Phoneme Crossfade AI button in the main window is pressed"""

        logging.info("Crfai button callback")
        self.crfaiUi = CrfaiUi(tkinter.Toplevel())
        self.crfaiUi.mainloop()
    
    def onPredaiPress(self) -> None:
        """opens Spectral Prediction AI UI window when Spectral Prediction AI button in the main window is pressed"""
        
        logging.info("Predai button callback")
        self.crfaiUi = PredaiUi(tkinter.Toplevel())
        self.crfaiUi.mainloop()
    
    def onWorddictPress(self) -> None:
        logging.info("Worddict button callback")

    def onUtauimportPress(self) -> None:
        """opens the UTAU import tool when the UTAU import tool button in the main window is pressed"""

        logging.info("UTAU import button callback")
        self.utauImportUi = UtauImportUi(tkinter.Toplevel())
        self.utauImportUi.mainloop()

    def onAdvSettingsPress(self) -> None:
        """opens the UTAU import tool when the UTAU import tool button in the main window is pressed"""

        logging.info("UTAU import button callback")
        self.advSettingsUi = AdvSettingsUi(tkinter.Toplevel())
        self.advSettingsUi.mainloop()

    def onDestroy(self, event) -> None:
        logging.info("Root UI destroyed")
        if hasattr(self, 'metadataUi'):
            self.metadataUi.master.destroy()
        if hasattr(self, 'phonemedictUi'):
            self.phonemedictUi.master.destroy()
        if hasattr(self, 'crfaiUi'):
            self.crfaiUi.master.destroy()
    
    def onSavePress(self) -> None:
        """Saves the currently loaded Voicebank to a .nvvb file"""

        logging.info("save button callback")
        global loadedVB
        global loadedVBPath
        filepath = tkinter.filedialog.asksaveasfilename(defaultextension = ".nvvb", filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
        if filepath != "":
            loadedVBPath = filepath
            loadedVB.save(filepath)
            self.master.wm_title(loadedVBPath)
        
    def onOpenPress(self) -> None:
        """opens a Voicebank and loads all of its data"""

        logging.info("open button callback")
        global loadedVB
        global loadedVBPath
        if "loadedVB" not in globals():
            loadedVBPath = None
        if ("loadedVB" not in globals()) or tkinter.messagebox.askokcancel(loc["warning"], loc["vb_discard_msg"], icon = "warning"):
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
            if filepath != "":
                loadedVBPath = filepath
                loadedVB = Voicebank(filepath, self.device)
                self.metadataButton["state"] = "active"
                self.phonemedictButton["state"] = "active"
                self.crfaiButton["state"] = "active"
                self.predaiButton["state"] = "active"
                self.worddictButton["state"] = "active"
                self.utauimportButton["state"] = "active"
                self.advSettingsButton["state"] = "active"
                self.saveButton["state"] = "active"
                self.master.wm_title(loadedVBPath)
    
    def onNewPress(self) -> None:
        """creates a new, empty Voicebank object in memory"""
        
        logging.info("new button callback")
        global loadedVB
        if tkinter.messagebox.askokcancel(loc["warning"], loc["vb_discard_msg"], icon = "warning"):
            loadedVB = Voicebank(None, self.device)
            self.metadataButton["state"] = "active"
            self.phonemedictButton["state"] = "active"
            self.crfaiButton["state"] = "active"
            self.predaiButton["state"] = "active"
            self.worddictButton["state"] = "active"
            self.utauimportButton["state"] = "active"
            self.advSettingsButton["state"] = "active"
            self.saveButton["state"] = "active"
            self.master.wm_title(loc["unsaved_vb"])
