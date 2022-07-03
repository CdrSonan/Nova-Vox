import logging
import tkinter
import torch
import sys

from Backend.VB_Components.Voicebank import Voicebank
import global_consts
from MiddleLayer.IniParser import readSettings
from Locale.devkit_locale import getLocale
loc = getLocale()

from UI.code.devkit.Metadata import MetadataUi
from UI.code.devkit.PhonemeDict import PhonemedictUi
from UI.code.devkit.CrfAi import CrfaiUi
from UI.code.devkit.UtauImport import UtauImportUi

global loadedVB
loadedVB = None

class RootUi(tkinter.Frame):
    """Class of the Devkit main window"""

    def __init__(self, master=tkinter.Tk()) -> None:
        """Initialize a new main window. Called once during devkit startup"""

        logging.info("initializing Root UI")
        tkinter.Frame.__init__(self, master)
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
        elif accelerator == "Hybrid":
            self.device = torch.device("cuda")
        elif accelerator == "GPU":
            self.device = torch.device("cuda")
        else:
            print("could not read accelerator setting. Accelerator has been set to CPU by default.")
            self.device = torch.device("cpu")
        
    def createWidgets(self) -> None:
        """Initialize all widgets of the main window. Called once during main window initialization."""

        self.infoDisplay = tkinter.Label(self)
        self.infoDisplay["text"] = loc["version_label"] + global_consts.version
        self.infoDisplay.pack(side = "top", fill = "x", padx = 20, pady = 20)
        
        self.metadataButton = tkinter.Button(self)
        self.metadataButton["text"] = loc["metadat_btn"]
        self.metadataButton["command"] = self.onMetadataPress
        self.metadataButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.metadataButton["state"] = "disabled"
        
        self.phonemedictButton = tkinter.Button(self)
        self.phonemedictButton["text"] = loc["phon_btn"]
        self.phonemedictButton["command"] = self.onPhonemedictPress
        self.phonemedictButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.phonemedictButton["state"] = "disabled"
        
        self.crfaiButton = tkinter.Button(self)
        self.crfaiButton["text"] = loc["crfai_btn"]
        self.crfaiButton["command"] = self.onCrfaiPress
        self.crfaiButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.crfaiButton["state"] = "disabled"
        
        self.parameterButton = tkinter.Button(self)
        self.parameterButton["text"] = loc["param_btn"]
        self.parameterButton["command"] = self.onParameterPress
        self.parameterButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.parameterButton["state"] = "disabled"
        
        self.worddictButton = tkinter.Button(self)
        self.worddictButton["text"] = loc["dict_btn"]
        self.worddictButton["command"] = self.onWorddictPress
        self.worddictButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.worddictButton["state"] = "disabled"

        self.utauimportButton = tkinter.Button(self)
        self.utauimportButton["text"] = loc["utau_btn"]
        self.utauimportButton["command"] = self.onUtauimportPress
        self.utauimportButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.utauimportButton["state"] = "disabled"
        
        self.saveButton = tkinter.Button(self)
        self.saveButton["text"] = loc["save_as"]
        self.saveButton["command"] = self.onSavePress
        self.saveButton.pack(side = "right", expand = True)
        self.saveButton["state"] = "disabled"
        
        self.openButton = tkinter.Button(self)
        self.openButton["text"] = loc["open"]
        self.openButton["command"] = self.onOpenPress
        self.openButton.pack(side = "right", expand = True)
        
        self.newButton = tkinter.Button(self)
        self.newButton["text"] = loc["new"]
        self.newButton["command"] = self.onNewPress
        self.newButton.pack(side = "right", expand = True)

        self.bind("<Destroy>", self.onDestroy)
        
    def onMetadataPress(self) -> None:
        """opens Metadata UI window when Metadata button in the main window is pressed"""

        logging.info("Metadata button callback")
        self.metadataUi = MetadataUi(tkinter.Tk())
        self.metadataUi.mainloop()
    
    def onPhonemedictPress(self) -> None:
        """opens Phoneme Dict UI window when Phoneme Dict button in the main window is pressed"""

        logging.info("PhonemeDict button callback")
        self.phonemedictUi = PhonemedictUi(tkinter.Tk())
        self.phonemedictUi.mainloop()
    
    def onCrfaiPress(self) -> None:
        """opens Phoneme Crossfade AI UI window when Phoneme Crossfade AI button in the main window is pressed"""

        logging.info("Crfai button callback")
        self.crfaiUi = CrfaiUi(tkinter.Tk())
        self.crfaiUi.mainloop()
    
    def onParameterPress(self) -> None:
        logging.info("Parameter button callback")
    
    def onWorddictPress(self) -> None:
        logging.info("Worddict button callback")

    def onUtauimportPress(self) -> None:
        """opens the UTAU import tool when the UTAU import tool button in the main window is pressed"""

        logging.info("UTAU import button callback")
        self.utauImportUi = UtauImportUi(tkinter.Tk())
        self.utauImportUi.mainloop()

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
                self.parameterButton["state"] = "active"
                self.worddictButton["state"] = "active"
                self.utauimportButton["state"] = "active"
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
            self.parameterButton["state"] = "active"
            self.worddictButton["state"] = "active"
            self.utauimportButton["state"] = "active"
            self.saveButton["state"] = "active"
            self.master.wm_title(loc["unsaved_vb"])
