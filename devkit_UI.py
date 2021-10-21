# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:51:29 2021

@author: CdrSonan
"""

import tkinter
import tkinter.filedialog
import tkinter.simpledialog
import tkinter.messagebox
from tkinter.ttk import Progressbar
from os import path
import logging
import torch
import csv
import sys
import global_consts
import Backend.VB_Components.Voicebank
Voicebank = Backend.VB_Components.Voicebank.Voicebank
import Backend.ESPER.PitchCalculator
calculatePitch = Backend.ESPER.PitchCalculator.calculatePitch
import Backend.ESPER.SpectralCalculator
calculateSpectra = Backend.ESPER.SpectralCalculator.calculateSpectra
import Locale.devkit_locale
loc = Locale.devkit_locale.getLocale()
from Backend.DataHandler.UtauSample import UtauSample
from Backend.UtauImport import fetchSamples

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class RootUi(tkinter.Frame):
    """Class of the Devkit main window"""
    def __init__(self, master=tkinter.Tk()):
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

        settings = {}
        with open("settings.ini", 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(" ")
                settings[line[0]] = line[1]
        if settings["accelerator"] == "CPU":
            self.device = torch.device("cpu")
        elif settings["accelerator"] == "Hybrid":
            self.device = torch.device("cuda")
        elif settings["accelerator"] == "GPU":
            self.device = torch.device("cuda")
        else:
            print("could not read accelerator setting. Accelerator has been set to CPU by default.")
            self.device = torch.device("cpu")
        
    def createWidgets(self):
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
        
    def onMetadataPress(self):
        """opens Metadata UI window when Metadata button in the main window is pressed"""
        logging.info("Metadata button callback")
        self.metadataUi = MetadataUi(tkinter.Tk())
        self.metadataUi.mainloop()
    
    def onPhonemedictPress(self):
        """opens Phoneme Dict UI window when Phoneme Dict button in the main window is pressed"""
        logging.info("PhonemeDict button callback")
        self.phonemedictUi = PhonemedictUi(tkinter.Tk())
        self.phonemedictUi.mainloop()
    
    def onCrfaiPress(self):
        """opens Phoneme Crossfade AI UI window when Phoneme Crossfade AI button in the main window is pressed"""
        logging.info("Crfai button callback")
        self.crfaiUi = CrfaiUi(tkinter.Tk())
        self.crfaiUi.mainloop()
    
    def onParameterPress(self):
        logging.info("Parameter button callback")
    
    def onWorddictPress(self):
        logging.info("Worddict button callback")

    def onUtauimportPress(self):
        """opens the UTAU import tool when the UTAU import tool button in the main window is pressed"""
        logging.info("UTAU import button callback")
        self.utauImportUi = UtauImportUi(tkinter.Tk())
        self.utauImportUi.mainloop()

    def onDestroy(self, event):
        logging.info("Root UI destroyed")
        if hasattr(self, 'metadataUi'):
            self.metadataUi.master.destroy()
        if hasattr(self, 'phonemedictUi'):
            self.phonemedictUi.master.destroy()
        if hasattr(self, 'crfaiUi'):
            self.crfaiUi.master.destroy()
    
    def onSavePress(self):
        """Saves the currently loaded Voicebank to a .nvvb file"""
        logging.info("save button callback")
        global loadedVB
        global loadedVBPath
        filepath = tkinter.filedialog.asksaveasfilename(defaultextension = ".nvvb", filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
        if filepath != "":
            loadedVBPath = filepath
            loadedVB.save(filepath)
            self.master.wm_title(loadedVBPath)
        
    def onOpenPress(self):
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
    
    def onNewPress(self):
        """creates a new, empty Voicebank object in memory"""
        logging.info("new button callback")
        global loadedVB
        if ("loadedVB" not in globals()) or tkinter.messagebox.askokcancel(loc["warning"], loc["vb_discard_msg"], icon = "warning"):
            loadedVB = Voicebank(None, self.device)
            self.metadataButton["state"] = "active"
            self.phonemedictButton["state"] = "active"
            self.crfaiButton["state"] = "active"
            self.parameterButton["state"] = "active"
            self.worddictButton["state"] = "active"
            self.utauimportButton["state"] = "active"
            self.saveButton["state"] = "active"
            self.master.wm_title(loc["unsaved_vb"])
            
class MetadataUi(tkinter.Frame):
    """Class of the Metadata window"""
    def __init__(self, master=None):
        logging.info("Initializing Metadata UI")
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20)
        self.createWidgets()
        self.master.wm_title(loc["metadat_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
    def createWidgets(self):
        """initializes all widgets of the Metadata window. Called once during initialization"""
        global loadedVB
        self.name = tkinter.Frame(self)
        self.name.variable = tkinter.StringVar(self.name)
        self.name.variable.set(loadedVB.metadata.name)
        self.name.entry = tkinter.Entry(self.name)
        self.name.entry["textvariable"] = self.name.variable
        self.name.entry.pack(side = "right", fill = "x", expand = True)
        self.name.display = tkinter.Label(self.name)
        self.name.display["text"] = loc["name"]
        self.name.display.pack(side = "right", fill = "x")
        self.name.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = tkinter.Button(self)
        self.loadButton["text"] = loc["load_other_VB"]
        self.loadButton["command"] = self.onLoadPress
        self.loadButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)
        
    def onOkPress(self):
        logging.info("Metadata OK button callback")
        """Applies all changes and closes the window when the OK button is pressed"""
        global loadedVB
        loadedVB.metadata.name = self.name.variable.get()
        loadedVB.metadata.sampleRate = global_consts.sampleRate
        self.master.destroy()

    def onLoadPress(self):
        logging.info("Metadata load button callback")
        """Opens a file browser, and loads the Voicebank metadata from a specified .nvvb file"""
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
        loadedVB.loadMetadata(filepath)
        self.name.variable.set(loadedVB.metadata.name)
        
class PhonemedictUi(tkinter.Frame):
    """Class of the phoneme dictionnary UI window"""
    def __init__(self, master=None):
        logging.info("Initializing Phonemedict UI")
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["phon_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
    def createWidgets(self):
        """Initializes all widgets of the Phoneme Dict UI window."""
        global loadedVB

        self.diagram = tkinter.LabelFrame(self, text = loc["diag_lbl"])
        self.diagram.fig = Figure(figsize=(4, 4))
        self.diagram.ax = self.diagram.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.diagram.ax.set_xlim([0, global_consts.sampleRate / 2])
        self.diagram.ax.set_xlabel(loc["freq_lbl"], fontsize = 8)
        self.diagram.ax.set_ylabel(loc["amp_lbl"], fontsize = 8)
        self.diagram.canvas = FigureCanvasTkAgg(self.diagram.fig, self.diagram)
        self.diagram.canvas.get_tk_widget().pack(side = "top", fill = "both", expand = True)
        self.diagram.timeSlider = tkinter.Scale(self.diagram, from_ = 0, to = 0, orient = "horizontal", length = 600, command = self.onSliderMove)
        self.diagram.timeSlider.pack(side = "left", fill = "both", expand = True, padx = 5, pady = 2)
        self.diagram.pack(side = "right", fill = "y", padx = 5, pady = 2)
        
        self.phonemeList = tkinter.LabelFrame(self, text = loc["phon_list"])
        self.phonemeList.list = tkinter.Frame(self.phonemeList)
        self.phonemeList.list.lb = tkinter.Listbox(self.phonemeList.list)
        self.phonemeList.list.lb.pack(side = "left",fill = "both", expand = True)
        self.phonemeList.list.sb = tkinter.Scrollbar(self.phonemeList.list)
        self.phonemeList.list.sb.pack(side = "left", fill = "y")
        self.phonemeList.list.lb["selectmode"] = "single"
        self.phonemeList.list.lb["yscrollcommand"] = self.phonemeList.list.sb.set
        self.phonemeList.list.lb.bind("<<ListboxSelect>>", self.onSelectionChange)
        self.phonemeList.list.lb.bind("<FocusOut>", self.onListFocusOut)
        self.phonemeList.list.sb["command"] = self.phonemeList.list.lb.yview
        self.phonemeList.list.pack(side = "top", fill = "x", expand = True, padx = 5, pady = 2)
        for i in loadedVB.phonemeDict.keys():
            self.phonemeList.list.lb.insert("end", i)
        self.phonemeList.removeButton = tkinter.Button(self.phonemeList)
        self.phonemeList.removeButton["text"] = loc["remove"]
        self.phonemeList.removeButton["command"] = self.onRemovePress
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = tkinter.Button(self.phonemeList)
        self.phonemeList.addButton["text"] = loc["add"]
        self.phonemeList.addButton["command"] = self.onAddPress
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar = tkinter.LabelFrame(self, text = loc["per_ph_set"])
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        self.sideBar.key = tkinter.Frame(self.sideBar)
        self.sideBar.key.variable = tkinter.StringVar(self.sideBar.key)
        self.sideBar.key.entry = tkinter.Entry(self.sideBar.key)
        self.sideBar.key.entry["textvariable"] = self.sideBar.key.variable
        self.sideBar.key.entry.bind("<FocusOut>", self.onKeyChange)
        self.sideBar.key.entry.bind("<KeyRelease-Return>", self.onKeyChange)
        self.sideBar.key.entry.pack(side = "right", fill = "x")
        self.sideBar.key.display = tkinter.Label(self.sideBar.key)
        self.sideBar.key.display["text"] = loc["phon_key"]
        self.sideBar.key.display.pack(side = "right", fill = "x")
        self.sideBar.key.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.expPitch = tkinter.Frame(self.sideBar)
        self.sideBar.expPitch.variable = tkinter.DoubleVar(self.sideBar.expPitch)
        self.sideBar.expPitch.entry = tkinter.Entry(self.sideBar.expPitch)
        self.sideBar.expPitch.entry["textvariable"] = self.sideBar.expPitch.variable
        self.sideBar.expPitch.entry.bind("<FocusOut>", self.onPitchUpdateTrigger)
        self.sideBar.expPitch.entry.bind("<KeyRelease-Return>", self.onPitchUpdateTrigger)
        self.sideBar.expPitch.entry.pack(side = "right", fill = "x")
        self.sideBar.expPitch.display = tkinter.Label(self.sideBar.expPitch)
        self.sideBar.expPitch.display["text"] = loc["est_pit"]
        self.sideBar.expPitch.display.pack(side = "right", fill = "x")
        self.sideBar.expPitch.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.pSearchRange = tkinter.Frame(self.sideBar)
        self.sideBar.pSearchRange.variable = tkinter.DoubleVar(self.sideBar.pSearchRange)
        self.sideBar.pSearchRange.entry = tkinter.Spinbox(self.sideBar.pSearchRange, from_ = 0.05, to = 0.5, increment = 0.05)
        self.sideBar.pSearchRange.entry["textvariable"] = self.sideBar.pSearchRange.variable
        self.sideBar.pSearchRange.entry.bind("<FocusOut>", self.onPitchUpdateTrigger)
        self.sideBar.pSearchRange.entry.bind("<KeyRelease-Return>", self.onPitchUpdateTrigger)
        self.sideBar.pSearchRange.entry.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.display = tkinter.Label(self.sideBar.pSearchRange)
        self.sideBar.pSearchRange.display["text"] = loc["psearchr"]
        self.sideBar.pSearchRange.display.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.voicedFilter = tkinter.Frame(self.sideBar)
        self.sideBar.voicedFilter.variable = tkinter.DoubleVar(self.sideBar.voicedFilter)
        self.sideBar.voicedFilter.entry = tkinter.Spinbox(self.sideBar.voicedFilter, from_ = 0, to = 5, increment = 0.05)
        self.sideBar.voicedFilter.entry["textvariable"] = self.sideBar.voicedFilter.variable
        self.sideBar.voicedFilter.entry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.voicedFilter.entry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.voicedFilter.entry.pack(side = "right", fill = "x")
        self.sideBar.voicedFilter.display = tkinter.Label(self.sideBar.voicedFilter)
        self.sideBar.voicedFilter.display["text"] = loc["vfilter"]
        self.sideBar.voicedFilter.display.pack(side = "right", fill = "x")
        self.sideBar.voicedFilter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.unvoicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.unvoicedIter.variable = tkinter.IntVar(self.sideBar.unvoicedIter)
        self.sideBar.unvoicedIter.entry = tkinter.Spinbox(self.sideBar.unvoicedIter, from_ = 1, to = 100)
        self.sideBar.unvoicedIter.entry["textvariable"] = self.sideBar.unvoicedIter.variable
        self.sideBar.unvoicedIter.entry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.unvoicedIter.entry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.unvoicedIter.entry.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIter.display = tkinter.Label(self.sideBar.unvoicedIter)
        self.sideBar.unvoicedIter.display["text"] = loc["uviter"]
        self.sideBar.unvoicedIter.display.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.fileButton = tkinter.Button(self.sideBar)
        self.sideBar.fileButton["text"] = loc["cng_file"]
        self.sideBar.fileButton["command"] = self.onFilechangePress
        self.sideBar.fileButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.finalizeButton = tkinter.Button(self.sideBar)
        self.sideBar.finalizeButton["text"] = loc["finalize"]
        self.sideBar.finalizeButton["command"] = self.onFinalizePress
        self.sideBar.finalizeButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = tkinter.Button(self)
        self.loadButton["text"] = loc["load_other_VB"]
        self.loadButton["command"] = self.onLoadPress
        self.loadButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.phonemeList.list.lastFocusedIndex = None
        
    def onSelectionChange(self, event):
        """Adjusts the per-phoneme part of the UI to display the correct values when the selected Phoneme in the Phoneme list changes"""
        logging.info("Phonemedict selection change callback")
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            index = self.phonemeList.list.lastFocusedIndex
            key = self.phonemeList.list.lb.get(index)
            self.sideBar.key.variable.set(key)
            if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
                self.sideBar.expPitch.variable.set(loadedVB.phonemeDict[key].expectedPitch)
                self.sideBar.pSearchRange.variable.set(loadedVB.phonemeDict[key].searchRange)
                self.sideBar.voicedFilter.variable.set(loadedVB.phonemeDict[key].voicedFilter)
                self.sideBar.unvoicedIter.variable.set(loadedVB.phonemeDict[key].unvoicedIterations)
                self.enableButtons()
            else:
                self.sideBar.expPitch.variable.set(None)
                self.sideBar.pSearchRange.variable.set(None)
                self.sideBar.voicedFilter.variable.set(None)
                self.sideBar.unvoicedIter.variable.set(None)
                self.disableButtons()
            self.updateSlider()
            self.onSliderMove(0)
                
    def onListFocusOut(self, event):
        """Helper function for retaining information about the last focused element of the Phoneme list when Phoneme list loses entry focus"""
        logging.info("Phonemedict list focus loss callback")
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
        
    def disableButtons(self):
        """Disables the per-phoneme settings buttons"""
        self.sideBar.expPitch.entry["state"] = "disabled"
        self.sideBar.pSearchRange.entry["state"] = "disabled"
        self.sideBar.voicedFilter.entry["state"] = "disabled"
        self.sideBar.unvoicedIter.entry["state"] = "disabled"
        self.sideBar.fileButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
    
    def enableButtons(self):
        """Enables the per-phoneme settings buttons"""
        self.sideBar.expPitch.entry["state"] = "normal"
        self.sideBar.pSearchRange.entry["state"] = "normal"
        self.sideBar.voicedFilter.entry["state"] = "normal"
        self.sideBar.unvoicedIter.entry["state"] = "normal"
        self.sideBar.fileButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
    
    def onAddPress(self):
        """UI Frontend function for adding a phoneme to the Voicebank"""
        logging.info("Phonemedict add button callback")
        global loadedVB
        key = tkinter.simpledialog.askstring(loc["new_phon"], loc["phon_key_sel"])
        if (key != "") & (key != None):
            if key in loadedVB.phonemeDict.keys():
                key += "#"
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")))
            if filepath != "":
                loadedVB.addPhoneme(key, filepath)
                calculatePitch(loadedVB.phonemeDict[key])
                calculateSpectra(loadedVB.phonemeDict[key])
                self.phonemeList.list.lb.insert("end", key)
        
    def onRemovePress(self):
        """UI Frontend function for removing a phoneme from the Voicebank"""
        logging.info("Phonemedict remove button callback")
        global loadedVB
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            key = self.phonemeList.list.lb.get(index)
            loadedVB.delPhoneme(key)
            self.phonemeList.list.lb.delete(index)
            if index == self.phonemeList.list.lb.size():
                self.phonemeList.list.lb.selection_set(index - 1)
            else:
                self.phonemeList.list.lb.selection_set(index)
            if self.phonemeList.list.lb.size() == 0:
                self.disableButtons()

    def onSliderMove(self, value):
        logging.info("Phonemedict slider movement callback")
        global loadedVB
        value = int(value)
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        spectrum = loadedVB.phonemeDict[key].spectrum + loadedVB.phonemeDict[key].spectra[value]
        window = torch.hann_window(global_consts.tripleBatchSize)
        voicedExcitation = torch.stft(loadedVB.phonemeDict[key].voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
        voicedExcitation = torch.abs(voicedExcitation[:, value]) * spectrum
        excitation = torch.abs(loadedVB.phonemeDict[key].excitation[value]) * spectrum
        xScale = torch.linspace(0, global_consts.sampleRate / 2, global_consts.halfTripleBatchSize + 1)
        self.diagram.ax.plot(xScale, excitation, label = loc["excitation"])
        self.diagram.ax.plot(xScale, voicedExcitation, label = loc["vExcitation"])
        self.diagram.ax.plot(xScale, spectrum, label = loc["spectrum"])
        self.diagram.ax.set_xlim([0, global_consts.sampleRate / 2])
        self.diagram.ax.set_xlabel(loc["freq_lbl"], fontsize = 8)
        self.diagram.ax.set_ylabel(loc["amp_lbl"], fontsize = 8)
        self.diagram.ax.legend(loc = "upper right", fontsize = 8)
        self.diagram.canvas.draw()
        self.diagram.ax.clear()

    def updateSlider(self):
        logging.info("Phonemedict slider properties update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        maxValue = loadedVB.phonemeDict[key].spectra.size()[0] - 2
        self.diagram.timeSlider.destroy()
        self.diagram.timeSlider = tkinter.Scale(self.diagram, from_ = 0, to = maxValue, orient = "horizontal", length = 600, command = self.onSliderMove)
        self.diagram.timeSlider.pack(side = "left", fill = "both", expand = True, padx = 5, pady = 2)
            
    def onKeyChange(self, event):
        """UI Frontend function for changing the key of a phoneme"""
        logging.info("Phonemedict key change callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        newKey = self.sideBar.key.variable.get()
        if key != newKey:
            loadedVB.changePhonemeKey(key, newKey)
            self.phonemeList.list.lb.delete(index)
            self.phonemeList.list.lb.insert(index, newKey)
        
    def onPitchUpdateTrigger(self, event):
        """UI Frontend function for updating the pitch of a phoneme"""
        logging.info("Phonemedict pitch update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
            if (loadedVB.phonemeDict[key].expectedPitch != self.sideBar.expPitch.variable.get()) or (loadedVB.phonemeDict[key].searchRange != self.sideBar.pSearchRange.variable.get()):
                loadedVB.phonemeDict[key].expectedPitch = self.sideBar.expPitch.variable.get()
                loadedVB.phonemeDict[key].searchRange = self.sideBar.pSearchRange.variable.get()
                calculatePitch(loadedVB.phonemeDict[key])
                calculateSpectra(loadedVB.phonemeDict[key])
        
    def onSpectralUpdateTrigger(self, event):
        """UI Frontend function for updating the spectral and excitation data of a phoneme"""
        logging.info("Phonemedict spectral update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
            if (loadedVB.phonemeDict[key].voicedFilter != self.sideBar.voicedFilter.variable.get()) or (loadedVB.phonemeDict[key].unvoicedIterations != self.sideBar.unvoicedIter.variable.get()):
                loadedVB.phonemeDict[key].voicedFilter = self.sideBar.voicedFilter.variable.get()
                loadedVB.phonemeDict[key].unvoicedIterations = self.sideBar.unvoicedIter.variable.get()
                calculateSpectra(loadedVB.phonemeDict[key])
                self.onSliderMove(self.diagram.timeSlider.get())
        
    def onFilechangePress(self, event):
        """UI Frontend function for changing the file associated with a phoneme"""
        logging.info("Phonemedict file change button callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")))
        if filepath != "":
            loadedVB.changePhonemeFile(key, filepath)
            calculatePitch(loadedVB.phonemeDict[key])
            calculateSpectra(loadedVB.phonemeDict[key])
            self.phonemeList.list.lb.delete(index)
            self.phonemeList.list.lb.insert(index, key)
            
        
    def onFinalizePress(self):
        """UI Frontend function for finalizing a phoneme"""
        logging.info("Phonemedict finalize button callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)[0]
        loadedVB.finalizePhoneme(key)
        self.sideBar.expPitch.variable.set(None)
        self.sideBar.pSearchRange.variable.set(None)
        self.sideBar.voicedFilter.variable.set(None)
        self.sideBar.unvoicedIter.variable.set(None)
        self.disableButtons()
        
        
    def onOkPress(self):
        """Updates the last selected phoneme and closes the Phoneme Dict UI window when the OK button is pressed"""
        logging.info("Phonemedict OK button callback")
        global loadedVB
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            if index != None:
                key = self.phonemeList.list.lb.get(index)
                self.onKeyChange(None)
                if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
                    self.onPitchUpdateTrigger(None)
                    self.onSpectralUpdateTrigger(None)
        self.master.destroy()

    def onLoadPress(self):
        """UI Frontend function for loading the phoneme dict of a different Voicebank"""
        logging.info("Phonemedict load button callback")
        global loadedVB
        additive =  tkinter.messagebox.askyesnocancel(loc["warning"], loc["additive_msg"], icon = "question")
        if additive != None:
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
            if filepath != "":
                loadedVB.loadPhonemeDict(filepath, additive)
                for i in range(self.phonemeList.list.lb.size()):
                    self.phonemeList.list.lb.delete(0)
                for i in loadedVB.phonemeDict.keys():
                    self.phonemeList.list.lb.insert("end", i)
        
class CrfaiUi(tkinter.Frame):
    """Class of the Spectral Crossfade AI UI window"""
    def __init__(self, master=None):
        logging.info("Initializing Crfai UI")
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["crfai_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
    def createWidgets(self):
        """creates all widgets of the window. Called once during initialization"""
        global loadedVB
        
        self.phonemeList = tkinter.LabelFrame(self, text = loc["ai_samp_list"])
        self.phonemeList.list = tkinter.Frame(self.phonemeList)
        self.phonemeList.list.lb = tkinter.Listbox(self.phonemeList.list)
        self.phonemeList.list.lb.pack(side = "left",fill = "both", expand = True)
        self.phonemeList.list.sb = tkinter.Scrollbar(self.phonemeList.list)
        self.phonemeList.list.sb.pack(side = "left", fill = "y")
        self.phonemeList.list.lb["selectmode"] = "single"
        self.phonemeList.list.lb["yscrollcommand"] = self.phonemeList.list.sb.set
        self.phonemeList.list.lb.bind("<<ListboxSelect>>", self.onSelectionChange)
        self.phonemeList.list.lb.bind("<FocusOut>", self.onListFocusOut)
        self.phonemeList.list.sb["command"] = self.phonemeList.list.lb.yview
        self.phonemeList.list.pack(side = "top", fill = "x", expand = True, padx = 5, pady = 2)
        for i in loadedVB.stagedTrainSamples:
            self.phonemeList.list.lb.insert("end", i.filepath)
        self.phonemeList.removeButton = tkinter.Button(self.phonemeList)
        self.phonemeList.removeButton["text"] = loc["remove"]
        self.phonemeList.removeButton["command"] = self.onRemovePress
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = tkinter.Button(self.phonemeList)
        self.phonemeList.addButton["text"] = loc["add"]
        self.phonemeList.addButton["command"] = self.onAddPress
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar = tkinter.LabelFrame(self, text = loc["ai_settings"])
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        self.sideBar.unvoicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.unvoicedIter.variable = tkinter.IntVar(self.sideBar.unvoicedIter)
        self.sideBar.unvoicedIter.variable.set(10)
        self.sideBar.unvoicedIter.entry = tkinter.Spinbox(self.sideBar.unvoicedIter, from_ = 0, to = 100)
        self.sideBar.unvoicedIter.entry["textvariable"] = self.sideBar.unvoicedIter.variable
        self.sideBar.unvoicedIter.entry.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIter.display = tkinter.Label(self.sideBar.unvoicedIter)
        self.sideBar.unvoicedIter.display["text"] = loc["uviter"]
        self.sideBar.unvoicedIter.display.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.epochs = tkinter.Frame(self.sideBar)
        self.sideBar.epochs.variable = tkinter.IntVar(self.sideBar.epochs)
        self.sideBar.epochs.variable.set(1)
        self.sideBar.epochs.entry = tkinter.Spinbox(self.sideBar.epochs, from_ = 1, to = 100)
        self.sideBar.epochs.entry["textvariable"] = self.sideBar.epochs.variable
        self.sideBar.epochs.entry.pack(side = "right", fill = "x")
        self.sideBar.epochs.display = tkinter.Label(self.sideBar.epochs)
        self.sideBar.epochs.display["text"] = loc["epochs"]
        self.sideBar.epochs.display.pack(side = "right", fill = "x")
        self.sideBar.epochs.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.trainButton = tkinter.Button(self.sideBar)
        self.sideBar.trainButton["text"] = loc["train"]
        self.sideBar.trainButton["command"] = self.onTrainPress
        self.sideBar.trainButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.finalizeButton = tkinter.Button(self.sideBar)
        self.sideBar.finalizeButton["text"] = loc["finalize"]
        self.sideBar.finalizeButton["command"] = self.onFinalizePress
        self.sideBar.finalizeButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        if loadedVB.crfAi.epoch == None:
            epoch = loc["varying"]
        else:
            epoch = str(loadedVB.crfAi.epoch)
        self.statusVar = tkinter.StringVar(self, loc["AI_stat_1"] + epoch + loc["AI_stat_2"] + str(loadedVB.crfAi.sampleCount) + loc["AI_stat_3"])
        self.statusLabel = tkinter.Label(self, textvariable = self.statusVar)
        self.statusLabel.pack(side = "top", fill = "x", expand = True, padx = 5)

        self.okButton = tkinter.Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = tkinter.Button(self)
        self.loadButton["text"] = loc["load_other_VB"]
        self.loadButton["command"] = self.onLoadPress
        self.loadButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)
        
        if type(loadedVB.crfAi).__name__ == "SavedSpecCrfAi":
            self.disableButtons()
        
    def onSelectionChange(self, event):
        """Helper function for retaining information about the last selected transition sample when the selected item in the transition sample list changes"""
        logging.info("Crfai selection change callback")
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            
    def onListFocusOut(self, event):
        """Helper function for retaining information about the last selected transition sample when the transition sample list loses entry focus"""
        logging.info("Crfai list focus loss callback")
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
    
    def onAddPress(self):
        """UI Frontend function for adding a new transition sample to the list of staged AI training samples"""
        logging.info("Crfai add button callback")
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")), multiple = True)
        if filepath != ():
            for i in filepath:
                loadedVB.addTrainSample(i)
                self.phonemeList.list.lb.insert("end", i)
        
    def onRemovePress(self):
        """UI Frontend function for removing a transition sample from the list of staged AI training samples"""
        logging.info("Crfai remove button callback")
        global loadedVB
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            loadedVB.delTrainSample(index)
            self.phonemeList.list.lb.delete(index)
            if index == self.phonemeList.list.lb.size():
                self.phonemeList.list.lb.selection_set(index - 1)
            else:
                self.phonemeList.list.lb.selection_set(index)
                
    def onTrainPress(self):
        """UI Frontend function for training the AI with the specified settings and samples"""
        logging.info("Crfai train button callback")
        global loadedVB
        self.statusVar.set("training Ai...")
        self.update()
        loadedVB.trainCrfAi(self.sideBar.epochs.variable.get(), True, self.sideBar.unvoicedIter.variable.get())
        numIter = self.phonemeList.list.lb.size()
        for i in range(numIter):
            loadedVB.delTrainSample(0)
            self.phonemeList.list.lb.delete(0)
        self.statusVar.set("AI trained with " + str(loadedVB.crfAi.epoch) + " epochs and " + str(loadedVB.crfAi.sampleCount) + " samples")
        logging.info("Crfai train button callback completed")
        
    def onFinalizePress(self):
        """UI Frontend function for finalizing the phoneme crossfade AI"""
        logging.info("Crfai finalize button callback")
        global loadedVB
        loadedVB.finalizCrfAi()
        self.disableButtons()
        
    def disableButtons(self):
        """disables the AI settings buttons"""
        self.sideBar.unvoicedIter.entry["state"] = "disabled"
        self.sideBar.epochs.entry["state"] = "disabled"
        self.sideBar.trainButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
    
    def enableButtons(self):
        """enables the AI settings buttons"""
        self.sideBar.unvoicedIter.entry["state"] = "normal"
        self.sideBar.epochs.entry["state"] = "normal"
        self.sideBar.trainButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
        
    def onOkPress(self):
        """closes the window when the OK button is pressed"""
        logging.info("Crfai OK button callback")
        global loadedVB
        self.master.destroy()

    def onLoadPress(self):
        """UI Frontend function for loading the AI state from the Phoneme Crossfade AI of a different Voicebank"""
        logging.info("Crfai load button callback")
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
        if filepath != "":
            loadedVB.loadCrfWeights(filepath)
            self.sidebar.statusVar.set("AI trained with " + loadedVB.crfAi.epoch + " epochs and " + loadedVB.crfAi.samples + " samples")

class UtauImportUi(tkinter.Frame):
    """Class of the UTAU import UI window"""
    def __init__(self, master=None):
        logging.info("Initializing UTAU import UI")
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["utau_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
        self.sampleList = []
        
    def createWidgets(self):
        """Initializes all widgets of the UTAU import UI window."""
        global loadedVB

        self.diagram = tkinter.LabelFrame(self, text = loc["utau_diag_lbl"], width = 600)
        self.diagram.fig = Figure(figsize=(4, 4))
        self.diagram.ax = self.diagram.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.diagram.ax.set_xlim([0, global_consts.sampleRate / 2])
        self.diagram.ax.set_xlabel(loc["time_lbl"], fontsize = 8)
        self.diagram.ax.set_ylabel(loc["amp_lbl"], fontsize = 8)
        self.diagram.canvas = FigureCanvasTkAgg(self.diagram.fig, self.diagram)
        self.diagram.canvas.get_tk_widget().pack(side = "top", fill = "both", expand = True)
        self.diagram.pack(side = "right", fill = "y", padx = 5, pady = 2)
        
        self.phonemeList = tkinter.LabelFrame(self, text = loc["smp_list"])
        self.phonemeList.list = tkinter.Frame(self.phonemeList)
        self.phonemeList.list.lb = tkinter.Listbox(self.phonemeList.list)
        self.phonemeList.list.lb.pack(side = "left",fill = "both", expand = True)
        self.phonemeList.list.sb = tkinter.Scrollbar(self.phonemeList.list)
        self.phonemeList.list.sb.pack(side = "left", fill = "y")
        self.phonemeList.list.lb["selectmode"] = "single"
        self.phonemeList.list.lb["yscrollcommand"] = self.phonemeList.list.sb.set
        self.phonemeList.list.lb.bind("<<ListboxSelect>>", self.onSelectionChange)
        self.phonemeList.list.lb.bind("<FocusOut>", self.onListFocusOut)
        self.phonemeList.list.sb["command"] = self.phonemeList.list.lb.yview
        self.phonemeList.list.pack(side = "top", fill = "x", expand = True, padx = 5, pady = 2)
        self.phonemeList.removeButton = tkinter.Button(self.phonemeList)
        self.phonemeList.removeButton["text"] = loc["remove"]
        self.phonemeList.removeButton["command"] = self.onRemovePress
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = tkinter.Button(self.phonemeList)
        self.phonemeList.addButton["text"] = loc["add"]
        self.phonemeList.addButton["command"] = self.onAddPress
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar = tkinter.LabelFrame(self, text = loc["per_smp_set"])
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        self.sideBar._type = tkinter.Frame(self.sideBar)
        self.sideBar._type.variable = tkinter.BooleanVar(self.sideBar._type)
        self.sideBar._type.entry = tkinter.Frame(self.sideBar._type)
        self.sideBar._type.entry.button1 = tkinter.Radiobutton(self.sideBar._type.entry, text = loc["smp_phoneme"], value = 0, variable = self.sideBar._type.variable)
        self.sideBar._type.entry.button1.pack(side = "right", fill = "x")
        self.sideBar._type.entry.button2 = tkinter.Radiobutton(self.sideBar._type.entry, text = loc["smp_transition"], value = 1, variable = self.sideBar._type.variable)
        self.sideBar._type.entry.button2.pack(side = "right", fill = "x")
        self.sideBar._type.entry.pack(side = "right", fill = "x")
        self.sideBar._type.display = tkinter.Label(self.sideBar._type)
        self.sideBar._type.display["text"] = loc["smpl_type"]
        self.sideBar._type.display.pack(side = "right", fill = "x")
        self.sideBar._type.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.key = tkinter.Frame(self.sideBar)
        self.sideBar.key.variable = tkinter.StringVar(self.sideBar.key)
        self.sideBar.key.entry = tkinter.Entry(self.sideBar.key)
        self.sideBar.key.entry["textvariable"] = self.sideBar.key.variable
        self.sideBar.key.entry.bind("<FocusOut>", self.onKeyChange)
        self.sideBar.key.entry.bind("<KeyRelease-Return>", self.onKeyChange)
        self.sideBar.key.entry.pack(side = "right", fill = "x")
        self.sideBar.key.display = tkinter.Label(self.sideBar.key)
        self.sideBar.key.display["text"] = loc["phon_key"]
        self.sideBar.key.display.pack(side = "right", fill = "x")
        self.sideBar.key.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.start = tkinter.Frame(self.sideBar)
        self.sideBar.start.variable = tkinter.DoubleVar(self.sideBar.start)
        self.sideBar.start.entry = tkinter.Spinbox(self.sideBar.start, from_ = 0, increment = 0.05)
        self.sideBar.start.entry["textvariable"] = self.sideBar.start.variable
        self.sideBar.start.entry.bind("<FocusOut>", self.onFrameUpdateTrigger)
        self.sideBar.start.entry.bind("<KeyRelease-Return>", self.onFrameUpdateTrigger)
        self.sideBar.start.entry.pack(side = "right", fill = "x")
        self.sideBar.start.display = tkinter.Label(self.sideBar.start)
        self.sideBar.start.display["text"] = loc["start"]
        self.sideBar.start.display.pack(side = "right", fill = "x")
        self.sideBar.start.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.end = tkinter.Frame(self.sideBar)
        self.sideBar.end.variable = tkinter.DoubleVar(self.sideBar.end)
        self.sideBar.end.entry = tkinter.Spinbox(self.sideBar.end, from_ = 0, increment = 0.05)
        self.sideBar.end.entry["textvariable"] = self.sideBar.end.variable
        self.sideBar.end.entry.bind("<FocusOut>", self.onFrameUpdateTrigger)
        self.sideBar.end.entry.bind("<KeyRelease-Return>", self.onFrameUpdateTrigger)
        self.sideBar.end.entry.pack(side = "right", fill = "x")
        self.sideBar.end.display = tkinter.Label(self.sideBar.end)
        self.sideBar.end.display["text"] = loc["end"]
        self.sideBar.end.display.pack(side = "right", fill = "x")
        self.sideBar.end.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.importButton = tkinter.Button(self.sideBar)
        self.sideBar.importButton["text"] = loc["smp_import"]
        self.sideBar.importButton["command"] = self.onImportPress
        self.sideBar.importButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = tkinter.Button(self)
        self.loadButton["text"] = loc["load_oto"]
        self.loadButton["command"] = self.onLoadPress
        self.loadButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.importButton = tkinter.Button(self)
        self.importButton["text"] = loc["all_import"]
        self.importButton["command"] = self.onImportAllPress
        self.importButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.phonemeList.list.lastFocusedIndex = None
        
    def onSelectionChange(self, event):
        """Adjusts the per-sample part of the UI to display the correct values when the selected sample in the SampleList list changes"""
        logging.info("UTAU sample list selection change callback")
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            index = self.phonemeList.list.lastFocusedIndex
            sample = self.sampleList[index]
            self.sideBar._type.variable.set(sample._type)
            self.sideBar.key.variable.set(sample.key)
            if sample.key == None:
                self.disableButtons()
            else:
                self.enableButtons()
            self.sideBar.start.variable.set(sample.start)
            self.sideBar.end.variable.set(sample.end)
            self.updateDiagram()

                
    def onListFocusOut(self, event):
        """Helper function for retaining information about the last focused element of the SampleList when the SampleList loses entry focus"""
        logging.info("UTAU sample list focus loss callback")
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
        
    def disableButtons(self):
        """Disables the key button"""
        self.sideBar.key.entry["state"] = "disabled"
    
    def enableButtons(self):
        """Enables the key button"""
        self.sideBar.key.entry["state"] = "normal"
    
    def onAddPress(self):
        """UI Frontend function for adding a sample to the SampleList"""
        logging.info("UTAU sample add button callback")
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")))
        if filepath != "":
            for s in self.sampleList:
                if filepath == s.audioSample.filepath:
                    sample = UtauSample(filepath, 0, None, 0, None, s.offset, s.fixed, s.blank, s.preuttr, s.overlap)
                    break
            else:
                sample = UtauSample(filepath, 0, None, 0, None, 0, 0, 0, 0, 0)
            self.sampleList.append(sample)
            self.phonemeList.list.lb.insert("end", sample.handle)
        
    def onRemovePress(self):
        """UI Frontend function for removing a sample from the SampleList"""
        logging.info("UTAU sample remove button callback")
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            del self.sampleList[index]
            self.phonemeList.list.lb.delete(index)
            if index == self.phonemeList.list.lb.size():
                self.phonemeList.list.lb.selection_set(index - 1)
            else:
                self.phonemeList.list.lb.selection_set(index)
            self.updateDiagram()

    def updateDiagram(self):
        logging.info("UTAU diagram update callback")
        index = self.phonemeList.list.lastFocusedIndex
        sample = self.sampleList[index]
        waveform = sample.audioSample.waveform
        timesize = waveform.size()[0] * 1000 / global_consts.sampleRate
        if sample.blank >= 0:
            endpoint = timesize - sample.blank
        else:
            endpoint = sample.offset - sample.blank
        xScale = torch.linspace(0, timesize, waveform.size()[0])
        self.diagram.ax.plot(xScale, waveform, label = loc["waveform"], color = (0., 0.5, 1.), alpha = 0.75)
        self.diagram.ax.axvspan(0, sample.offset, ymin = 0.5, facecolor=(0.75, 0.75, 1.), alpha=1., label = loc["offset/blank"])
        self.diagram.ax.axvspan(endpoint, timesize, ymin = 0.5, facecolor=(0.75, 0.75, 1.), alpha=1.)
        self.diagram.ax.axvspan(sample.offset, sample.offset + sample.fixed, ymin = 0.5, facecolor=(1., 0.75, 1.), alpha=1., label = loc["fixed"])
        self.diagram.ax.axvline(sample.offset + sample.overlap, ymin = 0.5, color = (0., 1., 0.), alpha = 0.9, label = loc["overlap"])
        self.diagram.ax.axvline(sample.offset + sample.preuttr, ymin = 0.5, color = (1., 0., 0.), alpha = 0.9, label = loc["preuttr"])
        self.diagram.ax.axvspan(sample.start, sample.end, ymax = 0.5, facecolor=(0.4, 0.1, 1.), alpha=1., label = loc["NV_area"])
        self.diagram.ax.set_xlim([0, timesize])
        self.diagram.ax.set_ylim([-1, 1])
        self.diagram.ax.set_xlabel(loc["time_lbl"], fontsize = 8)
        self.diagram.ax.set_ylabel(loc["amp_lbl"], fontsize = 8)
        self.diagram.ax.legend(loc = "upper right", fontsize = 8)
        self.diagram.canvas.draw()
        self.diagram.ax.clear()
            
    def onKeyChange(self, event):
        """UI Frontend function for changing the key of a phoneme-type UTAU sample"""
        logging.info("UTAU sample key change callback")
        index = self.phonemeList.list.lastFocusedIndex
        key = self.sampleList[index].key
        newKey = self.sideBar.key.variable.get()
        if key != newKey:
            self.sampleList[index].key = newKey
        
    def onFrameUpdateTrigger(self, event):
        """UI Frontend function for updating the start and end data of a sample"""
        logging.info("UTAU sample frame update callback")
        index = self.phonemeList.list.lastFocusedIndex
        sample = self.sampleList[index]
        sample.start = self.sideBar.start.variable.get()
        sample.end = self.sideBar.end.variable.get()
        sample.updateHandle()
        self.phonemeList.list.lb.delete(index)
        self.phonemeList.list.lb.insert(index, sample.handle)   
        self.updateDiagram()
        
    def onImportPress(self):
        """UI Frontend function for importing an UTAU sample from the SampleList as phoneme or transition sample"""
        logging.info("UTAU import button callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        if self.sampleList[index]._type == 0:
            loadedVB.addPhonemeUtau(self.sampleList[index])
        else:
            loadedVB.addTrainSampleUtau(self.sampleList[index])
        self.sideBar._type.variable.set(0)
        self.sideBar.key.variable.set(None)
        self.sideBar.start.variable.set(None)
        self.sideBar.end.variable.set(None)
        del self.sampleList[index]
        self.phonemeList.list.lb.delete(index)
        if index == self.phonemeList.list.lb.size():
            self.phonemeList.list.lb.selection_set(index - 1)
        else:
            self.phonemeList.list.lb.selection_set(index)
        if self.phonemeList.list.lb.size() > 0:
            self.updateDiagram()

    def onImportAllPress(self):
        """UI Frontend function for importing the entire SampleList as phoneme and transition samples"""
        logging.info("UTAU import all button callback")
        global loadedVB
        sampleCount = len(self.sampleList)
        for i in range(sampleCount):
            self.update()
            if self.sampleList[0]._type == 0:
                loadedVB.addPhonemeUtau(self.sampleList[0])
            else:
                loadedVB.addTrainSampleUtau(self.sampleList[0])
            del self.sampleList[0]
            self.phonemeList.list.lb.delete(0)
        self.update()
        self.sideBar._type.variable.set(0)
        self.sideBar.key.variable.set(None)
        self.sideBar.start.variable.set(None)
        self.sideBar.end.variable.set(None)
        self.phonemeList.list.lb.selection_set(0)
        
    def onOkPress(self):
        """Updates the last selected sample and closes the UTAU import UI window when the OK button is pressed"""
        logging.info("UTAU OK button callback")
        global loadedVB
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            if index != None:
                self.onKeyChange(None)
                self.onFrameUpdateTrigger(None)
        self.master.destroy()

    def onLoadPress(self):
        """UI Frontend function for parsing an oto.ini file, and loading its samples into the SampleList"""
        logging.info("oto.ini load button callback")
        global loadedVB
        custom_phonemes = tkinter.messagebox.askyesnocancel(loc["warning"], loc["utau_cstm_phn_msg"], icon = "question", default='no')
        if custom_phonemes != None:
            if custom_phonemes:
                phonemepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".txt_desc"], ".txt"), (loc["all_files_desc"], "*")))
            else:
                phonemepath = "Backend/UtauDefaultPhonemes.ini"
            phonemes = []
            types = []
            reader = csv.reader(open(phonemepath), delimiter = " ")
            for row in reader:
                phonemes.append(row[0])
                types.append(row[1])
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc["oto.ini_desc"], ".ini"), (loc["all_files_desc"], "*")))
            if filepath != "":

                f = open(filepath, "r", encoding = "Shift_JIS")
                reader = csv.reader(f, delimiter = ",")
                rowCount = sum(1 for row in reader)
                f.close()
                i = 0

                reader = csv.reader(open(filepath, encoding = "Shift_JIS"), delimiter = "=")
                otoPath = path.split(filepath)[0]

                for row in reader:
                    print("processing " + str(row))
                    i += 1
                    self.update()
                    filename = row[0]
                    properties = row[1].split(",")
                    occuredError = None
                    try:
                        fetchedSamples = fetchSamples(filename, properties, phonemes, types, otoPath)
                        for sample in fetchedSamples:
                            if sample._type == 1:
                                self.sampleList.append(sample)
                                self.phonemeList.list.lb.insert("end", sample.handle)
                            else:
                                for i in range(len(self.sampleList)):
                                    if self.sampleList[i].key == sample.key:
                                        if sample.end - sample.start > self.sampleList[i].end - self.sampleList[i].start:
                                            self.sampleList[i] = sample
                                            self.sampleList[i].updateHandle()
                                            self.phonemeList.list.lb.delete(i)
                                            self.phonemeList.list.lb.insert(i, sample.handle)
                                        break
                                else:
                                    self.sampleList.append(sample)
                                    self.phonemeList.list.lb.insert("end", sample.handle)

                    except LookupError as error:
                        if occuredError == None:
                            occuredError = 0
                        logging.warning(error)
                    except Exception as error:
                        occuredError = 1
                        logging.error(error)
                self.update()
                if occuredError == 0:
                    tkinter.messagebox.showinfo(loc["info"], loc["oto_msng_phn"])
                elif occuredError == 1:
                    tkinter.messagebox.showerror(loc["error"], loc["oto_error"])