import logging
import tkinter
import torch
import sys
from os import path, system

import global_consts
from Locale.devkit_locale import getLocale
from MiddleLayer.IniParser import readSettings
loc = getLocale()

class CrfaiUi(tkinter.Frame):
    """Class of the Spectral Crossfade AI UI window"""

    def __init__(self, master=None) -> None:
        logging.info("Initializing Crfai UI")
        global loadedVB
        from UI.code.devkit.Main import loadedVB
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["crfai_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
    def createWidgets(self) -> None:
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
        for i in loadedVB.stagedCrfTrainSamples:
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
        
        self.sideBar.expPitch = tkinter.Frame(self.sideBar)
        self.sideBar.expPitch.variable = tkinter.DoubleVar(self.sideBar.expPitch, global_consts.defaultExpectedPitch)
        self.sideBar.expPitch.entry = tkinter.Entry(self.sideBar.expPitch)
        self.sideBar.expPitch.entry["textvariable"] = self.sideBar.expPitch.variable
        self.sideBar.expPitch.entry.pack(side = "right", fill = "x")
        self.sideBar.expPitch.display = tkinter.Label(self.sideBar.expPitch)
        self.sideBar.expPitch.display["text"] = loc["est_pit"]
        self.sideBar.expPitch.display.pack(side = "right", fill = "x")
        self.sideBar.expPitch.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.pSearchRange = tkinter.Frame(self.sideBar)
        self.sideBar.pSearchRange.variable = tkinter.DoubleVar(self.sideBar.pSearchRange, global_consts.defaultSearchRange)
        self.sideBar.pSearchRange.entry = tkinter.Spinbox(self.sideBar.pSearchRange, from_ = 0.35, to = 0.95, increment = 0.05)
        self.sideBar.pSearchRange.entry["textvariable"] = self.sideBar.pSearchRange.variable
        self.sideBar.pSearchRange.entry.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.display = tkinter.Label(self.sideBar.pSearchRange)
        self.sideBar.pSearchRange.display["text"] = loc["psearchr"]
        self.sideBar.pSearchRange.display.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.unvoicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.unvoicedIter.variable = tkinter.IntVar(self.sideBar.unvoicedIter, global_consts.defaultUnvoicedIterations)
        self.sideBar.unvoicedIter.variable.set(20)
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

        self.sideBar.logging = tkinter.Frame(self.sideBar)
        self.sideBar.logging.variable = tkinter.BooleanVar(self.sideBar.logging, False)
        self.sideBar.logging.entry = tkinter.Checkbutton(self.sideBar.logging)
        self.sideBar.logging.entry["variable"] = self.sideBar.logging.variable
        self.sideBar.logging.entry.pack(side = "right", fill = "x")
        self.sideBar.logging.display = tkinter.Label(self.sideBar.logging)
        self.sideBar.logging.display["text"] = loc["logging"]
        self.sideBar.logging.display.pack(side = "right", fill = "x")
        self.sideBar.logging.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.exportButton = tkinter.Button(self.sideBar)
        self.sideBar.exportButton["text"] = loc["ai_smp_export"]
        self.sideBar.exportButton["command"] = self.onExportPress
        self.sideBar.exportButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        self.sideBar.importButton = tkinter.Button(self.sideBar)
        self.sideBar.importButton["text"] = loc["ai_smp_import"]
        self.sideBar.importButton["command"] = self.onImportPress
        self.sideBar.importButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.trainButton = tkinter.Button(self.sideBar)
        self.sideBar.trainButton["text"] = loc["train"]
        self.sideBar.trainButton["command"] = self.onTrainPress
        self.sideBar.trainButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.finalizeButton = tkinter.Button(self.sideBar)
        self.sideBar.finalizeButton["text"] = loc["finalize"]
        self.sideBar.finalizeButton["command"] = self.onFinalizePress
        self.sideBar.finalizeButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        self.sideBar.tensorBoardButton = tkinter.Button(self.sideBar)
        self.sideBar.tensorBoardButton["text"] = loc["open_tb"]
        self.sideBar.tensorBoardButton["command"] = lambda : system("start tensorboard.exe --logdir " + path.join(readSettings()["dataDir"], "Nova-Vox", "Logs"))
        self.sideBar.tensorBoardButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        if loadedVB.ai.crfAi.epoch == None:
            epoch = loc["varying"]
        else:
            epoch = str(loadedVB.ai.crfAi.epoch)
        self.statusVar = tkinter.StringVar(self, loc["AI_stat_1"] + epoch + loc["AI_stat_2"] + str(loadedVB.ai.crfAi.sampleCount) + loc["AI_stat_3"])
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
        
        if loadedVB.ai.final:
            self.disableButtons()
        
    def onSelectionChange(self, event) -> None:
        """Helper function for retaining information about the last selected transition sample when the selected item in the transition sample list changes"""

        logging.info("Crfai selection change callback")
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            
    def onListFocusOut(self, event) -> None:
        """Helper function for retaining information about the last selected transition sample when the transition sample list loses entry focus"""

        logging.info("Crfai list focus loss callback")
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
    
    def onAddPress(self) -> None:
        """UI Frontend function for adding a new transition sample to the list of staged AI training samples"""

        logging.info("Crfai add button callback")
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")), multiple = True)
        if filepath != ():
            for i in filepath:
                loadedVB.addCrfTrainSample(i)
                self.phonemeList.list.lb.insert("end", i)
        
    def onRemovePress(self) -> None:
        """UI Frontend function for removing a transition sample from the list of staged AI training samples"""

        logging.info("Crfai remove button callback")
        global loadedVB
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            loadedVB.delCrfTrainSample(index)
            self.phonemeList.list.lb.delete(index)
            if index == self.phonemeList.list.lb.size():
                self.phonemeList.list.lb.selection_set(index - 1)
            else:
                self.phonemeList.list.lb.selection_set(index)

    def onExportPress(self) -> None:
        """UI Frontend function for exporting an AI training dataset using Pickle."""

        logging.info("Crfai dataset export callback")
        global loadedVB
        filepath = tkinter.filedialog.asksaveasfilename(defaultextension = ".dat", filetypes = ((".dat", ".dat"), (loc["all_files_desc"], "*")))
        torch.save(loadedVB.stagedCrfTrainSamples, filepath)

    def onImportPress(self) -> None:
        """UI Frontend function for importing a previously saved AI training dataset. Overwrites any previously staged training samples."""

        logging.info("Crfai dataset export callback")
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc["all_files_desc"], "*"), ), multiple = True)
        loadedVB.stagedCrfTrainSamples = torch.load(filepath)
                
    def onTrainPress(self) -> None:
        """UI Frontend function for training the AI with the specified settings and samples"""

        logging.info("Crfai train button callback")
        global loadedVB
        self.statusVar.set("training Ai...")
        self.update()
        loadedVB.trainCrfAi(self.sideBar.epochs.variable.get(), True, self.sideBar.unvoicedIter.variable.get(), self.sideBar.expPitch.variable.get(), self.sideBar.pSearchRange.variable.get(), self.sideBar.logging.variable.get())
        numIter = self.phonemeList.list.lb.size()
        for i in range(numIter):
            loadedVB.delCrfTrainSample(0)
            self.phonemeList.list.lb.delete(0)
        self.statusVar.set(loc["AI_stat_1"] + str(loadedVB.ai.crfAi.epoch) + loc["AI_stat_2"] + str(loadedVB.ai.crfAi.sampleCount) + loc["AI_stat_3"])
        logging.info("Crfai train button callback completed")
        
    def onFinalizePress(self) -> None:
        """UI Frontend function for finalizing the phoneme crossfade AI, reducing its size when saving the Voicebnank to a file."""

        logging.info("Crfai finalize button callback")
        global loadedVB
        loadedVB.ai.finalize()
        self.disableButtons()
        
    def disableButtons(self) -> None:
        """Utility function for disabling the AI settings buttons"""

        self.sideBar.unvoicedIter.entry["state"] = "disabled"
        self.sideBar.epochs.entry["state"] = "disabled"
        self.sideBar.trainButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
    
    def enableButtons(self) -> None:
        """Utility function for enabling the AI settings buttons"""

        self.sideBar.unvoicedIter.entry["state"] = "normal"
        self.sideBar.epochs.entry["state"] = "normal"
        self.sideBar.trainButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
        
    def onOkPress(self) -> None:
        """closes the window when the OK button is pressed"""

        logging.info("Crfai OK button callback")
        global loadedVB
        self.master.destroy()

    def onLoadPress(self) -> None:
        """UI Frontend function for loading the AI state from the Phoneme Crossfade AI of a different Voicebank"""
        
        logging.info("Crfai load button callback")
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
        if filepath != "":
            loadedVB.loadCrfWeights(filepath)
            self.statusVar.set(loc["AI_stat_1"] + str(loadedVB.ai.crfAi.epoch) + loc["AI_stat_2"] + str(loadedVB.ai.crfAi.sampleCount) + loc["AI_stat_3"])
