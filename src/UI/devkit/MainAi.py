#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import tkinter
import torch
import sys

from UI.devkit.Widgets import *
import global_consts
from Localization.devkit_localization import getLanguage
loc = getLanguage()

class MainaiUi(Frame):
    """Class of the Spectral Prediction AI UI window"""

    def __init__(self, master=None) -> None:
        logging.info("Initializing Predai UI")
        global loadedVB
        from UI.devkit.Main import loadedVB
        Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["predai_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("assets/icon/nova-vox-logo-black.ico")
        
    def createWidgets(self) -> None:
        """creates all widgets of the window. Called once during initialization"""

        global loadedVB
        
        self.phonemeList = LabelFrame(self, text = loc["ai_samp_list"])
        self.phonemeList.list = Frame(self.phonemeList)
        self.phonemeList.list.lb = Listbox(self.phonemeList.list)
        self.phonemeList.list.lb.pack(side = "left",fill = "both", expand = True)
        self.phonemeList.list.sb = tkinter.ttk.Scrollbar(self.phonemeList.list)
        self.phonemeList.list.sb.pack(side = "left", fill = "y")
        self.phonemeList.list.lb["selectmode"] = "single"
        self.phonemeList.list.lb["yscrollcommand"] = self.phonemeList.list.sb.set
        self.phonemeList.list.lb.bind("<<ListboxSelect>>", self.onSelectionChange)
        self.phonemeList.list.lb.bind("<FocusOut>", self.onListFocusOut)
        self.phonemeList.list.sb["command"] = self.phonemeList.list.lb.yview
        self.phonemeList.list.pack(side = "top", fill = "x", expand = True, padx = 5, pady = 2)
        for i in loadedVB.stagedMainTrainSamples:
            self.phonemeList.list.lb.insert("end", i.filepath)
        self.phonemeList.removeButton = SlimButton(self.phonemeList)
        self.phonemeList.removeButton["text"] = loc["remove"]
        self.phonemeList.removeButton["command"] = self.onRemovePress
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = SlimButton(self.phonemeList)
        self.phonemeList.addButton["text"] = loc["add"]
        self.phonemeList.addButton["command"] = self.onAddPress
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar = LabelFrame(self, text = loc["ai_settings"])
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        self.sideBar.expPitch = Frame(self.sideBar)
        self.sideBar.expPitch.variable = tkinter.DoubleVar(self.sideBar.expPitch, global_consts.defaultExpectedPitch)
        self.sideBar.expPitch.entry = Entry(self.sideBar.expPitch)
        self.sideBar.expPitch.entry["textvariable"] = self.sideBar.expPitch.variable
        self.sideBar.expPitch.entry.pack(side = "right", fill = "x")
        self.sideBar.expPitch.display = Label(self.sideBar.expPitch)
        self.sideBar.expPitch.display["text"] = loc["est_pit"]
        self.sideBar.expPitch.display.pack(side = "right", fill = "x")
        self.sideBar.expPitch.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.pSearchRange = Frame(self.sideBar)
        self.sideBar.pSearchRange.variable = tkinter.DoubleVar(self.sideBar.pSearchRange, global_consts.defaultSearchRange)
        self.sideBar.pSearchRange.entry = Spinbox(self.sideBar.pSearchRange, from_ = 0.35, to = 0.95, increment = 0.05)
        self.sideBar.pSearchRange.entry["textvariable"] = self.sideBar.pSearchRange.variable
        self.sideBar.pSearchRange.entry.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.display = Label(self.sideBar.pSearchRange)
        self.sideBar.pSearchRange.display["text"] = loc["psearchr"]
        self.sideBar.pSearchRange.display.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.pBroadcastButton = Button(self.sideBar)
        self.sideBar.pBroadcastButton["text"] = loc["pit_brdc"]
        self.sideBar.pBroadcastButton["command"] = self.onPitBrdcPress
        self.sideBar.pBroadcastButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        self.sideBar.voicedThrh = Frame(self.sideBar)
        self.sideBar.voicedThrh.variable = tkinter.DoubleVar(self.sideBar.voicedThrh, global_consts.defaultVoicedThrh)
        self.sideBar.voicedThrh.entry = Spinbox(self.sideBar.voicedThrh, from_ = 0.35, to = 0.95, increment = 0.05)
        self.sideBar.voicedThrh.entry["textvariable"] = self.sideBar.voicedThrh.variable
        self.sideBar.voicedThrh.entry.pack(side = "right", fill = "x")
        self.sideBar.voicedThrh.display = Label(self.sideBar.voicedThrh)
        self.sideBar.voicedThrh.display["text"] = loc["voicedThrh"]
        self.sideBar.voicedThrh.display.pack(side = "right", fill = "x")
        self.sideBar.voicedThrh.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.specSmooth = Frame(self.sideBar)
        self.sideBar.specSmooth.widthVariable = tkinter.IntVar(self.sideBar.specSmooth, global_consts.defaultSpecWidth)
        self.sideBar.specSmooth.widthEntry = Spinbox(self.sideBar.specSmooth, from_ = 1, to = 100)
        self.sideBar.specSmooth.widthEntry["textvariable"] = self.sideBar.specSmooth.widthVariable
        self.sideBar.specSmooth.widthEntry.pack(side = "right", fill = "x")
        self.sideBar.specSmooth.depthVariable = tkinter.IntVar(self.sideBar.specSmooth, global_consts.defaultSpecDepth)
        self.sideBar.specSmooth.depthEntry = Spinbox(self.sideBar.specSmooth, from_ = 0, to = 100)
        self.sideBar.specSmooth.depthEntry["textvariable"] = self.sideBar.specSmooth.depthVariable
        self.sideBar.specSmooth.depthEntry.pack(side = "right", fill = "x")
        self.sideBar.specSmooth.display = Label(self.sideBar.specSmooth)
        self.sideBar.specSmooth.display["text"] = loc["specSmooth"]
        self.sideBar.specSmooth.display.pack(side = "right", fill = "x")
        self.sideBar.specSmooth.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.tempSmooth = Frame(self.sideBar)
        self.sideBar.tempSmooth.widthVariable = tkinter.IntVar(self.sideBar.tempSmooth, global_consts.defaultTempWidth)
        self.sideBar.tempSmooth.widthEntry = Spinbox(self.sideBar.tempSmooth, from_ = 1, to = 100)
        self.sideBar.tempSmooth.widthEntry["textvariable"] = self.sideBar.tempSmooth.widthVariable
        self.sideBar.tempSmooth.widthEntry.pack(side = "right", fill = "x")
        self.sideBar.tempSmooth.depthVariable = tkinter.IntVar(self.sideBar.tempSmooth, global_consts.defaultTempDepth)
        self.sideBar.tempSmooth.depthEntry = Spinbox(self.sideBar.tempSmooth, from_ = 0, to = 100)
        self.sideBar.tempSmooth.depthEntry["textvariable"] = self.sideBar.tempSmooth.depthVariable
        self.sideBar.tempSmooth.depthEntry.pack(side = "right", fill = "x")
        self.sideBar.tempSmooth.display = Label(self.sideBar.tempSmooth)
        self.sideBar.tempSmooth.display["text"] = loc["tempSmooth"]
        self.sideBar.tempSmooth.display.pack(side = "right", fill = "x")
        self.sideBar.tempSmooth.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.sBroadcastButton = Button(self.sideBar)
        self.sideBar.sBroadcastButton["text"] = loc["spec_brdc"]
        self.sideBar.sBroadcastButton["command"] = self.onSpecBrdcPress
        self.sideBar.sBroadcastButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.exprKey = Frame(self.sideBar)
        self.sideBar.exprKey.variable = tkinter.StringVar(self.sideBar.exprKey, "")
        self.sideBar.exprKey.entry = Entry(self.sideBar.exprKey)
        self.sideBar.exprKey.entry["textvariable"] = self.sideBar.exprKey.variable
        self.sideBar.exprKey.entry.pack(side = "right", fill = "x")
        self.sideBar.exprKey.display = Label(self.sideBar.exprKey)
        self.sideBar.exprKey.display["text"] = loc["exprKey"]
        self.sideBar.exprKey.display.pack(side = "right", fill = "x")
        self.sideBar.exprKey.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.epochs = Frame(self.sideBar)
        self.sideBar.epochs.variable = tkinter.IntVar(self.sideBar.epochs, 1)
        self.sideBar.epochs.entry = Spinbox(self.sideBar.epochs, from_ = 1, to = 100)
        self.sideBar.epochs.entry["textvariable"] = self.sideBar.epochs.variable
        self.sideBar.epochs.entry.pack(side = "right", fill = "x")
        self.sideBar.epochs.display = Label(self.sideBar.epochs)
        self.sideBar.epochs.display["text"] = loc["epochs"]
        self.sideBar.epochs.display.pack(side = "right", fill = "x")
        self.sideBar.epochs.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.additive = Frame(self.sideBar)
        self.sideBar.additive.variable = tkinter.BooleanVar(self.sideBar.additive, True)
        self.sideBar.additive.entry = Checkbutton(self.sideBar.additive)
        self.sideBar.additive.entry["variable"] = self.sideBar.additive.variable
        self.sideBar.additive.entry.pack(side = "right", fill = "x")
        self.sideBar.additive.display = Label(self.sideBar.additive)
        self.sideBar.additive.display["text"] = loc["additive"]
        self.sideBar.additive.display.pack(side = "right", fill = "x")
        self.sideBar.additive.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.generatorMode = Frame(self.sideBar)
        self.sideBar.generatorMode.variable = tkinter.StringVar(self.sideBar.generatorMode, "reclist")
        self.sideBar.generatorMode.entry = tkinter.ttk.OptionMenu(self.sideBar.generatorMode, self.sideBar.generatorMode.variable, "reclist", "reclist (strict vowels)", "dictionary", "dictionary (syllables)", "dataset file")
        self.sideBar.generatorMode.entry.pack(side = "right", fill = "x")
        self.sideBar.generatorMode.display = Label(self.sideBar.generatorMode)
        self.sideBar.generatorMode.display["text"] = loc["generator_mode"]
        self.sideBar.generatorMode.display.pack(side = "right", fill = "x")
        self.sideBar.generatorMode.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.logging = Frame(self.sideBar)
        self.sideBar.logging.variable = tkinter.BooleanVar(self.sideBar.logging, False)
        self.sideBar.logging.entry = Checkbutton(self.sideBar.logging)
        self.sideBar.logging.entry["variable"] = self.sideBar.logging.variable
        self.sideBar.logging.entry.pack(side = "right", fill = "x")
        self.sideBar.logging.display = Label(self.sideBar.logging)
        self.sideBar.logging.display["text"] = loc["logging"]
        self.sideBar.logging.display.pack(side = "right", fill = "x")
        self.sideBar.logging.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.exportButton = Button(self.sideBar)
        self.sideBar.exportButton["text"] = loc["ai_smp_export"]
        self.sideBar.exportButton["command"] = self.onExportPress
        self.sideBar.exportButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        self.sideBar.importButton = Button(self.sideBar)
        self.sideBar.importButton["text"] = loc["ai_smp_import"]
        self.sideBar.importButton["command"] = self.onImportPress
        self.sideBar.importButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.trainButton = Button(self.sideBar)
        self.sideBar.trainButton["text"] = loc["train"]
        self.sideBar.trainButton["command"] = self.onTrainPress
        self.sideBar.trainButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.finalizeButton = Button(self.sideBar)
        self.sideBar.finalizeButton["text"] = loc["finalize"]
        self.sideBar.finalizeButton["command"] = self.onFinalizePress
        self.sideBar.finalizeButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        if loadedVB.ai.mainAi.epoch == None:
            epoch = loc["varying"]
        else:
            epoch = str(loadedVB.ai.mainAi.epoch)
        self.statusVar = tkinter.StringVar(self, loc["AI_stat_1"] + epoch + loc["AI_stat_2"] + str(loadedVB.ai.mainAi.sampleCount) + loc["AI_stat_3"])
        self.statusLabel = Label(self)
        self.statusLabel["textvariable"] = self.statusVar
        self.statusLabel.pack(side = "top", fill = "x", expand = True, padx = 5)

        self.okButton = Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = Button(self)
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
            index = self.phonemeList.list.lastFocusedIndex
            self.sideBar.expPitch.variable.set(loadedVB.stagedMainTrainSamples[index].expectedPitch)
            self.sideBar.pSearchRange.variable.set(loadedVB.stagedMainTrainSamples[index].searchRange)
            self.sideBar.voicedThrh.variable.set(loadedVB.stagedMainTrainSamples[index].voicedThrh)
            self.sideBar.specSmooth.widthVariable.set(loadedVB.stagedMainTrainSamples[index].specWidth)
            self.sideBar.specSmooth.depthVariable.set(loadedVB.stagedMainTrainSamples[index].specDepth)
            self.sideBar.tempSmooth.widthVariable.set(loadedVB.stagedMainTrainSamples[index].tempWidth)
            self.sideBar.tempSmooth.depthVariable.set(loadedVB.stagedMainTrainSamples[index].tempDepth)
            self.sideBar.exprKey.variable.set(loadedVB.stagedMainTrainSamples[index].key)
            
    def onListFocusOut(self, event) -> None:
        """Helper function for retaining information about the last selected transition sample when the transition sample list loses entry focus"""

        logging.info("Crfai list focus loss callback")
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]

    def onUpdateTrigger(self) -> None:
        """Updates the pitch and spectral processing settings of a phoneme"""

        logging.info("crf staged phoneme update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        loadedVB.stagedMainTrainSamples[index].expectedPitch = self.sideBar.expPitch.variable.get()
        loadedVB.stagedMainTrainSamples[index].searchRange = self.sideBar.pSearchRange.variable.get()
        loadedVB.stagedMainTrainSamples[index].voicedThrh =  self.sideBar.voicedThrh.variable.get()
        loadedVB.stagedMainTrainSamples[index].specWidth = self.sideBar.specSmooth.widthVariable.get()
        loadedVB.stagedMainTrainSamples[index].specDepth = self.sideBar.specSmooth.depthVariable.get()
        loadedVB.stagedMainTrainSamples[index].tempWidth = self.sideBar.tempSmooth.widthVariable.get()
        loadedVB.stagedMainTrainSamples[index].tempDepth = self.sideBar.tempSmooth.depthVariable.get()
        loadedVB.stagedMainTrainSamples[index].key = self.sideBar.exprKey.variable.get()

    def onPitBrdcPress(self) -> None:
        """UI Frontend function for applying/broadcasting the pitch search settings of the currently selected sample to all samples"""

        pitch = self.sideBar.expPitch.variable.get()
        pitchRange = self.sideBar.pSearchRange.variable.get()
        for i in loadedVB.stagedMainTrainSamples:
            i.expectedPitch = pitch
            i.searchRange = pitchRange

    def onSpecBrdcPress(self) -> None:
        """UI Frontend function for applying/broadcasting the spectral filtering & analysis settings of the currently selected sample to all samples"""

        newValues = [
            self.sideBar.voicedThrh.variable.get(),
            self.sideBar.specSmooth.widthVariable.get(),
            self.sideBar.specSmooth.depthVariable.get(),
            self.sideBar.tempSmooth.widthVariable.get(),
            self.sideBar.tempSmooth.depthVariable.get(),
        ]
        for i in loadedVB.stagedMainTrainSamples:
            i.voicedThrh = newValues[0]
            i.specWidth = newValues[1]
            i.specDepth = newValues[2]
            i.tempWidth = newValues[3]
            i.tempDepth = newValues[4]
    
    def onAddPress(self) -> None:
        """UI Frontend function for adding a new transition sample to the list of staged AI training samples"""

        logging.info("Crfai add button callback")
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")), multiple = True)
        if filepath != ():
            for i in filepath:
                loadedVB.addMainTrainSample(i)
                self.phonemeList.list.lb.insert("end", i)
        
    def onRemovePress(self) -> None:
        """UI Frontend function for removing a transition sample from the list of staged AI training samples"""

        logging.info("Crfai remove button callback")
        global loadedVB
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            loadedVB.delMainTrainSample(index)
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
        torch.save(loadedVB.stagedMainTrainSamples, filepath)

    def onImportPress(self) -> None:
        """UI Frontend function for importing a previously saved AI training dataset. Overwrites any previously staged training samples."""

        logging.info("Crfai dataset export callback")
        global loadedVB
        filepaths = tkinter.filedialog.askopenfilename(filetypes = ((loc["all_files_desc"], "*"), ), multiple = True)
        for i in filepaths:
            loadedVB.stagedMainTrainSamples.extend(torch.load(i, map_location = torch.device("cpu")))
                
    def onTrainPress(self) -> None:
        """UI Frontend function for training the AI with the specified settings and samples"""

        logging.info("Crfai train button callback")
        global loadedVB
        self.statusVar.set("training Ai...")
        self.update()
        loadedVB.trainMainAi(
            self.sideBar.epochs.variable.get(),
            self.sideBar.additive.variable.get(),
            self.sideBar.generatorMode.variable.get(),
            self.sideBar.logging.variable.get()
        )
        numIter = self.phonemeList.list.lb.size()
        for i in range(numIter):
            self.phonemeList.list.lb.delete(0)
        self.statusVar.set(loc["AI_stat_1"] + str(loadedVB.ai.mainAi.epoch) + loc["AI_stat_2"] + str(loadedVB.ai.mainAi.sampleCount) + loc["AI_stat_3"])
        logging.info("Crfai train button callback completed")
        
    def onFinalizePress(self) -> None:
        """UI Frontend function for finalizing the phoneme crossfade AI, reducing its size when saving the Voicebnank to a file."""

        logging.info("Crfai finalize button callback")
        global loadedVB
        loadedVB.ai.finalize()
        self.disableButtons()
        
    def disableButtons(self) -> None:
        """Utility function for disabling the AI settings buttons"""

        self.sideBar.voicedThrh.entry["state"] = "disabled"
        self.sideBar.specSmooth.widthEntry["state"] = "disabled"
        self.sideBar.specSmooth.depthEntry["state"] = "disabled"
        self.sideBar.tempSmooth.widthEntry["state"] = "disabled"
        self.sideBar.tempSmooth.depthEntry["state"] = "disabled"
        self.sideBar.exprKey.entry["state"] = "disabled"
        self.sideBar.epochs.entry["state"] = "disabled"
        self.sideBar.trainButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
    
    def enableButtons(self) -> None:
        """Utility function for enabling the AI settings buttons"""

        self.sideBar.voicedThrh.entry["state"] = "normal"
        self.sideBar.specSmooth.widthEntry["state"] = "normal"
        self.sideBar.specSmooth.depthEntry["state"] = "normal"
        self.sideBar.tempSmooth.widthEntry["state"] = "normal"
        self.sideBar.tempSmooth.depthEntry["state"] = "normal"
        self.sideBar.exprKey.entry["state"] = "normal"
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
            loadedVB.loadMainWeights(filepath)
            self.statusVar.set(loc["AI_stat_1"] + str(loadedVB.ai.mainAi.epoch) + loc["AI_stat_2"] + str(loadedVB.ai.mainAi.sampleCount) + loc["AI_stat_3"])
