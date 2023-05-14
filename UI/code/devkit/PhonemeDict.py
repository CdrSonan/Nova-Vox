#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import tkinter
import torch
import sys

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from UI.code.devkit.Widgets import *
import global_consts
from Localization.devkit_localization import getLanguage
loc = getLanguage()
from Backend.ESPER.PitchCalculator import calculatePitch
from Backend.ESPER.SpectralCalculator import calculateSpectra

class PhonemedictUi(Frame):
    """Class of the phoneme dictionnary UI window"""

    def __init__(self, master=None) -> None:
        logging.info("Initializing Phonemedict UI")
        global loadedVB
        from UI.code.devkit.Main import loadedVB
        Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["phon_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
    def createWidgets(self) -> None:
        """Initializes all widgets of the Phoneme Dict UI window."""

        global loadedVB

        self.diagram = LabelFrame(self, text = loc["diag_lbl"])
        self.diagram.fig = Figure(figsize=(4, 4))
        self.diagram.ax = self.diagram.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.diagram.ax.set_xlim([0, global_consts.sampleRate / 2])
        self.diagram.ax.set_xlabel(loc["freq_lbl"], fontsize = 8)
        self.diagram.ax.set_ylabel(loc["amp_lbl"], fontsize = 8)
        self.diagram.canvas = FigureCanvasTkAgg(self.diagram.fig, self.diagram)
        self.diagram.canvas.get_tk_widget().pack(side = "top", fill = "both", expand = True)
        self.diagram.timeSlider = tkinter.ttk.Scale(self.diagram, from_ = 0, to = 0, orient = "horizontal", length = 600, command = self.onSliderMove)
        self.diagram.timeSlider.pack(side = "left", fill = "both", expand = True, padx = 5, pady = 2)
        self.diagram.pack(side = "right", fill = "y", padx = 5, pady = 2)
        
        self.phonemeList = LabelFrame(self, text = loc["phon_list"])
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
        for i in loadedVB.phonemeDict.keys():
            self.phonemeList.list.lb.insert("end", i)
        self.phonemeList.removeButton = SlimButton(self.phonemeList)
        self.phonemeList.removeButton["text"] = loc["remove"]
        self.phonemeList.removeButton["command"] = self.onRemovePress
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = SlimButton(self.phonemeList)
        self.phonemeList.addButton["text"] = loc["add"]
        self.phonemeList.addButton["command"] = self.onAddPress
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar = LabelFrame(self, text = loc["per_ph_set"])
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        self.sideBar.key = Frame(self.sideBar)
        self.sideBar.key.variable = tkinter.StringVar(self.sideBar.key)
        self.sideBar.key.entry = Entry(self.sideBar.key)
        self.sideBar.key.entry["textvariable"] = self.sideBar.key.variable
        self.sideBar.key.entry.bind("<FocusOut>", self.onKeyChange)
        self.sideBar.key.entry.bind("<KeyRelease-Return>", self.onKeyChange)
        self.sideBar.key.entry.pack(side = "right", fill = "x")
        self.sideBar.key.display = Label(self.sideBar.key)
        self.sideBar.key.display["text"] = loc["phon_key"]
        self.sideBar.key.display.pack(side = "right", fill = "x")
        self.sideBar.key.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.expPitch = Frame(self.sideBar)
        self.sideBar.expPitch.variable = tkinter.DoubleVar(self.sideBar.expPitch, global_consts.defaultExpectedPitch)
        self.sideBar.expPitch.entry = Entry(self.sideBar.expPitch)
        self.sideBar.expPitch.entry["textvariable"] = self.sideBar.expPitch.variable
        self.sideBar.expPitch.entry.bind("<FocusOut>", self.onPitchUpdateTrigger)
        self.sideBar.expPitch.entry.bind("<KeyRelease-Return>", self.onPitchUpdateTrigger)
        self.sideBar.expPitch.entry.pack(side = "right", fill = "x")
        self.sideBar.expPitch.display = Label(self.sideBar.expPitch)
        self.sideBar.expPitch.display["text"] = loc["est_pit"]
        self.sideBar.expPitch.display.pack(side = "right", fill = "x")
        self.sideBar.expPitch.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.pSearchRange = Frame(self.sideBar)
        self.sideBar.pSearchRange.variable = tkinter.DoubleVar(self.sideBar.pSearchRange, global_consts.defaultSearchRange)
        self.sideBar.pSearchRange.entry = Spinbox(self.sideBar.pSearchRange, from_ = 0.35, to = 0.95, increment = 0.05)
        self.sideBar.pSearchRange.entry["textvariable"] = self.sideBar.pSearchRange.variable
        self.sideBar.pSearchRange.entry.bind("<FocusOut>", self.onPitchUpdateTrigger)
        self.sideBar.pSearchRange.entry.bind("<KeyRelease-Return>", self.onPitchUpdateTrigger)
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
        self.sideBar.voicedThrh.entry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.voicedThrh.entry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.voicedThrh.entry.pack(side = "right", fill = "x")
        self.sideBar.voicedThrh.display = Label(self.sideBar.voicedThrh)
        self.sideBar.voicedThrh.display["text"] = loc["voicedThrh"]
        self.sideBar.voicedThrh.display.pack(side = "right", fill = "x")
        self.sideBar.voicedThrh.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.specSmooth = Frame(self.sideBar)
        self.sideBar.specSmooth.widthVariable = tkinter.IntVar(self.sideBar.specSmooth, global_consts.defaultSpecWidth)
        self.sideBar.specSmooth.widthEntry = Spinbox(self.sideBar.specSmooth, from_ = 1, to = 100)
        self.sideBar.specSmooth.widthEntry["textvariable"] = self.sideBar.specSmooth.widthVariable
        self.sideBar.specSmooth.widthEntry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.specSmooth.widthEntry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.specSmooth.widthEntry.pack(side = "right", fill = "x")
        self.sideBar.specSmooth.depthVariable = tkinter.IntVar(self.sideBar.specSmooth, global_consts.defaultSpecDepth)
        self.sideBar.specSmooth.depthEntry = Spinbox(self.sideBar.specSmooth, from_ = 0, to = 100)
        self.sideBar.specSmooth.depthEntry["textvariable"] = self.sideBar.specSmooth.depthVariable
        self.sideBar.specSmooth.depthEntry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.specSmooth.depthEntry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.specSmooth.depthEntry.pack(side = "right", fill = "x")
        self.sideBar.specSmooth.display = Label(self.sideBar.specSmooth)
        self.sideBar.specSmooth.display["text"] = loc["specSmooth"]
        self.sideBar.specSmooth.display.pack(side = "right", fill = "x")
        self.sideBar.specSmooth.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.tempSmooth = Frame(self.sideBar)
        self.sideBar.tempSmooth.widthVariable = tkinter.IntVar(self.sideBar.tempSmooth, global_consts.defaultTempWidth)
        self.sideBar.tempSmooth.widthEntry = Spinbox(self.sideBar.tempSmooth, from_ = 1, to = 100)
        self.sideBar.tempSmooth.widthEntry["textvariable"] = self.sideBar.tempSmooth.widthVariable
        self.sideBar.tempSmooth.widthEntry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.tempSmooth.widthEntry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.tempSmooth.widthEntry.pack(side = "right", fill = "x")
        self.sideBar.tempSmooth.depthVariable = tkinter.IntVar(self.sideBar.tempSmooth, global_consts.defaultTempDepth)
        self.sideBar.tempSmooth.depthEntry = Spinbox(self.sideBar.tempSmooth, from_ = 0, to = 100)
        self.sideBar.tempSmooth.depthEntry["textvariable"] = self.sideBar.tempSmooth.depthVariable
        self.sideBar.tempSmooth.depthEntry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.tempSmooth.depthEntry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.tempSmooth.depthEntry.pack(side = "right", fill = "x")
        self.sideBar.tempSmooth.display = Label(self.sideBar.tempSmooth)
        self.sideBar.tempSmooth.display["text"] = loc["tempSmooth"]
        self.sideBar.tempSmooth.display.pack(side = "right", fill = "x")
        self.sideBar.tempSmooth.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.sBroadcastButton = Button(self.sideBar)
        self.sideBar.sBroadcastButton["text"] = loc["spec_brdc"]
        self.sideBar.sBroadcastButton["command"] = self.onSpecBrdcPress
        self.sideBar.sBroadcastButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        self.sideBar.isVoiced = Frame(self.sideBar)
        self.sideBar.isVoiced.variable = tkinter.BooleanVar(self.sideBar.isVoiced, True)
        self.sideBar.isVoiced.entry = Checkbutton(self.sideBar.isVoiced)
        self.sideBar.isVoiced.entry["variable"] = self.sideBar.isVoiced.variable
        self.sideBar.isVoiced.entry["command"] = self.onVoicedUpdateTrigger
        self.sideBar.isVoiced.entry.pack(side = "right", fill = "x")
        self.sideBar.isVoiced.display = Label(self.sideBar.isVoiced)
        self.sideBar.isVoiced.display["text"] = loc["voiced"]
        self.sideBar.isVoiced.display.pack(side = "right", fill = "x")
        self.sideBar.isVoiced.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.isPlosive = Frame(self.sideBar)
        self.sideBar.isPlosive.variable = tkinter.BooleanVar(self.sideBar.isPlosive, True)
        self.sideBar.isPlosive.entry = Checkbutton(self.sideBar.isPlosive)
        self.sideBar.isPlosive.entry["variable"] = self.sideBar.isPlosive.variable
        self.sideBar.isPlosive.entry["command"] = self.onPlosiveUpdateTrigger
        self.sideBar.isPlosive.entry.pack(side = "right", fill = "x")
        self.sideBar.isPlosive.display = Label(self.sideBar.isPlosive)
        self.sideBar.isPlosive.display["text"] = loc["plosive"]
        self.sideBar.isPlosive.display.pack(side = "right", fill = "x")
        self.sideBar.isPlosive.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.fileButton = Button(self.sideBar)
        self.sideBar.fileButton["text"] = loc["cng_file"]
        self.sideBar.fileButton["command"] = self.onFilechangePress
        self.sideBar.fileButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.finalizeButton = Button(self.sideBar)
        self.sideBar.finalizeButton["text"] = loc["finalize"]
        self.sideBar.finalizeButton["command"] = self.onFinalizePress
        self.sideBar.finalizeButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.okButton = Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = Button(self)
        self.loadButton["text"] = loc["load_other_VB"]
        self.loadButton["command"] = self.onLoadPress
        self.loadButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.phonemeList.list.lastFocusedIndex = None
        
    def onSelectionChange(self, event) -> None:
        """Adjusts the per-phoneme part of the UI to display the correct values when the selected Phoneme in the Phoneme list changes"""

        logging.info("Phonemedict selection change callback")
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            index = self.phonemeList.list.lastFocusedIndex
            key = self.phonemeList.list.lb.get(index)
            self.sideBar.key.variable.set(key)
            if type(loadedVB.phonemeDict[key][0]).__name__ == "AudioSample":
                self.sideBar.expPitch.variable.set(loadedVB.phonemeDict[key][0].expectedPitch)
                self.sideBar.pSearchRange.variable.set(loadedVB.phonemeDict[key][0].searchRange)
                self.sideBar.voicedThrh.variable.set(loadedVB.phonemeDict[key][0].voicedThrh)
                self.sideBar.specSmooth.widthVariable.set(loadedVB.phonemeDict[key][0].specWidth)
                self.sideBar.specSmooth.depthVariable.set(loadedVB.phonemeDict[key][0].specDepth)
                self.sideBar.tempSmooth.widthVariable.set(loadedVB.phonemeDict[key][0].tempWidth)
                self.sideBar.tempSmooth.depthVariable.set(loadedVB.phonemeDict[key][0].tempDepth)
                self.sideBar.isVoiced.variable.set(loadedVB.phonemeDict[key][0].isVoiced)
                self.sideBar.isPlosive.variable.set(loadedVB.phonemeDict[key][0].isPlosive)
                self.enableButtons()
            else:
                self.sideBar.expPitch.variable.set(None)
                self.sideBar.pSearchRange.variable.set(None)
                self.sideBar.voicedThrh.variable.set(None)
                self.sideBar.specSmooth.widthVariable.set(None)
                self.sideBar.specSmooth.depthVariable.set(None)
                self.sideBar.tempSmooth.widthVariable.set(None)
                self.sideBar.tempSmooth.depthVariable.set(None)
                self.sideBar.isVoiced.variable.set(False)
                self.sideBar.isPlosive.variable.set(False)
                self.disableButtons()
            self.updateSlider()
            self.onSliderMove(0)
                
    def onListFocusOut(self, event) -> None:
        """Helper function for retaining information about the last focused element of the Phoneme list when Phoneme list loses entry focus"""

        logging.info("Phonemedict list focus loss callback")
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
        
    def disableButtons(self) -> None:
        """Helper function disabling all per-phoneme settings buttons"""

        self.sideBar.expPitch.entry["state"] = "disabled"
        self.sideBar.pSearchRange.entry["state"] = "disabled"
        self.sideBar.voicedThrh.entry["state"] = "disabled"
        self.sideBar.specSmooth.widthEntry["state"] = "disabled"
        self.sideBar.specSmooth.depthEntry["state"] = "disabled"
        self.sideBar.tempSmooth.widthEntry["state"] = "disabled"
        self.sideBar.tempSmooth.depthEntry["state"] = "disabled"
        self.sideBar.fileButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
        self.sideBar.isVoiced.entry["state"] = "disabled"
        self.sideBar.isPlosive.entry["state"] = "disabled"
    
    def enableButtons(self) -> None:
        """Helper function enabling all per-phoneme settings buttons"""

        self.sideBar.expPitch.entry["state"] = "normal"
        self.sideBar.pSearchRange.entry["state"] = "normal"
        self.sideBar.voicedThrh.entry["state"] = "normal"
        self.sideBar.specSmooth.widthEntry["state"] = "normal"
        self.sideBar.specSmooth.depthEntry["state"] = "normal"
        self.sideBar.tempSmooth.widthEntry["state"] = "normal"
        self.sideBar.tempSmooth.depthEntry["state"] = "normal"
        self.sideBar.fileButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
        self.sideBar.isVoiced.entry["state"] = "normal"
        self.sideBar.isPlosive.entry["state"] = "normal"
    
    def onAddPress(self) -> None:
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
                self.phonemeList.list.lb.insert("end", key)
        
    def onRemovePress(self) -> None:
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

    def onSliderMove(self, value:float) -> None:
        """Updates the phoneme diagram to display the correct time frame when the time slider is moved"""

        logging.info("Phonemedict slider movement callback")
        global loadedVB
        value = int(float(value))
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        spectrum = loadedVB.phonemeDict[key][0].avgSpecharm[int(global_consts.nHarmonics / 2) + 1:] + loadedVB.phonemeDict[key][0].specharm[value, global_consts.nHarmonics + 2:]
        harmonics = loadedVB.phonemeDict[key][0].avgSpecharm[:int(global_consts.nHarmonics / 2) + 1] + loadedVB.phonemeDict[key][0].specharm[value, :int(global_consts.nHarmonics / 2) + 1]
        excitation = torch.abs(loadedVB.phonemeDict[key][0].excitation[value]) * spectrum
        xScale = torch.linspace(0, global_consts.sampleRate / 2, global_consts.halfTripleBatchSize + 1)
        harmScale = torch.linspace(0, global_consts.nHarmonics / 2 * global_consts.sampleRate / loadedVB.phonemeDict[key][0].pitchDeltas[value], int(global_consts.nHarmonics / 2) + 1)
        self.diagram.ax.plot(xScale, excitation.cpu(), label = loc["excitation"], color = "red")
        self.diagram.ax.vlines(harmScale, 0., torch.sqrt(harmonics).cpu(), label = loc["vExcitation"], color = "blue")
        self.diagram.ax.plot(xScale, spectrum.cpu(), label = loc["spectrum"], color = "orange")
        self.diagram.ax.set_xlim([0, global_consts.sampleRate / 2])
        self.diagram.ax.set_xlabel(loc["freq_lbl"], fontsize = 8)
        self.diagram.ax.set_ylabel(loc["amp_lbl"], fontsize = 8)
        self.diagram.ax.legend(loc = "upper right", fontsize = 8)
        self.diagram.canvas.draw()
        self.diagram.ax.clear()

    def updateSlider(self) -> None:
        """replaces the time slider with a new slider object of the same class, but different maximum, when a new sample with possibly different length is selected from the sample list"""

        logging.info("Phonemedict slider properties update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        maxValue = loadedVB.phonemeDict[key][0].specharm.size()[0] - 2
        self.diagram.timeSlider.destroy()
        self.diagram.timeSlider = tkinter.ttk.Scale(self.diagram, from_ = 0, to = maxValue, orient = "horizontal", length = 600, command = self.onSliderMove)
        self.diagram.timeSlider.pack(side = "left", fill = "both", expand = True, padx = 5, pady = 2)
            
    def onKeyChange(self, event) -> None:
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

    def onPitBrdcPress(self) -> None:
        """UI Frontend function for applying/broadcasting the pitch search settings of the currently selected sample to all samples"""

        pitch = self.sideBar.expPitch.variable.get()
        pitchRange = self.sideBar.pSearchRange.variable.get()
        for i in loadedVB.phonemeDict:
            j = loadedVB.phonemeDict[i][0]
            if type(j).__name__ == "AudioSample":
                if (j.expectedPitch != pitch) or (j.searchRange != pitchRange):
                    j.expectedPitch = pitch
                    j.searchRange = pitchRange
                    calculatePitch(j, True)
                    calculateSpectra(j, True)

    def onSpecBrdcPress(self) -> None:
        """UI Frontend function for applying/broadcasting the spectral filtering & analysis settings of the currently selected sample to all samples"""

        newValues = [
            self.sideBar.voicedThrh.variable.get(),
            self.sideBar.specSmooth.widthVariable.get(),
            self.sideBar.specSmooth.depthVariable.get(),
            self.sideBar.tempSmooth.widthVariable.get(),
            self.sideBar.tempSmooth.depthVariable.get(),
        ]
        for i in loadedVB.phonemeDict:
            phoneme = loadedVB.phonemeDict[i][0]
            oldValues = [
                phoneme.voicedThrh,
                phoneme.specWidth,
                phoneme.specDepth,
                phoneme.tempWidth,
                phoneme.tempDepth
            ]
            reset = False
            if type(phoneme).__name__ == "AudioSample":
                for j in range(len(oldValues)):
                    if oldValues[j] != newValues[j]:
                        reset = True
                        break
            if reset:
                phoneme.voicedThrh = newValues[0]
                phoneme.specWidth = newValues[1]
                phoneme.specDepth = newValues[2]
                phoneme.tempWidth = newValues[3]
                phoneme.tempDepth = newValues[4]
                calculateSpectra(phoneme, True)
        self.onSliderMove(self.diagram.timeSlider.get())
        
    def onPitchUpdateTrigger(self, event) -> None:
        """Updates the pitch and phase data of a phoneme"""

        logging.info("Phonemedict pitch update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key][0]).__name__ == "AudioSample":
            if (loadedVB.phonemeDict[key][0].expectedPitch != self.sideBar.expPitch.variable.get()) or (loadedVB.phonemeDict[key][0].searchRange != self.sideBar.pSearchRange.variable.get()):
                loadedVB.phonemeDict[key][0].expectedPitch = self.sideBar.expPitch.variable.get()
                loadedVB.phonemeDict[key][0].searchRange = self.sideBar.pSearchRange.variable.get()
                calculatePitch(loadedVB.phonemeDict[key][0], True)
                calculateSpectra(loadedVB.phonemeDict[key][0], True)
    
    def onVoicedUpdateTrigger(self) -> None:
        """UI Frontend function for updating the "Voiced" flag of a phoneme"""

        logging.info("Phonemedict voicing update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key][0]).__name__ == "AudioSample":
            if loadedVB.phonemeDict[key][0].isVoiced != self.sideBar.isVoiced.variable.get():
                loadedVB.phonemeDict[key][0].isVoiced = self.sideBar.isVoiced.variable.get()
                calculateSpectra(loadedVB.phonemeDict[key][0], True)
                self.onSliderMove(self.diagram.timeSlider.get())

    def onPlosiveUpdateTrigger(self) -> None:
        """UI Frontend function for updating the "Plosive" flag of a phoneme"""

        logging.info("Phonemedict voicing update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key][0]).__name__ == "AudioSample":
            if loadedVB.phonemeDict[key][0].isPlosive != self.sideBar.isPlosive.variable.get():
                loadedVB.phonemeDict[key][0].isPlosive = self.sideBar.isPlosive.variable.get()
                calculateSpectra(loadedVB.phonemeDict[key][0], True)
                self.onSliderMove(self.diagram.timeSlider.get())
        
    def onSpectralUpdateTrigger(self, event) -> None:
        """updates the spectral and excitation data of a phoneme"""

        logging.info("Phonemedict spectral update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        newValues = [
            self.sideBar.voicedThrh.variable.get(),
            self.sideBar.specSmooth.widthVariable.get(),
            self.sideBar.specSmooth.depthVariable.get(),
            self.sideBar.tempSmooth.widthVariable.get(),
            self.sideBar.tempSmooth.depthVariable.get(),
        ]
        phoneme = loadedVB.phonemeDict[key][0]
        oldValues = [
            phoneme.voicedThrh,
            phoneme.specWidth,
            phoneme.specDepth,
            phoneme.tempWidth,
            phoneme.tempDepth
        ]
        reset = False
        if type(phoneme).__name__ == "AudioSample":
            for j in range(len(oldValues)):
                if oldValues[j] != newValues[j]:
                    reset = True
                    break
        if reset:
            phoneme.voicedThrh = newValues[0]
            phoneme.specWidth = newValues[1]
            phoneme.specDepth = newValues[2]
            phoneme.tempWidth = newValues[3]
            phoneme.tempDepth = newValues[4]
            calculateSpectra(phoneme, True)
        self.onSliderMove(self.diagram.timeSlider.get())
        
    def onFilechangePress(self, event) -> None:
        """UI Frontend function for changing the file associated with a phoneme"""

        logging.info("Phonemedict file change button callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")))
        if filepath != "":
            loadedVB.changePhonemeFile(key, filepath)
            calculatePitch(loadedVB.phonemeDict[key][0], True)
            calculateSpectra(loadedVB.phonemeDict[key][0], True)
            self.phonemeList.list.lb.delete(index)
            self.phonemeList.list.lb.insert(index, key)
            
        
    def onFinalizePress(self) -> None:
        """UI Frontend function for finalizing a phoneme, reducing its size when saving the VOicebank to a file"""

        logging.info("Phonemedict finalize button callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        loadedVB.finalizePhoneme(key)
        self.sideBar.expPitch.variable.set(None)
        self.sideBar.pSearchRange.variable.set(None)
        self.sideBar.voicedThrh.variable.set(None)
        self.sideBar.specSmooth.widthVariable.set(None)
        self.sideBar.specSmooth.depthVariable.set(None)
        self.sideBar.tempSmooth.widthVariable.set(None)
        self.sideBar.tempSmooth.depthVariable.set(None)
        self.disableButtons()
        
    def onOkPress(self) -> None:
        """Updates the last selected phoneme and closes the Phoneme Dict UI window when the OK button is pressed"""

        logging.info("Phonemedict OK button callback")
        global loadedVB
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            if index != None:
                key = self.phonemeList.list.lb.get(index)
                self.onKeyChange(None)
                if type(loadedVB.phonemeDict[key][0]).__name__ == "AudioSample":
                    self.onPitchUpdateTrigger(None)
                    self.onSpectralUpdateTrigger(None)
        self.master.destroy()

    def onLoadPress(self) -> None:
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
