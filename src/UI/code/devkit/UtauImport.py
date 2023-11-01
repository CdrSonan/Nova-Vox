#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import tkinter
import torch
import sys
import os
import csv
import sounddevice
from tqdm.auto import tqdm

import matplotlib
#set up matplotlib to use more efficient plotting settings
matplotlib.use('TkAgg')
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from UI.code.devkit.Widgets import *
from Backend.DataHandler.UtauSample import UtauSample
from Backend.UtauImport import fetchSamples
import global_consts
from MiddleLayer.IniParser import readSettings
from Localization.devkit_localization import getLanguage
loc = getLanguage()

class UtauImportUi(Frame):
    """Class of the UTAU import UI window"""

    def __init__(self, master=None) -> None:
        logging.info("Initializing UTAU import UI")
        global loadedVB
        from UI.code.devkit.Main import loadedVB
        Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.phoneticsPath = os.path.join(readSettings()["datadir"], "Devkit_Phonetics")
        self.createWidgets()
        self.master.wm_title(loc["utau_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("assets/icon/nova-vox-logo-black.ico")
        self.sampleList = []
        self.playing = False
        
    def createWidgets(self) -> None:
        """Initializes all widgets of the UTAU import UI window."""

        global loadedVB

        #Audio file waveform diagram
        self.diagram = LabelFrame(self, text = loc["utau_diag_lbl"])
        self.diagram.fig = Figure(figsize=(4, 4))
        self.diagram.ax = self.diagram.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.diagram.ax.set_xlim([0, global_consts.sampleRate / 2])
        self.diagram.ax.set_xlabel(loc["time_lbl"], fontsize = 8)
        self.diagram.ax.set_ylabel(loc["amp_lbl"], fontsize = 8)
        self.diagram.canvas = FigureCanvasTkAgg(self.diagram.fig, self.diagram)
        self.diagram.canvas.get_tk_widget().pack(side = "top", fill = "both", expand = True)
        self.diagram.playButton = Button(self.diagram)
        self.diagram.playButton["text"] = loc["play"]
        self.diagram.playButton["command"] = self.play
        self.diagram.playButton.config(width = 20)
        self.diagram.playButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        self.diagram.pack(side = "right", fill = "y", padx = 5, pady = 2)
        
        #UTAU phoneme list and related controls
        self.phonemeList = LabelFrame(self, text = loc["smp_list"])
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
        self.phonemeList.removeButton = Button(self.phonemeList)
        self.phonemeList.removeButton["text"] = loc["remove"]
        self.phonemeList.removeButton["command"] = self.onRemovePress
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = Button(self.phonemeList)
        self.phonemeList.addButton["text"] = loc["add"]
        self.phonemeList.addButton["command"] = self.onAddPress
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        #per-sample settings panel
        self.sideBar = LabelFrame(self, text = loc["per_smp_set"])
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        #sample type selection
        self.sideBar._type = Frame(self.sideBar)
        self.sideBar._type.variable = tkinter.BooleanVar(self.sideBar._type)
        self.sideBar._type.entry = Frame(self.sideBar._type)
        self.sideBar._type.entry.button1 = Radiobutton(self.sideBar._type.entry, text = loc["smp_phoneme"], value = 0, variable = self.sideBar._type.variable, command = self.onTypeChange)
        self.sideBar._type.entry.button1.pack(side = "right", fill = "x")
        self.sideBar._type.entry.button2 = Radiobutton(self.sideBar._type.entry, text = loc["smp_transition"], value = 1, variable = self.sideBar._type.variable, command = self.onTypeChange)
        self.sideBar._type.entry.button2.pack(side = "right", fill = "x")
        self.sideBar._type.entry.button3 = Radiobutton(self.sideBar._type.entry, text = loc["smp_sequence"], value = 2, variable = self.sideBar._type.variable, command = self.onTypeChange)
        self.sideBar._type.entry.button3.pack(side = "right", fill = "x")
        self.sideBar._type.entry.pack(side = "right", fill = "x")
        self.sideBar._type.display = Label(self.sideBar._type)
        self.sideBar._type.display["text"] = loc["smpl_type"]
        self.sideBar._type.display.pack(side = "right", fill = "x")
        self.sideBar._type.pack(side = "top", fill = "x", padx = 5, pady = 2)

        #key selection for phoneme samples
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
        
        #start point selection
        self.sideBar.start = Frame(self.sideBar)
        self.sideBar.start.variable = tkinter.DoubleVar(self.sideBar.start)
        self.sideBar.start.entry = Spinbox(self.sideBar.start, from_ = 0, to = None, increment = 0.05)
        self.sideBar.start.entry["textvariable"] = self.sideBar.start.variable
        self.sideBar.start.entry.bind("<FocusOut>", self.onFrameUpdateTrigger)
        self.sideBar.start.entry.bind("<KeyRelease-Return>", self.onFrameUpdateTrigger)
        self.sideBar.start.entry.pack(side = "right", fill = "x")
        self.sideBar.start.display = Label(self.sideBar.start)
        self.sideBar.start.display["text"] = loc["start"]
        self.sideBar.start.display.pack(side = "right", fill = "x")
        self.sideBar.start.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        #end point selection
        self.sideBar.end = Frame(self.sideBar)
        self.sideBar.end.variable = tkinter.DoubleVar(self.sideBar.end)
        self.sideBar.end.entry = Spinbox(self.sideBar.end, from_ = 0, to = None, increment = 0.05)
        self.sideBar.end.entry["textvariable"] = self.sideBar.end.variable
        self.sideBar.end.entry.bind("<FocusOut>", self.onFrameUpdateTrigger)
        self.sideBar.end.entry.bind("<KeyRelease-Return>", self.onFrameUpdateTrigger)
        self.sideBar.end.entry.pack(side = "right", fill = "x")
        self.sideBar.end.display = Label(self.sideBar.end)
        self.sideBar.end.display["text"] = loc["end"]
        self.sideBar.end.display.pack(side = "right", fill = "x")
        self.sideBar.end.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        #sample import button
        self.sideBar.importButton = Button(self.sideBar)
        self.sideBar.importButton["text"] = loc["smp_import"]
        self.sideBar.importButton["command"] = self.onImportPress
        self.sideBar.importButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        #oto.ini parser settings panel
        self.otoSettings = LabelFrame(self, text = loc["oto_set"])
        self.otoSettings.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)

        #prefix selection
        self.otoSettings.stripPrefix = Frame(self.otoSettings)
        self.otoSettings.stripPrefix.variable = tkinter.StringVar(self.otoSettings.stripPrefix)
        self.otoSettings.stripPrefix.entry = Entry(self.otoSettings.stripPrefix)
        self.otoSettings.stripPrefix.entry["textvariable"] = self.otoSettings.stripPrefix.variable
        self.otoSettings.stripPrefix.entry.pack(side = "right", fill = "x")
        self.otoSettings.stripPrefix.display = Label(self.otoSettings.stripPrefix)
        self.otoSettings.stripPrefix.display["text"] = loc["stripPrefix"]
        self.otoSettings.stripPrefix.display.pack(side = "right", fill = "x")
        self.otoSettings.stripPrefix.pack(side = "top", fill = "x", padx = 5, pady = 2)

        #postfix selection
        self.otoSettings.stripPostfix = Frame(self.otoSettings)
        self.otoSettings.stripPostfix.variable = tkinter.StringVar(self.otoSettings.stripPostfix)
        self.otoSettings.stripPostfix.entry = Entry(self.otoSettings.stripPostfix)
        self.otoSettings.stripPostfix.entry["textvariable"] = self.otoSettings.stripPostfix.variable
        self.otoSettings.stripPostfix.entry.pack(side = "right", fill = "x")
        self.otoSettings.stripPostfix.display = Label(self.otoSettings.stripPostfix)
        self.otoSettings.stripPostfix.display["text"] = loc["stripPostfix"]
        self.otoSettings.stripPostfix.display.pack(side = "right", fill = "x")
        self.otoSettings.stripPostfix.pack(side = "top", fill = "x", padx = 5, pady = 2)

        #NV expression selection
        self.otoSettings.addExpr = Frame(self.otoSettings)
        self.otoSettings.addExpr.variable = tkinter.StringVar(self.otoSettings.addExpr)
        self.otoSettings.addExpr.entry = Entry(self.otoSettings.addExpr)
        self.otoSettings.addExpr.entry["textvariable"] = self.otoSettings.addExpr.variable
        self.otoSettings.addExpr.entry.pack(side = "right", fill = "x")
        self.otoSettings.addExpr.display = Label(self.otoSettings.addExpr)
        self.otoSettings.addExpr.display["text"] = loc["addExpr"]
        self.otoSettings.addExpr.display.pack(side = "right", fill = "x")
        self.otoSettings.addExpr.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        #pitch selection
        self.otoSettings.pitch = Frame(self.otoSettings)
        self.otoSettings.pitch.variable = tkinter.DoubleVar(self.otoSettings.pitch)
        self.otoSettings.pitch.entry = Spinbox(self.otoSettings.pitch, from_ = 60, to = 1200, increment = 1)
        self.otoSettings.pitch.entry["textvariable"] = self.otoSettings.pitch.variable
        self.otoSettings.pitch.entry.pack(side = "right", fill = "x")
        self.otoSettings.pitch.display = Label(self.otoSettings.pitch)
        self.otoSettings.pitch.display["text"] = loc["pitch"]
        self.otoSettings.pitch.display.pack(side = "right", fill = "x")
        self.otoSettings.pitch.pack(side = "top", fill = "x", padx = 5, pady = 2)

        #language selection
        languages = os.listdir(os.path.join(self.phoneticsPath, "Lists"))
        self.otoSettings.language = Frame(self.otoSettings)
        self.otoSettings.language.variable = tkinter.StringVar(self.otoSettings.language)
        self.otoSettings.language.entry = tkinter.ttk.OptionMenu(self.otoSettings.language, self.otoSettings.language.variable, *languages)
        self.otoSettings.language.entry.pack(side = "right", fill = "x")
        self.otoSettings.language.display = Label(self.otoSettings.language)
        self.otoSettings.language.display["text"] = loc["phon_def"]
        self.otoSettings.language.display.pack(side = "right", fill = "x")
        self.otoSettings.language.pack(side = "top", fill = "x", padx = 5, pady = 2)

        #shift-JIS decoder flag
        self.otoSettings.forceJIS = Frame(self.otoSettings)
        self.otoSettings.forceJIS.variable = tkinter.BooleanVar(self.otoSettings.forceJIS, True)
        self.otoSettings.forceJIS.entry = Checkbutton(self.otoSettings.forceJIS)
        self.otoSettings.forceJIS.entry["variable"] = self.otoSettings.forceJIS.variable
        self.otoSettings.forceJIS.entry.pack(side = "right", fill = "x")
        self.otoSettings.forceJIS.display = Label(self.otoSettings.forceJIS)
        self.otoSettings.forceJIS.display["text"] = loc["force_jis"]
        self.otoSettings.forceJIS.display.pack(side = "right", fill = "x")
        self.otoSettings.forceJIS.pack(side = "top", fill = "x", padx = 5, pady = 2)

        #per-type sample exclusions
        self.otoSettings.typeSelector = Frame(self.otoSettings)
        self.otoSettings.typeSelector.variablePhon = tkinter.BooleanVar(self.otoSettings.typeSelector, True)
        self.otoSettings.typeSelector.entryPhon = Checkbutton(self.otoSettings.typeSelector)
        self.otoSettings.typeSelector.entryPhon["variable"] = self.otoSettings.typeSelector.variablePhon
        self.otoSettings.typeSelector.entryPhon.pack(side = "right", fill = "x")
        self.otoSettings.typeSelector.displayPhon = Label(self.otoSettings.typeSelector)
        self.otoSettings.typeSelector.displayPhon["text"] = loc["sample_phoneme"]
        self.otoSettings.typeSelector.displayPhon.pack(side = "right", fill = "x")
        self.otoSettings.typeSelector.variableTrans = tkinter.BooleanVar(self.otoSettings.typeSelector, True)
        self.otoSettings.typeSelector.entryTrans = Checkbutton(self.otoSettings.typeSelector)
        self.otoSettings.typeSelector.entryTrans["variable"] = self.otoSettings.typeSelector.variableTrans
        self.otoSettings.typeSelector.entryTrans.pack(side = "right", fill = "x")
        self.otoSettings.typeSelector.displayTrans = Label(self.otoSettings.typeSelector)
        self.otoSettings.typeSelector.displayTrans["text"] = loc["sample_transition"]
        self.otoSettings.typeSelector.displayTrans.pack(side = "right", fill = "x")
        self.otoSettings.typeSelector.variableSeq = tkinter.BooleanVar(self.otoSettings.typeSelector, True)
        self.otoSettings.typeSelector.entrySeq = Checkbutton(self.otoSettings.typeSelector)
        self.otoSettings.typeSelector.entrySeq["variable"] = self.otoSettings.typeSelector.variableSeq
        self.otoSettings.typeSelector.entrySeq.pack(side = "right", fill = "x")
        self.otoSettings.typeSelector.displaySeq = Label(self.otoSettings.typeSelector)
        self.otoSettings.typeSelector.displaySeq["text"] = loc["sample_sequence"]
        self.otoSettings.typeSelector.displaySeq.pack(side = "right", fill = "x")
        self.otoSettings.typeSelector.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        #window controls
        self.okButton = Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = Button(self)
        self.loadButton["text"] = loc["load_oto"]
        self.loadButton["command"] = self.onLoadPress
        self.loadButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.importButton = Button(self)
        self.importButton["text"] = loc["all_import"]
        self.importButton["command"] = self.onImportAllPress
        self.importButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.phonemeList.list.lastFocusedIndex = None

    def play(self) -> None:
        """plays an UTAU sample from its configured start point to end point"""

        sample = self.sampleList[self.phonemeList.list.lastFocusedIndex]
        wave = sample.audioSample.waveform[int(sample.start * global_consts.sampleRate / 1000):int(sample.end * global_consts.sampleRate / 1000)]
        sounddevice.play(wave, samplerate=global_consts.sampleRate)
        
    def onSelectionChange(self, event) -> None:
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

                
    def onListFocusOut(self, event) -> None:
        """Helper function for retaining information about the last focused element of the SampleList when the SampleList loses entry focus"""

        logging.info("UTAU sample list focus loss callback")
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
        
    def disableButtons(self) -> None:
        """Helper function disabling the key button"""

        self.sideBar.key.entry["state"] = "disabled"
    
    def enableButtons(self) -> None:
        """Helper function enabling the key button"""

        self.sideBar.key.entry["state"] = "normal"

    def onTypeChange(self, event = None) -> None:
        """UI Frontend function for changing the type (phoneme or transition) of a sample"""

        logging.info("Utau sample type change callback")
        index = self.phonemeList.list.lastFocusedIndex
        self.sampleList[index]._type = self.sideBar._type.variable.get()
        if self.sideBar._type.variable.get():
            self.disableButtons()
        else:
            self.enableButtons()
    
    def onAddPress(self) -> None:
        """UI Frontend function for adding a sample to the SampleList"""

        logging.info("UTAU sample add button callback")
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")))
        if filepath != "":
            for s in self.sampleList:
                if filepath == s.audioSample.filepath:
                    sample = UtauSample(filepath, 1, None, 0, None, s.offset, s.fixed, s.blank, s.preuttr, s.overlap, True, False, 0)
                    break
            else:
                sample = UtauSample(filepath, 1, None, 0, None, 0, 0, 0, 0, 0, True, False, 0)
            self.sampleList.append(sample)
            self.phonemeList.list.lb.insert("end", sample.handle)
        
    def onRemovePress(self) -> None:
        """UI Frontend function for removing a sample from the SampleList"""

        logging.info("UTAU sample remove button callback")
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            del self.sampleList[index]
            self.phonemeList.list.lb.delete(index)
            if index == self.phonemeList.list.lb.size() - 1:
                self.phonemeList.list.lb.selection_set(index - 1)
            else:
                self.phonemeList.list.lb.selection_set(index)
            self.updateDiagram()

    def updateDiagram(self) -> None:
        """updates the diagram displaying sample waveform, UTAU and Nova-Vox timing markers, when a new sample is selected from the list"""

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
        self.diagram.ax.plot(xScale, waveform, label = loc["waveform"], color = (0., 0.5, 1.), alpha = 0.75, solid_joinstyle = "bevel", markevery=100)
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
            
    def onKeyChange(self, event) -> None:
        """UI Frontend function for changing the key of a phoneme-type UTAU sample"""

        logging.info("UTAU sample key change callback")
        index = self.phonemeList.list.lastFocusedIndex
        key = self.sampleList[index].key
        newKey = self.sideBar.key.variable.get()
        if key != newKey:
            self.sampleList[index].key = newKey
        
    def onFrameUpdateTrigger(self, event) -> None:
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
        
    def onImportPress(self) -> None:
        """UI Frontend function for importing an UTAU sample from the SampleList as phoneme or transition sample"""

        logging.info("UTAU import button callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        if self.sampleList[index]._type == 0:
            loadedVB.addPhonemeUtau(self.sampleList[index])
        elif self.sampleList[0]._type == 2:
            loadedVB.addMainTrainSampleUtau(self.sampleList[index])
        else:
            loadedVB.addTrTrainSampleUtau(self.sampleList[index])
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

    def onImportAllPress(self) -> None:
        """UI Frontend function for importing the entire SampleList as phoneme and transition samples"""

        logging.info("UTAU import all button callback")
        global loadedVB
        sampleCount = len(self.sampleList)
        for i in tqdm(range(sampleCount), unit = "samples"):
            self.update()
            if self.sampleList[0]._type == 0:
                loadedVB.addPhonemeUtau(self.sampleList[0])
            elif self.sampleList[0]._type == 2:
                loadedVB.addMainTrainSampleUtau(self.sampleList[0])
            else:
                loadedVB.addTrTrainSampleUtau(self.sampleList[0])
            del self.sampleList[0]
            self.phonemeList.list.lb.delete(0)
        self.update()
        self.sideBar._type.variable.set(0)
        self.sideBar.key.variable.set(None)
        self.sideBar.start.variable.set(None)
        self.sideBar.end.variable.set(None)
        self.phonemeList.list.lb.selection_set(0)
        
    def onOkPress(self) -> None:
        """Updates the last selected sample and closes the UTAU import UI window when the OK button is pressed"""

        logging.info("UTAU OK button callback")
        global loadedVB
        if self.phonemeList.list.lb.size() > 0:
            index = self.phonemeList.list.lastFocusedIndex
            if index != None:
                self.onKeyChange(None)
                self.onFrameUpdateTrigger(None)
        self.master.destroy()

    def onLoadPress(self) -> None:
        """UI Frontend function for parsing an oto.ini file, and loading its samples into the SampleList"""
        
        logging.info("oto.ini load button callback")
        global loadedVB
        phonemePath = os.path.join(self.phoneticsPath, "Lists", self.otoSettings.language.variable.get())
        if os.path.isfile(os.path.join(self.phoneticsPath, "UtauConversions", self.otoSettings.language.variable.get())):
            conversionPath = os.path.join(self.phoneticsPath, "UtauConversions", self.otoSettings.language.variable.get())
        else:
            conversionPath = None
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc["oto.ini_desc"], ".ini"), (loc["all_files_desc"], "*")))
        if filepath != "":
            f = open(filepath, "r", encoding = "Shift_JIS")
            reader = csv.reader(f, delimiter = ",")
            f.close()
            i = 0

            file = open(filepath, encoding = "Shift_JIS")
            reader = csv.reader(file, delimiter = "=")
            length = len(list(reader))
            file.seek(0)
            otoPath = os.path.split(filepath)[0]
            samplePaths = []
            for row in tqdm(reader, total = length, unit = "oto defs"):
                #tqdm.write("processing " + str(row))
                i += 1
                self.update()
                filename = row[0]
                properties = row[1].split(",")
                occuredError = None
                try:
                    fetchedSamples = fetchSamples(filename,
                                                  properties,
                                                  otoPath,
                                                  self.otoSettings.stripPrefix.variable.get(),
                                                  self.otoSettings.stripPostfix.variable.get(),
                                                  self.otoSettings.addExpr.variable.get(),
                                                  self.otoSettings.pitch.variable.get(),
                                                  phonemePath,
                                                  conversionPath,
                                                  self.otoSettings.forceJIS.variable.get())
                    for sample in fetchedSamples:
                        if sample._type == 1 and self.otoSettings.typeSelector.variableTrans.get():
                            #transition sample
                            self.sampleList.append(sample)
                            self.phonemeList.list.lb.insert("end", sample.handle)
                        elif sample._type == 2 and sample.audioSample.filepath not in samplePaths and self.otoSettings.typeSelector.variableSeq.get():
                            #sequence sample
                            samplePaths.append(sample.audioSample.filepath)
                            self.sampleList.append(sample)
                            self.phonemeList.list.lb.insert("end", sample.handle)
                        elif self.otoSettings.typeSelector.variablePhon.get():
                            #phoneme sample
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
                    tqdm.write(repr(error))
                    if occuredError == None:
                        occuredError = 0
                    logging.warning(error)
                except Exception as error:
                    tqdm.write(error)
                    occuredError = 1
                    logging.error(error)
            self.update()
            if occuredError == 0:
                tkinter.messagebox.showinfo(loc["info"], loc["oto_msng_phn"])
            elif occuredError == 1:
                tkinter.messagebox.showerror(loc["error"], loc["oto_error"])
