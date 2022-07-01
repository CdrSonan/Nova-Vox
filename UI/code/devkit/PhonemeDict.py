import logging
import tkinter
import torch
import sys

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import global_consts
from Locale.devkit_locale import getLocale
loc = getLocale()
from Backend.ESPER.PitchCalculator import calculatePitch
from Backend.ESPER.SpectralCalculator import calculateSpectra

class PhonemedictUi(tkinter.Frame):
    """Class of the phoneme dictionnary UI window"""

    def __init__(self, master=None):
        logging.info("Initializing Phonemedict UI")
        global loadedVB
        from UI.code.devkit.Main import loadedVB
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
        self.sideBar.pSearchRange.entry = tkinter.Spinbox(self.sideBar.pSearchRange, from_ = 0.35, to = 0.95, increment = 0.05)
        self.sideBar.pSearchRange.entry["textvariable"] = self.sideBar.pSearchRange.variable
        self.sideBar.pSearchRange.entry.bind("<FocusOut>", self.onPitchUpdateTrigger)
        self.sideBar.pSearchRange.entry.bind("<KeyRelease-Return>", self.onPitchUpdateTrigger)
        self.sideBar.pSearchRange.entry.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.display = tkinter.Label(self.sideBar.pSearchRange)
        self.sideBar.pSearchRange.display["text"] = loc["psearchr"]
        self.sideBar.pSearchRange.display.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.pBroadcastButton = tkinter.Button(self.sideBar)
        self.sideBar.pBroadcastButton["text"] = loc["pit_brdc"]
        self.sideBar.pBroadcastButton["command"] = self.onPitBrdcPress
        self.sideBar.pBroadcastButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.voicedFilter = tkinter.Frame(self.sideBar)
        self.sideBar.voicedFilter.variable = tkinter.IntVar(self.sideBar.voicedFilter)
        self.sideBar.voicedFilter.entry = tkinter.Spinbox(self.sideBar.voicedFilter, from_ = 1, to = 50)
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

        self.sideBar.sBroadcastButton = tkinter.Button(self.sideBar)
        self.sideBar.sBroadcastButton["text"] = loc["spec_brdc"]
        self.sideBar.sBroadcastButton["command"] = self.onSpecBrdcPress
        self.sideBar.sBroadcastButton.pack(side = "top", fill = "x", expand = True, padx = 5)

        self.sideBar.isVoiced = tkinter.Frame(self.sideBar)
        self.sideBar.isVoiced.variable = tkinter.BooleanVar(self.sideBar.isVoiced, True)
        self.sideBar.isVoiced.entry = tkinter.Checkbutton(self.sideBar.isVoiced)
        self.sideBar.isVoiced.entry["variable"] = self.sideBar.isVoiced.variable
        self.sideBar.isVoiced.entry["command"] = self.onVoicedUpdateTrigger
        self.sideBar.isVoiced.entry.pack(side = "right", fill = "x")
        self.sideBar.isVoiced.display = tkinter.Label(self.sideBar.isVoiced)
        self.sideBar.isVoiced.display["text"] = loc["voiced"]
        self.sideBar.isVoiced.display.pack(side = "right", fill = "x")
        self.sideBar.isVoiced.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.sideBar.isPlosive = tkinter.Frame(self.sideBar)
        self.sideBar.isPlosive.variable = tkinter.BooleanVar(self.sideBar.isPlosive, True)
        self.sideBar.isPlosive.entry = tkinter.Checkbutton(self.sideBar.isPlosive)
        self.sideBar.isPlosive.entry["variable"] = self.sideBar.isPlosive.variable
        self.sideBar.isPlosive.entry["command"] = self.onPlosiveUpdateTrigger
        self.sideBar.isPlosive.entry.pack(side = "right", fill = "x")
        self.sideBar.isPlosive.display = tkinter.Label(self.sideBar.isPlosive)
        self.sideBar.isPlosive.display["text"] = loc["plosive"]
        self.sideBar.isPlosive.display.pack(side = "right", fill = "x")
        self.sideBar.isPlosive.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
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
                self.sideBar.isVoiced.variable.set(loadedVB.phonemeDict[key].isVoiced)
                self.sideBar.isPlosive.variable.set(loadedVB.phonemeDict[key].isPlosive)
                self.enableButtons()
            else:
                self.sideBar.expPitch.variable.set(None)
                self.sideBar.pSearchRange.variable.set(None)
                self.sideBar.voicedFilter.variable.set(None)
                self.sideBar.unvoicedIter.variable.set(None)
                self.sideBar.isVoiced.variable.set(False)
                self.sideBar.isPlosive.variable.set(False)
                self.disableButtons()
            self.updateSlider()
            self.onSliderMove(0)
                
    def onListFocusOut(self, event):
        """Helper function for retaining information about the last focused element of the Phoneme list when Phoneme list loses entry focus"""

        logging.info("Phonemedict list focus loss callback")
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
        
    def disableButtons(self):
        """Helper function disabling all per-phoneme settings buttons"""

        self.sideBar.expPitch.entry["state"] = "disabled"
        self.sideBar.pSearchRange.entry["state"] = "disabled"
        self.sideBar.voicedFilter.entry["state"] = "disabled"
        self.sideBar.unvoicedIter.entry["state"] = "disabled"
        self.sideBar.fileButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
        self.sideBar.isVoiced.entry["state"] = "disabled"
        self.sideBar.isPlosive.entry["state"] = "disabled"
    
    def enableButtons(self):
        """Helper function enabling all per-phoneme settings buttons"""

        self.sideBar.expPitch.entry["state"] = "normal"
        self.sideBar.pSearchRange.entry["state"] = "normal"
        self.sideBar.voicedFilter.entry["state"] = "normal"
        self.sideBar.unvoicedIter.entry["state"] = "normal"
        self.sideBar.fileButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
        self.sideBar.isVoiced.entry["state"] = "normal"
        self.sideBar.isPlosive.entry["state"] = "normal"
    
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
        """Updates the phoneme diagram to display the correct time frame when the time slider is moved"""

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
        """replaces the time slider with a new slider object of the same class, but different maximum, when a new sample with possibly different length is selected from the sample list"""

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

    def onPitBrdcPress(self):
        """UI Frontend function for applying/broadcasting the pitch search settings of the currently selected sample to all samples"""

        pitch = self.sideBar.expPitch.variable.get()
        pitchRange = self.sideBar.pSearchRange.variable.get()
        for i in loadedVB.phonemeDict:
            j = loadedVB.phonemeDict[i]
            if type(j).__name__ == "AudioSample":
                if (j.expectedPitch != pitch) or (j.searchRange != pitchRange):
                    j.expectedPitch = pitch
                    j.searchRange = pitchRange
                    calculatePitch(j)
                    calculateSpectra(j)

    def onSpecBrdcPress(self):
        """UI Frontend function for applying/broadcasting the spectral filtering & analysis settings of the currently selected sample to all samples"""

        voicedFilter = self.sideBar.voicedFilter.variable.get()
        unvoicedIter = self.sideBar.unvoicedIter.variable.get()
        for i in loadedVB.phonemeDict:
            j = loadedVB.phonemeDict[i]
            if type(j).__name__ == "AudioSample":
                if (j.voicedFilter != voicedFilter) or (j.unvoicedIterations != unvoicedIter):
                    j.voicedFilter = voicedFilter
                    j.unvoicedIterations = unvoicedIter
                    calculateSpectra(j)
        self.onSliderMove(self.diagram.timeSlider.get())
        
    def onPitchUpdateTrigger(self, event):
        """Updates the pitch and phase data of a phoneme"""

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
    
    def onVoicedUpdateTrigger(self):
        """UI Frontend function for updating the "Voiced" flag of a phoneme"""

        logging.info("Phonemedict voicing update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
            if loadedVB.phonemeDict[key].isVoiced != self.sideBar.isVoiced.variable.get():
                loadedVB.phonemeDict[key].isVoiced = self.sideBar.isVoiced.variable.get()
                calculateSpectra(loadedVB.phonemeDict[key])
                self.onSliderMove(self.diagram.timeSlider.get())

    def onPlosiveUpdateTrigger(self):
        """UI Frontend function for updating the "Plosive" flag of a phoneme"""

        logging.info("Phonemedict voicing update callback")
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
            if loadedVB.phonemeDict[key].isPlosive != self.sideBar.isPlosive.variable.get():
                loadedVB.phonemeDict[key].isPlosive = self.sideBar.isPlosive.variable.get()
                calculateSpectra(loadedVB.phonemeDict[key])
                self.onSliderMove(self.diagram.timeSlider.get())
        
    def onSpectralUpdateTrigger(self, event):
        """updates the spectral and excitation data of a phoneme"""

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
        """UI Frontend function for finalizing a phoneme, reducing its size when saving the VOicebank to a file"""

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
