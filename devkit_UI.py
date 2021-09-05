# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:51:29 2021

@author: CdrSonan
"""

import tkinter
import tkinter.filedialog
import tkinter.simpledialog

import global_consts
import devkit_pipeline
import devkit_locale
loc = devkit_locale.getLocale()

class RootUi(tkinter.Frame):
    """Class of the Devkit main window"""
    def __init__(self, master=tkinter.Tk()):
        """Initialize a new main window. Called once during devkit startup"""
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["no_vb"])
        self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
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
        self.metadataUi = MetadataUi(tkinter.Tk())
        self.metadataUi.mainloop()
    
    def onPhonemedictPress(self):
        """opens Phoneme Dict UI window when Phoneme Dict button in the main window is pressed"""
        self.phonemedictUi = PhonemedictUi(tkinter.Tk())
        self.phonemedictUi.mainloop()
    
    def onCrfaiPress(self):
        """opens Phoneme Crossfade AI UI window when Phoneme Crossfade AI button in the main window is pressed"""
        self.crfaiUi = CrfaiUi(tkinter.Tk())
        self.crfaiUi.mainloop()
    
    def onParameterPress(self):
        pass
    
    def onWorddictPress(self):
        pass

    def onDestroy(self, event):
        if hasattr(self, 'metadataUi'):
            self.metadataUi.master.destroy()
        if hasattr(self, 'phonemedictUi'):
            self.phonemedictUi.master.destroy()
        if hasattr(self, 'crfaiUi'):
            self.crfaiUi.master.destroy()
    
    def onSavePress(self):
        """Saves the currently loaded Voicebank to a .nvvb file"""
        global loadedVB
        global loadedVBPath
        filepath = tkinter.filedialog.asksaveasfilename(defaultextension = ".nvvb", filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")), initialfile = loadedVBPath)
        if filepath != "":
            loadedVBPath = filepath
            loadedVB.save(filepath)
            self.master.wm_title(loadedVBPath)
        
    def onOpenPress(self):
        """opens a Voicebank and loads all of its data"""
        global loadedVB
        global loadedVBPath
        if "loadedVB" not in globals():
            loadedVBPath = None
        if ("loadedVB" not in globals()) or tkinter.messagebox.askokcancel(loc["warning"], loc["vb_discard_msg"], icon = "warning"):
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")), initialfile = loadedVBPath)
            if filepath != "":
                loadedVBPath = filepath
                loadedVB = devkit_pipeline.Voicebank(filepath)
                self.metadataButton["state"] = "active"
                self.phonemedictButton["state"] = "active"
                self.crfaiButton["state"] = "active"
                self.parameterButton["state"] = "active"
                self.worddictButton["state"] = "active"
                self.saveButton["state"] = "active"
                self.master.wm_title(loadedVBPath)
    
    def onNewPress(self):
        """creates a new, empty Voicebank object in memory"""
        global loadedVB
        if ("loadedVB" not in globals()) or tkinter.messagebox.askokcancel(loc["warning"], loc["vb_discard_msg"], icon = "warning"):
            loadedVB = devkit_pipeline.Voicebank(None)
            self.metadataButton["state"] = "active"
            self.phonemedictButton["state"] = "active"
            self.crfaiButton["state"] = "active"
            self.parameterButton["state"] = "active"
            self.worddictButton["state"] = "active"
            self.saveButton["state"] = "active"
            self.master.wm_title(loc["unsaved_vb"])
            
class MetadataUi(tkinter.Frame):
    """Class of the Metadata window"""
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20)
        self.createWidgets()
        self.master.wm_title(loc["metadat_lbl"])
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
        """Applies all changes and closes the window when the OK button is pressed"""
        global loadedVB
        loadedVB.metadata.name = self.name.variable.get()
        loadedVB.metadata.sampleRate = global_consts.sampleRate
        self.master.destroy()

    def onLoadPress(self):
        """Opens a file browser, and loads the Voicebank metadata from a specified .nvvb file"""
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
        loadedVB.loadMetadata(filepath)
        self.name.variable.set(loadedVB.metadata.name)
        
class PhonemedictUi(tkinter.Frame):
    """Class of the phoneme dictionnary UI window"""
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["phon_lbl"])
        self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        self.disableButtons()
        
    def createWidgets(self):
        """Initializes all widgets of the Phoneme Dict UI window."""
        global loadedVB
        
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
        
        self.sideBar.voicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.voicedIter.variable = tkinter.IntVar(self.sideBar.voicedIter)
        self.sideBar.voicedIter.entry = tkinter.Spinbox(self.sideBar.voicedIter, from_ = 0, to = 10)
        self.sideBar.voicedIter.entry["textvariable"] = self.sideBar.voicedIter.variable
        self.sideBar.voicedIter.entry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.voicedIter.entry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.voicedIter.entry.pack(side = "right", fill = "x")
        self.sideBar.voicedIter.display = tkinter.Label(self.sideBar.voicedIter)
        self.sideBar.voicedIter.display["text"] = loc["viter"]
        self.sideBar.voicedIter.display.pack(side = "right", fill = "x")
        self.sideBar.voicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.unvoicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.unvoicedIter.variable = tkinter.IntVar(self.sideBar.unvoicedIter)
        self.sideBar.unvoicedIter.entry = tkinter.Spinbox(self.sideBar.unvoicedIter, from_ = 0, to = 100)
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
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            index = self.phonemeList.list.lastFocusedIndex
            key = self.phonemeList.list.lb.get(index)
            self.sideBar.key.variable.set(key)
            if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
                self.sideBar.expPitch.variable.set(loadedVB.phonemeDict[key].expectedPitch)
                self.sideBar.pSearchRange.variable.set(loadedVB.phonemeDict[key].searchRange)
                self.sideBar.voicedIter.variable.set(loadedVB.phonemeDict[key].voicedIterations)
                self.sideBar.unvoicedIter.variable.set(loadedVB.phonemeDict[key].unvoicedIterations)
                self.enableButtons()
            else:
                self.sideBar.expPitch.variable.set(None)
                self.sideBar.pSearchRange.variable.set(None)
                self.sideBar.voicedIter.variable.set(None)
                self.sideBar.unvoicedIter.variable.set(None)
                self.disableButtons()
                
    def onListFocusOut(self, event):
        """Helper function for retaining information about the last focused element of the Phoneme list when Phoneme list loses entry focus"""
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
        
    def disableButtons(self):
        """Disables the per-phoneme settings buttons"""
        self.sideBar.expPitch.entry["state"] = "disabled"
        self.sideBar.pSearchRange.entry["state"] = "disabled"
        self.sideBar.voicedIter.entry["state"] = "disabled"
        self.sideBar.unvoicedIter.entry["state"] = "disabled"
        self.sideBar.fileButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
    
    def enableButtons(self):
        """Enables the per-phoneme settings buttons"""
        self.sideBar.expPitch.entry["state"] = "normal"
        self.sideBar.pSearchRange.entry["state"] = "normal"
        self.sideBar.voicedIter.entry["state"] = "normal"
        self.sideBar.unvoicedIter.entry["state"] = "normal"
        self.sideBar.fileButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
    
    def onAddPress(self):
        """UI Frontend function for adding a phoneme to the Voicebank"""
        global loadedVB
        key = tkinter.simpledialog.askstring(loc["new_phon"], loc["phon_key_sel"])
        if (key != "") & (key != None):
            if key in loadedVB.phonemeDict.keys():
                key += "#"
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")))
            if filepath != "":
                loadedVB.addPhoneme(key, filepath)
                loadedVB.phonemeDict[key].calculatePitch()
                loadedVB.phonemeDict[key].calculateSpectra()
                loadedVB.phonemeDict[key].calculateExcitation()
                self.phonemeList.list.lb.insert("end", key)
        
    def onRemovePress(self):
        """UI Frontend function for removing a phoneme from the Voicebank"""
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
            
    def onKeyChange(self, event):
        """UI Frontend function for changing the key of a phoneme"""
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
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
            if (loadedVB.phonemeDict[key].expectedPitch != self.sideBar.expPitch.variable.get()) or (loadedVB.phonemeDict[key].searchRange != self.sideBar.pSearchRange.variable.get()):
                loadedVB.phonemeDict[key].expectedPitch = self.sideBar.expPitch.variable.get()
                loadedVB.phonemeDict[key].searchRange = self.sideBar.pSearchRange.variable.get()
                loadedVB.phonemeDict[key].calculatePitch()
        
    def onSpectralUpdateTrigger(self, event):
        """UI Frontend function for updating the spectral and excitation data of a phoneme"""
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
            if (loadedVB.phonemeDict[key].voicedIterations != self.sideBar.voicedIter.variable.get()) or (loadedVB.phonemeDict[key].unvoicedIterations != self.sideBar.unvoicedIter.variable.get()):
                loadedVB.phonemeDict[key].filterWidth = global_consts.spectralFilterWidth
                loadedVB.phonemeDict[key].voicedIterations = self.sideBar.voicedIter.variable.get()
                loadedVB.phonemeDict[key].unvoicedIterations = self.sideBar.unvoicedIter.variable.get()
                loadedVB.phonemeDict[key].calculateSpectra()
                loadedVB.phonemeDict[key].calculateExcitation()
        
    def onFilechangePress(self, event):
        """UI Frontend function for changing the file associated with a phoneme"""
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")))
        if filepath != "":
            loadedVB.changePhonemeFile(key, filepath)
            loadedVB.phonemeDict[key].calculatePitch()
            loadedVB.phonemeDict[key].calculateSpectra()
            loadedVB.phonemeDict[key].calculateExcitation()
            self.phonemeList.list.lb.delete(index)
            self.phonemeList.list.lb.insert(index, key)
            
        
    def onFinalizePress(self):
        """UI Frontend function for finalizing a phoneme"""
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)[0]
        loadedVB.finalizePhoneme(key)
        self.sideBar.expPitch.variable.set(None)
        self.sideBar.pSearchRange.variable.set(None)
        self.sideBar.voicedIter.variable.set(None)
        self.sideBar.unvoicedIter.variable.set(None)
        self.disableButtons()
        
        
    def onOkPress(self):
        """Updates the last selected phoneme and closes the Phoneme Dict UI window when the OK button is pressed"""
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
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["crfai_lbl"])
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
        
        self.sideBar.voicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.voicedIter.variable = tkinter.IntVar(self.sideBar.voicedIter)
        self.sideBar.voicedIter.variable.set(2)
        self.sideBar.voicedIter.entry = tkinter.Spinbox(self.sideBar.voicedIter, from_ = 0, to = 10)
        self.sideBar.voicedIter.entry["textvariable"] = self.sideBar.voicedIter.variable
        self.sideBar.voicedIter.entry.pack(side = "right", fill = "x")
        self.sideBar.voicedIter.display = tkinter.Label(self.sideBar.voicedIter)
        self.sideBar.voicedIter.display["text"] = loc["viter"]
        self.sideBar.voicedIter.display.pack(side = "right", fill = "x")
        self.sideBar.voicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
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
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            
    def onListFocusOut(self, event):
        """Helper function for retaining information about the last selected transition sample when the transition sample list loses entry focus"""
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
    
    def onAddPress(self):
        """UI Frontend function for adding a new transition sample to the list of staged AI training samples"""
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".wav_desc"], ".wav"), (loc["all_files_desc"], "*")), multiple = True)
        if filepath != ():
            for i in filepath:
                loadedVB.addTrainSample(i)
                self.phonemeList.list.lb.insert("end", i)
        
    def onRemovePress(self):
        """UI Frontend function for removing a transition sample from the list of staged AI training samples"""
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
        global loadedVB
        loadedVB.trainCrfAi(self.sideBar.epochs.variable.get(), True, global_consts.spectralFilterWidth, self.sideBar.voicedIter.variable.get(), self.sideBar.unvoicedIter.variable.get())
        numIter = self.phonemeList.list.lb.size()
        for i in range(numIter):
            loadedVB.delTrainSample(0)
            self.phonemeList.list.lb.delete(0)
        
    def onFinalizePress(self):
        """UI Frontend function for finalizing the phoneme crossfade AI"""
        global loadedVB
        loadedVB.finalizCrfAi()
        self.disableButtons()
        
    def disableButtons(self):
        """disables the AI settings buttons"""
        self.sideBar.voicedIter.entry["state"] = "disabled"
        self.sideBar.unvoicedIter.entry["state"] = "disabled"
        self.sideBar.epochs.entry["state"] = "disabled"
        self.sideBar.trainButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
    
    def enableButtons(self):
        """enables the AI settings buttons"""
        self.sideBar.voicedIter.entry["state"] = "normal"
        self.sideBar.unvoicedIter.entry["state"] = "normal"
        self.sideBar.epochs.entry["state"] = "normal"
        self.sideBar.trainButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
        
    def onOkPress(self):
        """closes the window when the OK button is pressed"""
        global loadedVB
        self.master.destroy()

    def onLoadPress(self):
        """UI Frontend function for loading the AI state from the Phoneme Crossfade AI of a different Voicebank"""
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
        if filepath != "":
            loadedVB.loadCrfWeights(filepath, additive)

