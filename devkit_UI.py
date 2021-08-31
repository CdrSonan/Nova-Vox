# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:51:29 2021

@author: CdrSonan
"""

import tkinter
import tkinter.filedialog
import tkinter.simpledialog

import devkit_pipeline
import devkit_locale

class RootUi(tkinter.Frame):
    def __init__(self, locale, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.locale = locale
        self.createWidgets()
        self.master.wm_title(self.locale["no_vb"])
        
    def createWidgets(self):
        self.infoDisplay = tkinter.Label(self)
        self.infoDisplay["text"] = self.locale["version_label"]
        self.infoDisplay.pack(side = "top", fill = "x", padx = 20, pady = 20)
        
        self.metadataButton = tkinter.Button(self)
        self.metadataButton["text"] = self.locale["metadat_btn"]
        self.metadataButton["command"] = self.onMetadataPress
        self.metadataButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.metadataButton["state"] = "disabled"
        
        self.phonemedictButton = tkinter.Button(self)
        self.phonemedictButton["text"] = self.locale["phon_btn"]
        self.phonemedictButton["command"] = self.onPhonemedictPress
        self.phonemedictButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.phonemedictButton["state"] = "disabled"
        
        self.crfaiButton = tkinter.Button(self)
        self.crfaiButton["text"] = self.locale["crfai_btn"]
        self.crfaiButton["command"] = self.onCrfaiPress
        self.crfaiButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.crfaiButton["state"] = "disabled"
        
        self.parameterButton = tkinter.Button(self)
        self.parameterButton["text"] = self.locale["param_btn"]
        self.parameterButton["command"] = self.onParameterPress
        self.parameterButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.parameterButton["state"] = "disabled"
        
        self.worddictButton = tkinter.Button(self)
        self.worddictButton["text"] = self.locale["dict_btn"]
        self.worddictButton["command"] = self.onWorddictPress
        self.worddictButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.worddictButton["state"] = "disabled"
        
        self.saveButton = tkinter.Button(self)
        self.saveButton["text"] = self.locale["save_as"]
        self.saveButton["command"] = self.onSavePress
        self.saveButton.pack(side = "right", expand = True)
        self.saveButton["state"] = "disabled"
        
        self.openButton = tkinter.Button(self)
        self.openButton["text"] = self.locale["open"]
        self.openButton["command"] = self.onOpenPress
        self.openButton.pack(side = "right", expand = True)
        
        self.newButton = tkinter.Button(self)
        self.newButton["text"] = self.locale["new"]
        self.newButton["command"] = self.onNewPress
        self.newButton.pack(side = "right", expand = True)
        
    def onMetadataPress(self):
        metadataUi = MetadataUi(self.locale, tkinter.Tk())
        metadataUi.mainloop()
    
    def onPhonemedictPress(self):
        phonemedictUi = PhonemedictUi(self.locale, tkinter.Tk())
        phonemedictUi.mainloop()
    
    def onCrfaiPress(self):
        crfaiUi = CrfaiUi(self.locale, tkinter.Tk())
        crfaiUi.mainloop()
    
    def onParameterPress(self):
        pass
    
    def onWorddictPress(self):
        pass
    
    def onSavePress(self):
        global loadedVB
        global loadedVBPath
        filepath = tkinter.filedialog.asksaveasfilename(defaultextension = ".nvvb", filetypes = ((self.locale[".nvvb_desc"], ".nvvb"), (self.locale["all_files_desc"], "*")), initialfile = loadedVBPath)
        if filepath != "":
            loadedVBPath = filepath
            loadedVB.save(filepath)
            self.master.wm_title(loadedVBPath)
        
    def onOpenPress(self):
        global loadedVB
        global loadedVBPath
        if (loadedVB == None) or tkinter.messagebox.askokcancel(self.locale["warning"], self.locale["vb_discard_msg"], icon = "warning"):
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((self.locale[".nvvb_desc"], ".nvvb"), (self.locale["all_files_desc"], "*")), initialfile = loadedVBPath)
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
        global loadedVB
        if (loadedVB == None) or tkinter.messagebox.askokcancel(self.locale["warning"], self.locale["vb_discard_msg"], icon = "warning"):
            loadedVB = devkit_pipeline.Voicebank(None)
            self.metadataButton["state"] = "active"
            self.phonemedictButton["state"] = "active"
            self.crfaiButton["state"] = "active"
            self.parameterButton["state"] = "active"
            self.worddictButton["state"] = "active"
            self.saveButton["state"] = "active"
            self.master.wm_title(self.locale["unsaved_vb"])
            
class MetadataUi(tkinter.Frame):
    def __init__(self, locale, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20)
        self.locale = locale
        self.createWidgets()
        self.master.wm_title(self.locale["metadat_lbl"])
        
    def createWidgets(self):
        global loadedVB
        self.name = tkinter.Frame(self)
        self.name.variable = tkinter.StringVar(self.name)
        self.name.variable.set(loadedVB.metadata.name)
        self.name.entry = tkinter.Entry(self.name)
        self.name.entry["textvariable"] = self.name.variable
        self.name.entry.pack(side = "right", fill = "x", expand = True)
        self.name.display = tkinter.Label(self.name)
        self.name.display["text"] = self.locale["name"]
        self.name.display.pack(side = "right", fill = "x")
        self.name.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        #self.sampleRate = tkinter.Frame(self)
        #self.sampleRate.variable = tkinter.IntVar(self.sampleRate)
        #self.sampleRate.variable.set(loadedVB.metadata.sampleRate)
        #self.sampleRate.entry = tkinter.Spinbox(self.sampleRate)
        #self.sampleRate.entry["values"] = (44100, 48000, 96000, 192000)
        #self.sampleRate.entry["textvariable"] = self.sampleRate.variable
        #self.sampleRate.entry.pack(side = "right", fill = "x", expand = True)
        #self.sampleRate.display = tkinter.Label(self.sampleRate)
        #self.sampleRate.display["text"] = self.locale["smp_rate"]
        #self.sampleRate.display.pack(side = "right", fill = "x")
        #self.sampleRate.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = self.locale["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "top", fill = "x", padx = 50, pady = 20)
        
    def onOkPress(self):
        global loadedVB
        loadedVB.metadata.name = self.name.variable.get()
        loadedVB.metadata.sampleRate = 49000#self.sampleRate.variable.get()
        self.master.destroy()
        
class PhonemedictUi(tkinter.Frame):
    def __init__(self, locale, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.locale = locale
        self.createWidgets()
        self.master.wm_title(self.locale["phon_lbl"])
        self.disableButtons()
        
    def createWidgets(self):
        global loadedVB
        
        self.phonemeList = tkinter.LabelFrame(self, text = self.locale["phon_list"])
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
        self.phonemeList.removeButton["text"] = self.locale["remove"]
        self.phonemeList.removeButton["command"] = self.onRemovePress
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = tkinter.Button(self.phonemeList)
        self.phonemeList.addButton["text"] = self.locale["add"]
        self.phonemeList.addButton["command"] = self.onAddPress
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar = tkinter.LabelFrame(self, text = self.locale["per_ph_set"])
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        self.sideBar.key = tkinter.Frame(self.sideBar)
        self.sideBar.key.variable = tkinter.StringVar(self.sideBar.key)
        self.sideBar.key.entry = tkinter.Entry(self.sideBar.key)
        self.sideBar.key.entry["textvariable"] = self.sideBar.key.variable
        self.sideBar.key.entry.bind("<FocusOut>", self.onKeyChange)
        self.sideBar.key.entry.bind("<KeyRelease-Return>", self.onKeyChange)
        self.sideBar.key.entry.pack(side = "right", fill = "x")
        self.sideBar.key.display = tkinter.Label(self.sideBar.key)
        self.sideBar.key.display["text"] = self.locale["phon_key"]
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
        self.sideBar.expPitch.display["text"] = self.locale["est_pit"]
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
        self.sideBar.pSearchRange.display["text"] = self.locale["psearchr"]
        self.sideBar.pSearchRange.display.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        #self.sideBar.fWidth = tkinter.Frame(self.sideBar)
        #self.sideBar.fWidth.variable = tkinter.IntVar(self.sideBar.fWidth)
        #self.sideBar.fWidth.entry = tkinter.Spinbox(self.sideBar.fWidth, from_ = 0, to = 100)
        #self.sideBar.fWidth.entry["textvariable"] = self.sideBar.fWidth.variable
        #self.sideBar.fWidth.entry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        #self.sideBar.fWidth.entry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        #self.sideBar.fWidth.entry.pack(side = "right", fill = "x")
        #self.sideBar.fWidth.display = tkinter.Label(self.sideBar.fWidth)
        #self.sideBar.fWidth.display["text"] = self.locale["fwidth"]
        #self.sideBar.fWidth.display.pack(side = "right", fill = "x")
        #self.sideBar.fWidth.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.voicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.voicedIter.variable = tkinter.IntVar(self.sideBar.voicedIter)
        self.sideBar.voicedIter.entry = tkinter.Spinbox(self.sideBar.voicedIter, from_ = 0, to = 10)
        self.sideBar.voicedIter.entry["textvariable"] = self.sideBar.voicedIter.variable
        self.sideBar.voicedIter.entry.bind("<FocusOut>", self.onSpectralUpdateTrigger)
        self.sideBar.voicedIter.entry.bind("<KeyRelease-Return>", self.onSpectralUpdateTrigger)
        self.sideBar.voicedIter.entry.pack(side = "right", fill = "x")
        self.sideBar.voicedIter.display = tkinter.Label(self.sideBar.voicedIter)
        self.sideBar.voicedIter.display["text"] = self.locale["viter"]
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
        self.sideBar.unvoicedIter.display["text"] = self.locale["uviter"]
        self.sideBar.unvoicedIter.display.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.fileButton = tkinter.Button(self.sideBar)
        self.sideBar.fileButton["text"] = self.locale["cng_file"]
        self.sideBar.fileButton["command"] = self.onFilechangePress
        self.sideBar.fileButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.finalizeButton = tkinter.Button(self.sideBar)
        self.sideBar.finalizeButton["text"] = self.locale["finalize"]
        self.sideBar.finalizeButton["command"] = self.onFinalizePress
        self.sideBar.finalizeButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = self.locale["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "top", fill = "x", expand = True, padx = 10, pady = 10)

        self.phonemeList.list.lastFocusedIndex = None
        
    def onSelectionChange(self, event):
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            index = self.phonemeList.list.lastFocusedIndex
            key = self.phonemeList.list.lb.get(index)
            self.sideBar.key.variable.set(key)
            if type(loadedVB.phonemeDict[key]).__name__ == "AudioSample":
                self.sideBar.expPitch.variable.set(loadedVB.phonemeDict[key].expectedPitch)
                self.sideBar.pSearchRange.variable.set(loadedVB.phonemeDict[key].searchRange)
                #self.sideBar.fWidth.variable.set(loadedVB.phonemeDict[key].filterWidth)
                self.sideBar.voicedIter.variable.set(loadedVB.phonemeDict[key].voicedIterations)
                self.sideBar.unvoicedIter.variable.set(loadedVB.phonemeDict[key].unvoicedIterations)
                self.enableButtons()
            else:
                self.sideBar.expPitch.variable.set(None)
                self.sideBar.pSearchRange.variable.set(None)
                #self.sideBar.fWidth.variable.set(None)
                self.sideBar.voicedIter.variable.set(None)
                self.sideBar.unvoicedIter.variable.set(None)
                self.disableButtons()
                
    def onListFocusOut(self, event):
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
        
    def disableButtons(self):
        self.sideBar.expPitch.entry["state"] = "disabled"
        self.sideBar.pSearchRange.entry["state"] = "disabled"
        #self.sideBar.fWidth.entry["state"] = "disabled"
        self.sideBar.voicedIter.entry["state"] = "disabled"
        self.sideBar.unvoicedIter.entry["state"] = "disabled"
        self.sideBar.fileButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
    
    def enableButtons(self):
        self.sideBar.expPitch.entry["state"] = "normal"
        self.sideBar.pSearchRange.entry["state"] = "normal"
        #self.sideBar.fWidth.entry["state"] = "normal"
        self.sideBar.voicedIter.entry["state"] = "normal"
        self.sideBar.unvoicedIter.entry["state"] = "normal"
        self.sideBar.fileButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
    
    def onAddPress(self):
        global loadedVB
        key = tkinter.simpledialog.askstring(self.locale["new_phon"], self.locale["phon_key_sel"])
        if (key != "") & (key != None):
            if key in loadedVB.phonemeDict.keys():
                key += "#"
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((self.locale[".wav_desc"], ".wav"), (self.locale["all_files_desc"], "*")))
            if filepath != "":
                loadedVB.addPhoneme(key, filepath)
                loadedVB.phonemeDict[key].calculatePitch()
                loadedVB.phonemeDict[key].calculateSpectra()
                loadedVB.phonemeDict[key].calculateExcitation()
                self.phonemeList.list.lb.insert("end", key)
        
    def onRemovePress(self):
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
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        newKey = self.sideBar.key.variable.get()
        if key != newKey:
            loadedVB.changePhonemeKey(key, newKey)
            self.phonemeList.list.lb.delete(index)
            self.phonemeList.list.lb.insert(index, newKey)
        
    def onPitchUpdateTrigger(self, event):
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key]).__name__ == "loadedAudioSample":
            if (loadedVB.phonemeDict[key].expectedPitch != self.sideBar.expPitch.variable.get()) or (loadedVB.phonemeDict[key].searchRange != self.sideBar.pSearchRange.variable.get()):
                loadedVB.phonemeDict[key].expectedPitch = self.sideBar.expPitch.variable.get()
                loadedVB.phonemeDict[key].searchRange = self.sideBar.pSearchRange.variable.get()
                loadedVB.phonemeDict[key].calculatePitch()
        
    def onSpectralUpdateTrigger(self, event):
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        if type(loadedVB.phonemeDict[key]).__name__ == "loadedAudioSample":
            #loadedVB.phonemeDict[key].filterWidth != self.sideBar.fWidth.variable.get()) or 
            if (loadedVB.phonemeDict[key].voicedIterations != self.sideBar.voicedIter.variable.get()) or (loadedVB.phonemeDict[key].unvoicedIterations != self.sideBar.unvoicedIter.variable.get()):
                loadedVB.phonemeDict[key].filterWidth = 10#self.sideBar.fWidth.variable.get()
                loadedVB.phonemeDict[key].voicedIterations = self.sideBar.voicedIter.variable.get()
                loadedVB.phonemeDict[key].unvoicedIterations = self.sideBar.unvoicedIter.variable.get()
                loadedVB.phonemeDict[key].calculateSpectra()
                loadedVB.phonemeDict[key].calculateExcitation()
        
    def onFilechangePress(self, event):
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((self.locale[".wav_desc"], ".wav"), (self.locale["all_files_desc"], "*")))
        if filepath != "":
            loadedVB.changePhonemeFile(key, filepath)
            loadedVB.phonemeDict[key].calculatePitch()
            loadedVB.phonemeDict[key].calculateSpectra()
            loadedVB.phonemeDict[key].calculateExcitation()
            self.phonemeList.list.lb.delete(index)
            self.phonemeList.list.lb.insert(index, key)
            
        
    def onFinalizePress(self):
        global loadedVB
        index = self.phonemeList.list.lastFocusedIndex
        key = self.phonemeList.list.lb.get(index)[0]
        loadedVB.finalizePhoneme(key)
        self.sideBar.expPitch.variable.set(None)
        self.sideBar.pSearchRange.variable.set(None)
        #self.sideBar.fWidth.variable.set(None)
        self.sideBar.voicedIter.variable.set(None)
        self.sideBar.unvoicedIter.variable.set(None)
        self.disableButtons()
        
        
    def onOkPress(self):
        global loadedVB
        #if self.phonemeList.list.lastFocusedIndex != None:
        if self.phonemeList.list.lb.size() > 0:
            self.onKeyChange(None)
            self.onPitchUpdateTrigger(None)
            self.onSpectralUpdateTrigger(None)
        self.master.destroy()
        
class CrfaiUi(tkinter.Frame):
    def __init__(self, locale, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.locale = locale
        self.createWidgets()
        self.master.wm_title(self.locale["crfai_lbl"])
        
    def createWidgets(self):
        global loadedVB
        
        self.phonemeList = tkinter.LabelFrame(self, text = self.locale["ai_samp_list"])
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
        self.phonemeList.removeButton["text"] = self.locale["remove"]
        self.phonemeList.removeButton["command"] = self.onRemovePress
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = tkinter.Button(self.phonemeList)
        self.phonemeList.addButton["text"] = self.locale["add"]
        self.phonemeList.addButton["command"] = self.onAddPress
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar = tkinter.LabelFrame(self, text = self.locale["ai_settings"])
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        #self.sideBar.fWidth = tkinter.Frame(self.sideBar)
        #self.sideBar.fWidth.variable = tkinter.IntVar(self.sideBar.fWidth)
        #self.sideBar.fWidth.variable.set(10)
        #self.sideBar.fWidth.entry = tkinter.Spinbox(self.sideBar.fWidth, from_ = 0, to = 100)
        #self.sideBar.fWidth.entry["textvariable"] = self.sideBar.fWidth.variable
        #self.sideBar.fWidth.entry.pack(side = "right", fill = "x")
        #self.sideBar.fWidth.display = tkinter.Label(self.sideBar.fWidth)
        #self.sideBar.fWidth.display["text"] = self.locale["fwidth"]
        #self.sideBar.fWidth.display.pack(side = "right", fill = "x")
        #self.sideBar.fWidth.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.voicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.voicedIter.variable = tkinter.IntVar(self.sideBar.voicedIter)
        self.sideBar.voicedIter.variable.set(2)
        self.sideBar.voicedIter.entry = tkinter.Spinbox(self.sideBar.voicedIter, from_ = 0, to = 10)
        self.sideBar.voicedIter.entry["textvariable"] = self.sideBar.voicedIter.variable
        self.sideBar.voicedIter.entry.pack(side = "right", fill = "x")
        self.sideBar.voicedIter.display = tkinter.Label(self.sideBar.voicedIter)
        self.sideBar.voicedIter.display["text"] = self.locale["viter"]
        self.sideBar.voicedIter.display.pack(side = "right", fill = "x")
        self.sideBar.voicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.unvoicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.unvoicedIter.variable = tkinter.IntVar(self.sideBar.unvoicedIter)
        self.sideBar.unvoicedIter.variable.set(10)
        self.sideBar.unvoicedIter.entry = tkinter.Spinbox(self.sideBar.unvoicedIter, from_ = 0, to = 100)
        self.sideBar.unvoicedIter.entry["textvariable"] = self.sideBar.unvoicedIter.variable
        self.sideBar.unvoicedIter.entry.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIter.display = tkinter.Label(self.sideBar.unvoicedIter)
        self.sideBar.unvoicedIter.display["text"] = self.locale["uviter"]
        self.sideBar.unvoicedIter.display.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.epochs = tkinter.Frame(self.sideBar)
        self.sideBar.epochs.variable = tkinter.IntVar(self.sideBar.epochs)
        self.sideBar.epochs.variable.set(1)
        self.sideBar.epochs.entry = tkinter.Spinbox(self.sideBar.epochs, from_ = 1, to = 100)
        self.sideBar.epochs.entry["textvariable"] = self.sideBar.epochs.variable
        self.sideBar.epochs.entry.pack(side = "right", fill = "x")
        self.sideBar.epochs.display = tkinter.Label(self.sideBar.epochs)
        self.sideBar.epochs.display["text"] = self.locale["epochs"]
        self.sideBar.epochs.display.pack(side = "right", fill = "x")
        self.sideBar.epochs.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.trainButton = tkinter.Button(self.sideBar)
        self.sideBar.trainButton["text"] = self.locale["train"]
        self.sideBar.trainButton["command"] = self.onTrainPress
        self.sideBar.trainButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.finalizeButton = tkinter.Button(self.sideBar)
        self.sideBar.finalizeButton["text"] = self.locale["finalize"]
        self.sideBar.finalizeButton["command"] = self.onFinalizePress
        self.sideBar.finalizeButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = self.locale["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "top", fill = "x", expand = True, padx = 10, pady = 10)
        
        if type(loadedVB.crfAi).__name__ == "SavedSpecCrfAi":
            self.disableButtons()
        
    def onSelectionChange(self, event):
        global loadedVB
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
            
    def onListFocusOut(self, event):
        if len(self.phonemeList.list.lb.curselection()) > 0:
            self.phonemeList.list.lastFocusedIndex = self.phonemeList.list.lb.curselection()[0]
    
    def onAddPress(self):
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((self.locale[".wav_desc"], ".wav"), (self.locale["all_files_desc"], "*")), multiple = True)
        if filepath != ():
            for i in filepath:
                loadedVB.addTrainSample(i)
                self.phonemeList.list.lb.insert("end", i)
        
    def onRemovePress(self):
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
        global loadedVB
        loadedVB.trainCrfAi(self.sideBar.epochs.variable.get(), True, 10, self.sideBar.voicedIter.variable.get(), self.sideBar.unvoicedIter.variable.get())
        numIter = self.phonemeList.list.lb.size()
        for i in range(numIter):
            loadedVB.delTrainSample(0)
            self.phonemeList.list.lb.delete(0)
        
    def onFinalizePress(self):
        global loadedVB
        loadedVB.finalizCrfAi()
        self.disableButtons()
        
    def disableButtons(self):
        #self.sideBar.fWidth.entry["state"] = "disabled"
        self.sideBar.voicedIter.entry["state"] = "disabled"
        self.sideBar.unvoicedIter.entry["state"] = "disabled"
        self.sideBar.epochs.entry["state"] = "disabled"
        self.sideBar.traineButton["state"] = "disabled"
        self.sideBar.finalizeButton["state"] = "disabled"
    
    def enableButtons(self):
        #self.sideBar.fWidth.entry["state"] = "normal"
        self.sideBar.voicedIter.entry["state"] = "normal"
        self.sideBar.unvoicedIter.entry["state"] = "normal"
        self.sideBar.epochs.entry["state"] = "normal"
        self.sideBar.traineButton["state"] = "normal"
        self.sideBar.finalizeButton["state"] = "normal"
        
    def onOkPress(self):
        global loadedVB
        self.master.destroy()
        
loc = devkit_locale.LocaleDict("en").locale
loadedVB = None
loadedVBPath = None
rootUi = RootUi(loc, tkinter.Tk())
rootUi.mainloop()
