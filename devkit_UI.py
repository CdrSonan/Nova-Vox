# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:51:29 2021

@author: CdrSonan
"""

import tkinter
import tkinter.filedialog

import devkit_pipeline

class RootUi(tkinter.Frame):
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        
        self.master.wm_title("no Voicebank loaded")
        
    def createWidgets(self):
        self.infoDisplay = tkinter.Label(self)
        self.infoDisplay["text"] = "NovaVox Devkit ALPHA 0.1.0"
        self.infoDisplay.pack(side = "top", fill = "x", padx = 20, pady = 20)
        
        self.metadataButton = tkinter.Button(self)
        self.metadataButton["text"] = "edit Metadata"
        self.metadataButton["command"] = self.onMetadataPress
        self.metadataButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.metadataButton["state"] = "disabled"
        
        self.phonemedictButton = tkinter.Button(self)
        self.phonemedictButton["text"] = "edit Phonemes"
        self.phonemedictButton["command"] = self.onPhonemedictPress
        self.phonemedictButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.phonemedictButton["state"] = "disabled"
        
        self.crfaiButton = tkinter.Button(self)
        self.crfaiButton["text"] = "edit Phoneme Crossfade Ai"
        self.crfaiButton["command"] = self.onCrfaiPress
        self.crfaiButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.crfaiButton["state"] = "disabled"
        
        self.parameterButton = tkinter.Button(self)
        self.parameterButton["text"] = "edit Ai-driven Parameters"
        self.parameterButton["command"] = self.onParameterPress
        self.parameterButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.parameterButton["state"] = "disabled"
        
        self.worddictButton = tkinter.Button(self)
        self.worddictButton["text"] = "edit Dictionary"
        self.worddictButton["command"] = self.onWorddictPress
        self.worddictButton.pack(side = "top", fill = "x", padx = 10, pady = 5)
        self.worddictButton["state"] = "disabled"
        
        self.saveButton = tkinter.Button(self)
        self.saveButton["text"] = "Save as..."
        self.saveButton["command"] = self.onSavePress
        self.saveButton.pack(side = "right", expand = True)
        self.saveButton["state"] = "disabled"
        
        self.openButton = tkinter.Button(self)
        self.openButton["text"] = "Open..."
        self.openButton["command"] = self.onOpenPress
        self.openButton.pack(side = "right", expand = True)
        
        self.newButton = tkinter.Button(self)
        self.newButton["text"] = "New..."
        self.newButton["command"] = self.onNewPress
        self.newButton.pack(side = "right", expand = True)
        
    def onMetadataPress(self):
        metadataUi = MetadataUi(tkinter.Tk())
        metadataUi.mainloop()
    
    def onPhonemedictPress(self):
        phonemedictUi = PhonemedictUi(tkinter.Tk())
        phonemedictUi.mainloop()
    
    def onCrfaiPress(self):
        pass
    
    def onParameterPress(self):
        pass
    
    def onWorddictPress(self):
        pass
    
    def onSavePress(self):
        global loadedVB
        global loadedVBPath
        filepath = tkinter.filedialog.asksaveasfilename(defaultextension = ".nvvb", filetypes = (("NovaVox Voicebanks", ".nvvb"), ("All files", "*")), initialfile = loadedVBPath)
        if filepath != "":
            loadedVBPath = filepath
            loadedVB.save(filepath)
            self.master.wm_title(loadedVBPath)
        
    def onOpenPress(self):
        global loadedVB
        global loadedVBPath
        if (loadedVB == None) or tkinter.messagebox.askokcancel("Warning", "Creating a new Voicebank will discard all unsaved changes to the currently opened one. Continue?", icon = "warning"):
            filepath = tkinter.filedialog.askopenfilename(filetypes = (("NovaVox Voicebanks", ".nvvb"), ("All files", "*")), initialfile = loadedVBPath)
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
        if (loadedVB == None) or tkinter.messagebox.askokcancel("Warning", "Creating a new Voicebank will discard all unsaved changes to the currently opened one. Continue?", icon = "warning"):
            loadedVB = devkit_pipeline.Voicebank(None)
            self.metadataButton["state"] = "active"
            self.phonemedictButton["state"] = "active"
            self.crfaiButton["state"] = "active"
            self.parameterButton["state"] = "active"
            self.worddictButton["state"] = "active"
            self.saveButton["state"] = "active"
            self.master.wm_title("unsaved Voicebank")
            
class MetadataUi(tkinter.Frame):
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20)
        self.createWidgets()
        
    def createWidgets(self):
        global loadedVB
        self.name = tkinter.Frame(self)
        self.name.variable = tkinter.StringVar(self.name)
        self.name.variable.set(loadedVB.metadata.name)
        self.name.entry = tkinter.Entry(self.name)
        self.name.entry["textvariable"] = self.name.variable
        self.name.entry.pack(side = "right", fill = "x", expand = True)
        self.name.display = tkinter.Label(self.name)
        self.name.display["text"] = "Name:"
        self.name.display.pack(side = "right", fill = "x")
        self.name.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sampleRate = tkinter.Frame(self)
        self.sampleRate.variable = tkinter.IntVar(self.sampleRate)
        self.sampleRate.variable.set(loadedVB.metadata.sampleRate)
        self.sampleRate.entry = tkinter.Spinbox(self.sampleRate)
        self.sampleRate.entry["values"] = (44100, 48000, 96000, 192000)
        self.sampleRate.entry["textvariable"] = self.sampleRate.variable
        self.sampleRate.entry.pack(side = "right", fill = "x", expand = True)
        self.sampleRate.display = tkinter.Label(self.sampleRate)
        self.sampleRate.display["text"] = "Name:"
        self.sampleRate.display.pack(side = "right", fill = "x")
        self.sampleRate.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = "OK"
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "top", fill = "x", padx = 50, pady = 20)
        
    def onOkPress(self):
        global loadedVB
        loadedVB.metadata.name = self.name.variable.get()
        loadedVB.metadata.sampleRate = self.sampleRate.variable.get()
        self.master.destroy()
        
class PhonemedictUi(tkinter.Frame):
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        
    def createWidgets(self):
        global loadedVB
        
        self.phonemeList = tkinter.LabelFrame(self, text = "phoneme list")
        self.phonemeList.list = tkinter.Frame(self.phonemeList)
        self.phonemeList.list.lb = tkinter.Listbox(self.phonemeList.list)
        self.phonemeList.list.lb.pack(side = "left",fill = "both", expand = True)
        self.phonemeList.list.sb = tkinter.Scrollbar(self.phonemeList.list)
        self.phonemeList.list.sb.pack(side = "left", fill = "y")
        self.phonemeList.list.lb["yscrollcommand"] = self.phonemeList.list.sb.set
        self.phonemeList.list.sb["command"] = self.phonemeList.list.lb.yview
        self.phonemeList.list.pack(side = "top", fill = "x", expand = True, padx = 5, pady = 2)
        for i in loadedVB.phonemeDict.keys():
            if type(self.phonemeDict[i]).__name__ == "AudioSample":
                self.phonemeList.list.lb.insert("end", i + " (not finalized)")
            else:
                self.phonemeList.list.lb.insert("end", i)
        
        self.phonemeList.removeButton = tkinter.Button(self.phonemeList)
        self.phonemeList.removeButton["text"] = "remove"
        self.phonemeList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.phonemeList.addButton = tkinter.Button(self.phonemeList)
        self.phonemeList.addButton["text"] = "add"
        self.phonemeList.addButton.pack(side = "right", fill = "x", expand = True)
        
        self.phonemeList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar = tkinter.LabelFrame(self, text = "per-phoneme settings")
        self.sideBar.pack(side = "top", fill = "x", padx = 5, pady = 2, ipadx = 5, ipady = 10)
        
        self.sideBar.key = tkinter.Frame(self.sideBar)
        self.sideBar.keyEntry = tkinter.Entry(self.sideBar.key)
        self.sideBar.keyEntry.pack(side = "right", fill = "x")
        self.sideBar.keyDisplay = tkinter.Label(self.sideBar.key)
        self.sideBar.keyDisplay["text"] = "phoneme key:"
        self.sideBar.keyDisplay.pack(side = "right", fill = "x")
        self.sideBar.key.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.expPitch = tkinter.Frame(self.sideBar)
        self.sideBar.expPitchEntry = tkinter.Entry(self.sideBar.expPitch)
        self.sideBar.expPitchEntry.pack(side = "right", fill = "x")
        self.sideBar.expPitchDisplay = tkinter.Label(self.sideBar.expPitch)
        self.sideBar.expPitchDisplay["text"] = "estimated pitch:"
        self.sideBar.expPitchDisplay.pack(side = "right", fill = "x")
        self.sideBar.expPitch.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.pSearchRange = tkinter.Frame(self.sideBar)
        self.sideBar.pSearchRangeEntry = tkinter.Spinbox(self.sideBar.pSearchRange, from_ = 0.05, to = 0.5, increment = 0.05)
        self.sideBar.pSearchRangeEntry.pack(side = "right", fill = "x")
        self.sideBar.pSearchRangeDisplay = tkinter.Label(self.sideBar.pSearchRange)
        self.sideBar.pSearchRangeDisplay["text"] = "pitch search range:"
        self.sideBar.pSearchRangeDisplay.pack(side = "right", fill = "x")
        self.sideBar.pSearchRange.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.fWidth = tkinter.Frame(self.sideBar)
        self.sideBar.fWidthEntry = tkinter.Spinbox(self.sideBar.fWidth, from_ = 0, to = 100)
        self.sideBar.fWidthEntry.pack(side = "right", fill = "x")
        self.sideBar.fWidthDisplay = tkinter.Label(self.sideBar.fWidth)
        self.sideBar.fWidthDisplay["text"] = "spectral filter width:"
        self.sideBar.fWidthDisplay.pack(side = "right", fill = "x")
        self.sideBar.fWidth.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.voicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.voicedIterEntry = tkinter.Spinbox(self.sideBar.voicedIter, from_ = 0, to = 10)
        self.sideBar.voicedIterEntry.pack(side = "right", fill = "x")
        self.sideBar.voicedIterDisplay = tkinter.Label(self.sideBar.voicedIter)
        self.sideBar.voicedIterDisplay["text"] = "voiced excitation filter iterations:"
        self.sideBar.voicedIterDisplay.pack(side = "right", fill = "x")
        self.sideBar.voicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.unvoicedIter = tkinter.Frame(self.sideBar)
        self.sideBar.unvoicedIterEntry = tkinter.Spinbox(self.sideBar.unvoicedIter, from_ = 0, to = 100)
        self.sideBar.unvoicedIterEntry.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIterDisplay = tkinter.Label(self.sideBar.unvoicedIter)
        self.sideBar.unvoicedIterDisplay["text"] = "unvoiced excitation filter iterations:"
        self.sideBar.unvoicedIterDisplay.pack(side = "right", fill = "x")
        self.sideBar.unvoicedIter.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.sideBar.fileButton = tkinter.Button(self.sideBar)
        self.sideBar.fileButton["text"] = "change file"
        self.sideBar.fileButton["command"] = self.onFinalizePress
        self.sideBar.fileButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        self.sideBar.finalizeButton = tkinter.Button(self.sideBar)
        self.sideBar.finalizeButton["text"] = "Finalize"
        self.sideBar.finalizeButton["command"] = self.onFinalizePress
        self.sideBar.finalizeButton.pack(side = "top", fill = "x", expand = True, padx = 5)
        
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = "OK"
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "top", fill = "x", expand = True, padx = 10, pady = 10)
        
        
    def onFinalizePress(self):
        global loadedVB
        
    def onOkPress(self):
        global loadedVB
        #loadedVB.metadata.name = self.name.variable.get()
        #loadedVB.metadata.sampleRate = self.sampleRate.variable.get()
        self.master.destroy()

loadedVB = None
loadedVBPath = None
rootUi = RootUi(tkinter.Tk())
rootUi.mainloop()