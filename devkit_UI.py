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
        
        self.okButton = tkinter.Button(self)
        self.okButton["text"] = "OK"
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, pady = 20)
        
        self.finalizeButton = tkinter.Button(self)
        self.finalizeButton["text"] = "Finalize"
        self.finalizeButton["command"] = self.onFinalizePress
        self.finalizeButton.pack(side = "right", fill = "x", expand = True, pady = 20)
        
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