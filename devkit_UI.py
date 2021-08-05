# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:51:29 2021

@author: CdrSonan
"""

import tkinter
import tkinter.filedialog

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
        pass
    
    def onPhonemedictPress(self):
        pass
    
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
        
    def onOpenPress(self):
        global loadedVB
        global loadedVBPath
        if (loadedVB == None) or tkinter.messagebox.askokcancel("Warning", "Creating a new Voicebank will discard all unsaved changes to the currently opened one. Continue?", icon = "warning"):
            filepath = tkinter.filedialog.askopenfilename(filetypes = (("NovaVox Voicebanks", ".nvvb"), ("All files", "*")), initialfile = loadedVBPath)
            if filepath != "":
                loadedVBPath = filepath
                loadedVB = 1
                self.metadataButton["state"] = "active"
                self.phonemedictButton["state"] = "active"
                self.crfaiButton["state"] = "active"
                self.parameterButton["state"] = "active"
                self.worddictButton["state"] = "active"
                self.saveButton["state"] = "active"
    
    def onNewPress(self):
        global loadedVB
        if (loadedVB == None) or tkinter.messagebox.askokcancel("Warning", "Creating a new Voicebank will discard all unsaved changes to the currently opened one. Continue?", icon = "warning"):
            loadedVB = 1
            self.metadataButton["state"] = "active"
            self.phonemedictButton["state"] = "active"
            self.crfaiButton["state"] = "active"
            self.parameterButton["state"] = "active"
            self.worddictButton["state"] = "active"
            self.saveButton["state"] = "active"

loadedVB = None
loadedVBPath = None
app = RootUi(tkinter.Tk())
app.mainloop()