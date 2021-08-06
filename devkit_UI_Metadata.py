# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:12:43 2021

@author: CdrSonan
"""

import tkinter

class MetadataUi(tkinter.Frame):
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
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
        self.okButton.pack(side = "top", expand = True)
        
    def onOkPress(self):
        global loadedVB
        loadedVB.metadata.name = self.name.variable
        loadedVB.metadata.sampleRate = self.sampleRate.variable
        self.destroy()