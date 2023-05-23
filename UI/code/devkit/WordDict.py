#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import tkinter
import sys

from UI.code.devkit.Widgets import *
from Localization.devkit_localization import getLanguage
loc = getLanguage()

class WorddictUI(Frame):
    """UI class for the main Voicebank Dictionary"""
    
    def __init__(self, master=None) -> None:
        logging.info("Initializing Worddict UI")
        global loadedVB
        from UI.code.devkit.Main import loadedVB
        Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["worddict_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
    
    def createWidgets(self):
        """initialize all widgets belonging to the UI"""
        
        global loadedVB
        
        self.overrideList = LabelFrame(self, text = loc["override_list"])
        self.overrideList.list = Frame(self.overrideList)
        self.overrideList.list.lb = Listbox(self.overrideList.list)
        self.overrideList.list.lb.pack(side = "left",fill = "both", expand = True)
        self.overrideList.list.sb = tkinter.ttk.Scrollbar(self.overrideList.list)
        self.overrideList.list.sb.pack(side = "left", fill = "y")
        self.overrideList.list.lb["selectmode"] = "single"
        self.overrideList.list.lb["yscrollcommand"] = self.overrideList.list.sb.set
        self.overrideList.list.lb.bind("<<ListboxSelect>>", self.onOverrideSelectionChange)
        self.overrideList.list.lb.bind("<FocusOut>", self.onOverrideListFocusOut)
        self.overrideList.list.sb["command"] = self.overrideList.list.lb.yview
        self.overrideList.list.pack(side = "top", fill = "x", expand = True, padx = 5, pady = 2)
        for i in loadedVB.phonemeDict[0].keys():
            self.overrideList.list.lb.insert("end", i)
        self.overrideList.removeButton = SlimButton(self.overrideList)
        self.overrideList.removeButton["text"] = loc["remove"]
        self.overrideList.removeButton["command"] = self.onOverrideRemovePress
        self.overrideList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.overrideList.addButton = SlimButton(self.overrideList)
        self.overrideList.addButton["text"] = loc["add"]
        self.overrideList.addButton["command"] = self.onOverrideAddPress
        self.overrideList.addButton.pack(side = "right", fill = "x", expand = True)
        self.overrideList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.overrideSettings = LabelFrame(self, text = loc["override_settings"])
        self.overrideSettings.word = Frame(self.overrideSettings)
        self.overrideSettings.word.variable = tkinter.StringVar(self.overrideSettings.word)
        self.overrideSettings.word.entry = tkinter.Entry(self.overrideSettings.word, textvariable = self.overrideSettings.word.variable)
        self.overrideSettings.word.entry.pack(side = "right", fill = "x", expand = True)
        self.overrideSettings.word.entry.bind("<FocusOut>", self.overrideWordUpdate)
        self.overrideSettings.word.entry.bind("<KeyRelease-Return>", self.overrideWordUpdate)
        self.overrideSettings.word.entry.pack(side = "right", fill = "x", expand = True)
        self.overrideSettings.word.display = Label(self.overrideSettings.word)
        self.overrideSettings.word.display["text"] = loc["override_word"]
        self.overrideSettings.word.display.pack(side = "right", fill = "x", expand = True)
        self.overrideSettings.word.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.overrideSettings.mapping = Frame(self.overrideSettings)
        self.overrideSettings.mapping.variable = tkinter.StringVar(self.overrideSettings.mapping)
        self.overrideSettings.mapping.entry = tkinter.Entry(self.overrideSettings.mapping, textvariable = self.overrideSettings.mapping.variable)
        self.overrideSettings.mapping.entry.bind("<FocusOut>", self.overrideMappingUpdate)
        self.overrideSettings.mapping.entry.bind("<KeyRelease-Return>", self.overrideMappingUpdate)
        self.overrideSettings.mapping.entry.pack(side = "right", fill = "x", expand = True)
        self.overrideSettings.mapping.display = Label(self.overrideSettings.mapping)
        self.overrideSettings.mapping.display["text"] = loc["override_mapping"]
        self.overrideSettings.mapping.display.pack(side = "right", fill = "x", expand = True)
        self.overrideSettings.mapping.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.overrideSettings.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.syllableList = LabelFrame(self, text = loc["syllable_list"])
        self.syllableList.list = Frame(self.syllableList)
        self.syllableList.list.lb = Listbox(self.syllableList.list)
        self.syllableList.list.lb.pack(side = "left",fill = "both", expand = True)
        self.syllableList.list.sb = tkinter.ttk.Scrollbar(self.syllableList.list)
        self.syllableList.list.sb.pack(side = "left", fill = "y")
        self.syllableList.list.lb["selectmode"] = "single"
        self.syllableList.list.lb["yscrollcommand"] = self.syllableList.list.sb.set
        self.syllableList.list.lb.bind("<<ListboxSelect>>", self.onSyllableSelectionChange)
        self.syllableList.list.lb.bind("<FocusOut>", self.onSyllableListFocusOut)
        self.syllableList.list.sb["command"] = self.syllableList.list.lb.yview
        self.syllableList.list.pack(side = "top", fill = "x", expand = True, padx = 5, pady = 2)
        for i in loadedVB.wordDict[1]:
            for j in i.keys():
                self.syllableList.list.lb.insert("end", i)
        self.syllableList.removeButton = SlimButton(self.syllableList)
        self.syllableList.removeButton["text"] = loc["remove"]
        self.syllableList.removeButton["command"] = self.onSyllableRemovePress
        self.syllableList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.syllableList.addButton = SlimButton(self.syllableList)
        self.syllableList.addButton["text"] = loc["add"]
        self.syllableList.addButton["command"] = self.onSyllableAddPress
        self.syllableList.addButton.pack(side = "right", fill = "x", expand = True)
        self.syllableList.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.syllableSettings = LabelFrame(self, text = loc["syllable_settings"])
        self.syllableSettings.syllable = Frame(self.syllableSettings)
        self.syllableSettings.syllable.variable = tkinter.StringVar(self.syllableSettings.syllable)
        self.syllableSettings.syllable.entry = tkinter.Entry(self.syllableSettings.syllable, textvariable = self.syllableSettings.syllable.variable)
        self.syllableSettings.syllable.entry.pack(side = "right", fill = "x", expand = True)
        self.syllableSettings.syllable.display = Label(self.syllableSettings.syllable)
        self.syllableSettings.syllable.display["text"] = loc["syllable_sykey"]
        self.syllableSettings.syllable.display.pack(side = "right", fill = "x", expand = True)
        self.syllableSettings.syllable.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.syllableSettings.mapping = Frame(self.syllableSettings)
        self.syllableSettings.mapping.variable = tkinter.StringVar(self.syllableSettings.mapping)
        self.syllableSettings.mapping.entry = tkinter.Entry(self.syllableSettings.mapping, textvariable = self.syllableSettings.mapping.variable)
        self.syllableSettings.mapping.entry.pack(side = "right", fill = "x", expand = True)
        self.syllableSettings.mapping.display = Label(self.syllableSettings.mapping)
        self.syllableSettings.mapping.display["text"] = loc["syllable_mapping"]
        self.syllableSettings.mapping.display.pack(side = "right", fill = "x", expand = True)
        self.syllableSettings.mapping.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.syllableSettings.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        #window controls
        self.okButton = Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = Button(self)
        self.loadButton["text"] = loc["load"]
        self.loadButton["command"] = self.onLoadPress
        self.loadButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.overrideList.list.lastFocusedIndex = None
        self.syllableList.list.lastFocusedIndex = None
        
    def onOverrideAddPress(self):
        """add a new override to the list"""
        
        global loadedVB
        
        if "new" in loadedVB.wordDict[0].keys():
            i = 1
            while "new" + str(i) in loadedVB.wordDict[0].keys():
                i += 1
            self.overrideList.list.lb.insert("end", "new" + str(i))
            loadedVB.wordDict[0]["new" + str(i)] = ""
        self.overrideList.list.lb.insert("end", "new")
        loadedVB.wordDict[0]["new"] = ""
        
    def onOverrideRemovePress(self):
        """remove the selected override from the list"""
        
        global loadedVB
        
        if self.overrideList.list.lastFocusedIndex != None:
            del loadedVB.wordDict[0][self.overrideList.list.lb.get(self.overrideList.list.lastFocusedIndex)]
            self.overrideList.list.lb.delete(self.overrideList.list.lastFocusedIndex)
            self.overrideList.list.lastFocusedIndex = None
            
    def onOverrideSelectionChange(self, event):
        """update the override settings when the selection changes"""
        
        global loadedVB
        
        if len(self.overrideList.list.lb.curselection()) > 0:
            self.overrideList.list.lastFocusedIndex = self.overrideList.list.lb.curselection()[0]
            index = self.overrideList.list.lastFocusedIndex
            word = self.overrideList.list.lb.get(index)
            self.overrideSettings.word.variable.set(word)
            self.overrideSettings.mapping.variable.set(loadedVB.wordDict[0][word])
            
    def overrideWordUpdate(self):
        """updates the loaded Voicebank's wordDict when an override word is changed"""
        
        global loadedVB
        
        if self.overrideList.list.lastFocusedIndex != None:
            index = self.overrideList.list.lastFocusedIndex
            word = self.overrideList.list.lb.get(index)
            newWord = self.overrideSettings.word.variable.get()
            if word != newWord:
                if newWord in loadedVB.wordDict[0].keys():
                    tkinter.messagebox.showerror(loc["error"], loc["worddict_override_word_exists_error"])
                    self.overrideSettings.word.variable.set(word)
                    return
                loadedVB.wordDict[0][newWord] = loadedVB.wordDict[0].pop(word)
                del loadedVB.wordDict[0][word]
                self.overrideList.list.lb.delete(index)
                self.overrideList.list.lb.insert(index, newWord)
                
    def overrideMappingUpdate(self):
        """updates the loaded Voicebank's wordDict when an override mapping is changed"""
        
        global loadedVB
        
        if self.overrideList.list.lastFocusedIndex != None:
            index = self.overrideList.list.lastFocusedIndex
            word = self.overrideList.list.lb.get(index)
            loadedVB.wordDict[0][word] = self.overrideSettings.mapping.variable.get()
            
            
            
            
    def onSyllableAddPress(self):
        """add a new syllable mapping to the list"""
        
        global loadedVB
        
        self.syllableList.list.lb.insert("end", "new")
        loadedVB.wordDict[1].append({"new": "new"})
            
            
            
            
    def onOverrideAddPress(self):
        """add a new override to the list"""
        
        global loadedVB
        
        self.overrideList.list.lb.insert("end", "new")
        loadedVB.wordDict[0]["new"] = "new"
        
    def onOverrideRemovePress(self):
        """remove the selected override from the list"""
        
        global loadedVB
        
        if self.overrideList.list.lastFocusedIndex != None:
            del loadedVB.wordDict[0][self.overrideList.list.lb.get(self.overrideList.list.lastFocusedIndex)]
            self.overrideList.list.lb.delete(self.overrideList.list.lastFocusedIndex)
            self.overrideList.list.lastFocusedIndex = None
            
    def onOverrideSelectionChange(self, event):
        """update the override settings when the selection changes"""
        
        global loadedVB
        
        if len(self.overrideList.list.lb.curselection()) > 0:
            self.overrideList.list.lastFocusedIndex = self.overrideList.list.lb.curselection()[0]
            index = self.overrideList.list.lastFocusedIndex
            word = self.overrideList.list.lb.get(index)
            self.overrideSettings.word.variable.set(word)
            self.overrideSettings.mapping.variable.set(loadedVB.wordDict[0][word])
            
    def overrideWordUpdate(self):
        """updates the loaded Voicebank's wordDict when an override word is changed"""
        
        global loadedVB
        
        if self.overrideList.list.lastFocusedIndex != None:
            index = self.overrideList.list.lastFocusedIndex
            word = self.overrideList.list.lb.get(index)
            newWord = self.overrideSettings.word.variable.get()
            if word != newWord:
                loadedVB.wordDict[0][newWord] = loadedVB.wordDict[0].pop(word)
                del loadedVB.wordDict[0][word]
                self.overrideList.list.lb.delete(index)
                self.overrideList.list.lb.insert(index, newWord)
                
    def overrideMappingUpdate(self):
        """updates the loaded Voicebank's wordDict when an override mapping is changed"""
        
        global loadedVB
        
        if self.overrideList.list.lastFocusedIndex != None:
            index = self.overrideList.list.lastFocusedIndex
            word = self.overrideList.list.lb.get(index)
            loadedVB.wordDict[0][word] = self.overrideSettings.mapping.variable.get()