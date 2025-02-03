#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import tkinter
import sys

from UI.devkit.Widgets import *
from Localization.devkit_localization import getLanguage
loc = getLanguage()

class WorddictUi(Frame):
    """UI class for the main Voicebank Dictionary"""

    def __init__(self, master=None) -> None:
        logging.info("Initializing Worddict UI")
        global loadedVB
        from UI.devkit.Main import loadedVB
        Frame.__init__(self, master)
        self.pack(ipadx = 20, ipady = 20)
        self.createWidgets()
        self.master.wm_title(loc["worddict_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("assets/icon/nova-vox-logo-black.ico")

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
        for i in loadedVB.wordDict[0].keys():
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
        
        self.overrideMappingList = LabelFrame(self, text = loc["override_mapping_list"])
        self.overrideMappingList.list = Frame(self.overrideMappingList)
        self.overrideMappingList.list.lb = Listbox(self.overrideMappingList.list)
        self.overrideMappingList.list.lb.pack(side = "left",fill = "both", expand = True)
        self.overrideMappingList.list.sb = tkinter.ttk.Scrollbar(self.overrideMappingList.list)
        self.overrideMappingList.list.sb.pack(side = "left", fill = "y")
        self.overrideMappingList.list.lb["selectmode"] = "single"
        self.overrideMappingList.list.lb["yscrollcommand"] = self.overrideMappingList.list.sb.set
        self.overrideMappingList.list.lb.bind("<<ListboxSelect>>", self.onOverrideMappingSelectionChange)
        self.overrideMappingList.list.lb.bind("<FocusOut>", self.onOverrideMappingListFocusOut)
        self.overrideMappingList.list.sb["command"] = self.overrideMappingList.list.lb.yview
        self.overrideMappingList.list.pack(side = "top", fill = "x", expand = True, padx = 5, pady = 2)
        for i in loadedVB.wordDict[0].keys():
            self.overrideMappingList.list.lb.insert("end", i)
        self.overrideMappingList.removeButton = SlimButton(self.overrideMappingList)
        self.overrideMappingList.removeButton["text"] = loc["remove"]
        self.overrideMappingList.removeButton["command"] = self.onOverrideMappingRemovePress
        self.overrideMappingList.removeButton.pack(side = "right", fill = "x", expand = True)
        self.overrideMappingList.addButton = SlimButton(self.overrideMappingList)
        self.overrideMappingList.addButton["text"] = loc["add"]
        self.overrideMappingList.addButton["command"] = self.onOverrideMappingAddPress
        self.overrideMappingList.addButton.pack(side = "right", fill = "x", expand = True)
        self.overrideMappingList.pack(side = "top", fill = "x", padx = 5, pady = 2)

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
        self.overrideSettings.csvButton = Button(self.overrideSettings)
        self.overrideSettings.csvButton["text"] = loc["override_csv_import"]
        self.overrideSettings.csvButton["command"] = self.onOverrideCSVImportPress
        self.overrideSettings.csvButton.pack(side = "top", fill = "x", padx = 5, pady = 2)
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
                self.syllableList.list.lb.insert("end", j)
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
        self.syllableSettings.syllable.entry.bind("<FocusOut>", self.syllableUpdate)
        self.syllableSettings.syllable.entry.bind("<KeyRelease-Return>", self.syllableUpdate)
        self.syllableSettings.syllable.entry.pack(side = "right", fill = "x", expand = True)
        self.syllableSettings.syllable.display = Label(self.syllableSettings.syllable)
        self.syllableSettings.syllable.display["text"] = loc["syllable_key"]
        self.syllableSettings.syllable.display.pack(side = "right", fill = "x", expand = True)
        self.syllableSettings.syllable.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.syllableSettings.mapping = Frame(self.syllableSettings)
        self.syllableSettings.mapping.variable = tkinter.StringVar(self.syllableSettings.mapping)
        self.syllableSettings.mapping.entry = tkinter.Entry(self.syllableSettings.mapping, textvariable = self.syllableSettings.mapping.variable)
        self.syllableSettings.mapping.entry.bind("<FocusOut>", self.syllableMappingUpdate)
        self.syllableSettings.mapping.entry.bind("<KeyRelease-Return>", self.syllableMappingUpdate)
        self.syllableSettings.mapping.entry.pack(side = "right", fill = "x", expand = True)
        self.syllableSettings.mapping.display = Label(self.syllableSettings.mapping)
        self.syllableSettings.mapping.display["text"] = loc["syllable_mapping"]
        self.syllableSettings.mapping.display.pack(side = "right", fill = "x", expand = True)
        self.syllableSettings.csvButton = Button(self.syllableSettings)
        self.syllableSettings.csvButton["text"] = loc["syllable_csv_import"]
        self.syllableSettings.csvButton["command"] = self.onSyllableCSVImportPress
        self.syllableSettings.csvButton.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.syllableSettings.mapping.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.syllableSettings.pack(side = "top", fill = "x", padx = 5, pady = 2)

        #window controls
        self.okButton = Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = Button(self)
        self.loadButton["text"] = loc["load_other_VB"]
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
            loadedVB.wordDict[0]["new" + str(i)] = []
        else:
            self.overrideList.list.lb.insert("end", "new")
            loadedVB.wordDict[0]["new"] = []

    def onOverrideRemovePress(self):
        """remove the selected override from the list"""

        global loadedVB

        if self.overrideList.list.lastFocusedIndex != None:
            del loadedVB.wordDict[0][self.overrideList.list.lb.get(self.overrideList.list.lastFocusedIndex)]
            self.overrideList.list.lb.delete(self.overrideList.list.lastFocusedIndex)
            self.overrideList.list.lastFocusedIndex = None
            while self.overrideMappingList.list.lb.size() > 0:
                self.overrideMappingList.list.lb.delete(0, "end")
            self.overrideMappingList.list.lastFocusedIndex = None

    def onOverrideSelectionChange(self, event):
        """update the override settings when the selection changes"""

        global loadedVB

        if len(self.overrideList.list.lb.curselection()) > 0:
            self.overrideList.list.lastFocusedIndex = self.overrideList.list.lb.curselection()[0]
            index = self.overrideList.list.lastFocusedIndex
            word = self.overrideList.list.lb.get(index)
            self.overrideSettings.word.variable.set(word)
            print(self.overrideMappingList.list.lb.size())
            while self.overrideMappingList.list.lb.size() > 0:
                self.overrideMappingList.list.lb.delete(0, "end")
            for i in loadedVB.wordDict[0][word]:
                self.overrideMappingList.list.lb.insert("end", i)
            self.overrideMappingList.list.lastFocusedIndex = None

    def onOverrideListFocusOut(self, event) -> None:
        """Helper function for retaining information about the last focused element of the override list when override list loses entry focus"""

        logging.info("override list focus loss callback")
        if len(self.overrideList.list.lb.curselection()) > 0:
            self.overrideList.list.lastFocusedIndex = self.overrideList.list.lb.curselection()[0]

    def onOverrideMappingAddPress(self):
        """add a new override to the list"""

        global loadedVB

        if self.overrideList.list.lastFocusedIndex != None:
            self.overrideMappingList.list.lb.insert("end", "")
            loadedVB.wordDict[0][self.overrideList.list.lb.get(self.overrideList.list.lastFocusedIndex)].append("")

    def onOverrideMappingRemovePress(self):
        """remove the selected override from the list"""

        global loadedVB

        if self.overrideList.list.lastFocusedIndex != None and self.overrideMappingList.list.lastFocusedIndex != None:
            del loadedVB.wordDict[0][self.overrideList.list.lb.get(self.overrideList.list.lastFocusedIndex)][self.overrideMappingList.list.lastFocusedIndex]
            self.overrideMappingList.list.lb.delete(self.overrideMappingList.list.lastFocusedIndex)
            self.overrideMappingList.list.lastFocusedIndex = None

    def onOverrideMappingSelectionChange(self, event):
        """update the override settings when the selection changes"""

        global loadedVB

        if len(self.overrideMappingList.list.lb.curselection()) > 0:
            self.overrideMappingList.list.lastFocusedIndex = self.overrideMappingList.list.lb.curselection()[0]
            self.overrideSettings.mapping.variable.set(loadedVB.wordDict[0][self.overrideList.list.lb.get(self.overrideList.list.lastFocusedIndex)][self.overrideMappingList.list.lastFocusedIndex])

    def onOverrideMappingListFocusOut(self, event) -> None:
        """Helper function for retaining information about the last focused element of the override list when override list loses entry focus"""

        logging.info("override list focus loss callback")
        if len(self.overrideMappingList.list.lb.curselection()) > 0:
            self.overrideMappingList.list.lastFocusedIndex = self.overrideMappingList.list.lb.curselection()[0]

    def overrideWordUpdate(self, value):
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
                self.overrideList.list.lb.delete(index)
                self.overrideList.list.lb.insert(index, newWord)

    def overrideMappingUpdate(self, value):
        """updates the loaded Voicebank's wordDict when an override mapping is changed"""

        global loadedVB

        if self.overrideList.list.lastFocusedIndex != None and self.overrideMappingList.list.lastFocusedIndex != None:
            word = self.overrideList.list.lb.get(self.overrideList.list.lastFocusedIndex)
            index = self.overrideMappingList.list.lastFocusedIndex
            loadedVB.wordDict[0][word][index] = self.overrideSettings.mapping.variable.get()
            self.overrideMappingList.list.lb.delete(index)
            self.overrideMappingList.list.lb.insert(index, self.overrideSettings.mapping.variable.get())
    
    def onOverrideCSVImportPress(self):
        """imports a CSV file to the override list"""
        
        global loadedVB
        
        file = tkinter.filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv")])
        if file != "":
            loadedVB.wordDict[0] = {}
            try:
                with open(file, "r", encoding = "utf-8") as f:
                    for i in f.readlines():
                        if i != "\n":
                            word = i.split(",")[0]
                            mapping = i.split(",")[1].strip()
                            if word in loadedVB.wordDict[0].keys():
                                loadedVB.wordDict[0][word].append(mapping)
                            else:
                                loadedVB.wordDict[0][word] = [mapping,]
                                self.overrideList.list.lb.insert("end", word)
            except:
                tkinter.messagebox.showerror(loc["error"], loc["worddict_csv_import_error"])
                return

    def onSyllableAddPress(self):
        """add a new override to the list"""

        global loadedVB

        self.syllableUpdate("")
        self.syllableMappingUpdate("")
        if len(loadedVB.wordDict[1]) >= len("new") and "new" in loadedVB.wordDict[1][len("new") - 1].keys():
            i = 1
            while len(loadedVB.wordDict[1]) >= len("new" + str(i)) and "new" + str(i) in loadedVB.wordDict[1][len("new" + str(i)) - 1].keys():
                i += 1
            self.syllableList.list.lb.insert("end", "new" + str(i))
            while len(loadedVB.wordDict[1]) < len("new" + str(i)):
                loadedVB.wordDict[1].append(dict())
            loadedVB.wordDict[1][len("new" + str(i)) - 1]["new" + str(i)] = ""
            return
        self.syllableList.list.lb.insert("end", "new")
        while len(loadedVB.wordDict[1]) < len("new"):
            loadedVB.wordDict[1].append(dict())
        loadedVB.wordDict[1][len("new") - 1]["new"] = ""

    def onSyllableRemovePress(self):
        """remove the selected syllable mapping from the list"""

        global loadedVB

        if self.syllableList.list.lastFocusedIndex != None:
            item = self.syllableList.list.lb.get(self.syllableList.list.lastFocusedIndex)
            del loadedVB.wordDict[1][len(item) - 1][item]
            while len(loadedVB.wordDict[1][-1]) == 0:
                del loadedVB.wordDict[1][-1]
            self.syllableList.list.lb.delete(self.syllableList.list.lastFocusedIndex)
            self.syllableList.list.lastFocusedIndex = None

    def onSyllableSelectionChange(self, event):
        """update the syllable settings when the selection changes"""

        global loadedVB

        if len(self.syllableList.list.lb.curselection()) > 0:
            self.syllableList.list.lastFocusedIndex = self.syllableList.list.lb.curselection()[0]
            index = self.syllableList.list.lastFocusedIndex
            syllable = self.syllableList.list.lb.get(index)
            self.syllableSettings.syllable.variable.set(syllable)
            self.syllableSettings.mapping.variable.set(loadedVB.wordDict[1][len(syllable) - 1][syllable])

    def onSyllableListFocusOut(self, event) -> None:
        """Helper function for retaining information about the last focused element of the syllable list when syllable list loses entry focus"""

        logging.info("syllable list focus loss callback")
        if len(self.syllableList.list.lb.curselection()) > 0:
            self.syllableList.list.lastFocusedIndex = self.syllableList.list.lb.curselection()[0]

    def syllableUpdate(self, value):
        """updates the loaded Voicebank's wordDict when a syllable key is changed"""

        global loadedVB

        if self.syllableList.list.lastFocusedIndex != None:
            index = self.syllableList.list.lastFocusedIndex
            syllable = self.syllableList.list.lb.get(index)
            newSyllable = self.syllableSettings.syllable.variable.get()
            if syllable != newSyllable:
                while len(loadedVB.wordDict[1]) < len(newSyllable):
                    loadedVB.wordDict[1].append(dict())
                if newSyllable in loadedVB.wordDict[1][len(newSyllable) - 1].keys():
                    tkinter.messagebox.showerror(loc["error"], loc["worddict_syllable_exists_error"])
                    self.syllableSettings.syllable.variable.set(syllable)
                    return
                loadedVB.wordDict[1][len(newSyllable) - 1][newSyllable] = loadedVB.wordDict[1][len(syllable) - 1].pop(syllable)
                while len(loadedVB.wordDict[1][-1]) == 0:
                    del loadedVB.wordDict[1][-1]
                self.syllableList.list.lb.delete(index)
                self.syllableList.list.lb.insert(index, newSyllable)

    def syllableMappingUpdate(self, value):
        """updates the loaded Voicebank's wordDict when a syllable mapping is changed"""

        global loadedVB

        if self.syllableList.list.lastFocusedIndex != None:
            index = self.syllableList.list.lastFocusedIndex
            syllable = self.syllableList.list.lb.get(index)
            loadedVB.wordDict[1][len(syllable) - 1][syllable] = self.syllableSettings.mapping.variable.get()

    def onSyllableCSVImportPress(self):
        """imports a CSV file to the syllable list"""
        
        global loadedVB
        
        file = tkinter.filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv")])
        if file != "":
            loadedVB.wordDict[1] = []
            try:
                with open(file, "r", encoding = "utf-8") as f:
                    for i in f.readlines():
                        if i != "\n":
                            syllable = i.split(",")[0]
                            mapping = i.split(",")[1].strip()
                            while len(loadedVB.wordDict[1]) < len(syllable):
                                loadedVB.wordDict[1].append(dict())
                            if syllable in loadedVB.wordDict[1][len(syllable) - 1].keys():
                                tkinter.messagebox.showerror(loc["error"], loc["worddict_syllable_exists_error"])
                                return
                            loadedVB.wordDict[1][len(syllable) - 1][syllable] = mapping
                            self.syllableList.list.lb.insert("end", syllable)
            except:
                tkinter.messagebox.showerror(loc["error"], loc["worddict_csv_import_error"])
                return

    def onOkPress(self) -> None:
        """Updates the last selected override and syllable and their mappings, and closes the Dictionary UI window when the OK button is pressed"""

        logging.info("Wordict OK button callback")
        global loadedVB
        self.overrideWordUpdate(None)
        self.overrideMappingUpdate(None)
        self.syllableUpdate(None)
        self.syllableMappingUpdate(None)
        self.master.destroy()

    def onLoadPress(self) -> None:
        """UI Frontend function for loading the word dict of a different Voicebank"""

        logging.info("Worddict load button callback")
        global loadedVB
        additive =  tkinter.messagebox.askyesnocancel(loc["warning"], loc["additive_msg"], icon = "question")
        if additive != None:
            filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
            if filepath != "":
                loadedVB.loadWordDict(filepath, additive)
                for i in range(self.overrideList.list.lb.size()):
                    self.overrideList.list.lb.delete(0)
                for i in loadedVB.wordDict[0].keys():
                    self.overrideList.list.lb.insert("end", i)
                for i in range(self.syllableList.list.lb.size()):
                    self.syllableList.list.lb.delete(0)
                for i in loadedVB.wordDict[1]:
                    for j in i.keys():
                        self.syllableList.list.lb.insert("end", j)
        