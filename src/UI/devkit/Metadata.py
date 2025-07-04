#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import tkinter
import sys
from PIL import Image, ImageTk

from UI.devkit.Widgets import *
import global_consts
from Localization.devkit_localization import getLanguage
loc = getLanguage()

class MetadataUi(Frame):
    """Class of the Metadata window"""

    def __init__(self, master=None) -> None:
        logging.info("Initializing Metadata UI")
        global loadedVB
        from UI.devkit.Main import loadedVB
        Frame.__init__(self, master)
        self.pack(ipadx = 20)
        self.createWidgets()
        self.master.wm_title(loc["metadat_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("assets/icon/nova-vox-logo-black.ico")
        
    def createWidgets(self) -> None:
        """initializes all widgets of the Metadata window. Called once during initialization"""

        global loadedVB

        self.name = Frame(self)
        self.name.variable = tkinter.StringVar(self.name)
        self.name.variable.set(loadedVB.metadata.name)
        self.name.entry = Entry(self.name)
        self.name.entry["textvariable"] = self.name.variable
        self.name.entry.pack(side = "right", fill = "x", expand = True)
        self.name.display = Label(self.name)
        self.name.display["text"] = loc["name"]
        self.name.display.pack(side = "right", fill = "x")
        self.name.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.image = Frame(self)
        self.image.variable = tkinter.StringVar(self.image)
        self.image.variable.set("assets/UI/TrackList/SingerGrey04.png")
        self.image.entry = Button(self.image)
        self.image.entry["text"] = loc["change"]
        self.image.entry["command"] = self.onImagePress
        self.image.entry.pack(side = "bottom", fill = "x", expand = True)
        self.image.display = tkinter.Canvas(self.image,width=200,height=200)
        self.image.display.pack(side = "right", fill = "x")
        self.image.label = Label(self.image)
        self.image.label["text"] = loc["image"]
        self.image.label.pack(side = "left", fill = "x")
        self.image.pack(side = "top", fill = "x", padx = 5, pady = 2)
        self.applyImage()

        self.version = Frame(self)
        self.version.variable = tkinter.StringVar(self.version)
        self.version.variable.set(loadedVB.metadata.version)
        self.version.entry = Entry(self.version)
        self.version.entry["textvariable"] = self.version.variable
        self.version.entry.pack(side = "right", fill = "x", expand = True)
        self.version.display = Label(self.version)
        self.version.display["text"] = loc["version"]
        self.version.display.pack(side = "right", fill = "x")
        self.version.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.description = Frame(self)
        self.description.variable = tkinter.StringVar(self.description)
        self.description.variable.set(loadedVB.metadata.description)
        self.description.entry = Entry(self.description)
        self.description.entry["textvariable"] = self.description.variable
        self.description.entry.pack(side = "right", fill = "x", expand = True)
        self.description.display = Label(self.description)
        self.description.display["text"] = loc["description"]
        self.description.display.pack(side = "right", fill = "x")
        self.description.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.license = Frame(self)
        self.license.variable = tkinter.StringVar(self.license)
        self.license.variable.set(loadedVB.metadata.license)
        self.license.entry = Entry(self.license)
        self.license.entry["textvariable"] = self.license.variable
        self.license.entry.pack(side = "right", fill = "x", expand = True)
        self.license.display = Label(self.license)
        self.license.display["text"] = loc["license"]
        self.license.display.pack(side = "right", fill = "x")
        self.license.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.okButton = Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

        self.loadButton = Button(self)
        self.loadButton["text"] = loc["load_other_VB"]
        self.loadButton["command"] = self.onLoadPress
        self.loadButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

    def applyImage(self) -> None:
        """helper function for showing the image saved as part of the loaded Voicebank in the UI"""

        logging.info("Metadata Image change button callback")
        global loadedVB
        self.storedImage = ImageTk.PhotoImage(loadedVB.metadata.image, master = self.image.display)
        self.image.display.delete(1)
        self.image.display.create_image(100, 100, image = self.storedImage)
        

    def onImagePress(self) -> None:
        """opens a file browser to select a different image file for the Voicebank"""

        logging.info("Metadata Image change button callback")
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc["all_files_desc"], "*"),))
        if filepath == "":
            return
        pilImage = Image.open(filepath)
        pilImage = pilImage.resize((200, 200), resample = 3)
        loadedVB.metadata.image = pilImage
        self.applyImage()
        
    def onOkPress(self) -> None:
        """Applies all changes and closes the window when the OK button is pressed"""

        logging.info("Metadata OK button callback")
        global loadedVB
        loadedVB.metadata.name = self.name.variable.get()
        loadedVB.metadata.sampleRate = global_consts.sampleRate
        loadedVB.metadata.version = self.version.variable.get()
        loadedVB.metadata.description = self.description.variable.get()
        loadedVB.metadata.license = self.license.variable.get()
        self.master.destroy()

    def onLoadPress(self) -> None:
        """Opens a file browser, and loads the Voicebank metadata from a specified .nvvb file"""

        logging.info("Metadata load button callback")
        global loadedVB
        filepath = tkinter.filedialog.askopenfilename(filetypes = ((loc[".nvvb_desc"], ".nvvb"), (loc["all_files_desc"], "*")))
        loadedVB.loadMetadata(filepath)
        self.name.variable.set(loadedVB.metadata.name)
        self.applyImage()
