#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import tkinter
import tkinter.messagebox
import logging
import sys
from ast import literal_eval

import torch

from Backend.VB_Components.Ai.TrAi import TrAi
from Backend.VB_Components.Ai.MainAi import MainAi, MainCritic
from Localization.devkit_localization import getLanguage
from UI.devkit.Widgets import *
loc = getLanguage()

class AdvSettingsUi(Frame):
    """class of the advanced AI settings UI window"""

    def __init__(self, master=None) -> None:
        logging.info("Initializing adv. AI settings UI")
        global loadedVB
        from UI.devkit.Main import loadedVB
        Frame.__init__(self, master)
        self.pack(ipadx = 20)
        self.createWidgets()
        self.master.wm_title(loc["advsettings"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("assets/icon/nova-vox-logo-black.ico")
        
    def createWidgets(self) -> None:
        """initializes all widgets of the adv. AI settings window. Called once during initialization"""

        global loadedVB

        self.hparams = LabelFrame(self, text = loc["hparams"])

        self.hparams.tr_lr = Frame(self.hparams)
        self.hparams.tr_lr.variable = tkinter.DoubleVar(self.hparams.tr_lr)
        self.hparams.tr_lr.variable.set(loadedVB.ai.hparams["tr_lr"])
        self.hparams.tr_lr.entry = Entry(self.hparams.tr_lr)
        self.hparams.tr_lr.entry["textvariable"] = self.hparams.tr_lr.variable
        self.hparams.tr_lr.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.tr_lr.display = Label(self.hparams.tr_lr)
        self.hparams.tr_lr.display["text"] = loc["tr_lr"]
        self.hparams.tr_lr.display.pack(side = "right", fill = "x")
        self.hparams.tr_lr.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.tr_reg = Frame(self.hparams)
        self.hparams.tr_reg.variable = tkinter.DoubleVar(self.hparams.tr_reg)
        self.hparams.tr_reg.variable.set(loadedVB.ai.hparams["tr_reg"])
        self.hparams.tr_reg.entry = Entry(self.hparams.tr_reg)
        self.hparams.tr_reg.entry["textvariable"] = self.hparams.tr_reg.variable
        self.hparams.tr_reg.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.tr_reg.display = Label(self.hparams.tr_reg)
        self.hparams.tr_reg.display["text"] = loc["tr_reg"]
        self.hparams.tr_reg.display.pack(side = "right", fill = "x")
        self.hparams.tr_reg.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.tr_hlc = Frame(self.hparams)
        self.hparams.tr_hlc.variable = tkinter.IntVar(self.hparams.tr_hlc)
        self.hparams.tr_hlc.variable.set(loadedVB.ai.hparams["tr_hlc"])
        self.hparams.tr_hlc.entry = Spinbox(self.hparams.tr_hlc, from_ = 0, to = 128)
        self.hparams.tr_hlc.entry["textvariable"] = self.hparams.tr_hlc.variable
        self.hparams.tr_hlc.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.tr_hlc.display = Label(self.hparams.tr_hlc)
        self.hparams.tr_hlc.display["text"] = loc["tr_hlc"]
        self.hparams.tr_hlc.display.pack(side = "right", fill = "x")
        self.hparams.tr_hlc.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.tr_hls = Frame(self.hparams)
        self.hparams.tr_hls.variable = tkinter.IntVar(self.hparams.tr_hls)
        self.hparams.tr_hls.variable.set(loadedVB.ai.hparams["tr_hls"])
        self.hparams.tr_hls.entry = Spinbox(self.hparams.tr_hls, from_ = 16, to = 4096)
        self.hparams.tr_hls.entry["textvariable"] = self.hparams.tr_hls.variable
        self.hparams.tr_hls.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.tr_hls.display = Label(self.hparams.tr_hls)
        self.hparams.tr_hls.display["text"] = loc["tr_hls"]
        self.hparams.tr_hls.display.pack(side = "right", fill = "x")
        self.hparams.tr_hls.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.tr_def_thrh = Frame(self.hparams)
        self.hparams.tr_def_thrh.variable = tkinter.DoubleVar(self.hparams.tr_def_thrh)
        self.hparams.tr_def_thrh.variable.set(loadedVB.ai.hparams["tr_def_thrh"])
        self.hparams.tr_def_thrh.entry = Entry(self.hparams.tr_def_thrh)
        self.hparams.tr_def_thrh.entry["textvariable"] = self.hparams.tr_def_thrh.variable
        self.hparams.tr_def_thrh.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.tr_def_thrh.display = Label(self.hparams.tr_def_thrh)
        self.hparams.tr_def_thrh.display["text"] = loc["tr_def_thrh"]
        self.hparams.tr_def_thrh.display.pack(side = "right", fill = "x")
        self.hparams.tr_def_thrh.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.latent_dim = Frame(self.hparams)
        self.hparams.latent_dim.variable = tkinter.IntVar(self.hparams.latent_dim)
        self.hparams.latent_dim.variable.set(loadedVB.ai.hparams["latent_dim"])
        self.hparams.latent_dim.entry = Spinbox(self.hparams.latent_dim, from_ = 1, to = 4096)
        self.hparams.latent_dim.entry["textvariable"] = self.hparams.latent_dim.variable
        self.hparams.latent_dim.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.latent_dim.display = Label(self.hparams.latent_dim)
        self.hparams.latent_dim.display["text"] = loc["latent_dim"]
        self.hparams.latent_dim.display.pack(side = "right", fill = "x")
        self.hparams.latent_dim.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.main_blkA = Frame(self.hparams)
        self.hparams.main_blkA.variable = tkinter.StringVar(self.hparams.main_blkA)
        self.hparams.main_blkA.variable.set(str(loadedVB.ai.hparams["main_blkA"]))
        self.hparams.main_blkA.entry = Entry(self.hparams.main_blkA)
        self.hparams.main_blkA.entry["textvariable"] = self.hparams.main_blkA.variable
        self.hparams.main_blkA.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.main_blkA.display = Label(self.hparams.main_blkA)
        self.hparams.main_blkA.display["text"] = loc["main_blkA"]
        self.hparams.main_blkA.display.pack(side = "right", fill = "x")
        self.hparams.main_blkA.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.main_blkB = Frame(self.hparams)
        self.hparams.main_blkB.variable = tkinter.StringVar(self.hparams.main_blkB)
        self.hparams.main_blkB.variable.set(str(loadedVB.ai.hparams["main_blkB"]))
        self.hparams.main_blkB.entry = Entry(self.hparams.main_blkB)
        self.hparams.main_blkB.entry["textvariable"] = self.hparams.main_blkB.variable
        self.hparams.main_blkB.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.main_blkB.display = Label(self.hparams.main_blkB)
        self.hparams.main_blkB.display["text"] = loc["main_blkB"]
        self.hparams.main_blkB.display.pack(side = "right", fill = "x")
        self.hparams.main_blkB.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.main_blkC = Frame(self.hparams)
        self.hparams.main_blkC.variable = tkinter.StringVar(self.hparams.main_blkC)
        self.hparams.main_blkC.variable.set(str(loadedVB.ai.hparams["main_blkC"]))
        self.hparams.main_blkC.entry = Entry(self.hparams.main_blkC)
        self.hparams.main_blkC.entry["textvariable"] = self.hparams.main_blkC.variable
        self.hparams.main_blkC.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.main_blkC.display = Label(self.hparams.main_blkC)
        self.hparams.main_blkC.display["text"] = loc["main_blkC"]
        self.hparams.main_blkC.display.pack(side = "right", fill = "x")
        self.hparams.main_blkC.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.main_lr = Frame(self.hparams)
        self.hparams.main_lr.variable = tkinter.DoubleVar(self.hparams.main_lr)
        self.hparams.main_lr.variable.set(loadedVB.ai.hparams["main_lr"])
        self.hparams.main_lr.entry = Entry(self.hparams.main_lr)
        self.hparams.main_lr.entry["textvariable"] = self.hparams.main_lr.variable
        self.hparams.main_lr.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.main_lr.display = Label(self.hparams.main_lr)
        self.hparams.main_lr.display["text"] = loc["main_lr"]
        self.hparams.main_lr.display.pack(side = "right", fill = "x")
        self.hparams.main_lr.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.main_reg = Frame(self.hparams)
        self.hparams.main_reg.variable = tkinter.DoubleVar(self.hparams.main_reg)
        self.hparams.main_reg.variable.set(loadedVB.ai.hparams["main_reg"])
        self.hparams.main_reg.entry = Entry(self.hparams.main_reg)
        self.hparams.main_reg.entry["textvariable"] = self.hparams.main_reg.variable
        self.hparams.main_reg.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.main_reg.display = Label(self.hparams.main_reg)
        self.hparams.main_reg.display["text"] = loc["main_reg"]
        self.hparams.main_reg.display.pack(side = "right", fill = "x")
        self.hparams.main_reg.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.main_drp = Frame(self.hparams)
        self.hparams.main_drp.variable = tkinter.DoubleVar(self.hparams.main_drp)
        self.hparams.main_drp.variable.set(loadedVB.ai.hparams["main_drp"])
        self.hparams.main_drp.entry = Entry(self.hparams.main_drp)
        self.hparams.main_drp.entry["textvariable"] = self.hparams.main_drp.variable
        self.hparams.main_drp.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.main_drp.display = Label(self.hparams.main_drp)
        self.hparams.main_drp.display["text"] = loc["main_drp"]
        self.hparams.main_drp.display.pack(side = "right", fill = "x")
        self.hparams.main_drp.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.crt_blkA = Frame(self.hparams)
        self.hparams.crt_blkA.variable = tkinter.StringVar(self.hparams.crt_blkA)
        self.hparams.crt_blkA.variable.set(str(loadedVB.ai.hparams["crt_blkA"]))
        self.hparams.crt_blkA.entry = Entry(self.hparams.crt_blkA)
        self.hparams.crt_blkA.entry["textvariable"] = self.hparams.crt_blkA.variable
        self.hparams.crt_blkA.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crt_blkA.display = Label(self.hparams.crt_blkA)
        self.hparams.crt_blkA.display["text"] = loc["crt_blkA"]
        self.hparams.crt_blkA.display.pack(side = "right", fill = "x")
        self.hparams.crt_blkA.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.crt_blkB = Frame(self.hparams)
        self.hparams.crt_blkB.variable = tkinter.StringVar(self.hparams.crt_blkB)
        self.hparams.crt_blkB.variable.set(str(loadedVB.ai.hparams["crt_blkB"]))
        self.hparams.crt_blkB.entry = Entry(self.hparams.crt_blkB)
        self.hparams.crt_blkB.entry["textvariable"] = self.hparams.crt_blkB.variable
        self.hparams.crt_blkB.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crt_blkB.display = Label(self.hparams.crt_blkB)
        self.hparams.crt_blkB.display["text"] = loc["crt_blkB"]
        self.hparams.crt_blkB.display.pack(side = "right", fill = "x")
        self.hparams.crt_blkB.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.crt_blkC = Frame(self.hparams)
        self.hparams.crt_blkC.variable = tkinter.StringVar(self.hparams.crt_blkC)
        self.hparams.crt_blkC.variable.set(str(loadedVB.ai.hparams["crt_blkC"]))
        self.hparams.crt_blkC.entry = Entry(self.hparams.crt_blkC)
        self.hparams.crt_blkC.entry["textvariable"] = self.hparams.crt_blkC.variable
        self.hparams.crt_blkC.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crt_blkC.display = Label(self.hparams.crt_blkC)
        self.hparams.crt_blkC.display["text"] = loc["crt_blkC"]
        self.hparams.crt_blkC.display.pack(side = "right", fill = "x")
        self.hparams.crt_blkC.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.crt_lr = Frame(self.hparams)
        self.hparams.crt_lr.variable = tkinter.DoubleVar(self.hparams.crt_lr)
        self.hparams.crt_lr.variable.set(loadedVB.ai.hparams["crt_lr"])
        self.hparams.crt_lr.entry = Entry(self.hparams.crt_lr)
        self.hparams.crt_lr.entry["textvariable"] = self.hparams.crt_lr.variable
        self.hparams.crt_lr.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crt_lr.display = Label(self.hparams.crt_lr)
        self.hparams.crt_lr.display["text"] = loc["crt_lr"]
        self.hparams.crt_lr.display.pack(side = "right", fill = "x")
        self.hparams.crt_lr.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.crt_reg = Frame(self.hparams)
        self.hparams.crt_reg.variable = tkinter.DoubleVar(self.hparams.crt_reg)
        self.hparams.crt_reg.variable.set(loadedVB.ai.hparams["crt_reg"])
        self.hparams.crt_reg.entry = Entry(self.hparams.crt_reg)
        self.hparams.crt_reg.entry["textvariable"] = self.hparams.crt_reg.variable
        self.hparams.crt_reg.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crt_reg.display = Label(self.hparams.crt_reg)
        self.hparams.crt_reg.display["text"] = loc["crt_reg"]
        self.hparams.crt_reg.display.pack(side = "right", fill = "x")
        self.hparams.crt_reg.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.crt_drp = Frame(self.hparams)
        self.hparams.crt_drp.variable = tkinter.DoubleVar(self.hparams.crt_drp)
        self.hparams.crt_drp.variable.set(loadedVB.ai.hparams["crt_drp"])
        self.hparams.crt_drp.entry = Entry(self.hparams.crt_drp)
        self.hparams.crt_drp.entry["textvariable"] = self.hparams.crt_drp.variable
        self.hparams.crt_drp.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crt_drp.display = Label(self.hparams.crt_drp)
        self.hparams.crt_drp.display["text"] = loc["crt_drp"]
        self.hparams.crt_drp.display.pack(side = "right", fill = "x")
        self.hparams.crt_drp.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.gan_guide_wgt = Frame(self.hparams)
        self.hparams.gan_guide_wgt.variable = tkinter.DoubleVar(self.hparams.gan_guide_wgt)
        self.hparams.gan_guide_wgt.variable.set(loadedVB.ai.hparams["gan_guide_wgt"])
        self.hparams.gan_guide_wgt.entry = Entry(self.hparams.gan_guide_wgt)
        self.hparams.gan_guide_wgt.entry["textvariable"] = self.hparams.gan_guide_wgt.variable
        self.hparams.gan_guide_wgt.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.gan_guide_wgt.display = Label(self.hparams.gan_guide_wgt)
        self.hparams.gan_guide_wgt.display["text"] = loc["gan_guide_wgt"]
        self.hparams.gan_guide_wgt.display.pack(side = "right", fill = "x")
        self.hparams.gan_guide_wgt.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.gan_train_asym = Frame(self.hparams)
        self.hparams.gan_train_asym.variable = tkinter.IntVar(self.hparams.gan_train_asym)
        self.hparams.gan_train_asym.variable.set(loadedVB.ai.hparams["gan_train_asym"])
        self.hparams.gan_train_asym.entry = Spinbox(self.hparams.gan_train_asym, from_ = 1, to = 256)
        self.hparams.gan_train_asym.entry["textvariable"] = self.hparams.gan_train_asym.variable
        self.hparams.gan_train_asym.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.gan_train_asym.display = Label(self.hparams.gan_train_asym)
        self.hparams.gan_train_asym.display["text"] = loc["gan_train_asym"]
        self.hparams.gan_train_asym.display.pack(side = "right", fill = "x")
        self.hparams.gan_train_asym.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.fargan_interval = Frame(self.hparams)
        self.hparams.fargan_interval.variable = tkinter.IntVar(self.hparams.fargan_interval)
        self.hparams.fargan_interval.variable.set(loadedVB.ai.hparams["fargan_interval"])
        self.hparams.fargan_interval.entry = Spinbox(self.hparams.fargan_interval, from_ = 2, to = 256)
        self.hparams.fargan_interval.entry["textvariable"] = self.hparams.fargan_interval
        self.hparams.fargan_interval.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.fargan_interval.display = Label(self.hparams.fargan_interval)
        self.hparams.fargan_interval.display["text"] = loc["fargan_interval"]
        self.hparams.fargan_interval.display.pack(side = "right", fill = "x")
        self.hparams.fargan_interval.pack(side = "top", fill = "x", padx = 5, pady = 2)
        
        self.hparams.embeddingDim = Frame(self.hparams)
        self.hparams.embeddingDim.variable = tkinter.IntVar(self.hparams.embeddingDim)
        self.hparams.embeddingDim.variable.set(loadedVB.ai.hparams["embeddingDim"])
        self.hparams.embeddingDim.entry = Spinbox(self.hparams.embeddingDim, from_ = 1, to = 256)
        self.hparams.embeddingDim.entry["textvariable"] = self.hparams.embeddingDim.variable
        self.hparams.embeddingDim.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.embeddingDim.display = Label(self.hparams.embeddingDim)
        self.hparams.embeddingDim.display["text"] = loc["embeddingDim"]
        self.hparams.embeddingDim.display.pack(side = "right", fill = "x")
        self.hparams.embeddingDim.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.resamp = LabelFrame(self, text = loc["advResamp"])
        self.resamp.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.okButton = Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

    def onOkPress(self) -> None:
        """applies all changes and closes the window when the OK button is pressed"""

        self.applyHyperParams()
        self.master.destroy()

    def applyHyperParams(self) -> None:
        """applies all changes set by the user, and resets the AI components of the currently loaded Voicebank if necessary"""

        global loadedVB

        hparams = {
            "tr_lr": self.hparams.tr_lr.variable.get(),
            "tr_reg": self.hparams.tr_reg.variable.get(),
            "tr_hlc": self.hparams.tr_hlc.variable.get(),
            "tr_hls": self.hparams.tr_hls.variable.get(),
            "tr_def_thrh" : self.hparams.tr_def_thrh.variable.get(),
            "latent_dim": self.hparams.latent_dim.variable.get(),
            "main_blkA": literal_eval(self.hparams.main_blkA.variable.get()),
            "main_blkB": literal_eval(self.hparams.main_blkB.variable.get()),
            "main_blkC": literal_eval(self.hparams.main_blkC.variable.get()),
            "main_lr": self.hparams.main_lr.variable.get(),
            "main_reg": self.hparams.main_reg.variable.get(),
            "main_drp": self.hparams.main_drp.variable.get(),
            "crt_blkA": literal_eval(self.hparams.crt_blkA.variable.get()),
            "crt_blkB": literal_eval(self.hparams.crt_blkB.variable.get()),
            "crt_blkC": literal_eval(self.hparams.crt_blkC.variable.get()),
            "crt_lr": self.hparams.crt_lr.variable.get(),
            "crt_reg": self.hparams.crt_reg.variable.get(),
            "crt_drp": self.hparams.crt_drp.variable.get(),
            "gan_guide_wgt": self.hparams.gan_guide_wgt.variable.get(),
            "gan_train_asym": self.hparams.gan_train_asym.variable.get(),
            "fargan_interval": self.hparams.fargan_interval.variable.get(),
            "embeddingDim": self.hparams.embeddingDim.variable.get(),
        }
        if (loadedVB.ai.hparams["tr_hlc"] != hparams["tr_hlc"]) or (loadedVB.ai.hparams["tr_hls"] != hparams["tr_hls"]):
            resetTr = tkinter.messagebox.askokcancel(loc["warning"], loc["tr_warn"])
        else:
            resetTr = False
        if any([any([i != j for i, j in zip(loadedVB.ai.hparams["main_blkA"], hparams["main_blkA"])]),
                any([i != j for i, j in zip(loadedVB.ai.hparams["main_blkB"], hparams["main_blkB"])]),
                any([i != j for i, j in zip(loadedVB.ai.hparams["main_blkC"], hparams["main_blkC"])]),
                any([i != j for i, j in zip(loadedVB.ai.hparams["crt_blkA"], hparams["crt_blkA"])]),
                any([i != j for i, j in zip(loadedVB.ai.hparams["crt_blkB"], hparams["crt_blkB"])]),
                any([i != j for i, j in zip(loadedVB.ai.hparams["crt_blkC"], hparams["crt_blkC"])]),
                loadedVB.ai.hparams["latent_dim"] != hparams["latent_dim"],
                loadedVB.ai.hparams["embeddingDim"] != hparams["embeddingDim"]]
           ):
            resetMain = tkinter.messagebox.askokcancel(loc["warning"], loc["pred_warn"])
        else:
            resetMain = False
        if resetTr:
            resetTrOptim = True
        elif ((loadedVB.ai.hparams["tr_lr"] != hparams["tr_lr"]) or (loadedVB.ai.hparams["tr_reg"] != hparams["tr_reg"])):
            resetTrOptim = tkinter.messagebox.askokcancel(loc["warning"], loc["tr_optim_warn"])
        else:
            resetTrOptim = False
        if resetMain:
            resetMainOptim = True
        elif ((loadedVB.ai.hparams["main_lr"] != hparams["main_lr"]) or (loadedVB.ai.hparams["main_reg"] != hparams["main_reg"]) or loadedVB.ai.hparams["fargan_interval"] != hparams["fargan_interval"]):
            resetMainOptim = tkinter.messagebox.askokcancel(loc["warning"], loc["pred_optim_warn"])
        else:
            resetMainOptim = False
        loadedVB.ai.hparams["tr_def_thrh"] = hparams["tr_def_thrh"]
        loadedVB.ai.hparams["main_drp"] = hparams["main_drp"]
        loadedVB.ai.hparams["crt_drp"] = hparams["crt_drp"]
        loadedVB.ai.hparams["gan_guide_wgt"] = hparams["gan_guide_wgt"]
        loadedVB.ai.hparams["gan_train_asym"] = hparams["gan_train_asym"]
        if resetTr:
            loadedVB.ai.hparams["tr_hlc"] = hparams["tr_hlc"]
            loadedVB.ai.hparams["tr_hls"] = hparams["tr_hls"]
            loadedVB.ai.trAi = TrAi(device = loadedVB.ai.device, learningRate=loadedVB.ai.hparams["tr_lr"], regularization=loadedVB.ai.hparams["tr_reg"], hiddenLayerCount=loadedVB.ai.hparams["tr_hlc"], hiddenLayerSize=loadedVB.ai.hparams["tr_hls"])
        if resetMain:
            loadedVB.ai.hparams["latent_dim"] = hparams["latent_dim"]
            loadedVB.ai.hparams["embeddingDim"] = hparams["embeddingDim"]
            loadedVB.ai.hparams["main_blkA"] = hparams["main_blkA"]
            loadedVB.ai.hparams["main_blkB"] = hparams["main_blkB"]
            loadedVB.ai.hparams["main_blkC"] = hparams["main_blkC"]
            loadedVB.ai.hparams["crt_blkA"] = hparams["crt_blkA"]
            loadedVB.ai.hparams["crt_blkB"] = hparams["crt_blkB"]
            loadedVB.ai.hparams["crt_blkC"] = hparams["crt_blkC"]
            loadedVB.ai.mainAi = MainAi(device = loadedVB.ai.device,
                                        dim = loadedVB.ai.hparams["latent_dim"],
                                        embedDim = loadedVB.ai.hparams["embeddingDim"],
                                        blockA = loadedVB.ai.hparams["main_blkA"],
                                        blockB = loadedVB.ai.hparams["main_blkB"],
                                        blockC = loadedVB.ai.hparams["main_blkC"],
                                        learningRate=loadedVB.ai.hparams["main_lr"],
                                        regularization=loadedVB.ai.hparams["main_reg"],
                                        dropout = loadedVB.ai.hparams["main_drp"])
            loadedVB.ai.mainCritic = MainCritic(device = loadedVB.ai.device,
                                        dim = loadedVB.ai.hparams["latent_dim"],
                                        embedDim = loadedVB.ai.hparams["embeddingDim"],
                                        blockA = loadedVB.ai.hparams["main_blkA"],
                                        blockB = loadedVB.ai.hparams["main_blkB"],
                                        blockC = loadedVB.ai.hparams["main_blkC"],
                                        learningRate=loadedVB.ai.hparams["main_lr"],
                                        regularization=loadedVB.ai.hparams["main_reg"],
                                        dropout = loadedVB.ai.hparams["main_drp"])
        if resetTrOptim:
            loadedVB.ai.hparams["tr_lr"] = hparams["tr_lr"]
            loadedVB.ai.hparams["tr_reg"] = hparams["tr_reg"]
            loadedVB.ai.trAiOptimizer = torch.optim.NAdam(loadedVB.ai.trAi.parameters(), lr=loadedVB.ai.trAi.learningRate, weight_decay=loadedVB.ai.trAi.regularization)
        if resetMainOptim:
            loadedVB.ai.hparams["main_lr"] = hparams["main_lr"]
            loadedVB.ai.hparams["main_reg"] = hparams["main_reg"]
            loadedVB.ai.hparams["crt_lr"] = hparams["crt_lr"]
            loadedVB.ai.hparams["crt_reg"] = hparams["crt_reg"]
            loadedVB.ai.hparams["fargan_interval"] = hparams["fargan_interval"]
            loadedVB.ai.mainAiOptimizer = [torch.optim.AdamW([*loadedVB.ai.mainAi.baseEncoder.parameters(), *loadedVB.ai.mainAi.baseDecoder.parameters()], lr=loadedVB.ai.mainAi.learningRate, weight_decay=loadedVB.ai.mainAi.regularization),
                                    torch.optim.AdamW([*loadedVB.ai.mainAi.encoderA.parameters(), *loadedVB.ai.mainAi.decoderA.parameters()], lr=loadedVB.ai.mainAi.learningRate, weight_decay=loadedVB.ai.mainAi.regularization),
                                    torch.optim.AdamW([*loadedVB.ai.mainAi.encoderB.parameters(), *loadedVB.ai.mainAi.decoderB.parameters()], lr=loadedVB.ai.mainAi.learningRate * 4, weight_decay=loadedVB.ai.mainAi.regularization),
                                    torch.optim.AdamW([*loadedVB.ai.mainAi.encoderC.parameters(), *loadedVB.ai.mainAi.decoderC.parameters()], lr=loadedVB.ai.mainAi.learningRate * 16, weight_decay=loadedVB.ai.mainAi.regularization)]
            loadedVB.ai.mainCriticOptimizer = [torch.optim.AdamW([*loadedVB.ai.mainCritic.baseEncoder.parameters(), *loadedVB.ai.mainCritic.baseDecoder.parameters(), *loadedVB.ai.mainCritic.final.parameters()], lr=loadedVB.ai.mainCritic.learningRate, weight_decay=loadedVB.ai.mainCritic.regularization),
                                        torch.optim.AdamW([*loadedVB.ai.mainCritic.encoderA.parameters(), *loadedVB.ai.mainCritic.decoderA.parameters()], lr=loadedVB.ai.mainCritic.learningRate, weight_decay=loadedVB.ai.mainCritic.regularization),
                                        torch.optim.AdamW([*loadedVB.ai.mainCritic.encoderB.parameters(), *loadedVB.ai.mainCritic.decoderB.parameters()], lr=loadedVB.ai.mainCritic.learningRate * 4, weight_decay=loadedVB.ai.mainCritic.regularization),
                                        torch.optim.AdamW([*loadedVB.ai.mainCritic.encoderC.parameters(), *loadedVB.ai.mainCritic.decoderC.parameters()], lr=loadedVB.ai.mainCritic.learningRate * 16, weight_decay=loadedVB.ai.mainCritic.regularization)]

    def applyResampSettings(self) -> None:
        """placeholder function for saving resampler settings into the Voicebank file"""

        global loadedVB

        pass
