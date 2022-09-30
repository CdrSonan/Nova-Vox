import tkinter
import tkinter.simpledialog
import logging
import sys

import torch

from Backend.VB_Components.SpecCrfAi import SpecCrfAi, SpecPredAI
from Locale.devkit_locale import getLocale
loc = getLocale()

class AdvSettingsUi(tkinter.Frame):

    def __init__(self, master=None) -> None:
        logging.info("Initializing Metadata UI")
        global loadedVB
        from UI.code.devkit.Main import loadedVB
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20)
        self.createWidgets()
        self.master.wm_title(loc["metadat_lbl"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
    def createWidgets(self) -> None:
        """initializes all widgets of the Metadata window. Called once during initialization"""

        global loadedVB

        self.hparams = tkinter.LabelFrame(self, text = loc["hparams"])

        self.hparams.crf_lr = tkinter.Frame(self.hparams)
        self.hparams.crf_lr.variable = tkinter.DoubleVar(self.hparams.crf_lr)
        self.hparams.crf_lr.variable.set(loadedVB.ai.hparams["crf_lr"])
        self.hparams.crf_lr.entry = tkinter.Entry(self.hparams.crf_lr)
        self.hparams.crf_lr.entry["textvariable"] = self.hparams.crf_lr.variable
        self.hparams.crf_lr.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crf_lr.display = tkinter.Label(self.hparams.crf_lr)
        self.hparams.crf_lr.display["text"] = loc["crf_lr"]
        self.hparams.crf_lr.display.pack(side = "right", fill = "x")
        self.hparams.crf_lr.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.crf_reg = tkinter.Frame(self.hparams)
        self.hparams.crf_reg.variable = tkinter.DoubleVar(self.hparams.crf_reg)
        self.hparams.crf_reg.variable.set(loadedVB.ai.hparams["crf_reg"])
        self.hparams.crf_reg.entry = tkinter.Entry(self.hparams.crf_reg)
        self.hparams.crf_reg.entry["textvariable"] = self.hparams.crf_reg.variable
        self.hparams.crf_reg.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crf_reg.display = tkinter.Label(self.hparams.crf_reg)
        self.hparams.crf_reg.display["text"] = loc["crf_reg"]
        self.hparams.crf_reg.display.pack(side = "right", fill = "x")
        self.hparams.crf_reg.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.crf_hlc = tkinter.Frame(self.hparams)
        self.hparams.crf_hlc.variable = tkinter.IntVar(self.hparams.crf_hlc)
        self.hparams.crf_hlc.variable.set(loadedVB.ai.hparams["crf_hlc"])
        self.hparams.crf_hlc.entry = tkinter.Spinbox(self.hparams.crf_hlc, from_ = 0, to = 128)
        self.hparams.crf_hlc.entry["textvariable"] = self.hparams.crf_hlc.variable
        self.hparams.crf_hlc.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crf_hlc.display = tkinter.Label(self.hparams.crf_hlc)
        self.hparams.crf_hlc.display["text"] = loc["crf_hlc"]
        self.hparams.crf_hlc.display.pack(side = "right", fill = "x")
        self.hparams.crf_hlc.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.crf_hls = tkinter.Frame(self.hparams)
        self.hparams.crf_hls.variable = tkinter.IntVar(self.hparams.crf_hls)
        self.hparams.crf_hls.variable.set(loadedVB.ai.hparams["crf_hls"])
        self.hparams.crf_hls.entry = tkinter.Spinbox(self.hparams.crf_hls, from_ = 16, to = 4096)
        self.hparams.crf_hls.entry["textvariable"] = self.hparams.crf_hls.variable
        self.hparams.crf_hls.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crf_hls.display = tkinter.Label(self.hparams.crf_hls)
        self.hparams.crf_hls.display["text"] = loc["crf_hls"]
        self.hparams.crf_hls.display.pack(side = "right", fill = "x")
        self.hparams.crf_hls.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.pred_lr = tkinter.Frame(self.hparams)
        self.hparams.pred_lr.variable = tkinter.DoubleVar(self.hparams.pred_lr)
        self.hparams.pred_lr.variable.set(loadedVB.ai.hparams["pred_lr"])
        self.hparams.pred_lr.entry = tkinter.Entry(self.hparams.pred_lr)
        self.hparams.pred_lr.entry["textvariable"] = self.hparams.pred_lr.variable
        self.hparams.pred_lr.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.pred_lr.display = tkinter.Label(self.hparams.pred_lr)
        self.hparams.pred_lr.display["text"] = loc["pred_lr"]
        self.hparams.pred_lr.display.pack(side = "right", fill = "x")
        self.hparams.pred_lr.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.pred_reg = tkinter.Frame(self.hparams)
        self.hparams.pred_reg.variable = tkinter.DoubleVar(self.hparams.pred_reg)
        self.hparams.pred_reg.variable.set(loadedVB.ai.hparams["pred_reg"])
        self.hparams.pred_reg.entry = tkinter.Entry(self.hparams.pred_reg)
        self.hparams.pred_reg.entry["textvariable"] = self.hparams.pred_reg.variable
        self.hparams.pred_reg.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.pred_reg.display = tkinter.Label(self.hparams.pred_reg)
        self.hparams.pred_reg.display["text"] = loc["pred_reg"]
        self.hparams.pred_reg.display.pack(side = "right", fill = "x")
        self.hparams.pred_reg.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.pred_rs = tkinter.Frame(self.hparams)
        self.hparams.pred_rs.variable = tkinter.IntVar(self.hparams.pred_rs)
        self.hparams.pred_rs.variable.set(loadedVB.ai.hparams["pred_rs"])
        self.hparams.pred_rs.entry = tkinter.Spinbox(self.hparams.pred_rs, from_ = 16, to = 4096)
        self.hparams.pred_rs.entry["textvariable"] = self.hparams.pred_rs.variable
        self.hparams.pred_rs.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.pred_rs.display = tkinter.Label(self.hparams.pred_rs)
        self.hparams.pred_rs.display["text"] = loc["pred_rs"]
        self.hparams.pred_rs.display.pack(side = "right", fill = "x")
        self.hparams.pred_rs.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.resamp = tkinter.LabelFrame(self, text = loc["advResamp"])
        self.resamp.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.okButton = tkinter.Button(self)
        self.okButton["text"] = loc["ok"]
        self.okButton["command"] = self.onOkPress
        self.okButton.pack(side = "right", fill = "x", expand = True, padx = 10, pady = 10)

    def onOkPress(self) -> None:

        self.applyHyperParams()
        self.master.destroy()

    def applyHyperParams(self) -> None:

        global loadedVB

        hparams = {
            "crf_lr":self.hparams.crf_lr.variable.get(),
            "crf_reg":self.hparams.crf_reg.variable.get(),
            "crf_hlc":self.hparams.crf_hlc.variable.get(),
            "crf_hls":self.hparams.crf_hls.variable.get(),
            "pred_lr":self.hparams.pred_lr.variable.get(),
            "pred_reg":self.hparams.pred_reg.variable.get(),
            "pred_rs":self.hparams.pred_rs.variable.get()
        }
        resetCrf = False
        resetPred = False
        if (loadedVB.ai.hparams["crf_lr"] != hparams["crf_lr"]) or (loadedVB.ai.hparams["crf_reg"] != hparams["crf_reg"]) or (loadedVB.ai.hparams["crf_hlc"] != hparams["crf_hlc"]) or (loadedVB.ai.hparams["crf_hls"] != hparams["crf_hls"]):
            resetCrf = True
        if (loadedVB.ai.hparams["pred_lr"] != hparams["pred_lr"]) or (loadedVB.ai.hparams["pred_reg"] != hparams["pred_reg"]) or (loadedVB.ai.hparams["pred_hs"] != hparams["pred_hs"]):
            resetPred = True
        loadedVB.ai.hparams = hparams
        if resetCrf:
            loadedVB.ai.crfAi = SpecCrfAi(device = loadedVB.ai.device, learningRate=loadedVB.ai.hparams["crf_lr"], regularization=loadedVB.ai.hparams["crf_reg"], hiddenLayerCount=loadedVB.ai.hparams["crf_hlc"], hiddenLayerSize=loadedVB.ai.hparams["crf_hls"])
            loadedVB.ai.crfAiOptimizer = torch.optim.Adam(loadedVB.ai.crfAi.parameters(), lr=loadedVB.ai.crfAi.learningRate, weight_decay=loadedVB.ai.crfAi.regularization)
        if resetPred:
            loadedVB.ai.predAi = SpecPredAI(device = loadedVB.ai.device, learningRate=loadedVB.ai.hparams["pred_lr"], regularization=loadedVB.ai.hparams["pred_reg"], recSize=loadedVB.ai.hparams["rs"])
            loadedVB.ai.predAiOptimizer = torch.optim.Adam(loadedVB.ai.predAi.parameters(), lr=loadedVB.ai.predAi.learningRate, weight_decay=loadedVB.ai.predAi.regularization)

    def applyResampSettings(self) -> None:

        global loadedVB

        pass