import tkinter
import tkinter.messagebox
import logging
import sys

import torch

from Backend.VB_Components.SpecCrfAi import SpecCrfAi, SpecPredAi
from Locale.devkit_locale import getLocale
loc = getLocale()

class AdvSettingsUi(tkinter.Frame):

    def __init__(self, master=None) -> None:
        logging.info("Initializing adv. AI settings UI")
        global loadedVB
        from UI.code.devkit.Main import loadedVB
        tkinter.Frame.__init__(self, master)
        self.pack(ipadx = 20)
        self.createWidgets()
        self.master.wm_title(loc["advsettings"])
        if (sys.platform.startswith('win')): 
            self.master.iconbitmap("icon/nova-vox-logo-black.ico")
        
    def createWidgets(self) -> None:
        """initializes all widgets of the adv. AI settings window. Called once during initialization"""

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

        self.hparams.crf_def_thrh = tkinter.Frame(self.hparams)
        self.hparams.crf_def_thrh.variable = tkinter.DoubleVar(self.hparams.crf_def_thrh)
        self.hparams.crf_def_thrh.variable.set(loadedVB.ai.hparams["crf_def_thrh"])
        self.hparams.crf_def_thrh.entry = tkinter.Entry(self.hparams.crf_def_thrh)
        self.hparams.crf_def_thrh.entry["textvariable"] = self.hparams.crf_def_thrh.variable
        self.hparams.crf_def_thrh.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.crf_def_thrh.display = tkinter.Label(self.hparams.crf_def_thrh)
        self.hparams.crf_def_thrh.display["text"] = loc["crf_def_thrh"]
        self.hparams.crf_def_thrh.display.pack(side = "right", fill = "x")
        self.hparams.crf_def_thrh.pack(side = "top", fill = "x", padx = 5, pady = 2)

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

        self.hparams.pred_rlc = tkinter.Frame(self.hparams)
        self.hparams.pred_rlc.variable = tkinter.IntVar(self.hparams.pred_rlc)
        self.hparams.pred_rlc.variable.set(loadedVB.ai.hparams["pred_rlc"])
        self.hparams.pred_rlc.entry = tkinter.Entry(self.hparams.pred_rlc)
        self.hparams.pred_rlc.entry["textvariable"] = self.hparams.pred_rlc.variable
        self.hparams.pred_rlc.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.pred_rlc.display = tkinter.Label(self.hparams.pred_rlc)
        self.hparams.pred_rlc.display["text"] = loc["pred_rlc"]
        self.hparams.pred_rlc.display.pack(side = "right", fill = "x")
        self.hparams.pred_rlc.pack(side = "top", fill = "x", padx = 5, pady = 2)

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

        self.hparams.predh_lr = tkinter.Frame(self.hparams)
        self.hparams.predh_lr.variable = tkinter.DoubleVar(self.hparams.predh_lr)
        self.hparams.predh_lr.variable.set(loadedVB.ai.hparams["predh_lr"])
        self.hparams.predh_lr.entry = tkinter.Entry(self.hparams.predh_lr)
        self.hparams.predh_lr.entry["textvariable"] = self.hparams.predh_lr.variable
        self.hparams.predh_lr.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.predh_lr.display = tkinter.Label(self.hparams.predh_lr)
        self.hparams.predh_lr.display["text"] = loc["predh_lr"]
        self.hparams.predh_lr.display.pack(side = "right", fill = "x")
        self.hparams.predh_lr.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.predh_reg = tkinter.Frame(self.hparams)
        self.hparams.predh_reg.variable = tkinter.DoubleVar(self.hparams.predh_reg)
        self.hparams.predh_reg.variable.set(loadedVB.ai.hparams["predh_reg"])
        self.hparams.predh_reg.entry = tkinter.Entry(self.hparams.predh_reg)
        self.hparams.predh_reg.entry["textvariable"] = self.hparams.predh_reg.variable
        self.hparams.predh_reg.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.predh_reg.display = tkinter.Label(self.hparams.predh_reg)
        self.hparams.predh_reg.display["text"] = loc["predh_reg"]
        self.hparams.predh_reg.display.pack(side = "right", fill = "x")
        self.hparams.predh_reg.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.predh_rlc = tkinter.Frame(self.hparams)
        self.hparams.predh_rlc.variable = tkinter.IntVar(self.hparams.predh_rlc)
        self.hparams.predh_rlc.variable.set(loadedVB.ai.hparams["predh_rlc"])
        self.hparams.predh_rlc.entry = tkinter.Entry(self.hparams.predh_rlc)
        self.hparams.predh_rlc.entry["textvariable"] = self.hparams.predh_rlc.variable
        self.hparams.predh_rlc.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.predh_rlc.display = tkinter.Label(self.hparams.predh_rlc)
        self.hparams.predh_rlc.display["text"] = loc["predh_rlc"]
        self.hparams.predh_rlc.display.pack(side = "right", fill = "x")
        self.hparams.predh_rlc.pack(side = "top", fill = "x", padx = 5, pady = 2)

        self.hparams.predh_rs = tkinter.Frame(self.hparams)
        self.hparams.predh_rs.variable = tkinter.IntVar(self.hparams.predh_rs)
        self.hparams.predh_rs.variable.set(loadedVB.ai.hparams["predh_rs"])
        self.hparams.predh_rs.entry = tkinter.Spinbox(self.hparams.predh_rs, from_ = 16, to = 4096)
        self.hparams.predh_rs.entry["textvariable"] = self.hparams.predh_rs.variable
        self.hparams.predh_rs.entry.pack(side = "right", fill = "x", expand = True)
        self.hparams.predh_rs.display = tkinter.Label(self.hparams.predh_rs)
        self.hparams.predh_rs.display["text"] = loc["predh_rs"]
        self.hparams.predh_rs.display.pack(side = "right", fill = "x")
        self.hparams.predh_rs.pack(side = "top", fill = "x", padx = 5, pady = 2)

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
            "crf_lr": self.hparams.crf_lr.variable.get(),
            "crf_reg": self.hparams.crf_reg.variable.get(),
            "crf_hlc": self.hparams.crf_hlc.variable.get(),
            "crf_hls": self.hparams.crf_hls.variable.get(),
            "crf_def_thrh" : 0.05,
            "pred_lr": self.hparams.pred_lr.variable.get(),
            "pred_reg": self.hparams.pred_reg.variable.get(),
            "pred_rlc": self.hparams.pred_rlc.variable.get(),
            "pred_rs": self.hparams.pred_rs.variable.get(),
            "predh_lr": self.hparams.predh_lr.variable.get(),
            "predh_reg": self.hparams.predh_reg.variable.get(),
            "predh_rlc": self.hparams.predh_rlc.variable.get(),
            "predh_rs": self.hparams.predh_rs.variable.get()
        }
        resetCrf = False
        resetPred = False
        if (loadedVB.ai.hparams["crf_hlc"] != hparams["crf_hlc"]) or (loadedVB.ai.hparams["crf_hls"] != hparams["crf_hls"]):
            resetCrf = tkinter.messagebox.askokcancel(loc["warning"], loc["crf_warn"])
        if loadedVB.ai.hparams["pred_rs"] != hparams["pred_rs"]:
            resetPred = tkinter.messagebox.askokcancel(loc["warning"], loc["pred_warn"])
        if ((loadedVB.ai.hparams["crf_lr"] != hparams["crf_lr"]) or (loadedVB.ai.hparams["crf_reg"] != hparams["crf_reg"])) and (resetCrf == False):
            resetCrfOptim = tkinter.messagebox.askokcancel(loc["warning"], loc["crf_optim_warn"])
        else:
            resetCrfOptim = True
        if ((loadedVB.ai.hparams["pred_lr"] != hparams["pred_lr"]) or (loadedVB.ai.hparams["pred_reg"] != hparams["pred_reg"])) and (resetPred == False):
            resetPredOptim = tkinter.messagebox.askokcancel(loc["warning"], loc["pred_optim_warn"])
        else:
            resetPredOptim = True
        if resetCrf:
            loadedVB.ai.hparams["crf_hlc"] = hparams["crf_hlc"]
            loadedVB.ai.hparams["crf_hls"] = hparams["crf_hls"]
            loadedVB.ai.crfAi = SpecCrfAi(device = loadedVB.ai.device, learningRate=loadedVB.ai.hparams["crf_lr"], regularization=loadedVB.ai.hparams["crf_reg"], hiddenLayerCount=loadedVB.ai.hparams["crf_hlc"], hiddenLayerSize=loadedVB.ai.hparams["crf_hls"])
        if resetPred:
            loadedVB.ai.hparams["pred_rs"] = hparams["pred_rs"]
            loadedVB.ai.predAi = SpecPredAi(device = loadedVB.ai.device, learningRate=loadedVB.ai.hparams["pred_lr"], regularization=loadedVB.ai.hparams["pred_reg"], recSize=loadedVB.ai.hparams["rs"])
        if resetCrfOptim:
            loadedVB.ai.hparams["crf_lr"] = hparams["crf_lr"]
            loadedVB.ai.hparams["crf_reg"] = hparams["crf_reg"]
            loadedVB.ai.crfAiOptimizer = torch.optim.Adam(loadedVB.ai.crfAi.parameters(), lr=loadedVB.ai.crfAi.learningRate, weight_decay=loadedVB.ai.crfAi.regularization)
        if resetPredOptim:
            loadedVB.ai.hparams["pred_lr"] = hparams["pred_lr"]
            loadedVB.ai.hparams["pred_reg"] = hparams["pred_reg"]
            loadedVB.ai.predAiOptimizer = torch.optim.Adam(loadedVB.ai.predAi.parameters(), lr=loadedVB.ai.predAi.learningRate, weight_decay=loadedVB.ai.predAi.regularization)

    def applyResampSettings(self) -> None:

        global loadedVB

        pass