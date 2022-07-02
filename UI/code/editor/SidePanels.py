from kivy.core.image import Image as CoreImage

from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.uix.label import Label

from tkinter import Tk, filedialog

from io import BytesIO

import os
import torch
import subprocess

import sounddevice
import soundfile

import global_consts

from MiddleLayer.IniParser import readSettings, writeSettings

from UI.code.editor.Util import ListElement


class FileSidePanel(ModalView):
    def openRenderPopup(self):
        FileRenderPopup().open()

class FileRenderPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final = False
        self.format = "WAV"
        self.bitdepth = "PCM_24"
        self.sampleRate = "48000"
    def reloadBitdepths(self, format):
        self.children[0].children[0].children[0].children[1].children[1].values = soundfile.available_subtypes(format).keys()
        self.children[0].children[0].children[0].children[1].children[1].text = self.children[0].children[0].children[0].children[1].children[1].values[0]
        self.format = format
    def render(self, path):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        data = torch.zeros_like(middleLayer.audioBuffer[0])
        for i in range(len(middleLayer.audioBuffer)):
            data += middleLayer.audioBuffer[i] * middleLayer.trackList[i].volume
            data = data.numpy()
        with soundfile.SoundFile(path, "w", int(self.sampleRate), 1, self.bitdepth, None, self.format) as file:
            file.write(data)
    def finalisingClose(self):
        self.final = True
        self.dismiss()
    def on_dismiss(self):
        if self.final == False:
            return False
        tkui = Tk()
        tkui.withdraw()
        dir = filedialog.asksaveasfilename(filetypes = ((self.format, self.format.lower()), ("all files", "*")))
        tkui.destroy()
        if dir == "":
            self.final = False
            return True
        self.render(dir + "." + self.format.lower())
        return False

class SingerSidePanel(ModalView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voicebanks = []
        self.filepaths = []
        self.selectedIndex = None
    def listVoicebanks(self): 
        voicePath = os.path.join(readSettings()["dataDir"], "Voices")
        if os.path.isdir(voicePath) == False:
            popup = Popup(title = "error", content = Label(text = "no valid data directory"), size_hint = (None, None), size = (400, 400))
            popup.open()
            return
        files = os.listdir(voicePath)
        for file in files:
            if file.endswith(".nvvb"):
                data = torch.load(os.path.join(voicePath, file))
                self.voicebanks.append(data["metadata"])
                self.filepaths.append(os.path.join(voicePath, file))
        j = 0
        for i in self.voicebanks:
            self.ids["singers_list"].add_widget(ListElement(text = i.name, index = j))
            j += 1
    def detailElement(self, index):
        self.ids["singer_name"].text = self.voicebanks[index].name
        canvas_img = self.voicebanks[index].image
        data = BytesIO()
        canvas_img.save(data, format='png')
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext='png')
        self.ids["singer_image"].texture = im.texture
        self.ids["singer_version"].text = self.voicebanks[index].version
        self.ids["singer_description"].text = self.voicebanks[index].description
        self.ids["singer_license"].text = self.voicebanks[index].license
        self.selectedIndex = index
    def importVoicebank(self, path, name, image):
        global middleLayer
        middleLayer.importVoicebank(path, name, image)

class ParamSidePanel(ModalView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = []
        self.filepaths = []
        self.selectedIndex = None
    def listParams(self):
        paramPath = os.path.join(readSettings()["dataDir"], "Parameters")
        if os.path.isdir(paramPath) == False:
            popup = Popup(title = "error", content = Label(text = "no valid data directory"), size_hint = (None, None), size = (400, 400))
            popup.open()
            return
        files = os.listdir(paramPath)
        for file in files:
            if file.endswith(".nvpr"):
                data = torch.load(os.path.join(paramPath, file))
                self.parameters.append(data["metadata"])
                self.filepaths.append(os.path.join(paramPath, file))
        j = 0
        for i in self.parameters:
            self.ids["params_list"].add_widget(ListElement(text = i.name, index = j))
            j += 1
    def detailElement(self, index):
        self.ids["param_name"].text = self.voicebanks[index].name
        self.ids["param_type"].text = self.voicebanks[index]._type
        self.ids["param_capacity"].text = self.voicebanks[index].capacity
        self.ids["param_recurrency"].text = self.voicebanks[index].recurrency
        self.ids["param_version"].text = self.voicebanks[index].version
        self.ids["param_license"].text = self.voicebanks[index].license
        self.selectedIndex = index
    def importVoicebank(self, path, name):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.importParam(path, name)

class ScriptingSidePanel(ModalView):
    def openDevkit(self):
        try:
            subprocess.Popen("Devkit.exe")
        except:
            popup = Popup(title = "error", content = Label(text = "Devkit not installed"), size_hint = (None, None), size = (400, 400))
            popup.open()
    def runScript(self):
        try:
            exec(self.ids["scripting_editor"].text)
        except Exception as e:
            popup = Popup(title = "script error", content = Label(text = repr(e)), size_hint = (None, None), size = (400, 400))
            popup.open()
    def saveCache(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.scriptCache = self.ids["scripting_editor"].text
    def loadCache(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        self.ids["scripting_editor"].text = middleLayer.scriptCache
    

class SettingsSidePanel(ModalView):
    def __init__(self, **kwargs):
        audioApis = sounddevice.query_hostapis()
        self.audioApiNames =  []
        for i in audioApis:
            self.audioApiNames.append(i["name"])
        #self.refreshAudioDevices()
        self.audioDeviceNames = []
        super().__init__(**kwargs)
    def refreshAudioDevices(self, api):
        self.audioDeviceNames = []
        devices = sounddevice.query_hostapis(self.audioApiNames.index(api))["devices"]
        for i in devices:
            device = sounddevice.query_devices(i)
            if device["max_output_channels"]:
                self.audioDeviceNames.append(device["name"])
        self.ids["settings_audioDevice"].values = self.audioDeviceNames
    def restartAudioStream(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.audioStream.close()
        identifier = self.ids["settings_audioDevice"].text + ", " + self.ids["settings_audioApi"].text
        middleLayer.audioStream = sounddevice.OutputStream(global_consts.sampleRate, global_consts.audioBufferSize, identifier, callback = middleLayer.playCallback)
    def changeDataDir(self):
        tkui = Tk()
        tkui.withdraw()
        newDir = filedialog.askdirectory()
        if newDir != "":
            self.ids["settings_datadir"].text = newDir
        tkui.destroy()
    def readSettings(self):
        settings = readSettings()
        self.ids["settings_lang"].text = settings["language"]
        self.ids["settings_accel"].text = settings["accelerator"]
        self.ids["settings_tcores"].text = settings["tensorCores"]
        self.ids["settings_lowSpecMode"].text = settings["lowSpecMode"]
        self.ids["settings_cachingMode"].text = settings["cachingMode"]
        self.ids["settings_audioApi"].text = settings["audioApi"]
        self.refreshAudioDevices(settings["audioApi"])
        self.ids["settings_audioDevice"].text = settings["audioDevice"]
        self.ids["settings_loglevel"].text = settings["loglevel"]
        self.ids["settings_datadir"].text = settings["dataDir"]
    def writeSettings(self):
        if self.ids["settings_audioDevice"].text in self.audioDeviceNames:
            audioDevice = self.ids["settings_audioDevice"].text
        else:
            audioDevice = self.audioDeviceNames[0]
        writeSettings(None, self.ids["settings_lang"].text, self.ids["settings_accel"].text, self.ids["settings_tcores"].text, self.ids["settings_lowSpecMode"].text, self.ids["settings_cachingMode"].text, self.ids["settings_audioApi"].text, audioDevice, self.ids["settings_loglevel"].text, self.ids["settings_datadir"].text)
        self.restartAudioStream()