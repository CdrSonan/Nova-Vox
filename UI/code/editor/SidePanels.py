from kivy.core.image import Image as CoreImage

from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.app import App
from kivy.properties import NumericProperty, ColorProperty

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
    """Side panel for saving, loading, importing, exporting and rendering files"""

    uiScale = NumericProperty()
    toolColor = ColorProperty()
    accColor = ColorProperty()
    bgColor = ColorProperty()

    def openRenderPopup(self) -> None:
        """opens a popup containing the settings for rendering files"""

        FileRenderPopup(uiScale = self.uiScale, toolColor = self.toolCOlor, accColor = self.accColor, bgColor = self.bgColor).open()

class FileRenderPopup(Popup):
    """Popup triggered from the file side panel, containing settings specific to rendering files"""

    uiScale = NumericProperty()
    toolColor = ColorProperty()
    accColor = ColorProperty()
    bgColor = ColorProperty()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.final = False
        self.format = "WAV"
        self.bitdepth = "PCM_24"
        self.sampleRate = "48000"

    def reloadBitdepths(self, format:str) -> None:
        """reloades the available bitdepths when a new export format is selected"""

        self.children[0].children[0].children[0].children[1].children[1].values = soundfile.available_subtypes(format).keys()
        self.children[0].children[0].children[0].children[1].children[1].text = self.children[0].children[0].children[0].children[1].children[1].values[0]
        self.format = format

    def render(self, path:str) -> None:
        """renders the audio buffer held by the middle layer to a file using the specified settings"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        data = torch.zeros_like(middleLayer.audioBuffer[0])
        for i in range(len(middleLayer.audioBuffer)):
            data += middleLayer.audioBuffer[i] * middleLayer.trackList[i].volume
            data = data.numpy()
        with soundfile.SoundFile(path, "w", int(self.sampleRate), 1, self.bitdepth, None, self.format) as file:
            file.write(data)

    def finalisingClose(self) -> None:
        """sets the final flag, so the popup can be dismissed"""

        self.final = True
        self.dismiss()

    def on_dismiss(self) -> bool:
        """callback function cancelling the dismissal of the popup if necessary, and starting rendering otherwise"""

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
    """Side panel containing a list of installed Voicebanks, and options to display info about them and load them"""

    uiScale = NumericProperty()
    toolColor = ColorProperty()
    accColor = ColorProperty()
    bgColor = ColorProperty()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.voicebanks = []
        self.filepaths = []
        self.selectedIndex = None

    def listVoicebanks(self) -> None: 
        """reads a list of available Voicebanks from disk and displays it"""

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

    def detailElement(self, index:int) -> None:
        """displays more detailed information about a Voicebank selected from the list"""

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

    def importVoicebank(self, path:str, name:str, image) -> None:
        """adds a new vocal track using the selected Voicebank when the import button is pressed"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.importVoicebank(path, name, image)

class ParamSidePanel(ModalView):
    """Side panel containing a list of installed AI-driven parameters, and options to display info about them and load them"""

    uiScale = NumericProperty()
    toolColor = ColorProperty()
    accColor = ColorProperty()
    bgColor = ColorProperty()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.parameters = []
        self.filepaths = []
        self.selectedIndex = None

    def listParams(self) -> None:
        """reads a list of available parameters from disk and displays it"""

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

    def detailElement(self, index:int) -> None:
        """displays more detailed information about a parameter selected from the list"""

        self.ids["param_name"].text = self.voicebanks[index].name
        self.ids["param_type"].text = self.voicebanks[index]._type
        self.ids["param_capacity"].text = self.voicebanks[index].capacity
        self.ids["param_recurrency"].text = self.voicebanks[index].recurrency
        self.ids["param_version"].text = self.voicebanks[index].version
        self.ids["param_license"].text = self.voicebanks[index].license
        self.selectedIndex = index

    def importParameter(self, path:str, name:str) -> None:
        """imports the selected parameter, and attaches it to the node tree of the active track"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.importParam(path, name)

class ScriptingSidePanel(ModalView):
    """side panel containing options for scripting, loading and unloading addons, and opening the devkit"""

    uiScale = NumericProperty()
    toolColor = ColorProperty()
    accColor = ColorProperty()
    bgColor = ColorProperty()

    def openDevkit(self) -> None:
        """opens the devkit as a separate process through the OS"""

        try:
            subprocess.Popen("Devkit.exe")
        except:
            popup = Popup(title = "error", content = Label(text = "Devkit not installed"), size_hint = (None, None), size = (400, 400))
            popup.open()

    def runScript(self) -> None:
        """executes the script entered into the script editor"""

        try:
            exec(self.ids["scripting_editor"].text)
        except Exception as e:
            popup = Popup(title = "script error", content = Label(text = repr(e)), size_hint = (None, None), size = (400, 400))
            popup.open()

    def saveCache(self) -> None:
        """saves the content of the script editor to a cache when the side panel is closed"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.scriptCache = self.ids["scripting_editor"].text

    def loadCache(self) -> None:
        """loads the content of the script editor from cache when the side panel is opened"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        self.ids["scripting_editor"].text = middleLayer.scriptCache

class SettingsSidePanel(ModalView):
    """Class for a side panel displaying a settings menu for the program"""

    uiScale = NumericProperty()
    toolColor = ColorProperty()
    accColor = ColorProperty()
    bgColor = ColorProperty()

    def __init__(self, **kwargs) -> None:
        audioApis = sounddevice.query_hostapis()
        self.audioApiNames =  []
        for i in audioApis:
            self.audioApiNames.append(i["name"])
        self.audioDeviceNames = []
        super().__init__(**kwargs)

    def refreshAudioDevices(self, api:str) -> None:
        """refreshes the list of available audio devices"""

        self.audioDeviceNames = []
        devices = sounddevice.query_hostapis(self.audioApiNames.index(api))["devices"]
        for i in devices:
            device = sounddevice.query_devices(i)
            if device["max_output_channels"]:
                self.audioDeviceNames.append(device["name"])
        self.ids["settings_audioDevice"].values = self.audioDeviceNames

    def restartAudioStream(self) -> None:
        """restarts the audio stream after a device change"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.audioStream.close()
        identifier = self.ids["settings_audioDevice"].text + ", " + self.ids["settings_audioApi"].text
        middleLayer.audioStream = sounddevice.OutputStream(global_consts.sampleRate, global_consts.audioBufferSize, identifier, callback = middleLayer.playCallback)

    def changeDataDir(self) -> None:
        """Changes the directory that is searched for Voicebanks, Parameters etc."""

        tkui = Tk()
        tkui.withdraw()
        newDir = filedialog.askdirectory()
        if newDir != "":
            self.ids["settings_datadir"].text = newDir
        tkui.destroy()

    def applyColors(self) -> None:
        app = App.get_running_app()
        app.root.uiScale = self.ids["settings_uiScale"].value
        app.root.toolColor = self.ids["settings_toolColor"].color
        app.root.accColor = self.ids["settings_accColor"].color
        app.root.bgColor = self.ids["settings_bgColor"].color


    def readSettings(self) -> None:
        """reads the settings from the settings file"""

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
        self.ids["settings_uiScale"].value = settings["uiScale"]
        self.ids["settings_toolColor"].color = eval(settings["toolColor"])
        self.ids["settings_accColor"].color = eval(settings["accColor"])
        self.ids["settings_bgColor"].color = eval(settings["bgColor"])

    def writeSettings(self) -> None:
        """writes the settings to the settings file"""
        
        if self.ids["settings_audioDevice"].text in self.audioDeviceNames:
            audioDevice = self.ids["settings_audioDevice"].text
        else:
            audioDevice = self.audioDeviceNames[0]
        writeSettings(None, self.ids["settings_lang"].text, self.ids["settings_accel"].text, self.ids["settings_tcores"].text, self.ids["settings_lowSpecMode"].text, self.ids["settings_cachingMode"].text, self.ids["settings_audioApi"].text, audioDevice, self.ids["settings_loglevel"].text, self.ids["settings_datadir"].text, self.ids["settings_uiScale"].value, self.ids["settings_toolColor"].color, self.ids["settings_accColor"].color, self.ids["settings_bgColor"].color)
        self.restartAudioStream()
