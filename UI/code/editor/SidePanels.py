#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.core.image import Image as CoreImage

from kivy.uix.popup import Popup
from kivy.app import App

from tkinter import Tk, filedialog

from io import BytesIO

import os
import torch
import subprocess

import sounddevice
import soundfile

import global_consts

import API.Ops

from MiddleLayer.IniParser import readSettings, writeSettings
from MiddleLayer.FileIO import saveNVX

from UI.code.editor.Util import ListElement, CursorAwareView, ManagedPopup

from Localization.editor_localization import getLanguage
loc = getLanguage()

class FileSidePanel(CursorAwareView):
    """Side panel for saving, loading, importing, exporting and rendering files"""

    def save(self) -> None:
        """saves the current work to a .nvx file"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        tkui = Tk()
        tkui.withdraw()
        dir = filedialog.asksaveasfilename(defaultextension = "nvx", filetypes = (("NVX", "nvx"), (loc["all_files"], "*")))
        tkui.destroy()
        saveNVX(dir, middleLayer)

    def load(self) -> None:
        """loads a .nvx file into the editor"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        tkui = Tk()
        tkui.withdraw()
        dir = filedialog.askopenfilename(filetypes = (("NVX", "nvx"), (loc["all_files"], "*")))
        tkui.destroy()
        API.Ops.LoadNVX(dir)()


    def openRenderPopup(self) -> None:
        """opens a popup containing the settings for rendering files"""

        FileRenderPopup().open()

class FileRenderPopup(Popup):
    """Popup triggered from the file side panel, containing settings specific to rendering files"""

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
        """sets the final flag so the popup can be dismissed, and then dismisses it"""

        self.final = True
        self.dismiss()

    def on_dismiss(self) -> bool:
        """callback function cancelling the dismissal of the popup if necessary, and starting rendering otherwise"""

        if self.final == False:
            return False
        tkui = Tk()
        tkui.withdraw()
        dir = filedialog.asksaveasfilename(defaultextension = self.format.lower(), filetypes = ((self.format, self.format.lower()), (loc["all_files"], "*")))
        tkui.destroy()
        if dir == "":
            self.final = False
            return True
        self.render(dir + "." + self.format.lower())
        return False

class SingerSidePanel(CursorAwareView):
    """Side panel containing a list of installed Voicebanks, and options to display info about them and load them"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.voicebanks = []
        self.filepaths = []
        self.selectedIndex = None

    def listVoicebanks(self) -> None: 
        """reads a list of available Voicebanks from disk and displays it"""

        voicePath = os.path.join(readSettings()["datadir"], "Voices")
        if os.path.isdir(voicePath) == False:
            popup = ManagedPopup(title = loc["error"], message = loc["dataDir_err"])
            popup.open()
            return
        files = os.listdir(voicePath)
        for file in files:
            if file.endswith(".nvvb"):
                data = torch.load(os.path.join(voicePath, file), map_location = torch.device("cpu"))
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

        API.Ops.ImportVoicebank(path, name, image)()

class ParamSidePanel(CursorAwareView):
    """Side panel containing a list of installed parameters, and options to display info about them and load them"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.parameters = []
        self.filepaths = []
        self.selectedIndex = None

    def listParams(self) -> None:
        """reads a list of available parameters from disk and displays it"""

        paramPath = os.path.join(readSettings()["datadir"], "Parameters")
        if os.path.isdir(paramPath) == False:
            popup = ManagedPopup(title = loc["error"], message = loc["datadir_err"])
            popup.open()
            return
        files = os.listdir(paramPath)
        for file in files:
            if file.endswith(".nvpr"):
                data = torch.load(os.path.join(paramPath, file), map_location = torch.device("cpu"))
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
        API.Ops.AddParam(path, name)()

class ScriptingSidePanel(CursorAwareView):
    """side panel containing options for scripting, loading and unloading addons, and opening the devkit"""

    def openDevkit(self) -> None:
        """opens the devkit as a separate process through the OS"""

        try:
            subprocess.Popen("Devkit.exe")
        except:
            popup = ManagedPopup(title = loc["error"], message = loc["devkit_inst_err"])
            popup.open()

    def runScript(self) -> None:
        """executes the script entered into the script editor"""

        try:
            exec(self.ids["scripting_editor"].text)
        except Exception as e:
            popup = ManagedPopup(title = loc["script_err"], message = repr(e))
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

class SettingsSidePanel(CursorAwareView):
    """Class for a side panel displaying a settings menu for the program"""

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
        latency = float(self.ids["settings_audioLatency"].text)
        identifier = self.ids["settings_audioDevice"].text + ", " + self.ids["settings_audioApi"].text
        deviceinfo = sounddevice.query_devices(identifier)
        if deviceinfo["default_low_output_latency"] > latency:
            text = loc["latency_warn_1"] + str(deviceinfo["default_low_output_latency"]) + loc["latency_warn_2"] + str(deviceinfo["default_high_output_latency"]) + loc["latency_warn_3"]
            popup = ManagedPopup(title = "audio latency", message = text)
            popup.open()
        middleLayer.audioStream = sounddevice.OutputStream(global_consts.sampleRate, global_consts.audioBufferSize, identifier, callback = middleLayer.playCallback, latency = latency)

    def changeDataDir(self) -> None:
        """Changes the directory that is searched for Voicebanks, Parameters etc."""

        tkui = Tk()
        tkui.withdraw()
        newDir = filedialog.askdirectory()
        if newDir != "":
            self.ids["settings_datadir"].text = newDir
        tkui.destroy()

    def applyColors(self) -> None:
        """applies a new set of colors to the UI when one of the available color settings is changed"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        app = App.get_running_app()
        if self.ids["settings_uiScale"].text == "":
            uiScale = 1.
        else:
            uiScale = float(self.ids["settings_uiScale"].text)
        uiScale = max(min(uiScale, 5), 0.1)
        app.root.ids["pianoRoll"].xScale = app.root.ids["pianoRoll"].xScale * uiScale / app.root.uiScale
        app.root.ids["pianoRoll"].yScale = app.root.ids["pianoRoll"].yScale * uiScale / app.root.uiScale
        app.root.ids["pianoRoll"].updateTrack()
        app.root.uiScale = uiScale
        app.root.toolColor = self.ids["settings_toolColor"].color
        app.root.accColor = self.ids["settings_accColor"].color
        app.root.bgColor = self.ids["settings_bgColor"].color
        
    def restartRenderProcess(self)-> None:
        """restarts the render process after a change to accelerator, caching or TPU settings"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.manager.restart(middleLayer.trackList)

    def readSettings(self) -> None:
        """reads the settings from the settings file"""

        settings = readSettings()
        self.ids["settings_lang"].text = settings["language"]
        self.ids["settings_accel"].text = settings["accelerator"]
        if settings["tensorcores"] == "disabled":
            self.ids["settings_tcores"].text = loc["disabled"]
        else:
            self.ids["settings_tcores"].text = loc["enabled"]
        if settings["lowspecmode"] == "disabled":
            self.ids["settings_lowSpecMode"].text = loc["disabled"]
        else:
            self.ids["settings_lowSpecMode"].text = loc["enabled"]
        if settings["cachingmode"] == "save RAM":
            self.ids["settings_cachingMode"].text = loc["save_ram"]
        elif settings["cachingmode"] == "default":
            self.ids["settings_cachingMode"].text = loc["default"]
        else:
            self.ids["settings_cachingMode"].text = loc["render_speed"]
        self.ids["settings_audioApi"].text = settings["audioapi"]
        self.refreshAudioDevices(settings["audioapi"])
        self.ids["settings_audioDevice"].text = settings["audiodevice"]
        self.ids["settings_audioLatency"].text = settings["audiolatency"]
        self.ids["settings_undoLimit"].text = settings["undolimit"]
        self.ids["settings_loglevel"].text = settings["loglevel"]
        self.ids["settings_datadir"].text = settings["datadir"]
        self.ids["settings_uiScale"].text = settings["uiscale"]
        self.ids["settings_toolColor"].color = eval(settings["toolcolor"])
        self.ids["settings_accColor"].color = eval(settings["acccolor"])
        self.ids["settings_bgColor"].color = eval(settings["bgcolor"])

    def writeSettings(self) -> None:
        """writes the settings to the settings file"""
        
        if self.ids["settings_audioDevice"].text in self.audioDeviceNames:
            audioDevice = self.ids["settings_audioDevice"].text
        else:
            audioDevice = self.audioDeviceNames[0]
            
        accel = self.ids["settings_accel"].text
        if self.ids["settings_tcores"].text == loc["disabled"]:
            tcores = "disabled"
        else:
            tcores = "enabled"
        if self.ids["settings_lowSpecMode"].text == loc["disabled"]:
            lowspec = "disabled"
        else:
            lowspec = "enabled"
        if self.ids["settings_cachingMode"].text == loc["save_ram"]:
            caching = "save RAM"
        elif self.ids["settings_cachingMode"].text == loc["default"]:
            caching = "default"
        else:
            caching = "best rendering speed"
        writeSettings(None,
                      self.ids["settings_lang"].text,
                      accel,
                      tcores,
                      lowspec,
                      caching,
                      self.ids["settings_audioApi"].text,
                      audioDevice,
                      self.ids["settings_audioLatency"].text,
                      self.ids["settings_undoLimit"].text,
                      self.ids["settings_loglevel"].text,
                      self.ids["settings_datadir"].text,
                      self.ids["settings_uiScale"].text,
                      self.ids["settings_toolColor"].color,
                      self.ids["settings_accColor"].color,
                      self.ids["settings_bgColor"].color)
        self.restartAudioStream()
