#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch

from csv import reader, writer
from tkinter.messagebox import askyesno

from os import path
from ast import literal_eval

from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import NumericProperty, ColorProperty, ObjectProperty, DictProperty

from MiddleLayer.MiddleLayer import MiddleLayer
global middleLayer
middleLayer = MiddleLayer()
from MiddleLayer.IniParser import readSettings
from MiddleLayer.FileIO import saveNVX
from MiddleLayer.ErrorHandler import handleMainException, handleRendererException

from Localization.editor_localization import getLanguage

from UI.editor.AdaptiveSpace import *
from UI.editor.Headers import *
from UI.editor.PianoRoll import *
from UI.editor.Popups import *
from UI.editor.SidePanels import *
from UI.editor.NodeEditor import *
from UI.editor.Util import *

import API.Ops

class NovaVoxUI(Widget):
    """class of the Root UI of the Nova-Vox editor. At program startup or after a reset, a single instance of this class is created, which then creates all UI elements as its children"""

    settings = readSettings()
    uiCfg = {}
    try:
        with open(path.join(settings["datadir"], "ui.cfg"), "r") as f:
            uiCfgReader = reader(f)
            for line in uiCfgReader:
                if line == []:
                    continue
                uiCfg[line[0]] = literal_eval(line[1])
    except FileNotFoundError:
        uiCfg = {"mainSplitter": 0.25,
                 "sideSplitter": 0.75,
                 "pianoSplitter": 0.75}
    uiScale = NumericProperty(float(settings["uiscale"]))
    toolColor = ColorProperty(eval(settings["toolcolor"]))
    accColor = ColorProperty(eval(settings["acccolor"]))
    bgColor = ColorProperty(eval(settings["bgcolor"]))
    cursorSource = ObjectProperty()
    cursorPrio = NumericProperty()
    loc = DictProperty(getLanguage())
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        middleLayer.setUI(self)
        self.middleLayer = middleLayer
        self._keyboard = Window.request_keyboard(None, self, 'text')
        if self._keyboard.widget:
            pass
        self._keyboard.bind(on_key_down = self._on_keyboard_down)
        self._keyboard.bind(on_key_up = self._on_keyboard_up)
        self._keyboard.target = None
        Window.on_request_close = self._on_request_close

    def update(self, deltatime:float) -> None:
        """periodically called update function tied to UI updates that reads changes from the rendering process and passes them to the middleLayer"""

        change = middleLayer.manager.receiveChange()
        if change == None:
            return None
        try:
            if change.type == "status":
                #print("recv status update ", change.track, change.index, change.value)
                middleLayer.updateRenderStatus(change.track, change.index, change.value)
            elif change.type == "updateAudio":
                #print("recv audio update ", change.track, change.index, change.value)
                middleLayer.updateAudioBuffer(change.track, change.index, change.value)
            elif change.type == "zeroAudio":
                #print("recv audio zero ", change.track, change.index, change.value)
                if change.value < 0:
                    change.index += change.value
                    change.value *= -1
                middleLayer.updateAudioBuffer(change.track, change.index, torch.zeros([change.value,]))
            elif change.type == "deletion":
                middleLayer.deletions.pop(0)
            elif change.type == "offsetApplied":
                #print("recv offset applied ", change.track, change.index, change.value)
                middleLayer.trackList[change.track].offsets.pop(0)
            elif change.type == "error":
                handleRendererException(change.value)
        except Exception as e:
            handleMainException(e)
        return self.update(deltatime)
    
    def updateParamPanel(self) -> None:

        self.ids["paramList"].clear_widgets()
        self.ids["adaptiveSpace"].clear_widgets()
        if middleLayer.activeTrack == None:
            return
        if middleLayer.mode == "notes":
            self.ids["paramList"].add_widget(ParamPanel(name = "steadiness", switchable = True, sortable = False, deletable = False, index = -1, switchState = middleLayer.trackList[middleLayer.activeTrack].useSteadiness, visualName = loc["steadiness"], state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "breathiness", switchable = True, sortable = False, deletable = False, index = -1, switchState = middleLayer.trackList[middleLayer.activeTrack].useBreathiness, visualName = loc["breathiness"]))
            self.ids["paramList"].add_widget(ParamPanel(name = "AI balance", switchable = True, sortable = False, deletable = False, index = -1, switchState = middleLayer.trackList[middleLayer.activeTrack].useAIBalance, visualName = loc["ai_balance"]))
            self.ids["adaptiveSpace"].add_widget(ParamCurve())
            counter = 0
            for i in middleLayer.trackList[middleLayer.activeTrack].nodegraph.params.keys():
                self.ids["paramList"].add_widget(ParamPanel(name = i, switchable = True, sortable = True, deletable = False, index = counter, switchState = middleLayer.trackList[middleLayer.activeTrack].nodegraph.params[i].enabled, visualName = i))
                counter += 1
            API.Ops.SwitchParam("steadiness")()
        if middleLayer.mode == "timing":
            self.ids["paramList"].add_widget(ParamPanel(name = "loop overlap", switchable = False, sortable = False, deletable = False, index = -1, visualName = loc["loop_overlap"], state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "loop offset", switchable = False, sortable = False, deletable = False, index = -1, visualName = loc["loop_offset"]))
            self.ids["adaptiveSpace"].add_widget(TimingOptns())
            API.Ops.SwitchParam("loop overlap")()
        if middleLayer.mode == "pitch":
            self.ids["paramList"].add_widget(ParamPanel(name = "vibrato speed", switchable = True, sortable = False, deletable = False, index = -1, switchState = middleLayer.trackList[middleLayer.activeTrack].useVibratoSpeed, visualName = loc["vibrato_speed"], state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "vibrato strength", switchable = True, sortable = False, deletable = False, index = -1, switchState = middleLayer.trackList[middleLayer.activeTrack].useVibratoStrength, visualName = loc["vibrato_strength"]))
            self.ids["adaptiveSpace"].add_widget(PitchOptns())
            API.Ops.SwitchParam("vibrato speed")()
    
    def setMode(self, mode) -> None:
        """signals the middleLAyer a change of the input mode and prompts the required UI updates"""

        global middleLayer
        if middleLayer.activeTrack == None or middleLayer.mode == mode:
            return
        middleLayer.mode = mode
        self.updateParamPanel()
        middleLayer.changePianoRollMode()

    def setTool(self, tool) -> None:
        """signals the middleLAyer a change of the current tool"""

        global middleLayer
        middleLayer.tool = tool

    def play(self, state:bool) -> None:
        """signals the middle layer a state change of the playback toggle"""

        if state == "down":
            middleLayer.play(True)
        else:
            middleLayer.play(False)

    def spoolBack(self) -> None:
        """sets the playback head to the beginning of the active track"""

        if middleLayer.activeTrack == None or len(middleLayer.trackList[middleLayer.activeTrack].borders) == 0:
            middleLayer.mainAudioBufferPos = 0
            middleLayer.movePlayhead(0)
        else:
            middleLayer.mainAudioBufferPos = int(middleLayer.trackList[middleLayer.activeTrack].borders[0])
            middleLayer.movePlayhead(middleLayer.trackList[middleLayer.activeTrack].borders[0])
            
    def spoolForward(self) -> None:
        """sets the playback head to the end of the active track"""

        if middleLayer.activeTrack == None or len(middleLayer.trackList[middleLayer.activeTrack].borders) == 0:
            return
        else:
            middleLayer.mainAudioBufferPos = int(middleLayer.trackList[middleLayer.activeTrack].borders[-1])
            middleLayer.movePlayhead(middleLayer.trackList[middleLayer.activeTrack].borders[-1])

    def restart(self) -> None:
        """restarts the rendering process through its manager"""

        middleLayer.manager.restart(middleLayer.trackList)

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers) -> None:
        """universal function for processing keyboard shortcuts and keeping track of modifier keys"""

        if keyboard.target != None:
            return False
        if keycode[0] == 303 or keycode[0] == 304: 
            middleLayer.shift = True
        elif keycode[0] == 305 or keycode[0] == 306: 
            middleLayer.ctrl = True
        elif keycode[0] == 307 or keycode[0] == 308: 
            middleLayer.alt = True
        elif keycode[0] == 32:
            middleLayer.play()
        elif keycode[0] in (8, 127):
            middleLayer.delete()
        else:
            print("keycode pressed:", keycode[0])
        return True

    def _on_keyboard_up(self, keyboard, keycode) -> None:
        """signals the middleLayer when modifier keys have been let go of"""
        
        if keyboard.target != None:
            return False
        if keycode[0] == 303 or keycode[0] == 304: 
            middleLayer.shift = False
        if keycode[0] == 305 or keycode[0] == 306: 
            middleLayer.ctrl = False
        if keycode[0] == 307 or keycode[0] == 308: 
            middleLayer.alt = False
        return True

    def _on_request_close(self, *args) -> None:
        """called when the window is closed. Prompts the middleLayer to stop the rendering process, saves the current project, and asks the user for confirmation if there are unsaved changes."""

        middleLayer.manager.stop()
        tkui = Tk()
        tkui.withdraw()
        if middleLayer.unsavedChanges:
            if askyesno(loc["unsaved_changes"], loc["unsaved_changes_msg"]):
                dir = filedialog.asksaveasfilename(defaultextension = "nvx", filetypes = (("NVX", "nvx"), (loc["all_files"], "*")))
                if dir != "":
                    saveNVX(dir, middleLayer)
        tkui.destroy()
        uiCfg = {"windowWidth":int(Window.width / dp(1)),
                 "windowHeight":int(Window.height / dp(1)),
                 "windowState": Window.fullscreen,
                 "mainSplitter": self.ids["mainSplitter"].width / self.ids["mainSplitter"].parent.width,
                 "sideSplitter": self.ids["sideSplitter"].height / self.ids["sideSplitter"].parent.height,
                 "pianoSplitter": self.ids["pianoSplitter"].height / self.ids["pianoSplitter"].parent.height}
        with open(path.join(readSettings()["datadir"], "ui.cfg"), "w") as f:
            cfgWriter = writer(f)
            cfgWriter.writerows(uiCfg.items())
        return False
