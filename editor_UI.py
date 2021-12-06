from logging import root
from kivy.core.image import Image as CoreImage
from PIL import Image as PilImage, ImageDraw, ImageFont

from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty, ListProperty, OptionProperty
from kivy.graphics import Color, Line
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label

from io import BytesIO

from copy import deepcopy

import os
import torch
import subprocess

import MiddleLayer.DataHandlers as dh

class MiddleLayer(Widget):
    def __init__(self, ids, **kwargs):
        super().__init__(**kwargs)
        self.ids = ids
        self.trackList = []
        self.activeTrack = None
        self.activeParam = None
        self.mode = OptionProperty("notes", options = ["notes", "timing", "pitch"])
    def importVoicebank(self, path, name, inImage):
        self.trackList.append(dh.Track(path))
        canvas_img = inImage
        data = BytesIO()
        canvas_img.save(data, format='png')
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext='png')
        image = im.texture
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(self.trackList) - 1))
    def importParam(self, path, name):
        self.trackList[self.activeTrack].paramStack.append(dh.Parameter(path))
        self.ids["paramList"].add_widget(ParamPanel(name = name, switchable = True, sortable = True, deletable = True, index = len(self.trackList[self.activeTrack].paramStack) - 1))
    def changeTrack(self, index):
        self.activeTrack = index
    def copyTrack(self, index, name, inImage):
        self.trackList.append(dh.Track(None))
        self.trackList[len(self.trackList) - 1].voicebank = self.trackList[index].voicebank
        image = inImage
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(self.trackList) - 1))
    def deleteTrack(self, index):
        self.trackList.pop(index)
        if index <= self.activeTrack:
            self.changeTrack(self.activeTrack - 1)
        for i in self.ids["singerList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
            if i.index > index:
                i.index = i.index - 1
    def deleteParam(self, index):
        self.trackList[self.activeTrack].paramStack.pop(index)
        if index <= self.activeParam:
            self.changeParam(self.activeParam - 1)
        for i in self.ids["paramList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
            if i.index > index:
                i.index = i.index - 1
    def enableParam(self, index):
        if index == -1:
            pass
        else:
            self.trackList[self.activeTrack].paramStack[index].enabled = True
    def disableParam(self, index):
        if index == -1:
            pass
        else:
            self.trackList[self.activeTrack].paramStack[index].enabled = False
    def moveParam(self, name, switchable, sortable, deletable, index, delta):
        param = self.trackList[self.activeTrack].paramStack[index]
        if delta > 0:
            for i in range(delta):
                self.trackList[self.activeTrack].paramStack[index + i] = self.trackList[self.activeTrack].paramStack[index + i + 1]
            self.trackList[self.activeTrack].paramStack[index + delta] = param
        if delta < 0:
            for i in range(-delta):
                self.trackList[self.activeTrack].paramStack[index - i] = self.trackList[self.activeTrack].paramStack[index - i - 1]
            self.trackList[self.activeTrack].paramStack[index - delta] = param
        for i in self.ids["paramList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
                break
        self.ids["paramList"].add_widget(ParamPanel(name = name, switchable = switchable, sortable = sortable, deletable = deletable, index = index), index = index + delta)
        self.changeParam(index + delta)
    def updateParamPanel(self):
        self.ids["paramList"].clear_widgets()
        if self.mode == "notes":
            self.ids["paramList"].add_widget(ParamPanel(name = "steadiness", switchable = True, sortable = False, deletable = False, index = -1))
            self.ids["paramList"].add_widget(ParamPanel(name = "breathiness", switchable = True, sortable = False, deletable = False, index = -1))
            counter = 0
            for i in self.trackList[self.activeTrack].paramStack:
                self.ids["paramList"].add_widget(ParamPanel(name = i.name, index = counter))
                counter += 1
        if self.mode == "timing":
            self.ids["paramList"].add_widget(ParamPanel(name = "loop overlap", switchable = False, sortable = False, deletable = False, index = -1))
            self.ids["paramList"].add_widget(ParamPanel(name = "loop offset", switchable = False, sortable = False, deletable = False, index = -1))
        if self.mode == "pitch":
            self.ids["paramList"].add_widget(ParamPanel(name = "vibrato speed", switchable = True, sortable = False, deletable = False, index = -1))
            self.ids["paramList"].add_widget(ParamPanel(name = "vibrato strength", switchable = True, sortable = False, deletable = False, index = -1))
    def changeParam(self, index):
        if self.mode == "notes":
            if index == -1:
                pass
            else:
                self.activeParam = index
                self.ids["adaptiveSpace"].data = self.trackList[self.activeTrack].paramStack[index].curve

class ImageButton(ButtonBehavior, Image):
    imageNormal = StringProperty()
    imagePressed = StringProperty()
    function = ObjectProperty(None)
    def on_press(self):
        self.source = self.imagePressed
        if self.function != None:
            self.function()
        else:
            print("NONE function callback")
    def on_release(self):
        self.source = self.imageNormal

class ImageToggleButton(ToggleButtonBehavior, Image):
    imageNormal = StringProperty()
    imagePressed = StringProperty()
    function = ObjectProperty(None)
    def on_press(self):
        if self.function != None:
            self.function()
        else:
            print("NONE function callback")
    def on_release(self):
        pass

    def on_state(self, widget, value):
        if value == 'down':
            self.source = self.imagePressed
        else:
            self.source = self.imageNormal

class SingerPanel(AnchorLayout):
    name = StringProperty()
    image = ObjectProperty()
    index = NumericProperty()
    def changeTrack(self):
        global middleLayer
        middleLayer.changeTrack(self.index)
    def copyTrack(self):
        global middleLayer
        middleLayer.copyTrack(self.index, self.name, self.image)
    def deleteTrack(self):
        global middleLayer
        middleLayer.deleteTrack(self.index)

class ParamPanel(ToggleButton):
    def __init__(self, name, switchable, sortable, deletable, index, **kwargs):
        super().__init__(**kwargs)
        self.name = StringProperty()
        self.switchable = BooleanProperty()
        self.sortable = BooleanProperty()
        self.index = NumericProperty()
        self.deletable = BooleanProperty()
        self.name = name
        self.switchable = switchable
        self.sortable = sortable
        self.deletable = deletable
        self.index = index
        self.background_color = (1, 1, 1, 0.3)
        self.add_widget(Label(size_hint = (None, None), size = (self.width - 76, 30), pos = (self.x + 103, self.y + 3), text = self.name))
        if self.switchable:
            self.add_widget(ImageToggleButton(size_hint = (None, None), size = (30, 30), pos = (self.x + 3, self.y + 3), imageNormal = "UI/assets/ParamList/Adaptive02.png", imagePressed = "UI/assets/ParamList/Adaptive01.png", on_state = self.enableParam))
        if self.sortable:
            self.add_widget(ImageButton(size_hint = (None, None), size = (40, 30), pos = (self.x + 33, self.y + 3), imageNormal = "UI/assets/ParamList/Adaptive03.png", imagePressed = "UI/assets/ParamList/Adaptive03_clicked.png", on_release = self.moveParam))
        if self.deletable:
            self.add_widget(ImageButton(size_hint = (None, None), size = (30, 30), pos = (self.x + 73, self.y + 3), imageNormal = "UI/assets/TrackList/SingerGrey03.png", imagePressed = "UI/assets/TrackList/SingerGrey03_clicked.png", on_press = self.deleteParam))
    def enableParam(self):
        global middleLayer
        if self.state == "down":
            middleLayer.enableParam(self.index)
        else:
            middleLayer.disableParam(self.index)
    def moveParam(self):
        global middleLayer
        delta = 0
        middleLayer.moveParam(self.name, self.switchable, self.sortable, self.index, delta)
    def deleteParam(self):
        global middleLayer
        middleLayer.deleteParam(self.index)

class AdaptiveSpace(AnchorLayout):
    pass

class ParamCurve(ScrollView):
    xScale = NumericProperty()
    data = ListProperty([(0, 0), (20, 20), (100, -10), (5000, 30)])

class ParamBars(ScrollView):
    xScale = NumericProperty()
    data = ListProperty([(0, 0), (20, 20), (100, 100), (5000, 30)])

class PitchOptns(ScrollView):
    xScale = NumericProperty()
    data1 = ListProperty([(0, 0), (20, 20), (100, 100), (5000, 30)])
    data2 = ListProperty([(0, 0), (20, 20), (100, 100), (5000, 30)])

class TimingOptns(ScrollView):
    xScale = NumericProperty()
    data1 = ListProperty([(0, 0), (20, 20), (100, 100), (5000, 30)])
    data2 = ListProperty([(0, 0), (20, 20), (100, 100), (5000, 30)])

class Note(ToggleButton):
    #index = NumericProperty()
    selected = BooleanProperty(False)
    xPos = NumericProperty()
    yPos = NumericProperty()
    length = NumericProperty()
    def on_parent(self, screen, parent):
        self.pos = (self.parent.x + self.xPos * self.parent.parent.xScale, self.parent.y + self.yPos * self.parent.parent.yScale)
        with self.canvas:
            Color(1, 0, 0, 1)
            Line(points = ([self.x, self.parent.y], [self.x, self.parent.top]))

class PianoRollOctave(FloatLayout):
    pass

class PianoRoll(ScrollView):
    def __init__(self, **kwargs):
        super(PianoRoll, self).__init__(**kwargs)
        self.xScale = NumericProperty()
        self.yScale = NumericProperty()
        self.data = [{'text': str(x), "xPos": x, "yPos": x, "length": 10 * x} for x in range(10)]
    def generate_notes(self):
        for d in self.data:
            self.children[0].add_widget(Note(**d))

class ListElement(Button):
    index = NumericProperty()

class FileSidePanel(ModalView):
    pass

class SingerSidePanel(ModalView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voicebanks = []
        self.filepaths = []
        self.selectedIndex = None
    def listVoicebanks(self):
        files = os.listdir("Voices/")
        for file in files:
            if file.endswith(".nvvb"):
                data = torch.load(os.path.join("Voices/", file))
                self.voicebanks.append(data["metadata"])
                self.filepaths.append(os.path.join("Voices/", file))
        j = 0
        for i in self.voicebanks:
            self.ids["singers_list"].add_widget(ListElement(text = i.name, index = j))
            j += 1
    def detailElement(self, index):
        self.ids["singer_name"].text = self.voicebanks[index].name
        #self.ids["singer_image"].source = self.voicebanks[index].image
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
        files = os.listdir("Params/")
        for file in files:
            if file.endswith(".nvpr"):
                data = torch.load(os.path.join("Params/", file))
                self.parameters.append(data["metadata"])
                self.filepaths.append(os.path.join("Params/", file))
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
        middleLayer.importParam(path, name)

class ScriptingSidePanel(ModalView):
    def openDevkit(self):
        subprocess.Popen("Devkit.exe")
    def runScript(self):
        exec(self.ids["scripting_editor"].text)

class SettingsSidePanel(ModalView):
    def readSettings(self):
        settings = {}
        with open("settings.ini", 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(" ")
                settings[line[0]] = line[1]
        self.ids["settings_lang"].text = settings["language"]
        self.ids["settings_accel"].text = settings["accelerator"]
        self.ids["settings_tcores"].text = settings["tensorCores"]
        self.ids["settings_prerender"].text = settings["intermediateOutputs"]
        self.ids["settings_loglevel"].text = settings["loglevel"]
    def writeSettings(self):
        with open("settings.ini", 'w') as f:
            f.write("language " + self.ids["settings_lang"].text + "\n")
            f.write("accelerator " + self.ids["settings_accel"].text + "\n")
            f.write("tensorCores " + self.ids["settings_tcores"].text + "\n")
            f.write("intermediateOutputs " + self.ids["settings_prerender"].text + "\n")
            f.write("loglevel " + self.ids["settings_loglevel"].text + "\n")

class LicensePanel(Popup):
    pass

class NovaVoxUI(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global middleLayer
        middleLayer = MiddleLayer(self.ids)
    def update(self, deltatime):
        pass
    def setMode(self, mode):
        global middleLayer
        middleLayer.mode = mode
        middleLayer.updateParamPanel()