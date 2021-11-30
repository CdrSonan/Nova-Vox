from kivy.core.image import Image as CoreImage
from PIL import Image as PilImage, ImageDraw, ImageFont

from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty, ListProperty
from kivy.graphics import Color, Line
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.uix.button import Button

from io import BytesIO

import os
import torch
import subprocess

import MiddleLayer.DataHandlers as dh

class MiddleLayer:
    def __init__(self, ids, **kwargs):
        super().__init__(**kwargs)
        self.ids = ids
        self.trackList = []
        self.activeTrack = NumericProperty()
        self.activeParam = NumericProperty()
    def importVoicebank(self, path, name, inImage):
        self.trackList.append(dh.Track(path))
        canvas_img = inImage
        data = BytesIO()
        canvas_img.save(data, format='png')
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext='png')
        image = im.texture
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image))
    def importParam(self, path, name):
        self.trackList[self.activeTrack].paramStack.append(dh.Parameter(path))
        self.ids["paramList"].add_widget(ParamPanel, name = name)

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

class ParamPanel(ToggleButton):
    name = StringProperty()

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