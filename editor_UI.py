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

import os
import torch

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
    pass

class ParamPanel(ToggleButton):
    pass

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

class SingerDetails(GridLayout):
    pass

class ParamDetails(GridLayout):
    pass

class ListElement(Button):
    pass

class FileSidePanel(ModalView):
    pass

class SingerSidePanel(ModalView):
    def listVoicebanks(self):
        Voicebanks = {}
        files = os.listdir("Voices/")
        for file in files:
            if file.endswith(".nvpr"):
                data = torch.load(os.path.join("Voices/", file))
                Voicebanks.append(data["metadata"])

class ParamSidePanel(ModalView):
    def listParams(self):
        Parameters = {}
        files = os.listdir("Params/")
        for file in files:
            if file.endswith(".nvpr"):
                data = torch.load(os.path.join("Params/", file))
                Parameters.append(data["metadata"])

class ScriptingSidePanel(ModalView):
    pass

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
    def update(self, deltatime):
        pass