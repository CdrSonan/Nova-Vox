from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from kivy.graphics import Color, Line
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView

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

class ParamCurve(Widget):
    pass

class PitchOptns(Widget):
    pass

class TimingOptns(Widget):
    pass

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

class NovaVoxUI(Widget):
    def update(self, deltatime):
        pass