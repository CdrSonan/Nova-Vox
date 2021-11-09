from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, ObjectProperty

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
        self.source = self.imagePressed
        if self.function != None:
            self.function()
        else:
            print("NONE function callback")
    def on_release(self):
        self.source = self.imageNormal

class SingerListItem(Widget):
    pass

class ParamListItem(Widget):
    pass

class ParamCurve(Widget):
    pass

class PitchOptns(Widget):
    pass

class TimingOptns(Widget):
    pass

class Note(Widget):
    pass

class TimingBar(Widget):
    pass

class PitchLine(Widget):
    pass

class NovaVoxUI(Widget):
    def update(self, deltatime):
        pass