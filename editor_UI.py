from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview.views import RecycleDataViewBehavior

from kivy.uix.recyclelayout import RecycleLayout

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

class Note(RecycleDataViewBehavior, Label):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)
    xPos = NumericProperty()
    yPos = NumericProperty()

    def refresh_view_attrs(self, pianoRoll, index, data):
        ''' Catch and handle the view changes '''
        self.index = index
        return super(Note, self).refresh_view_attrs(pianoRoll, index, data)

    def on_touch_down(self, touch):
        ''' Add selection on touch down '''
        if super(Note, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, pianoRoll, index, is_selected):
        ''' Respond to the selection of items in the view. '''
        self.selected = is_selected
        if is_selected:
            print("selection changed to {0}".format(pianoRoll.data[index]))
        else:
            print("selection removed for {0}".format(pianoRoll.data[index]))

class PianoRoll(RecycleView):
    def __init__(self, **kwargs):
        super(PianoRoll, self).__init__(**kwargs)
        self.data = [{'text': str(x), "xPos": x, "yPos": x} for x in range(10)]


class SelectableRecycleLayout(FocusBehavior, LayoutSelectionBehavior, RecycleLayout):
    pass

class NovaVoxUI(Widget):
    def update(self, deltatime):
        pass