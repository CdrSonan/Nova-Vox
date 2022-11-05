from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import ObjectProperty, NumericProperty
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.bubble import BubbleButton
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.app import App

def fullRoot(widget):
    root = widget
    while root.parent != None and root.parent != root.parent.parent:
        root = root.parent
    return root

class ImageButton(ButtonBehavior, Image):
    """Class for a button displaying an image instead of text"""

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.collide_point(*pos):
            self.color = (0.5, 0.5, 0.5, 1.)
        else:
            self.color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        self.color = fullRoot(self).accColor
        if self.function != None:
            self.function()

    def on_release(self) -> None:
        self.on_mouseover(self, Window.mouse_pos)

class ImageToggleButton(ToggleButtonBehavior, Image):
    """Class for a toggle button displaying an image instead of text"""

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.state == 'down':
            self.color = fullRoot(self).accColor
        elif self.collide_point(*pos):
            self.color = (0.5, 0.5, 0.5, 1.)
        else:
            self.color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        if self.function != None:
            self.function()

    def on_state(self, widget, value) -> None:
        self.on_mouseover(widget, Window.mouse_pos)

class ManagedButton(Button):

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.collide_point(*pos):
            self.background_color = (0.5, 0.5, 0.5, 1.)
        else:
            self.background_color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        root = fullRoot(self)
        if "accColor" in root.__dict__.keys():
            self.background_color = root.accColor
        if self.function != None:
            self.function()

    def on_release(self) -> None:
        self.on_mouseover(self, Window.mouse_pos)

class ManagedToggleButton(ToggleButton):

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.state == 'down':
            root = fullRoot(self)
            if "accColor" in root.__dict__.keys():
                self.background_color = root.accColor
        elif self.collide_point(*pos):
            self.background_color = (0.5, 0.5, 0.5, 1.)
        else:
            self.background_color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        if self.function != None:
            self.function()

    def on_state(self, widget, value) -> None:
        self.on_mouseover(widget, Window.mouse_pos)

class NumberInput(TextInput):
    """A modified text input field that sanitizes the input, ensuring it is always an integer number"""

    def insert_text(self, substring:str, from_undo:bool=False) -> None:
        s = ""
        s += "".join(char for char in substring if char.isdigit())
        return super().insert_text(s, from_undo=from_undo)

class ReferencingButton(BubbleButton):
    """A button with an additional property for keeping a reference to a different, arbitrary object"""

    reference = ObjectProperty()

class ListElement(ManagedButton):
    """A button with an additional integer property representing its position in a list"""

    index = NumericProperty()
