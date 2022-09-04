from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, ObjectProperty, NumericProperty
from kivy.uix.button import Button
from kivy.uix.bubble import BubbleButton
from kivy.uix.textinput import TextInput
from kivy.uix.treeview import TreeViewNode

class ImageButton(ButtonBehavior, Image):
    """Class for a button displaying an image instead of text"""

    imageNormal = StringProperty()
    imagePressed = StringProperty()
    function = ObjectProperty(None)

    def on_press(self) -> None:
        self.source = self.imagePressed
        if self.function != None:
            self.function()

    def on_release(self) -> None:
        self.source = self.imageNormal

class ImageToggleButton(ToggleButtonBehavior, Image):
    """Class for a toggle button displaying an image instead of text"""

    imageNormal = StringProperty()
    imagePressed = StringProperty()
    function = ObjectProperty(None)

    def on_press(self) -> None:
        if self.function != None:
            self.function()

    def on_release(self) -> None:
        pass

    def on_state(self, widget, value) -> None:
        if value == 'down':
            self.source = self.imagePressed
        else:
            self.source = self.imageNormal

class NumberInput(TextInput):
    """A modified text input field that sanitizes the input, ensuring it is always an integer number"""

    def insert_text(self, substring:str, from_undo:bool=False) -> None:
        s = ""
        s += "".join(char for char in substring if char.isdigit())
        return super().insert_text(s, from_undo=from_undo)

class ReferencingButton(BubbleButton):
    """A button with an additional property for keeping a reference to a different, arbitrary object"""

    reference = ObjectProperty()

class ListElement(Button):
    """A button with an additional integer property representing its position in a list"""

    index = NumericProperty()
