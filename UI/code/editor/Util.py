#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import ObjectProperty, NumericProperty
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.modalview import ModalView
from kivy.core.window import Window
from kivy.app import App

def fullRoot(widget):
    root = widget
    while root.parent != None and root.parent != root.parent.parent:
        root = root.parent
    return root

class ImageButton(ButtonBehavior, Image):
    """a "managed" button that displays an image instead of text, automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.collide_point(*self.to_widget(*pos)):
            self.color = (0.5, 0.5, 0.5, 1.)
        else:
            self.color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        root = App.get_running_app().root
        if "accColor" in dir(root):
            self.color = root.accColor
        if self.function != None:
            self.function()

    def on_release(self) -> None:
        self.on_mouseover(self, Window.mouse_pos)

class ImageToggleButton(ToggleButtonBehavior, Image):
    """a "managed" toggle button that displays an image instead of text, automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.state == 'down':
            root = App.get_running_app().root
            if "accColor" in dir(root):
                self.color = root.accColor
        elif self.collide_point(*self.to_widget(*pos)):
            self.color = (0.5, 0.5, 0.5, 1.)
        else:
            self.color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        if self.function != None:
            self.function()

    def on_state(self, widget, value) -> None:
        self.on_mouseover(widget, Window.mouse_pos)

class ManagedButton(Button):
    """a "managed" button that automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.collide_point(*self.to_widget(*pos)):
            self.background_color = (0.5, 0.5, 0.5, 1.)
        else:
            self.background_color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        root = App.get_running_app().root
        if "accColor" in dir(root):
            self.background_color = root.accColor
        if self.function != None:
            self.function()

    def on_release(self) -> None:
        self.on_mouseover(self, Window.mouse_pos)

class ManagedToggleButton(ToggleButton):
    """a "managed" toggle button that automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.state == 'down':
            root = App.get_running_app().root
            if "accColor" in dir(root):
                self.background_color = root.accColor
        elif self.collide_point(*self.to_widget(*pos)):
            self.background_color = (0.5, 0.5, 0.5, 1.)
        else:
            self.background_color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        if self.function != None:
            self.function()

    def on_state(self, widget, value) -> None:
        self.on_mouseover(widget, Window.mouse_pos)

class ManagedSpinnerOptn(SpinnerOption):
    """a "managed" spinner option button that automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.collide_point(*self.to_widget(*pos)):
            self.background_color = (0.5, 0.5, 0.5, 1.)
        else:
            self.background_color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        root = App.get_running_app().root
        if "accColor" in dir(root):
            self.background_color = root.accColor

    def on_release(self) -> None:
        self.on_mouseover(self, Window.mouse_pos)

class ManagedSpinner(Spinner):
    """a "managed" spinner that automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.is_open:
            root = App.get_running_app().root
            if "accColor" in dir(root):
                self.background_color = root.accColor
        elif self.collide_point(*self.to_widget(*pos)):
            self.background_color = (0.5, 0.5, 0.5, 1.)
        else:
            self.background_color = (1., 1., 1., 1.)

    def on_is_open(self, widget, value) -> None:
        super().on_is_open(widget, value)
        self.on_mouseover(widget, Window.mouse_pos)

class ManagedSplitterStrip(Button):
    """a "managed" splitter strip that sets its appeareance to match the app theme, and changes the mouse cursor and opacity when it is hovered over"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        root = App.get_running_app().root
        if self.collide_point(*self.to_widget(*pos)) and root.cursorPrio <= 1:
            self.background_color = (0.5, 0.5, 0.5, 1.)
            if self.parent.sizable_from[0] in ('t', 'b'):
                Window.set_system_cursor("size_ns")
            else:
                Window.set_system_cursor("size_we")
            root.cursorSource = self
            root.cursorPrio = 1
        else:
            self.background_color = (1., 1., 1., 1.)
            if root.cursorSource == self:
                Window.set_system_cursor("arrow")
                root.cursorSource = root
                root.cursorPrio = 0

    def on_press(self) -> None:
        root = App.get_running_app().root
        if "accColor" in dir(root):
            self.background_color = root.accColor
        super().on_press()

    def on_release(self) -> None:
        super().on_release()
        self.on_mouseover(self, Window.mouse_pos)
        
class CursorAwareView(ModalView):
    """A modified ModalView that overrides mouse cursor changes from widgets with lower priority"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        root = App.get_running_app().root
        if self.collide_point(*self.to_widget(*pos)) and root.cursorPrio <= 2:
            Window.set_system_cursor("arrow")
            root.cursorSource = self
            root.cursorPrio = 2
        elif root.cursorSource == self:
            root.cursorSource = root
            root.cursorPrio = 0

class NumberInput(TextInput):
    """A modified text input field that sanitizes the input, ensuring it is always an integer number"""

    def insert_text(self, substring:str, from_undo:bool=False) -> None:
        s = ""
        s += "".join(char for char in substring if char.isdigit())
        return super().insert_text(s, from_undo=from_undo)

class ReferencingButton(ManagedButton):
    """A button with an additional property for keeping a reference to a different, arbitrary object"""

    reference = ObjectProperty()

class ListElement(ManagedButton):
    """A button with an additional integer property representing its position in a list"""

    index = NumericProperty()
