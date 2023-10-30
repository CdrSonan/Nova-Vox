#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import re

from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.app import App
from kivy.metrics import dp

def fullRoot(widget):
    root = widget
    while root.parent != None and root.parent != root.parent.parent:
        root = root.parent
    return root

class Tooltip(Label):
    pass

class TooltipBehavior(object):
    tooltip_txt = StringProperty('')
    tooltip_cls = ObjectProperty(Tooltip)
    
    def __init__(self, **kwargs):
        self._tooltip = None
        super(TooltipBehavior, self).__init__(**kwargs)
        self.fbind('tooltip_cls', self._build_tooltip)
        self.fbind('tooltip_txt', self._update_tooltip)
        Window.bind(mouse_pos=self.on_mouse_pos)
        self._build_tooltip()
    
    def _build_tooltip(self, *largs):
        if self._tooltip:
            self._tooltip = None
        self._tooltip = self.tooltip_cls()
        self._update_tooltip()
    
    def _update_tooltip(self, *largs):
        txt = self.tooltip_txt
        if txt:
            self._tooltip.text = txt
        else:
            self._tooltip.text = ''
    
    def on_mouse_pos(self, *args):
        if not self.get_root_window() or self._tooltip.text == '':
            return
        pos = args[1]
        self._tooltip.pos = pos
        Clock.unschedule(self.display_tooltip) # cancel scheduled event since I moved the cursor
        self.close_tooltip() # close if it's opened
        if self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))):
            Clock.schedule_once(self.display_tooltip, 1)

    def close_tooltip(self, *args):
        Window.remove_widget(self._tooltip)

    def display_tooltip(self, *args):
        Window.add_widget(self._tooltip)

class ImageButton(ButtonBehavior, Image, TooltipBehavior):
    """a "managed" button that displays an image instead of text, automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))):
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

class ImageToggleButton(ToggleButtonBehavior, Image, TooltipBehavior):
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
        elif self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))):
            self.color = (0.5, 0.5, 0.5, 1.)
        else:
            self.color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        if self.function != None:
            self.function()

    def on_state(self, widget, value) -> None:
        self.on_mouseover(widget, Window.mouse_pos)

class ManagedButton(Button, TooltipBehavior):
    """a "managed" button that automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    function = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))):
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

class ManagedToggleButton(ToggleButton, TooltipBehavior):
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
        elif self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))):
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
        if self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))):
            self.background_color = (0.5, 0.5, 0.5, 1.)
        else:
            self.background_color = (1., 1., 1., 1.)

    def on_press(self) -> None:
        root = App.get_running_app().root
        if "accColor" in dir(root):
            self.background_color = root.accColor

    def on_release(self) -> None:
        self.on_mouseover(self, Window.mouse_pos)

class ManagedSpinner(Spinner, TooltipBehavior):
    """a "managed" spinner that automatically sets its appeareance to match the app theme, and includes mouseover functionality"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.is_open:
            root = App.get_running_app().root
            if "accColor" in dir(root):
                self.background_color = root.accColor
        elif self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))):
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
        if self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))) and root.cursorPrio <= 1:
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
        if self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))) and root.cursorPrio <= 2:
            Window.set_system_cursor("arrow")
            root.cursorSource = self
            root.cursorPrio = 2
        elif root.cursorSource == self:
            root.cursorSource = root
            root.cursorPrio = 0

class NumberInput(TextInput, TooltipBehavior):
    """A modified text input field that sanitizes the input, ensuring it is always an integer number"""

    def insert_text(self, substring:str, from_undo:bool=False) -> None:
        s = ""
        s += "".join(char for char in substring if char.isdigit())
        return super().insert_text(s, from_undo=from_undo)

class FloatInput(TextInput, TooltipBehavior):

    pat = re.compile('[^0-9]')
    def insert_text(self, substring, from_undo=False):
        pat = self.pat
        if '.' in self.text:
            s = re.sub(pat, '', substring)
        else:
            s = '.'.join(
                re.sub(pat, '', s)
                for s in substring.split('.', 1)
            )
        return super().insert_text(s, from_undo=from_undo)

class IntInput(TextInput, TooltipBehavior):
    
    pat = re.compile('[^0-9]')
    def insert_text(self, substring, from_undo=False):
        pat = self.pat
        s = re.sub(pat, '', substring)
        return super().insert_text(s, from_undo)

class ReferencingButton(ManagedButton, TooltipBehavior):
    """A button with an additional property for keeping a reference to a different, arbitrary object"""

    reference = ObjectProperty()

class ListElement(ManagedButton, TooltipBehavior):
    """A button with an additional integer property representing its position in a list"""

    index = NumericProperty()

class ManagedPopup(Popup):

    message = StringProperty()
