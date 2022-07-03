import torch

from kivy.uix.widget import Widget
from kivy.core.window import Window

from MiddleLayer.MiddleLayer import MiddleLayer

from UI.code.editor.AdaptiveSpace import *
from UI.code.editor.Headers import *
from UI.code.editor.PianoRoll import *
from UI.code.editor.Popups import *
from UI.code.editor.SidePanels import *
from UI.code.editor.Util import *

class NovaVoxUI(Widget):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        global middleLayer
        middleLayer = MiddleLayer(self.ids)
        self._keyboard = Window.request_keyboard(None, self, 'text')
        if self._keyboard.widget:
            pass
        self._keyboard.bind(on_key_down = self._on_keyboard_down)
        self._keyboard.bind(on_key_up = self._on_keyboard_up)
        self._keyboard.target = None

    def update(self, deltatime:float) -> None:
        """periodically called update function tied to UI updates that reads changes from the rendering process and passes them to the middleLayer"""

        change = middleLayer.manager.receiveChange()
        if change == None:
            return None
        if change.type == False:
            middleLayer.updateRenderStatus(change.track, change.index, change.value)
        elif change.type == "updateAudio":
            middleLayer.updateAudioBuffer(change.track, change.index, change.value)
        elif change.type == "zeroAudio":
            if change.value < 0:
                change.index += change.value
                change.value *= -1
            middleLayer.updateAudioBuffer(change.track, change.index, torch.zeros([change.value,]))
        else:
            middleLayer.deletions.pop(0)
        return self.update(deltatime)

    def setMode(self, mode) -> None:
        """signals the middleLAyer a change of the input mode and prompts the required UI updates"""

        global middleLayer
        middleLayer.mode = mode
        if middleLayer.activeTrack == None:
            return
        middleLayer.updateParamPanel()
        middleLayer.changePianoRollMode()

    def setTool(self, tool) -> None:
        """signals the middleLAyer a change of the current tool"""

        global middleLayer
        middleLayer.tool = tool

    def play(self, state:bool) -> None:
        """signals the middle layer a state change of the playback toggle"""

        if state == "down":
            middleLayer.play(True)
        else:
            middleLayer.play(False)

    def spoolBack(self) -> None:
        """sets the playback head to the beginning of the active track"""

        if middleLayer.activeTrack == None or len(middleLayer.trackList[middleLayer.activeTrack].borders) == 0:
            middleLayer.mainAudioBufferPos = 0
            middleLayer.movePlayhead(0)
        else:
            middleLayer.mainAudioBufferPos = middleLayer.trackList[middleLayer.activeTrack].borders[0]
            middleLayer.movePlayhead(middleLayer.trackList[middleLayer.activeTrack].borders[0])
            
    def spoolForward(self) -> None:
        """sets the playback head to the end of the active track"""

        if middleLayer.activeTrack == None or len(middleLayer.trackList[middleLayer.activeTrack].borders) == 0:
            return
        else:
            middleLayer.mainAudioBufferPos = middleLayer.trackList[middleLayer.activeTrack].borders[-1]
            middleLayer.movePlayhead(middleLayer.trackList[middleLayer.activeTrack].borders[-1])

    def undo(self) -> None:
        """placeholder for undo function"""

        print("undo callback")

    def redo(self) -> None:
        """placeholder for redo function"""

        print("redo callback")

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers) -> None:
        """universal function for processing keyboard shortcuts and keeping track of modifier keys"""

        if keyboard.target != None:
            return False
        if keycode[0] == 303 or keycode[0] == 304: 
            middleLayer.shift = True
        elif keycode[0] == 32:
            middleLayer.play()
        else:
            print("keycode pressed:", keycode[0])
        return True

    def _on_keyboard_up(self, keyboard, keycode) -> None:
        """signals the middleLayer when modifier keys have been let go of"""
        
        if keyboard.target != None:
            return False
        if keycode[0] == 303 or keycode[0] == 304: 
            middleLayer.shift = False
        else:
            pass
        return True
