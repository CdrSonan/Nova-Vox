import torch

from kivy.uix.widget import Widget

from kivy.core.window import Window

from MiddleLayer.MiddleLayer import MiddleLayer

from UI.code.editor.PianoRoll import *

class NovaVoxUI(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global middleLayer
        middleLayer = MiddleLayer(self.ids)
        global manager
        from editor_runtime import manager
        self._keyboard = Window.request_keyboard(None, self, 'text')
        if self._keyboard.widget:
            pass
        self._keyboard.bind(on_key_down = self._on_keyboard_down)
        self._keyboard.bind(on_key_up = self._on_keyboard_up)
        self._keyboard.target = None
    def update(self, deltatime):
        change = manager.receiveChange()
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
    def setMode(self, mode):
        global middleLayer
        middleLayer.mode = mode
        if middleLayer.activeTrack == None:
            return
        middleLayer.updateParamPanel()
        middleLayer.changePianoRollMode()
    def setTool(self, tool):
        global middleLayer
        middleLayer.tool = tool
    def play(self, state):
        if state == "down":
            middleLayer.play(True)
        else:
            middleLayer.play(False)
    def spoolBack(self):
        if middleLayer.activeTrack == None or len(middleLayer.trackList[middleLayer.activeTrack].borders) == 0:
            middleLayer.mainAudioBufferPos = 0
            middleLayer.movePlayhead(0)
        else:
            middleLayer.mainAudioBufferPos = middleLayer.trackList[middleLayer.activeTrack].borders[0]
            middleLayer.movePlayhead(middleLayer.trackList[middleLayer.activeTrack].borders[0])
            
    def spoolForward(self):
        if middleLayer.activeTrack == None or len(middleLayer.trackList[middleLayer.activeTrack].borders) == 0:
            return
        else:
            middleLayer.mainAudioBufferPos = middleLayer.trackList[middleLayer.activeTrack].borders[-1]
            middleLayer.movePlayhead(middleLayer.trackList[middleLayer.activeTrack].borders[-1])
    def undo(self):
        print("undo callback")
    def redo(self):
        print("redo callback")
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keyboard.target != None:
            return False
        if keycode[0] == 303 or keycode[0] == 304: 
            middleLayer.shift = True
        elif keycode[0] == 32:
            middleLayer.play()
        else:
            print("keycode pressed:", keycode[0])
        return True
    def _on_keyboard_up(self, keyboard, keycode):
        if keyboard.target != None:
            return False
        if keycode[0] == 303 or keycode[0] == 304: 
            middleLayer.shift = False
        else:
            pass
        return True