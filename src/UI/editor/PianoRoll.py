#Copyright 2022-2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, BooleanProperty, NumericProperty, ListProperty
from kivy.graphics import Color, Line, Rectangle, InstructionGroup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.bubble import Bubble

import torch

from UI.editor.Util import ManagedToggleButton, ReferencingButton

import API.Ops

from kivy.clock import mainthread
from kivy.core.window import Window
from kivy.metrics import dp

from math import floor, ceil

import global_consts

from API.Addon import UIExtensions

from Localization.editor_localization import getLanguage
loc = getLanguage()

class NoteProperties(Bubble):
    """class for the context menu of a note"""

    reference = ObjectProperty()

    def on_parent(self, instance, value) -> None:
        global middleLayer, UIExtensions
        for i in UIExtensions["noteContextMenu"]:
            self.content.add_widget(i.instance)
        for i in self.content.children:
            i.reference = self.reference

class PhonemeSelector(Bubble):
    """class for the phoneme selection menu for notes not in phoneme mode with multiple available pronunciations"""

    reference = ObjectProperty()
    
    def __init__(self, options, index, word, **kwargs):
        super().__init__(**kwargs)
        self.options = options
        self.index = index
        self.word = word
        for i in range(len(self.options)):
            self.content.add_widget(ReferencingButton(text = self.options[i], reference = self, on_press = lambda a : API.Ops.ChangeLyrics(self.index, self.word, i)()))
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if not (self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))) or self.reference.collide_point(*self.children[0].to_widget(dp(pos[0]), dp(pos[1])))):
            self.reference.remove_widget(self)
        

class Note(ManagedToggleButton):
    """class for a note on the piano roll"""

    xPos = NumericProperty()
    yPos = NumericProperty()
    length = NumericProperty()
    inputMode = BooleanProperty(False)
    statusBars = ListProperty()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)
        self.propertiesBubble = None
        self.reference = None

    def on_mouseover(self, window, pos):
        super().on_mouseover(window, pos)
        if self.propertiesBubble and not (self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))) or self.children[0].collide_point(*self.children[0].to_widget(dp(pos[0]), dp(pos[1])))):
            self.remove_widget(self.propertiesBubble)
            self.propertiesBubble = None

    def on_parent(self, note, parent) -> None:
        """redraw call during initial note creation"""

        if parent != None:
            self.redraw()

    def redraw(self) -> None:
        """redraws the note"""

        self.pos = (self.xPos * self.parent.parent.xScale, self.yPos * self.parent.parent.yScale)
        self.width = self.length * self.parent.parent.xScale
        self.height = self.parent.parent.yScale
        self.redrawStatusBars(True)

    def updateStatus(self, index, status):
        """updates one of the status bars present on the note"""

        global middleLayer
        from UI.editor.Main import middleLayer
        if len(middleLayer.trackList[middleLayer.activeTrack].phonemes) == 0:
            return
        if index >= len(self.statusBars) or index < 0:
            return
        self.canvas.remove(self.statusBars[index])
        del self.statusBars[index]
        rectanglePos = (self.pos[0] + (index) / len(self.reference.phonemes) * self.width, self.pos[1] + self.height * status / 5.)
        rectangleSize = (self.width / len(self.reference.phonemes), self.height * (1. - status / 5.))
        group = InstructionGroup()
        group.add(Color(0., 0., 0., 0.5))
        group.add(Rectangle(pos = rectanglePos, size = rectangleSize))
        self.statusBars.insert(index, group)
        self.canvas.add(self.statusBars[index])

    def quantize(self, x:float, y:float = None) -> tuple:
        """adjusts the x coordinate of a touch to achieve the desired input quantization"""
        if self.parent.parent.quantization == None:
            xOut = x
        else:
            xOut = int(x / self.parent.parent.xScale / self.parent.parent.quantization + 0.5) * self.parent.parent.xScale
        if y == None:
            return xOut
        return (xOut, y)

    def on_touch_down(self, touch) -> bool:
        """callback function used for processing mouse input on the note"""

        global middleLayer
        from UI.editor.Main import middleLayer
        if middleLayer.mode != "notes":
            return False
        if self.collide_point(*touch.pos):
            coord = self.to_local(touch.x, touch.y)
            touch.ud["initialPos"] = coord
            index = middleLayer.trackList[middleLayer.activeTrack].notes.index(self.reference)
            touch.ud["noteIndex"] = index
            if coord[0] <= self.x + self.width and coord[0] > max(self.x, self.x + self.width - 10):
                touch.ud["grabMode"] = "end"
            elif coord[0] >= self.x and coord[0] < min(self.x + 10, self.x + self.width):
                touch.ud["grabMode"] = "start"
            else:
                touch.ud["grabMode"] = "mid"
                touch.ud["xOffset"] = (self.pos[0] - coord[0]) / self.parent.parent.xScale
                touch.ud["yOffset"] = (self.pos[1] - coord[1]) / self.parent.parent.yScale
            touch.ud['param'] = False
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch) -> bool:
        """callback function used for processing mouse input on the note"""

        global middleLayer
        from UI.editor.Main import middleLayer
        if middleLayer.mode != "notes" or touch.is_mouse_scrolling or "initialPos" not in touch.ud.keys():
            return False
        coord = self.to_local(touch.x, touch.y)
        if (abs(touch.ud["initialPos"][0] - coord[0]) < 4) and (abs(touch.ud["initialPos"][1] - coord[1]) < 4) and self.collide_point(*coord):
            if self.state == "down":
                super().on_touch_down(touch)
                super().on_touch_up(touch)
            else:
                self.trigger_action()
            return True
        return False

    def on_state(self, screen, state) -> None:
        """creates or removes the note's context menu when it is selected or deselected"""

        super().on_state(screen, state)
        if state == "normal" and self.propertiesBubble:
            self.remove_widget(self.propertiesBubble)
            self.propertiesBubble = None
        elif state == "down" and not self.propertiesBubble:
            self.add_widget(NoteProperties(reference = self))
            self.propertiesBubble = self.children[0]

    def changeInputMode(self) -> None:
        """switches the note's input mode between text and phonemes"""

        global middleLayer
        from UI.editor.Main import middleLayer
        self.inputMode = not self.inputMode
        self.reference.phonemeMode = self.inputMode
        index = middleLayer.trackList[middleLayer.activeTrack].notes.index(self.reference)
        API.Ops.ChangeLyrics(index, self.children[1].text, immediate = True)

    def delete(self) -> None:
        """deletes the note"""

        global middleLayer
        from UI.editor.Main import middleLayer
        self.parent.parent.notes.remove(self)
        self.parent.remove_widget(self)
        index = middleLayer.trackList[middleLayer.activeTrack].notes.index(self.reference)
        API.Ops.RemoveNote(index)()

    def changeLyrics(self, text:str, focus = False) -> None:
        """changes the lyrics of the note"""

        global middleLayer
        from UI.editor.Main import middleLayer
        if focus:
            return
        index = middleLayer.trackList[middleLayer.activeTrack].notes.index(self.reference)
        API.Ops.ChangeLyrics(index, text)()
        self.redrawStatusBars()

    def redrawStatusBars(self, complete = False) -> None:
        """redraws the statuse bars belonging to the note, updating their number, position, and size as necessary in the process.
        "complete" is a flag that indicates whether the status bars are to be drawn as if rendering of their respective phonemes had been completed.
        If the flag is not set, the status bars are instead drawn without any rendering progress."""
        
        phonemeLength = len(self.reference.phonemes)
        for i in range(len(self.statusBars)):
            self.canvas.remove(self.statusBars[-1])
            del self.statusBars[-1]
        for i in range(phonemeLength):
            if complete:
                rectanglePos = (self.pos[0] + i / phonemeLength * self.width, self.pos[1] + self.height)
                rectangleSize = (self.width / phonemeLength,0)
            else:
                rectanglePos = (self.pos[0] + i / phonemeLength * self.width, self.pos[1])
                rectangleSize = (self.width / phonemeLength, self.height)
            group = InstructionGroup()
            group.add(Color(0., 0., 0., 0.5))
            group.add(Rectangle(pos = rectanglePos, size = rectangleSize))
            self.statusBars.insert(i, group)
            self.canvas.add(self.statusBars[i])

class PianoRollOctave(FloatLayout):
    """header of a one octave region on the piano roll"""

    octave = NumericProperty()

class PianoRollOctaveBackground(FloatLayout):
    """background of the piano roll spanning one octave"""

    index = NumericProperty()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)
        self.tooltip = None

    def on_mouseover(self, window, pos):
        if self.parent.collide_point(*self.to_parent(dp(pos[0]), dp(pos[1]))):
            if self.tooltip != None:
                self.parent.remove_widget(self.tooltip)
            index = floor((self.to_widget(dp(pos[0]), dp(pos[1]))[1] - self.y) * 12 / self.height)
            if (index < 0) or (index > 11):
                return
            text = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            if index == 0:
                return
            if index in (1, 3, 6, 8, 10):
                self.tooltip = Label(pos = (self.to_widget(self.parent.parent.x + 25, 0)[0], self.y + self.height * (index + 0.5) / 12 - 5),
                                    size_hint = (None, None),
                                    size = [10, 10],
                                    text = text[index] + str(self.index),
                                    color = (1, 1, 1, 1))
            else:
                self.tooltip = Label(pos = (self.to_widget(self.parent.parent.x + 80, 0)[0], self.y + self.height * ([0, 2, 4, 5, 7, 9, 11].index(index) + 0.5) / 7 - 5),
                                    size_hint = (None, None),
                                    size = [10, 10],
                                    text = text[index] + str(self.index),
                                    color = (0, 0, 0, 1))
            self.parent.add_widget(self.tooltip)
        elif (not self.parent.parent.collide_point(*self.to_parent(dp(pos[0]), dp(pos[1])))) and self.tooltip != None:
            self.parent.remove_widget(self.tooltip)
            self.tooltip = None

class PlaybackHead(Widget):
    """playback head widget for the piano roll"""

    pass

class TimingBar(FloatLayout):
    """header of the piano roll, displaying the current measures and other timing information"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.markers = []

class TimingLabel(Label):
    """A label for a time on the timing bar and piano roll"""

    index = NumericProperty()
    reference = ObjectProperty()
    modulo = NumericProperty()

class PianoRoll(ScrollView):
    """Class for the editor piano roll"""

    length = NumericProperty(5000)
    xScale = NumericProperty(1)
    yScale = NumericProperty(10)

    def __init__(self, **kwargs) -> None:
        super(PianoRoll, self).__init__(**kwargs)
        self.measureSize = 4
        self.tempo = 125
        self.quantization = None
        self.timingMarkers = []
        self.timingZones = []
        self.timingHints = []
        self.notes = []
        self.pitchLine = None
        self.basePitchLine = None
        self.generateTimingMarkers()

    def generate_notes(self) -> None:
        """plots all notes of a track, fetching the required data from the middleLayer"""

        global middleLayer
        from UI.editor.Main import middleLayer
        for i in middleLayer.trackList[middleLayer.activeTrack].notes:
            note = Note(xPos = i.xPos, yPos = i.yPos, length = i.length, height = self.yScale, inputMode = i.phonemeMode)
            note.reference = i
            i.reference = note
            note.children[0].text = i.content
            self.children[0].add_widget(note)
            self.notes.append(note)

    @mainthread
    def generateTimingMarkers(self) -> None:
        """delayed call of updateTimingMarkers(). The delay is required during widget initialization, since not all properties required by updateTimingMarkers() are available during initialization yet."""

        self.updateTimingMarkers()

    def updateLength(self) -> None:
        """updates the length of the track according to the position and length of the last note"""

        if len(middleLayer.trackList) > 0 and len(middleLayer.trackList[middleLayer.activeTrack].notes) > 0:
            noteEnd = middleLayer.trackList[middleLayer.activeTrack].notes[-1].xPos + middleLayer.trackList[middleLayer.activeTrack].notes[-1].length
        else:
            noteEnd = 0
        self.length = noteEnd + 5000
        middleLayer.changeLength(self.length)
        self.updateTimingMarkers()

    def updateTempo(self, measureType:str, tempo:str, quantization:str) -> None:
        """callback function used or updating the tempo of the track"""

        measureSizes = {
            "4/4": 4,
            "3/4": 3,
            "2/4": 2,
            "6/8": 6,
            "1/1": 1
        }
        tempoMultipliers = {
            "4/4": 1.,
            "3/4": 1.,
            "2/4": 1.,
            "6/8": 3.,
            "1/1": 1.
        }
        quantMultipliers = {
            "4/4": 4.,
            "3/4": 4.,
            "2/4": 4.,
            "6/8": 8.,
            "1/1": 1.
        }
        quantFactors = {
            "Q: 1/1": 1.,
            "Q: 1/2": 2.,
            "Q: 1/4": 4.,
            "Q: 1/8": 8.,
            "Q: 1/16": 16.,
            "Q: 1/32": 32.,
            loc["quant_off"]: None
        }
        self.measureSize = measureSizes[measureType]
        if tempo != "":
            self.tempo = 15000. / (float(tempo) * tempoMultipliers[measureType])
        if quantFactors[quantization] == None:
            self.quantization = None
        else:
            self.quantization = self.tempo * quantMultipliers[measureType] / quantFactors[quantization]
        while len(self.ids["timingBar"].markers) > 0:
            self.ids["timingBar"].remove_widget(self.ids["timingBar"].markers.pop())
        self.updateTimingMarkers()

    def quantize(self, x:float, y:float = None) -> tuple:
        """adjusts the x coordinate of a touch to achieve the desired input quantization"""

        if self.quantization == None:
            xOut = x
        else:
            xOut = int(int(x / self.xScale / self.quantization + 0.5) * self.xScale * self.quantization)
        if y == None:
            return xOut
        return (xOut, y)

    def updateTimingMarkers(self) -> None:
        """plots all timing markers that are currently in view"""

        i = floor(self.length * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width / self.tempo)
        while len(self.ids["timingBar"].markers) > 0 and self.ids["timingBar"].markers[0].index < i:
            self.ids["timingBar"].remove_widget(self.ids["timingBar"].markers[0])
            self.ids["timingBar"].markers = self.ids["timingBar"].markers[1:]
        i = ceil(self.length * (self.scroll_x * (self.children[0].width - self.width) + self.width) / self.children[0].width / self.tempo)
        while len(self.ids["timingBar"].markers) > 0 and self.ids["timingBar"].markers[-1].index > i:
            self.ids["timingBar"].remove_widget(self.ids["timingBar"].markers[-1])
            self.ids["timingBar"].markers = self.ids["timingBar"].markers[:-1]

        if len(self.ids["timingBar"].markers) > 0:
            i = self.ids["timingBar"].markers[-1].index + 1
        else:
            i = floor(self.length * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width / self.tempo) + 1
        while i <= self.length * (self.scroll_x * (self.children[0].width - self.width) + self.width) / self.children[0].width / self.tempo:
            modulo = i % self.measureSize
            self.ids["timingBar"].markers.append(TimingLabel(index = i, reference = self.ids["timingBar"], modulo = modulo))
            self.ids["timingBar"].add_widget(self.ids["timingBar"].markers[-1])
            i += 1

        if len(self.ids["timingBar"].markers) > 0:
            i = self.ids["timingBar"].markers[0].index - 1
        else:
            i = floor(self.length * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width / self.tempo)
        while i >= self.length * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width / self.tempo:
            modulo = i % self.measureSize
            self.ids["timingBar"].markers.insert(0, TimingLabel(index = i, reference = self.ids["timingBar"], modulo = modulo))
            self.ids["timingBar"].add_widget(self.ids["timingBar"].markers[0])
            i -= 1

    def on_scroll_x(self, instance, scroll_x:float) -> None:
        """updates the displayed timing markers when the view is scrolled"""
        
        self.updateTimingMarkers()
        if middleLayer.mode == "pitch":
            self.redrawPitch()

    def on_xScale(self, instance, xScale:float) -> None:
        """updates the displayed timing markers, notes and other content when the x axis zoom level is changed"""

        self.updateTimingMarkers()
        self.updateTrack()

    def on_scroll_y(self, instance, scroll_y:float) -> None:
        for i in self.timingHints:
            i.y = (self.children[0].height - self.height) * scroll_y

    def changePlaybackPos(self, playbackPos:float) -> None:
        """changes the position of the playback head"""

        with self.ids["playbackHead"].canvas:
            points = self.ids["playbackHead"].canvas.children[-1].points
            points[0] = playbackPos * self.xScale
            points[2] = playbackPos * self.xScale
            del self.ids["playbackHead"].canvas.children[-1]
            Line(points = points)
            del self.ids["playbackHead"].canvas.children[-2]

    def redrawPitch(self) -> None:
        """redraws the pitch curve, fetching the required data from the middleLayer"""

        global middleLayer
        from UI.editor.Main import middleLayer
        start = floor(self.length * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width * 0.9)
        end = min(ceil(self.length * (self.scroll_x * (self.children[0].width - self.width) + self.width) / self.children[0].width / 0.9), self.length)
        data1 = middleLayer.trackList[middleLayer.activeTrack].pitch
        data2 = middleLayer.trackList[middleLayer.activeTrack].basePitch
        points1 = ((data1[start:end] + 0.5) * self.yScale).to(torch.int)
        points2 = ((data2[start:end] + 0.5) * self.yScale).to(torch.int)
        scale = torch.arange(start, end, dtype = torch.int) * self.xScale
        points1 = torch.cat((scale.unsqueeze(1), points1.unsqueeze(1)), dim = 1).flatten().tolist()
        points2 = torch.cat((scale.unsqueeze(1), points2.unsqueeze(1)), dim = 1).flatten().tolist()
        if self.pitchLine != None:
            self.children[0].canvas.remove(self.pitchLine)
        if self.basePitchLine != None:
            self.children[0].canvas.remove(self.basePitchLine)
        with self.children[0].canvas:
            Color(0.8, 0.8, 0.8, 1)
            self.basePitchLine = Line(points = points2)
            Color(1, 0, 0, 1)
            self.pitchLine = Line(points = points1)
            
    def timingMarkerGroup(self, pos):
        group = InstructionGroup()
        group.add(Color(1, 0, 0))
        group.add(Line(points = [self.xScale * pos, 0, self.xScale * pos, self.children[0].height]))
        return group
    
    def timingZoneGroup(self, pos, size):
        group = InstructionGroup()
        group.add(Color(1, 0, 1, 0.33))
        group.add(Rectangle(pos = (pos, 0), size = (size, self.children[0].height)))
        return group
    
    def drawTiming(self) -> None:
        """draws all border/timing markers and related UI elements"""
        for i in middleLayer.trackList[middleLayer.activeTrack].borders:
            self.timingMarkers.append(self.timingMarkerGroup(i))
            self.children[0].canvas.add(self.timingMarkers[-1])
        for i in range(len(middleLayer.trackList[middleLayer.activeTrack].phonemes) + 1):
            pos = self.xScale * middleLayer.trackList[middleLayer.activeTrack].borders[3 * i]
            size = self.xScale * middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 2] - pos
            self.timingZones.append(self.timingZoneGroup(pos, size))
            self.children[0].canvas.add(self.timingZones[-1])
        for i, phoneme in enumerate(middleLayer.trackList[middleLayer.activeTrack].phonemes):
            self.timingHints.append(Label(size_hint = (None, None),
                                           x = middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 2] * self.xScale,
                                           y = (self.children[0].height - self.height) * self.scroll_y,
                                           width = (middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 3] - middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 2]) * self.xScale,
                                           height = 30,
                                           text = phoneme,
                                           halign = "center"))
            self.children[0].add_widget(self.timingHints[-1], index = 5)

    """def updateTiming(self) -> None:
        draws and/or removes timing-related UI elements, so that the entire viewport is covered, but the regions beyond it are left unrendered

        start = floor(self.length * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width)
        end = min(ceil(self.length * (self.scroll_x * (self.children[0].width - self.width) + self.width) / self.children[0].width), self.length)
        while len(self.timingMarkers) > 0 and self.timingMarkers[0] < start:
            self.timingMarkers = self.timingMarkers[1:]
        while len(self.timingMarkers) > 0 and self.timingMarkers[-1] > end:
            self.timingMarkers = self.timingMarkers[:-1]"""
        
    def removeTiming(self) -> None:
        """removes all border/timing markers and related UI elements"""

        for i in self.timingMarkers:
            self.children[0].canvas.remove(i)
        del self.timingMarkers[:]
        for i in self.timingZones:
            self.children[0].canvas.remove(i)
        del self.timingZones[:]
        for i in self.timingHints:
            self.children[0].remove_widget(i)
        del self.timingHints[:]

    def changeMode(self) -> None:
        """prompts all UI changes required when the input mode changes"""

        global middleLayer
        from UI.editor.Main import middleLayer
        if middleLayer.mode == "notes":
            self.removeTiming()
            if self.pitchLine != None:
                self.children[0].canvas.remove(self.pitchLine)
                self.pitchLine = None
            if self.basePitchLine != None:
                self.children[0].canvas.remove(self.basePitchLine)
                self.basePitchLine = None
        if middleLayer.mode == "timing":
            for i in self.children[0].children:
                i.state = "normal"
            if self.pitchLine != None:
                self.children[0].canvas.remove(self.pitchLine)
                self.pitchLine = None
            if self.basePitchLine != None:
                self.children[0].canvas.remove(self.basePitchLine)
                self.basePitchLine = None
            self.drawTiming()
        if middleLayer.mode == "pitch":
            for i in self.children[0].children:
                i.state = "normal"
            self.removeTiming()
            self.redrawPitch()

    def updateTrack(self) -> None:
        """updates the piano roll as required when the active track changes"""

        global middleLayer
        from UI.editor.Main import middleLayer
        for i in self.notes:
            self.children[0].remove_widget(i)
        del self.notes[:]
        self.removeTiming()
        if self.pitchLine != None:
            self.children[0].canvas.remove(self.pitchLine)
            self.pitchLine = None
        if self.basePitchLine != None:
            self.children[0].canvas.remove(self.basePitchLine)
            self.basePitchLine = None
        if middleLayer.activeTrack == None:
            return
        self.generate_notes()
        if middleLayer.mode == "timing":
            self.drawTiming()
        if middleLayer.mode == "pitch":
            self.redrawPitch()

    def applyScroll(self, scrollValue:float) -> None:
        """sets the x scroll value of the  widget"""

        self.scroll_x = scrollValue
        
    def applyZoom(self, xScale:float) -> None:
        """sets the x zoom value of the  widget"""

        self.xScale = xScale

    def triggerScroll(self) -> None:
        """signals the middleLayer that the x scroll value of the widget has changed. Used for synchronizing scrolling between the adaptive space and piano roll."""

        global middleLayer
        from UI.editor.Main import middleLayer
        middleLayer.scrollValue = self.scroll_x
        middleLayer.applyScroll()
        
    def triggerZoom(self) -> None:
        """signals the middleLayer that the x scale value/zoom of the widget has changed. Used for synchronizing zoom level between the adaptive space and piano roll."""

        global middleLayer
        from UI.editor.Main import middleLayer
        middleLayer.xScale = self.xScale
        middleLayer.applyZoom()

    def on_touch_down(self, touch) -> bool:
        """Callback function used for processing mouse input on the piano roll"""

        global middleLayer
        from UI.editor.Main import middleLayer
        if touch.is_mouse_scrolling == False:
            if super(PianoRoll, self).on_touch_down(touch):
                return True
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if middleLayer.shift:
                    #horizontal scrolling
                    if touch.button == 'scrollup':
                        newvalue = self.scroll_x + self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                        if newvalue < 1:
                            self.scroll_x = newvalue
                        else:
                            self.scroll_x = 1.
                    elif touch.button == 'scrolldown':
                        newvalue = self.scroll_x - self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                        if newvalue > 0:
                            self.scroll_x = newvalue
                        else:
                            self.scroll_x = 0.
                    return True
                if middleLayer.ctrl:
                    #horizontal zoom
                    if touch.button == 'scrollup':
                        newvalue = self.xScale / 0.9
                        if newvalue < 10.:
                            self.xScale = newvalue
                        else:
                            self.xScale = 10.
                    elif touch.button == 'scrolldown':
                        newvalue = self.xScale * 0.9
                        if newvalue > 0.1:
                            self.xScale = newvalue
                        else:
                            self.xScale = 0.1
                    return True
                if middleLayer.alt:
                    #vertical zoom
                    if touch.button == 'scrollup':
                        newvalue = self.yScale / 0.9
                        if newvalue < 100.:
                            self.yScale = newvalue
                        else:
                            self.yScale = 100.
                    elif touch.button == 'scrolldown':
                        newvalue = self.yScale * 0.9
                        if newvalue > 1.:
                            self.yScale = newvalue
                        else:
                            self.yScale = 1.
                    self.updateTrack()
                    return True
                return super(PianoRoll, self).on_touch_down(touch)
            else:
                coord = self.to_local(touch.x, touch.y)
                x = int(coord[0] / self.xScale)
                y = int(coord[1] / self.yScale)
                if touch.y < self.y + self.height and touch.y > self.y + self.height - 20:
                    middleLayer.mainAudioBufferPos = x * global_consts.batchSize
                    middleLayer.movePlayhead(x)
                    return True
                if middleLayer.activeTrack == None:
                    return True
                if middleLayer.mode == "notes":
                    index = 0
                    xQuant = self.quantize(x)
                    for i in self.notes:
                        if i.xPos < xQuant:
                            index += 1
                        elif i.xPos == xQuant:
                            index += 1
                            xQuant += 1
                    touch.ud["newNote"] = (index, xQuant, y, coord)
                elif middleLayer.mode == "timing":
                    def getNearestBorder(x):
                        nextBorder = 0
                        for i in range(len(middleLayer.trackList[middleLayer.activeTrack].borders) - 1):
                            previousBorder = nextBorder
                            nextBorder = (middleLayer.trackList[middleLayer.activeTrack].borders[i] + middleLayer.trackList[middleLayer.activeTrack].borders[i + 1]) / 2
                            if previousBorder < x and x <= nextBorder:
                                return i
                        return len(middleLayer.trackList[middleLayer.activeTrack].borders) - 1
                    border = getNearestBorder(x)
                    touch.ud["border"] = border
                    touch.ud["offset"] = x - middleLayer.trackList[middleLayer.activeTrack].borders[border]
                    touch.ud["shift"] = middleLayer.shift
                elif middleLayer.mode == "pitch":
                    with self.children[0].canvas:
                        Color(0, 0, 1)
                        touch.ud['line'] = Line(points=self.to_local(touch.x, touch.y))
                        touch.ud['startPoint'] = self.to_local(touch.x, touch.y)
                        touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height * self.yScale)]
                        touch.ud['lastPoint'] = touch.ud['startPoint'][0]
                        touch.ud['startPointOffset'] = 0
                touch.ud['param'] = False
                self.on_touch_move(touch)
                return True
        return False

    def on_touch_move(self, touch) -> bool:
        """Callback function used for processing mouse input on the piano roll"""

        global middleLayer
        from UI.editor.Main import middleLayer
        if "param" not in touch.ud:
            return super().on_touch_move(touch)
        if self.collide_point(*touch.pos) == False or touch.ud['param']:
            return False
        if middleLayer.activeTrack == None:
            return
        if middleLayer.mode == "notes":
            coord = self.quantize(*self.to_local(touch.x, touch.y))
            x = int(coord[0] / self.xScale)
            y = int(coord[1] / self.yScale)
            yMod = coord[1]
            if "newNote" in touch.ud:
                if self.to_local(touch.x, touch.y) != touch.ud["newNote"][3]:
                    newNote = Note(xPos = touch.ud["newNote"][1], yPos = touch.ud["newNote"][2], length = 100, height = self.yScale)
                    API.Ops.AddNote(touch.ud["newNote"][0], touch.ud["newNote"][1], touch.ud["newNote"][2], newNote)()
                    self.children[0].add_widget(newNote, index = 5)
                    self.notes.append(newNote)
                    touch.ud["noteIndex"] = touch.ud["newNote"][0]
                    touch.ud["grabMode"] = "end"
                    touch.ud["initialPos"] = touch.ud["newNote"][3]
                    del touch.ud["newNote"]
                else:
                    for i in self.notes:
                        i.state = "normal"
            if "noteIndex" in touch.ud:
                note = middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference
                if abs(touch.ud["initialPos"][0] - coord[0]) < 4 and abs(touch.ud["initialPos"][1] - coord[1]) < 4:
                    return True
                if touch.ud["noteIndex"] == len(middleLayer.trackList[middleLayer.activeTrack].notes) - 1:
                    self.updateLength()
                if touch.ud["grabMode"] == "start":
                    length = max(note.xPos + note.length - x, 1)
                    note.length = length
                    note.xPos = x
                    switch = API.Ops.ChangeNoteLength(touch.ud["noteIndex"], x, length)()
                    if switch == True:
                        touch.ud["noteIndex"] += 1
                    elif switch == False:
                        touch.ud["noteIndex"] -= 1
                    note.redraw()
                elif touch.ud["grabMode"] == "mid":
                    xNonquant, yNonquant = self.to_local(touch.x, touch.y)
                    xNonquant /= self.xScale
                    yNonquant /= self.yScale
                    switch = API.Ops.MoveNote(touch.ud["noteIndex"], int(self.quantize(xNonquant + touch.ud["xOffset"] + 1)), int(yNonquant + touch.ud["yOffset"] + 0.5))()
                    if switch == True:
                        touch.ud["noteIndex"] += 1
                    elif switch == False:
                        touch.ud["noteIndex"] -= 1
                    note.redraw()
                elif touch.ud["grabMode"] == "end":
                    length = max(x - note.xPos, 1)
                    note.length = length
                    API.Ops.ChangeNoteLength(touch.ud["noteIndex"], note.xPos, length)()
                    note.redraw()
                return True
            else:
                return False
        else:
            coord = self.to_local(touch.x, touch.y)
            x = int(coord[0] / self.xScale)
            y = int(coord[1] / self.yScale)
            yMod = coord[1]
        if middleLayer.mode == "timing":

            def applyBorder(border, newPos, checkLeft, checkRight):
                if newPos < border:
                    return
                
                index = self.children[0].canvas.indexof(self.timingMarkers[border])
                self.children[0].canvas.remove(self.timingMarkers[border])
                del self.timingMarkers[border]
                self.timingMarkers.insert(border, self.timingMarkerGroup(newPos))
                self.children[0].canvas.insert(index, self.timingMarkers[border])
                zone = floor(border / 3)
                index = self.children[0].canvas.indexof(self.timingZones[zone])
                self.children[0].canvas.remove(self.timingZones[zone])
                del self.timingZones[zone]
                pos = self.xScale * middleLayer.trackList[middleLayer.activeTrack].borders[3 * zone]
                size = self.xScale * middleLayer.trackList[middleLayer.activeTrack].borders[3 * zone + 2] - pos
                self.timingZones.insert(zone, self.timingZoneGroup(pos, size))
                self.children[0].canvas.insert(index, self.timingZones[zone])
                
                if border % 3 == 2 and border < len(middleLayer.trackList[middleLayer.activeTrack].borders) - 1:
                    self.timingHints[int((border - 2) / 3)].x = self.xScale * newPos
                    self.timingHints[int((border - 2) / 3)].width = self.xScale * (middleLayer.trackList[middleLayer.activeTrack].borders[border + 1] - newPos)
                elif border % 3 == 0 and border > 0:
                    self.timingHints[int(border / 3) - 1].width = self.xScale * (newPos - middleLayer.trackList[middleLayer.activeTrack].borders[border - 1])
                API.Ops.MoveBorder(border, newPos)()
                if checkLeft and border > 0:
                    if middleLayer.trackList[middleLayer.activeTrack].borders[border - 1] >= middleLayer.trackList[middleLayer.activeTrack].borders[border]:
                        applyBorder(border - 1, newPos - 1, True, False)
                if checkRight and border < len(middleLayer.trackList[middleLayer.activeTrack].borders) - 1:
                    if middleLayer.trackList[middleLayer.activeTrack].borders[border] >= middleLayer.trackList[middleLayer.activeTrack].borders[border + 1]:
                        applyBorder(border + 1, newPos + 1, False, True)

            if touch.ud["shift"]:
                baseBorder = floor(touch.ud["border"] / 3) * 3
                offsets = []
                for i in range(3):
                    offsets.append(middleLayer.trackList[middleLayer.activeTrack].borders[baseBorder + i] - middleLayer.trackList[middleLayer.activeTrack].borders[touch.ud["border"]])
                for i in range(3):
                    newPos = x - touch.ud["offset"] + offsets[i]
                    applyBorder(baseBorder + i, newPos, True, True)
            else:
                newPos = x - touch.ud["offset"]
                applyBorder(touch.ud["border"], newPos, True, True)
            middleLayer.submitFinalize()
        elif middleLayer.mode == "pitch":
            p = x - int(touch.ud['startPoint'][0] / self.xScale)
            if middleLayer.tool == "draw":
                if p < 0:
                    for i in range(-p):
                        touch.ud['line'].points = [touch.ud['startPoint'][0] - i * self.xScale, yMod] + touch.ud['line'].points
                        touch.ud['startPoint'][0] -= 1
                    touch.ud['lastPoint'] = touch.ud['line'].points[0] / self.xScale
                elif p < int(len(touch.ud['line'].points) / 2):
                    points = touch.ud['line'].points
                    if x >= touch.ud['lastPoint']:
                        domain = range(int(touch.ud['lastPoint']) - int(touch.ud['startPoint'][0]), p + 1)
                    else:
                        domain = range(p, int(touch.ud['lastPoint']) - int(touch.ud['startPoint'][0]))
                    for i in domain:
                        points[2 * i + 1] = yMod
                    touch.ud['line'].points = points
                    touch.ud['lastPoint'] = points[2 * p] / self.xScale
                else:
                    diff = p - int(len(touch.ud['line'].points) / 2)
                    for i in range(diff):
                        touch.ud['line'].points += [(touch.ud['startPoint'][0] + int(len(touch.ud['line'].points) / 2)) * self.xScale, yMod]
                    touch.ud['lastPoint'] = touch.ud['line'].points[len(touch.ud['line'].points) - 2] / self.xScale
            elif middleLayer.tool == "line":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (yMod - touch.ud['startPoint'][1]) * -i / p]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (yMod - touch.ud['startPoint'][1]) * i / p]
            elif middleLayer.tool == "arch":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (yMod - touch.ud['startPoint'][1]) * (-i / p) * (-i / p)]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (yMod - touch.ud['startPoint'][1]) * (i / p) * (i / p)]
            elif middleLayer.tool == "reset":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, self.basePitchLine.points[2 * (-i + touch.ud['startPoint'][0]) + 1]]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, self.basePitchLine.points[2 * (i + touch.ud['startPoint'][0]) + 1]]
        else:
            return super().on_touch_move(touch)

    def on_touch_up(self, touch) -> bool:
        """Callback function used for processing mouse input on the piano roll"""
        
        global middleLayer
        from UI.editor.Main import middleLayer
        if "param" not in touch.ud:
            return super(PianoRoll, self).on_touch_up(touch)
        if 'startPoint' in touch.ud and touch.ud['param'] == False:
            data = []
            if touch.ud['startPointOffset'] == 0:
                for i in range(int(len(touch.ud['line'].points) / 2)):
                    data.append(touch.ud['line'].points[2 * i + 1] / self.yScale - 0.5)
            else:
                for i in range(int(len(touch.ud['line'].points) / 2)):
                    data.append(touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] / self.yScale - 0.5)
            API.Ops.ChangePitch(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'])()
            self.children[0].canvas.remove(touch.ud['line'])
            self.redrawPitch()
        else:
            return super(PianoRoll, self).on_touch_up(touch)
