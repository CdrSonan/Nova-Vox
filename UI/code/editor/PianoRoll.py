#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, BooleanProperty, NumericProperty, ListProperty
from kivy.graphics import Color, Line
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.bubble import Bubble

from kivy.clock import mainthread

from math import floor

import global_consts

class NoteProperties(Bubble):
    """class for the context menu of a note"""

    reference = ObjectProperty()

    def on_parent(self, instance, value) -> None:
        for i in self.content.children:
            i.reference = self.reference
        return super().on_parent(instance, value)

class Note(ToggleButton):
    """class for a note on the piano roll"""

    index = NumericProperty()
    xPos = NumericProperty()
    yPos = NumericProperty()
    length = NumericProperty()
    inputMode = BooleanProperty()

    def on_parent(self, note, parent) -> None:
        """redraw call during initial note creation"""

        if parent == None:
            return
        self.redraw()

    def redraw(self) -> None:
        """redraws the note"""

        self.pos = (self.xPos * self.parent.parent.xScale, self.yPos * self.parent.parent.yScale)
        self.width = self.length * self.parent.parent.xScale
        self.height = self.parent.parent.yScale

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
        from UI.code.editor.Main import middleLayer
        if middleLayer.mode != "notes":
            return False
        if self.collide_point(*touch.pos):
            coord = self.to_local(touch.x, touch.y)
            touch.ud["initialPos"] = coord
            touch.ud["noteIndex"] = self.index
            if coord[0] <= self.x + self.width and coord[0] > max(self.x, self.x + self.width - 10):
                touch.ud["grabMode"] = "end"
            elif coord[0] >= self.x and coord[0] < min(self.x + 10, self.x + self.width):
                touch.ud["grabMode"] = "start"
            else:
                touch.ud["grabMode"] = "mid"
                touch.ud["xOffset"] = (self.pos[0] - self.quantize(coord[0])) / self.parent.parent.xScale
                touch.ud["yOffset"] = (self.pos[1] - coord[1]) / self.parent.parent.yScale
            touch.ud['param'] = False
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch) -> bool:
        """callback function used for processing mouse input on the note"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
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

        if state == "normal":
            self.remove_widget(self.children[0])
        else:
            self.add_widget(NoteProperties(reference = self))

    def changeInputMode(self) -> None:
        """switches the note's input mode between text and phonemes"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        self.inputMode = not self.inputMode
        middleLayer.trackList[middleLayer.activeTrack].notes[self.index].phonemeMode = self.inputMode
        middleLayer.changeLyrics(self.index, self.children[1].text)

    def delete(self) -> None:
        """deletes the note"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.removeNote(self.index)
        for i in self.parent.children:
            if i.__class__.__name__ == "Note":
                if i.index > self.index:
                    i.index -= 1
        self.parent.remove_widget(self)

    def changeLyrics(self, text:str, focus = False) -> None:
        """changes the lyrics of the note"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if focus == False:
            middleLayer.changeLyrics(self.index, text)

class PianoRollOctave(FloatLayout):
    """header of a one octave region on the piano roll"""

    pass

class PianoRollOctaveBackground(FloatLayout):
    """background of the piano roll spanning one octave"""

    pass

class PlaybackHead(Widget):
    """playback head widget for the piano roll"""

    pass

class TimingBar(FloatLayout):
    """header of the piano roll, displaying the current measures and other timing information"""

    pass

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
        self.pitchLine = None
        self.basePitchLine = None
        self.generateTimingMarkers()

    def generate_notes(self) -> None:
        """plots all notes of a track, fetching the required data from the middleLayer"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        index = 0
        for i in middleLayer.trackList[middleLayer.activeTrack].notes:
            note = Note(index = index, xPos = i.xPos, yPos = i.yPos, length = i.length, height = self.yScale, inputMode = i.phonemeMode)
            note.children[0].text = i.content
            self.children[0].add_widget(note)
            middleLayer.trackList[middleLayer.activeTrack].notes[index].reference = note
            index += 1

    @mainthread
    def generateTimingMarkers(self) -> None:
        """delayed call of updateTimingMarkers(). The delay is required during widget initialization, since not all properties required by updateTimingMarkers() are available during initialization yet."""
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
            "Q: off": None
        }
        self.measureSize = measureSizes[measureType]
        if tempo != "":
            self.tempo = 15000. / (float(tempo) * tempoMultipliers[measureType])
        if quantFactors[quantization] == None:
            self.quantization = None
        else:
            self.quantization = self.tempo * quantMultipliers[measureType] / quantFactors[quantization]
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

        self.children[0].children[-global_consts.octaves - 1].clear_widgets()
        i = floor(self.length * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width / self.tempo)
        t = i * self.tempo
        while t <= self.length * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width + self.width:
            modulo = i % self.measureSize
            self.children[0].children[-global_consts.octaves - 1].add_widget(TimingLabel(index = i, reference = self.children[0].children[-global_consts.octaves - 1], modulo = modulo))
            t += self.tempo * self.xScale
            i += 1

    def on_scroll_x(self, instance, scroll_x:float) -> None:
        """updates the displayed timing markers when the view is scrolled"""
        
        self.updateTimingMarkers()

    def changePlaybackPos(self, playbackPos:float) -> None:
        """changes the position of the playback head"""

        with self.children[0].children[0].canvas:
            points = self.children[0].children[0].canvas.children[-1].points
            points[0] = playbackPos * self.xScale
            points[2] = playbackPos * self.xScale
            del self.children[0].children[0].canvas.children[-1]
            Line(points = points)
            del self.children[0].children[0].canvas.children[-2]

    def redrawPitch(self) -> None:
        """redraws the pitch curve, fetching the required data from the middleLayer"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        data1 = middleLayer.trackList[middleLayer.activeTrack].pitch
        data2 = middleLayer.trackList[middleLayer.activeTrack].basePitch
        points1 = []
        points2 = []
        c = 0
        for i in range(len(data1)):
            points1.append(c * self.xScale)
            points1.append((data1[i].item() + 0.5) * self.yScale)
            points2.append(c * self.xScale)
            points2.append((data2[i].item() + 0.5) * self.yScale)
            c += 1
        if self.pitchLine != None:
            self.children[0].canvas.remove(self.pitchLine)
        if self.basePitchLine != None:
            self.children[0].canvas.remove(self.basePitchLine)
        with self.children[0].canvas:
            Color(0.8, 0.8, 0.8, 1)
            self.basePitchLine = Line(points = points2)
            Color(1, 0, 0, 1)
            self.pitchLine = Line(points = points1)

    def changeMode(self) -> None:
        """prompts all UI changes required when the input mode changes"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.mode == "notes":
            for i in self.timingMarkers:
                self.children[0].canvas.remove(i)
            del self.timingMarkers[:]
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
            with self.children[0].canvas:
                Color(1, 0, 0)
                for i in middleLayer.trackList[middleLayer.activeTrack].borders:
                    self.timingMarkers.append(ObjectProperty())
                    self.timingMarkers[-1] = Line(points = [self.xScale * i, 0, self.xScale * i, self.children[0].height])
        if middleLayer.mode == "pitch":
            for i in self.children[0].children:
                i.state = "normal"
            for i in self.timingMarkers:
                self.children[0].canvas.remove(i)
            del self.timingMarkers[:]
            self.redrawPitch()

    def updateTrack(self) -> None:
        """updates the piano roll as required when the active track changes"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        removes = []
        for i in self.children[0].children:
            if i.__class__.__name__ == "Note":
                removes.append(i)
        for i in removes:
            self.children[0].remove_widget(i)
        if middleLayer.activeTrack == None:
            for i in self.timingMarkers:
                self.children[0].canvas.remove(i)
            del self.timingMarkers[:]
            if self.pitchLine != None:
                self.children[0].canvas.remove(self.pitchLine)
                self.pitchLine = None
            if self.basePitchLine != None:
                self.children[0].canvas.remove(self.basePitchLine)
                self.basePitchLine = None
            return
        self.generate_notes()
        if middleLayer.mode == "timing":
            with self.children[0].canvas:
                Color(1, 0, 0)
                for i in middleLayer.trackList[middleLayer.activeTrack].borders:
                    self.timingMarkers.append(ObjectProperty())
                    self.timingMarkers[-1] = Line(points = [self.xScale * i, 0, self.xScale * i, self.children[0].height])
        if middleLayer.mode == "pitch":
            self.redrawPitch()

    def applyScroll(self, scrollValue:float) -> None:
        """sets the x scroll value of the  widget"""

        self.scroll_x = scrollValue

    def triggerScroll(self) -> None:
        """signals the middleLayer that the x scroll value of the widget has changed. Used for synchronizing scrolling between the adaptive space and piano roll."""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.scrollValue = self.scroll_x
        middleLayer.applyScroll()

    def on_touch_down(self, touch) -> bool:
        """Callback function used for processing mouse input on the piano roll"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if touch.is_mouse_scrolling == False:
            if super(PianoRoll, self).on_touch_down(touch):
                return True
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                print("call")
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
                    self.updateTimingMarkers()
                    self.updateTrack()
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
                    for i in self.children[0].children:
                        if i.__class__.__name__ == "Note":
                            if i.xPos < xQuant:
                                index += 1
                            elif i.xPos > xQuant:
                                i.index += 1
                            else:
                                index += 1
                                xQuant += 1
                    newNote = Note(index = index, xPos = xQuant, yPos = y, length = 100, height = self.yScale)
                    middleLayer.addNote(index, x, y, newNote)
                    self.children[0].add_widget(newNote, index = 5)
                    touch.ud["noteIndex"] = index
                    touch.ud["grabMode"] = "end"
                    touch.ud["initialPos"] = coord
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
                    touch.ud["offset"] = self.quantize(x) - middleLayer.trackList[middleLayer.activeTrack].borders[border]
                elif middleLayer.mode == "pitch":
                    with self.children[0].canvas:
                        Color(0, 0, 1)
                        touch.ud['line'] = Line(points=self.quantize(*self.to_local(touch.x, touch.y)))
                        touch.ud['startPoint'] = self.quantize(*self.to_local(touch.x, touch.y))
                        touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height)]
                        touch.ud['lastPoint'] = touch.ud['startPoint'][0]
                        touch.ud['startPointOffset'] = 0
                touch.ud['param'] = False
                return True
        return False

    def on_touch_move(self, touch) -> bool:
        """Callback function used for processing mouse input on the piano roll"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if "param" not in touch.ud:
            return super().on_touch_move(touch)
        if self.collide_point(*touch.pos) == False or touch.ud['param']:
            return False
        if middleLayer.activeTrack == None:
            return
        coord = self.quantize(*self.to_local(touch.x, touch.y))
        x = int(coord[0] / self.xScale)
        y = int(coord[1] / self.yScale)
        yMod = coord[1]
        if middleLayer.mode == "notes":
            if "noteIndex" in touch.ud:
                note = middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference
                if abs(touch.ud["initialPos"][0] - coord[0]) < 4 and abs(touch.ud["initialPos"][1] - coord[1]) < 4:
                    return True
                if touch.ud["grabMode"] == "start":
                    length = max(note.xPos + note.length - x, 1)
                    note.length = length
                    note.xPos = x
                    switch = middleLayer.changeNoteLength(touch.ud["noteIndex"], x, length)
                    if switch == True:
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"] + 1].reference.index += 1
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference.index -= 1
                        touch.ud["noteIndex"] += 1
                    elif switch == False:
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"] - 1].reference.index -= 1
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference.index += 1
                        note.index -= 1
                        touch.ud["noteIndex"] -= 1
                    note.redraw()
                elif touch.ud["grabMode"] == "mid":
                    note.xPos = int(x + touch.ud["xOffset"] + 1)
                    note.yPos = int(y + touch.ud["yOffset"] + 1)
                    switch = middleLayer.moveNote(touch.ud["noteIndex"], int(x + touch.ud["xOffset"] + 1), int(y + touch.ud["yOffset"] + 1))
                    if switch == True:
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"] + 1].reference.index += 1
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference.index -= 1
                        touch.ud["noteIndex"] += 1
                    elif switch == False:
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"] - 1].reference.index -= 1
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference.index += 1
                        touch.ud["noteIndex"] -= 1
                    note.redraw()
                elif touch.ud["grabMode"] == "end":
                    length = max(x - note.xPos, 1)
                    note.length = length
                    middleLayer.changeNoteLength(touch.ud["noteIndex"], note.xPos, length)
                    note.redraw()
                return True
            else:
                return False
        elif middleLayer.mode == "timing":
            self.children[0].canvas.remove(self.timingMarkers[touch.ud["border"]])
            del self.timingMarkers[touch.ud["border"]]
            self.timingMarkers.insert(touch.ud["border"], ObjectProperty())
            newPos = x - touch.ud["offset"]
            with self.children[0].canvas:
                self.timingMarkers[touch.ud["border"]] = Line(points = [self.xScale * newPos, 0, self.xScale * newPos, self.children[0].height])
            middleLayer.changeBorder(touch.ud["border"], newPos)
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
        from UI.code.editor.Main import middleLayer
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
            middleLayer.applyPitchChanges(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'])
            self.children[0].canvas.remove(touch.ud['line'])
            self.redrawPitch()
        else:
            return super(PianoRoll, self).on_touch_up(touch)
