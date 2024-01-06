#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from math import floor, ceil
from kivy.properties import ObjectProperty, NumericProperty, ListProperty
from kivy.graphics import Color, Line, Rectangle
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.app import App
from kivy.metrics import dp

import API.Ops

class AdaptiveSpace(AnchorLayout):
    """Contains a ParamCurve, TimingOptions or PitchOptions widget depending on the current mode, and handles adressing and switching between them."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        root = App.get_running_app().root
        if self.collide_point(*self.to_widget(dp(pos[0]), dp(pos[1]))) and root.cursorPrio <= 1:
            Window.set_system_cursor("crosshair")
            root.cursorSource = self
            root.cursorPrio = 1
        elif root.cursorSource == self:
            Window.set_system_cursor("arrow")
            root.cursorSource = root
            root.cursorPrio = 0

    def on_children(self, instance, children) -> None:
        """synchronizes the scroll and zoom value of the widget with the piano roll when it is filled"""
        
        if len(children) == 0:
            return
        global middleLayer
        from UI.code.editor.Main import middleLayer
        self.applyScroll(middleLayer.scrollValue)
        self.applyZoom(middleLayer.xScale)

    def redraw(self) -> None:
        """redraws the currently widget currently displayed by the adaptive space. Used during several update procedures."""

        self.children[0].redraw()

    def applyScroll(self, scrollValue) -> None:
        """sets the scroll position of the widget currently displayed by the adaptive space. Used for keeping scrolling synchronized between adaptive space and piano roll."""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeTrack == None:
            return
        self.children[0].scroll_x = scrollValue

    def applyZoom(self, xScale) -> None:
        """sets the zoom level of the widget currently displayed by the adaptive space. Used for keeping scrolling synchronized between adaptive space and piano roll."""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeTrack == None:
            return
        self.children[0].xScale = xScale

    def applyLength(self, length) -> None:
        """adjusts the displayed information to account for a change of track length"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeTrack == None:
            return
        self.children[0].seqLength = length

    def triggerScroll(self) -> None:
        """sends the scroll value of the widget currently displayed by the adaptive space to the middle layer, for updating the piano roll scroll accordingly."""
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.scrollValue = self.children[0].scroll_x
        middleLayer.applyScroll()

    def triggerZoom(self) -> None:
        """sends the scroll value of the widget currently displayed by the adaptive space to the middle layer, for updating the piano roll scroll accordingly."""
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.xScale = self.children[0].xScale
        middleLayer.applyZoom()

    def updateFromBorder(self, border:int, pos:float) -> None:
        if self.children[0].__class__ == TimingOptns:
            self.children[0].updateFromBorder(border, pos)

class ParamCurve(ScrollView):
    """Widget displaying a single, editable curve used for controlling a tuning parameter"""

    xScale = NumericProperty(1)
    seqLength = NumericProperty(5000)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = Color(1, 0, 0, 1)
        with self.children[0].canvas:
            self.line = Line()

    def on_parent(self, instance, parent):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeTrack != None and self.parent:
            self.seqLength = middleLayer.trackList[middleLayer.activeTrack].length

    def redraw(self) -> None:
        """redraws the parameter curve, using data fetched from the middleLayer."""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeParam == "loop" or middleLayer.activeParam == "vibrato":
            return
        elif middleLayer.activeParam == "steadiness":
            data = middleLayer.trackList[middleLayer.activeTrack].steadiness
        elif middleLayer.activeParam == "breathiness":
            data = middleLayer.trackList[middleLayer.activeTrack].breathiness
        elif middleLayer.activeParam == "AI balance":
            data = middleLayer.trackList[middleLayer.activeTrack].aiBalance
        else:
            data = middleLayer.trackList[middleLayer.activeTrack].nodegraph.params[middleLayer.activeParam].curve
        start = floor(self.seqLength * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width * 0.9)
        end = min(ceil(self.seqLength * (self.scroll_x * (self.children[0].width - self.width) + self.width) / self.children[0].width / 0.9), self.seqLength)
        points = []
        for i in range(start, end):
            points.append(i * self.xScale)
            points.append((data[i].item() + 1) * self.height / 2)
        self.children[0].canvas.remove(self.color)
        self.children[0].canvas.remove(self.line)
        with self.children[0].canvas:
            self.color = Color(1, 0, 0, 1)
            self.line = Line(points = points)

    def on_xScale(self, instance, xScale:float) -> None:
        """updates the displayed curve when the x axis zoom level is changed"""

        self.redraw()

    def on_scroll_x(self, instance, xScale:float) -> None:
        """updates the displayed curves when the x axis scroll value is changed"""

        self.redraw()

    def on_seqLength(self, instance, length:float) -> None:
        """updates the displayed curve when the track length is changed"""

        self.redraw()

    def on_touch_down(self, touch) -> bool:
        """Callback function used for editing the curve"""

        if touch.is_mouse_scrolling == False:
            if super(ParamCurve, self).on_touch_down(touch):
                return True
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
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
            else:
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points=self.to_local(touch.x, touch.y))
                    touch.ud['startPoint'] = self.to_local(touch.x, touch.y)
                    touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height)]
                    touch.ud['lastPoint'] = touch.ud['startPoint'][0]
                    touch.ud['startPointOffset'] = 0
                    touch.ud['param'] = True
            return True
        else:
            return False

    def on_touch_move(self, touch) -> bool:
        """Callback function used for editing the curve"""

        if "param" not in touch.ud:
            return super(ParamCurve, self).on_touch_move(touch)
        if 'startPoint' in touch.ud and touch.ud['param']:
            global middleLayer
            from UI.code.editor.Main import middleLayer
            coord = self.to_local(touch.x, touch.y)
            x = int(coord[0] / self.xScale)
            y = min(max(coord[1], 0.), self.height)
            p = x - touch.ud['startPoint'][0]
            if middleLayer.tool == "draw":
                if p < 0:
                    for i in range(-p):
                        touch.ud['startPoint'][0] -= 1
                        touch.ud['line'].points = [touch.ud['startPoint'][0] * self.xScale, y] + touch.ud['line'].points
                    touch.ud['lastPoint'] = touch.ud['line'].points[0] / self.xScale
                elif p < int(len(touch.ud['line'].points) / 2):
                    points = touch.ud['line'].points
                    if x >= touch.ud['lastPoint']:
                        domain = range(int(touch.ud['lastPoint']) - int(touch.ud['startPoint'][0]), p + 1)
                    else:
                        domain = range(p, int(touch.ud['lastPoint']) - int(touch.ud['startPoint'][0]))
                    for i in domain:
                        points[2 * i + 1] = y
                    touch.ud['line'].points = points
                    touch.ud['lastPoint'] = points[2 * p] / self.xScale
                else:
                    diff = p - int(len(touch.ud['line'].points) / 2)
                    for i in range(diff):
                        touch.ud['line'].points += [(touch.ud['startPoint'][0] + int(len(touch.ud['line'].points) / 2)) * self.xScale, y]
                    touch.ud['lastPoint'] = touch.ud['line'].points[len(touch.ud['line'].points) - 2] / self.xScale
            elif middleLayer.tool == "line":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * -i / p]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * i / p]
            elif middleLayer.tool == "arch":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (-i / p) * (-i / p)]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (i / p) * (i / p)]
            elif middleLayer.tool == "reset":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
        else:
            return super(ParamCurve, self).on_touch_move(touch)

    def on_touch_up(self, touch) -> bool:
        """Callback function used for editing the curve"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if "param" not in touch.ud:
            return super(ParamCurve, self).on_touch_up(touch)
        if 'startPoint' in touch.ud:
            data = []
            if touch.ud['startPointOffset'] == 0:
                for i in range(int(len(touch.ud['line'].points) / 2)):
                    data.append((touch.ud['line'].points[2 * i + 1] * 2 / self.height) - 1)
            else:
                for i in range(int(len(touch.ud['line'].points) / 2)):
                    data.append((touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 2 / self.height) - 1)
            API.Ops.ChangeParam(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'])()
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(ParamCurve, self).on_touch_up(touch)

class TimingOptns(ScrollView):
    """Widget displaying the timing options of loop overlap and loop offset as bar diagrams"""

    xScale = NumericProperty(1)
    seqLength = NumericProperty(5000)
    points1 = ListProperty()
    points2 = ListProperty()
    rectangles1 = ListProperty()
    rectangles2 = ListProperty()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = Color(1, 0, 0, 1)

    def on_parent(self, instance, parent):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeTrack != None and self.parent:
            self.seqLength = middleLayer.trackList[middleLayer.activeTrack].length

    def redraw(self) -> None:
        """redraws the bar diagram, using data fetched from the middleLayer."""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        self.children[0].canvas.remove(self.color)
        for i in self.rectangles1:
            self.children[0].canvas.remove(i)
        del self.rectangles1[:]
        for i in self.rectangles2:
            self.children[0].canvas.remove(i)
        del self.rectangles2[:]
        self.points1 = []
        self.points2 = []
        for i, data in enumerate(zip(middleLayer.trackList[middleLayer.activeTrack].loopOverlap, middleLayer.trackList[middleLayer.activeTrack].loopOffset)):
            middle = self.xScale * (middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 2] + middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 3]) / 2
            if middle < floor(self.scroll_x * (self.children[0].width - self.width) * 0.9):
                continue
            elif middle > ceil((self.scroll_x * (self.children[0].width - self.width) + self.width) / 0.9):
                break
            self.points1.append((middle - 5, data[0] * self.height / 2))
            self.points2.append((middle - 5, data[1] * self.height / 2))
        with self.children[0].canvas:
            self.color = Color(1, 0, 0, 1)
            for i in self.points1:
                self.rectangles1.append(Rectangle(pos = (i[0], self.y + 0.5 * self.height), size = (10, i[1])))
            for i in self.points2:
                self.rectangles2.append(Rectangle(pos = (i[0], self.y), size = (10, i[1])))

    def updateFromBorder(self, border:int, pos:float) -> None:
        """when in timing mode, updates the bars for a single phoneme after a change of a border on the piano roll"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if border % 3 == 2 and border < len(middleLayer.trackList[middleLayer.activeTrack].borders) - 1:
            middle = (middleLayer.trackList[middleLayer.activeTrack].borders[border + 1] + pos) / 2
            index1 = self.children[0].canvas.indexof(self.rectangles1[int((border - 2) / 3)])
            index2 = self.children[0].canvas.indexof(self.rectangles2[int((border - 2) / 3)])
            self.children[0].canvas.remove(self.rectangles1[int((border - 2) / 3)])
            self.children[0].canvas.remove(self.rectangles2[int((border - 2) / 3)])
            self.rectangles1[int((border - 2) / 3)] = Rectangle(pos = (self.xScale * middle - 5, self.y + 0.5 * self.height), size = (10, middleLayer.trackList[middleLayer.activeTrack].loopOverlap[int((border - 2) / 3)] * self.height / 2))
            self.rectangles2[int((border - 2) / 3)] = Rectangle(pos = (self.xScale * middle - 5, self.y), size = (10, middleLayer.trackList[middleLayer.activeTrack].loopOverlap[int((border - 2) / 3)] * self.height / 2))
            self.children[0].canvas.insert(index1, self.rectangles1[int((border - 2) / 3)])
            self.children[0].canvas.insert(index2, self.rectangles2[int((border - 2) / 3)])
        elif border % 3 == 0 and border > 0:
            middle = (middleLayer.trackList[middleLayer.activeTrack].borders[border - 1] + pos) / 2
            index1 = self.children[0].canvas.indexof(self.rectangles1[int(border / 3) - 1])
            index2 = self.children[0].canvas.indexof(self.rectangles2[int(border / 3) - 1])
            self.children[0].canvas.remove(self.rectangles1[int(border / 3) - 1])
            self.children[0].canvas.remove(self.rectangles2[int(border / 3) - 1])
            self.rectangles1[int(border / 3) - 1] = Rectangle(pos = (self.xScale * middle - 5, self.y + 0.5 * self.height), size = (10, middleLayer.trackList[middleLayer.activeTrack].loopOverlap[int(border / 3) - 1] * self.height / 2))
            self.rectangles2[int(border / 3) - 1] = Rectangle(pos = (self.xScale * middle - 5, self.y), size = (10, middleLayer.trackList[middleLayer.activeTrack].loopOverlap[int(border / 3) - 1] * self.height / 2))
            self.children[0].canvas.insert(index1, self.rectangles1[int(border / 3) - 1])
            self.children[0].canvas.insert(index2, self.rectangles2[int(border / 3) - 1])

    def on_xScale(self, instance, xScale:float) -> None:
        """updates the displayed bar diagrams when the x axis zoom level is changed"""

        self.redraw()

    def on_scroll_x(self, instance, xScale:float) -> None:
        """updates the displayed curves when the x axis scroll value is changed"""

        self.redraw()

    def on_seqLength(self, instance, length:float) -> None:
        """updates the displayed bar diagrams when the track length is changed"""

        self.redraw()

    def on_touch_down(self, touch) -> bool:
        """Callback function used for editing the timing parameters"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if touch.is_mouse_scrolling == False:
            if super(TimingOptns, self).on_touch_down(touch):
                return True
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
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
            else:
                with self.children[0].canvas:
                    touch.ud['startPoint'] = self.to_local(touch.x, touch.y)
                    touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height)]
                    touch.ud['section'] = touch.ud['startPoint'][1] > self.height / 2
                    touch.ud['param'] = True
                    if middleLayer.tool != "draw":
                        Color(0, 0, 1)
                        touch.ud['line'] = Line(points=[touch.x, touch.y])
                        touch.ud['startPointOffset'] = 0
                        touch.ud['lastPoint'] = touch.ud['startPoint'][0]
            return True
        else:
            return False

    def on_touch_move(self, touch) -> bool:
        """Callback function used for editing the timing parameters"""

        def getPreviousBar(self, x:float) -> float:
            """utility function that returns the first bar to the left of a touch position"""

            if touch.ud['section']:
                for i in range(len(self.points1)):
                    j = len(self.points1) - i - 1
                    if x > self.points1[j][0]:
                        return j
            else:
                for i in range(len(self.points2)):
                    j = len(self.points2) - i - 1
                    if x > self.points2[j][0]:
                        return j
            return None

        def barToPos(self, bar:int, x:float) -> float:
            """utility function that returns the x position of a bar, or x if no bar index is passed"""

            if bar == None:
                return x
            if touch.ud['section']:
                return self.points1[bar][0]
            else:
                return self.points2[bar][0]

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if "param" not in touch.ud:
            return super(TimingOptns, self).on_touch_move(touch)
        if 'startPoint' in touch.ud:
            global middleLayer
            coord = self.to_local(touch.x, touch.y)
            x = int(coord[0] / self.xScale)
            if touch.ud['section']:
                y = min(max(coord[1], self.height / 2), self.height)
            else:
                y = min(max(coord[1], 0.), self.height / 2)
            p = x - touch.ud['startPoint'][0]
            if middleLayer.tool == "draw":
                bar = getPreviousBar(self, x)
                if bar == None:
                    return True
                if touch.ud["section"]:
                    API.Ops.ChangeParam([y * 2 / self.height - 1], bar, section = touch.ud['section'])()
                    self.points1[bar] = (self.points1[bar][0], y)
                    self.children[0].canvas.remove(self.rectangles1[bar])
                    del self.rectangles1[bar]
                    with self.children[0].canvas:
                        Color(1, 0, 0, 1)
                        self.rectangles1.insert(bar, ObjectProperty())
                        self.rectangles1[bar] = Rectangle(pos = (barToPos(self, bar, x), self.y + 0.5 * self.height), size = (10, self.points1[bar][1] - 0.5 * self.height))
                else:
                    API.Ops.ChangeParam([y * 2 / self.height], bar, section = touch.ud['section'])()
                    self.points2[bar] = (self.points2[bar][0], y)
                    self.children[0].canvas.remove(self.rectangles2[bar])
                    del self.rectangles2[bar]
                    with self.children[0].canvas:
                        Color(1, 0, 0, 1)
                        self.rectangles2.insert(bar, ObjectProperty)
                        self.rectangles2[bar] = Rectangle(pos = (barToPos(self, bar, x), self.y), size = (10, self.points2[bar][1]))
            elif middleLayer.tool == "line":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * -i / p]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * i / p]
            elif middleLayer.tool == "arch":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (-i / p) * (-i / p)]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (i / p) * (i / p)]
            elif middleLayer.tool == "reset":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
        else:
            return super(TimingOptns, self).on_touch_move(touch)

    def on_touch_up(self, touch) -> bool:
        """Callback function used for editing the timing parameters"""

        def getCurrentBar(self, x:int) -> int:
            """returns the index of the next bar to the left of a touch position, or None if no such bar exists"""

            if touch.ud['section']:
                for i in range(len(self.points1)):
                    j = len(self.points1) - i - 1
                    if x == self.points1[j][0]:
                        return j
                    if x > self.points1[j][0]:
                        break
            else:
                for i in range(len(self.points2)):
                    j = len(self.points2) - i - 1
                    if x == self.points2[j][0]:
                        return j
                    if x > self.points2[j][0]:
                        break
            return None

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if "param" not in touch.ud:
            return super(TimingOptns, self).on_touch_up(touch)
        if middleLayer.tool == "draw":
            pass
        elif 'startPoint' in touch.ud:
            data = []
            if touch.ud['section']:
                if touch.ud['startPointOffset'] == 0:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * i + 1] * 2 / self.height) - 1)
                else:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 2 / self.height) - 1)
            else:
                if touch.ud['startPointOffset'] == 0:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append(touch.ud['line'].points[2 * i + 1] * 2 / self.height)
                else:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append(touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 2 / self.height)
            finalData = []
            firstBar = None
            for i in range(len(data)):
                bar = getCurrentBar(self, touch.ud["startPoint"][0] - touch.ud["startPointOffset"] + i)
                if bar != None:
                    finalData.append(data[i])
                    if firstBar == None:
                        firstBar = bar
            if firstBar != None:
                API.Ops.ChangeParam(finalData, firstBar, section = touch.ud['section'])()
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(TimingOptns, self).on_touch_up(touch)

class PitchOptns(ScrollView):
    """Widget displaying two curves for controlling vibrato"""

    xScale = NumericProperty(1)
    seqLength = NumericProperty(5000)

    def on_parent(self, instance, parent):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeTrack != None and self.parent:
            self.seqLength = middleLayer.trackList[middleLayer.activeTrack].length

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        with self.children[0].canvas:
            self.line1 = Line()
            self.line2 = Line()
            self.color = Color(1, 0, 0, 1)

    def redraw(self) -> None:
        """redraws the curves using data fetched from the middleLayer"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        self.children[0].canvas.remove(self.color)
        self.children[0].canvas.remove(self.line1)
        self.children[0].canvas.remove(self.line2)
        start = floor(self.seqLength * self.scroll_x * (self.children[0].width - self.width) / self.children[0].width * 0.9)
        end = min(ceil(self.seqLength * (self.scroll_x * (self.children[0].width - self.width) + self.width) / self.children[0].width / 0.9), self.seqLength)
        data1 = middleLayer.trackList[middleLayer.activeTrack].vibratoStrength
        data2 = middleLayer.trackList[middleLayer.activeTrack].vibratoSpeed
        points1 = []
        points2 = []
        for i in range(start, end):
            points1.append((self.xScale * i, int((1 + data1[i]) * self.height / 4)))
            points2.append((self.xScale * i, int((3 + data2[i]) * self.height / 4)))
        with self.children[0].canvas:
            self.color = Color(1, 0, 0, 1)
            self.line1 = Line(points = points1)
            self.line2 = Line(points = points2)

    def on_xScale(self, instance, xScale:float) -> None:
        """updates the displayed curves when the x axis zoom level is changed"""

        self.redraw()

    def on_scroll_x(self, instance, xScale:float) -> None:
        """updates the displayed curves when the x axis scroll value is changed"""

        self.redraw()

    def on_seqLength(self, instance, length:float) -> None:
        """updates the displayed curves when the track length is changed"""

        self.redraw()

    def on_touch_down(self, touch) -> bool:
        """Callback function used for editing the curve"""

        if touch.is_mouse_scrolling == False:
            if super(PitchOptns, self).on_touch_down(touch):
                return True
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
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
            else:
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points=self.to_local(touch.x, touch.y))
                    touch.ud['startPoint'] = self.to_local(touch.x, touch.y)
                    touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height)]
                    touch.ud['lastPoint'] = touch.ud['startPoint'][0]
                    touch.ud['startPointOffset'] = 0
                    touch.ud['section'] = touch.ud['startPoint'][1] > self.height / 2
                    touch.ud['param'] = True
            return True
        else:
            return False

    def on_touch_move(self, touch) -> bool:
        """Callback function used for editing the curve"""

        if "param" not in touch.ud:
            return super(PitchOptns, self).on_touch_move(touch)
        if 'startPoint' in touch.ud and touch.ud['param']:
            global middleLayer
            from UI.code.editor.Main import middleLayer
            coord = self.to_local(touch.x, touch.y)
            x = int(coord[0] / self.xScale)
            if touch.ud['section']:
                y = min(max(coord[1], self.height / 2), self.height)
            else:
                y = min(max(coord[1], 0.), self.height / 2)
            p = x - touch.ud['startPoint'][0]
            if middleLayer.tool == "draw":
                if p < 0:
                    for i in range(-p):
                        touch.ud['startPoint'][0] -= 1
                        touch.ud['line'].points = [touch.ud['startPoint'][0] * self.xScale, y] + touch.ud['line'].points
                    touch.ud['lastPoint'] = touch.ud['line'].points[0] / self.xScale
                elif p < int(len(touch.ud['line'].points) / 2):
                    points = touch.ud['line'].points
                    if x >= touch.ud['lastPoint']:
                        domain = range(int(touch.ud['lastPoint']) - int(touch.ud['startPoint'][0]), p + 1)
                    else:
                        domain = range(p, int(touch.ud['lastPoint']) - int(touch.ud['startPoint'][0]))
                    for i in domain:
                        points[2 * i + 1] = y
                    touch.ud['line'].points = points
                    touch.ud['lastPoint'] = points[2 * p] / self.xScale
                else:
                    diff = p - int(len(touch.ud['line'].points) / 2)
                    for i in range(diff):
                        touch.ud['line'].points += [(touch.ud['startPoint'][0] + int(len(touch.ud['line'].points) / 2)) * self.xScale, y]
                    touch.ud['lastPoint'] = touch.ud['line'].points[len(touch.ud['line'].points) - 2] / self.xScale
            elif middleLayer.tool == "line":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * -i / p]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * i / p]
            elif middleLayer.tool == "arch":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (-i / p) * (-i / p)]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (i / p) * (i / p)]
            elif middleLayer.tool == "reset":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
        else:
            return super(PitchOptns, self).on_touch_move(touch)

    def on_touch_up(self, touch) -> bool:
        """Callback function used for editing the curve"""
        
        global middleLayer
        from UI.code.editor.Main import middleLayer
        if "param" not in touch.ud:
            return super(PitchOptns, self).on_touch_up(touch)
        if 'startPoint' in touch.ud and touch.ud['param']:
            data = []
            if touch.ud['section']:
                if touch.ud['startPointOffset'] == 0:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * i + 1] * 4 / self.height) - 3)
                else:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 4 / self.height) - 3)
            else:
                if touch.ud['startPointOffset'] == 0:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * i + 1] * 4 / self.height) - 1)
                else:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 4 / self.height) - 1)
            API.Ops.ChangeParam(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'], section = touch.ud['section'])()
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(PitchOptns, self).on_touch_up(touch)
