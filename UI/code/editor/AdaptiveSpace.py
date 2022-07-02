from kivy.properties import ObjectProperty, NumericProperty, ListProperty
from kivy.graphics import Color, Line, Rectangle
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.scrollview import ScrollView

class AdaptiveSpace(AnchorLayout):
    def redraw(self):
        self.children[0].redraw()
    def applyScroll(self, scrollValue):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeTrack == None:
            return
        self.children[0].scroll_x = scrollValue
    def triggerScroll(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.scrollValue = self.children[0].scroll_x
        middleLayer.applyScroll()

class ParamCurve(ScrollView):
    xScale = NumericProperty(1)
    seqLength = NumericProperty(5000)
    line = ObjectProperty()
    line = Line()
    def redraw(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        if middleLayer.activeParam == "steadiness":
            data = middleLayer.trackList[middleLayer.activeTrack].steadiness
        elif middleLayer.activeParam == "breathiness":
            data = middleLayer.trackList[middleLayer.activeTrack].breathiness
        else:
            data = middleLayer.trackList[middleLayer.activeTrack].paramStack[middleLayer.activeParam].curve
        points = []
        c = 0
        for i in data:
            points.append(c * self.xScale)
            points.append((i.item() + 1) * self.height / 2)
            c += 1
        self.children[0].canvas.remove(self.line)
        with self.children[0].canvas:
            Color(1, 0, 0, 1)
            self.line = Line(points = points)

    def on_touch_down(self, touch):
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
    def on_touch_move(self, touch):
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
    def on_touch_up(self, touch):
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
            middleLayer.applyParamChanges(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'])
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(ParamCurve, self).on_touch_up(touch)

class TimingOptns(ScrollView):
    xScale = NumericProperty(1)
    seqLength = NumericProperty(5000)
    points1 = ListProperty()
    points2 = ListProperty()
    rectangles1 = ListProperty()
    rectangles2 = ListProperty()
    def redraw(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        for i in self.rectangles1:
            self.children[0].canvas.remove(i)
        del self.rectangles1[:]
        for i in self.rectangles2:
            self.children[0].canvas.remove(i)
        del self.rectangles2[:]
        data1 = middleLayer.trackList[middleLayer.activeTrack].loopOverlap
        data2 = middleLayer.trackList[middleLayer.activeTrack].loopOffset
        self.points1 = []
        self.points2 = []
        for i in range(data1.size()[0]):
            self.points1.append((self.parent.xScale * middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 1], data1[i].item() * self.height / 2))
            self.points2.append((self.parent.xScale * middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 1],  (data2[i].item() * self.height) / 2))
        with self.children[0].canvas:
            Color(1, 0, 0, 1)
            for i in self.points1:
                self.rectangles1.append(ObjectProperty())
                self.rectangles1[-1] = Rectangle(pos = (i[0], self.y + 0.5 * self.height), size = (10, i[1]))
            for i in self.points2:
                self.rectangles2.append(ObjectProperty())
                self.rectangles2[-1] = Rectangle(pos = (i[0], self.y), size = (10, i[1]))
    def on_touch_down(self, touch):
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
    def on_touch_move(self, touch):
        def getPreviousBar(self, x):
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
        def barToPos(self, bar, x):
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
                    middleLayer.applyParamChanges([y * 2 / self.height - 1], bar, section = touch.ud['section'])
                    self.points1[bar] = (self.points1[bar][0], y)
                    self.children[0].canvas.remove(self.rectangles1[bar])
                    del self.rectangles1[bar]
                    with self.children[0].canvas:
                        Color(1, 0, 0, 1)
                        self.rectangles1.insert(bar, ObjectProperty())
                        self.rectangles1[bar] = Rectangle(pos = (barToPos(self, bar, x), self.y + 0.5 * self.height), size = (10, self.points1[bar][1] - 0.5 * self.height))
                else:
                    middleLayer.applyParamChanges([y * 2 / self.height], bar, section = touch.ud['section'])
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
    def on_touch_up(self, touch):
        def getCurrentBar(self, x):
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
                middleLayer.applyParamChanges(finalData, firstBar, section = touch.ud['section'])
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(TimingOptns, self).on_touch_up(touch)
class PitchOptns(ScrollView):
    xScale = NumericProperty(1)
    seqLength = NumericProperty(5000)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.line1 = ObjectProperty()
        self.line2 = ObjectProperty()
        with self.children[0].canvas:
            self.line1 = Line()
            self.line2 = Line()
    def redraw(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        self.children[0].canvas.remove(self.line1)
        self.children[0].canvas.remove(self.line2)
        data1 = middleLayer.trackList[middleLayer.activeTrack].vibratoStrength
        data2 = middleLayer.trackList[middleLayer.activeTrack].vibratoSpeed
        points1 = []
        points2 = []
        c = 0
        for i in data1:
            points1.append((self.parent.xScale * c, int((1 + i) * self.height / 4)))
            c += 1
        c = 0
        for i in data2:
            points2.append((self.parent.xScale * c, int((3 + i) * self.height / 4)))
            c += 1
        with self.children[0].canvas:
            Color(1, 0, 0, 1)
            self.line1 = Line(points = points1)
            self.line2 = Line(points = points2)
    def on_touch_down(self, touch):
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
    def on_touch_move(self, touch):
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
    def on_touch_up(self, touch):
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
            middleLayer.applyParamChanges(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'], section = touch.ud['section'])
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(PitchOptns, self).on_touch_up(touch)