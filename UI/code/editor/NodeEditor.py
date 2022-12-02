from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Line
from kivy.properties import NumericProperty
from kivy.clock import mainthread
from math import pow, log, floor, ceil

class NodeEditor(ScrollView):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.scale = NumericProperty()
        self.scale = 1.
        self.strictWidth = NumericProperty()
        self.strictWidth = 0
        self.strictHeight = NumericProperty()
        self.strictHeight = 0
        self.scroll_x = 0.5
        self.scroll_y = 0.5
        self.generateGrid()

    @mainthread
    def generateGrid(self):
        self.updateGrid()

    def updateGrid(self):
        interval = self.scale * 100 / pow(5, ceil(log(self.scale, 5)))
        self.children[0].size = (max(self.strictWidth * 2, self.width * 2) * self.scale, max(self.strictHeight * 2, self.height * 2) * self.scale)
        self.children[0].children[-1].canvas.clear()
        with self.children[0].children[-1].canvas:
            Color(0.4, 0.4, 0.4, 1.)
            t = floor(self.scroll_x * (self.children[0].width - self.width) / interval) * interval
            c = 0
            while t <= self.scroll_x * (self.children[0].width - self.width) + self.width:
                points = [t, self.scroll_y * (self.children[0].height - self.height), t, self.scroll_y * (self.children[0].height - self.height) + self.height]
                Line(points = points)
                t += interval
                c += 1
            t = floor(self.scroll_y * (self.children[0].height - self.height) / interval) * interval
            c = 0
            while t <= self.scroll_y * (self.children[0].height - self.height) + self.height:
                points = [self.scroll_x * (self.children[0].width - self.width), t, self.scroll_x * (self.children[0].width - self.width) + self.width, t]
                Line(points = points)
                t += interval
                c += 1
        for i in self.children[0].children[:-1]:
            i.recalculateSize()
            
    def on_touch_down(self, touch):
        if touch.is_mouse_scrolling:
            position = self.to_local(*touch.pos)
            xPos = position[0]
            yPos = position[1]
            leftBorder = self.scroll_x * (self.children[0].width - self.width)
            rightBorder = self.scroll_x * (self.children[0].width - self.width) + self.width
            lowerBorder = self.scroll_y * (self.children[0].height - self.height)
            upperBorder = self.scroll_y * (self.children[0].height - self.height) + self.height
            if touch.button == 'scrolldown':
                self.scale *= 1.1
                xPos *= 1.1
                yPos *= 1.1
                #x *= 1.1
                #y *= 1.1
            elif touch.button == 'scrollup':
                self.scale /= 1.1
                xPos /= 1.1
                yPos /= 1.1
                #x /= 1.1
                #y /= 1.1
            
            self.updateGrid()
            return True
        for i in self.children[0].children[:-1]:
            if i.collide_point(*self.to_local(*touch.pos)):
                print("collide")
                i.on_touch_down(touch)
                return False
        return super(NodeEditor, self).on_touch_down(touch)
