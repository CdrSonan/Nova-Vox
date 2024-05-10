#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Line
from kivy.properties import NumericProperty
from kivy.clock import mainthread
from kivy.metrics import dp
from math import pow, log, floor, ceil

class NodeEditor(ScrollView):
    """class of the audio processing node editor"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.scale = NumericProperty()
        self.scale = 2.
        self.scroll_x = 0.5
        self.scroll_y = 0.5
        self.generateGrid()

    @mainthread
    def generateGrid(self):
        """delayed call od updateGrid(). Required during widget initialization, when not all properties required by updateGrid() are available yet."""

        self.updateGrid()
    
    def updateNodes(self, scaleFactor = 1.):
        """updates the position of all nodes in the node editor to reflect changed scroll values or zoom levels"""

        for i in self.children[0].children[:-1]:
            i.pos = (i.pos[0] * scaleFactor, i.pos[1] * scaleFactor)
            i.recalculateRectangle()
            i.recalculateConnections()

    def updateGrid(self):
        """updates the background grid of the node editor to reflect changed scroll values or zoom levels"""

        interval = self.scale * 100 / pow(5, ceil(log(self.scale, 5)))
        self.children[0].size = (self.width * self.scale, self.height * self.scale)
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
        """callback function used for processing mouse input"""

        if touch.is_mouse_scrolling:
            oldScale = self.scale
            if touch.button == 'scrolldown':
                self.scale *= 1.1
                self.scale = min(self.scale, 10.)
            elif touch.button == 'scrollup':
                self.scale /= 1.1
                self.scale = max(self.scale, 1.)
            self.updateNodes(self.scale / oldScale)
            self.updateGrid()
            return True
        for i in self.children[0].children[:-1]:
            if i.collide_point(*self.to_local(*touch.pos)):
                i.on_touch_down(touch)
                return False
        return super(NodeEditor, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if "draggedConnFrom" in touch.ud:
            touch.ud["draggedConnFrom"].drawCurve(touch)
            return True
        return super(NodeEditor, self).on_touch_move(touch)
    
    def on_touch_up(self, touch):
        
        def processConnector(conn, touch):
            x, y = self.to_local(*touch.pos)
            print(conn.ellipse.pos, x, y)
            if conn.ellipse.pos[0] - 5 < x < conn.ellipse.pos[0] + 15 and conn.ellipse.pos[1] - 5 < y < conn.ellipse.pos[1] + 15:
                touch.ud["draggedConnFrom"].attach(conn)
                return True
            return False
        
        if "draggedConnFrom" in touch.ud:
            if touch.ud["draggedConnFrom"].out:
                for node in self.children[0].children[:-1]:
                    for conn in node.inputs.values():
                        if processConnector(conn, touch):
                            return True
            else:
                for node in self.children[0].children[:-1]:
                    for conn in node.outputs.values():
                        if processConnector(conn, touch):
                            return True
            touch.ud["draggedConnFrom"].curve.clear()
        return super(NodeEditor, self).on_touch_up(touch)
