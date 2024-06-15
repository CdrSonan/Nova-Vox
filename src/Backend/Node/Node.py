#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.behaviors.drag import DragBehavior
from kivy.properties import ObjectProperty
from kivy.graphics import Color, Ellipse, RoundedRectangle, Bezier, InstructionGroup
from kivy.clock import Clock

from Backend.Node.NodeBase import NodeBase, ConnectorBase, NodeAttachError, NodeLoopError, NodeTypeMismatchError
from Backend.Node.Types import getType

class Node(DragBehavior, BoxLayout):
    """base class of an audio processing node. All node classes should inherit from this class."""

    def __init__(self, base:NodeBase, **kwargs) -> None:
        """
        constructs an arbitrary node using the provided parameters. Designed to be called from the __init__ method of a child class using super().__init__(), after the required arguments have been set up.
        Arguments:
            inputs: dictionary using strings as keys, and node datatype classes as values. Each key-value-pair corresponds to the name and data type of an input port the node will have.

            outputs: dictionary using strings as keys, and node datatype classes as values. Each key-value-pair corresponds to the name and data type of an input port the node will have.

            func: callable performing the function of the node. Its arguments match the keys in the input dictionary, and it must return a dictionary with the same keys as the output dictionary,
            and values that can be converted to the respective node data type.

            timed: boolean flag indicating whether the node output is time-dependent even with constant input. Setting this flag will cause the node to be evaluated in every frame, instead of only when one of its inputs changes.

            **kwargs: keyword arguments passed to the Kivy constructors
        """


        super().__init__(**kwargs)
        self.base = base
        self.rectangle = ObjectProperty()
        with self.canvas:
            Color(0.322, 0.259, 0.463, 1.)
            self.rectangle = RoundedRectangle(pos = self.pos, size = self.size)
        self.inputs = dict()
        self.outputs = dict()
        self.add_widget(Label(text = self.name()[-1]))
        for i in self.base.inputs.keys():
            conn = Connector(self.base.inputs[i], self, self.base.inputs[i]._type)
            self.add_widget(conn)
            self.inputs[i] = conn
        for i in self.base.outputs.keys():
            conn = Connector(self.base.outputs[i], self, self.base.outputs[i]._type)
            self.add_widget(conn)
            self.outputs[i] = conn

    def recalculateSize(self):
        """recalculates the visual size of the node based on the current zoom level of the node editor"""

        self.size = (250 * self.parent.parent.scale, (len(self.inputs) + len(self.outputs) + 1) * 30 * self.parent.parent.scale)
        self.recalculateRectangle()
        self.recalculateConnections()

    def recalculateRectangle(self):
        """recalculates the position and size of the rectangle used for dragging the widget when the node is moved or zoom is used"""

        self.drag_rectangle = (*self.pos, *self.size)
        with self.canvas:
            self.rectangle.pos = self.pos
            self.rectangle.size = self.size
    
    def recalculateConnections(self):
        """recalculates the positions of all connections to the node when it is moved or zoom is used"""

        for i in self.outputs.keys():
            self.outputs[i].update()

    def on_parent(self, instance, parent):
        """call of recalculateSize() during initial widget creation"""

        if parent is not None:
            self.recalculateSize()

    def on_touch_down(self, touch):
        """processes initial Kivy touch input, and triggers scrolling or dragging accordingly"""
        
        def processConnector(conn:Connector, touch):
            if conn.ellipse.pos[0] - 5 < x < conn.ellipse.pos[0] + 15 and conn.ellipse.pos[1] - 5 < y < conn.ellipse.pos[1] + 15:
                if conn.base.attachedTo == None or conn.base.out:
                    touch.ud["draggedConnFrom"] = conn
                    conn.drawCurve(touch)
                else:
                    touch.ud["draggedConnFrom"] = conn.attachedTo
                    conn.detach()
                    touch.ud["draggedConnFrom"].drawCurve(touch)
                return True
            return False

        xx, yy, w, h = self.drag_rectangle
        x, y = self.parent.parent.to_local(*touch.pos)
        
        # check whether the touch collides with the ellipse of one of the connectors
        for i in self.inputs.values():
            if processConnector(i, touch):
                return True
        for i in self.outputs.values():
            if processConnector(i, touch):
                return True
        
        if not self.collide_point(x, y):
            touch.ud[self._get_uid('svavoid')] = True
            return super(DragBehavior, self).on_touch_down(touch)
        if self._drag_touch or ('button' in touch.profile and
                                touch.button.startswith('scroll')) or\
                not ((xx < x <= xx + w) and (yy < y <= yy + h)):
            return super(DragBehavior, self).on_touch_down(touch)

        # no mouse scrolling or ellipse hit, so the user is going to drag with this touch.
        self._drag_touch = touch
        uid = self._get_uid()
        touch.grab(self)
        touch.ud[uid] = {
            'mode': 'unknown',
            'dx': 0,
            'dy': 0}
        Clock.schedule_once(self._change_touch_mode,
                            self.drag_timeout / 1000.)
        return True

    def on_touch_move(self, touch):
        """updates the widget when it is being dragged"""

        self.recalculateRectangle()
        self.recalculateConnections()
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """updates the widget after a dragging operation for it has been completed"""

        self.recalculateRectangle()
        self.recalculateConnections()
        return super().on_touch_up(touch)

    @staticmethod
    def name() -> list:
        """returns the name of the node and its position in the available node browser, akin to a file path and name.
        Designed to be overwritten by classes inheriting from this class."""

        return ["",]

    def calculate(self) -> None:
        """evaluates the node, and recursively prompts evaluation of all nodes connected to its inputs, if required"""

        self.base.calculate()

    def checkStatic(self):
        """checks whether the node needs to be evaluated every frame, or can be considered static, returning a constant value"""

        self.base.checkStatic()

    def reset(self):
        """resets time-dependent components of the node when evaluation jumps to a different point of the track. Designed to be overloaded by inheriting classes."""

        pass

class Connector(BoxLayout):
    """Connector used for processing in- or output for a node. Always instantiated as child of a node."""

    def __init__(self, base:ConnectorBase, node:Node, dtype:str, **kwargs) -> None:
        """
        Constructor function.
        
        Arguments:
        out: Boolean flag indicating whether the connector is an output (True) or an input (False)

        node: reference to the node the connector is a part of

        dtype: node datatype class of the data type of the connector

        name: display name and key of the connector
        """


        super().__init__(**kwargs)
        self.dtype = getType(dtype)()
        self.base = base
        self.base._value = self.dtype.defaultValue
        self.multiline = False
        self.node = node
        self.orientation = "horizontal"
        self.attachedTo = [] if self.base.out else None
        self.add_widget(Label(text = self.base.name))
        if not self.base.out and self.dtype.hasWidget:
            self.dtype.make_widget(self, self.widget_setter)
        with self.canvas:
            self.curve = InstructionGroup()
            print(self.dtype.UIColor)
            Color(self.dtype.UIColor)
            self.ellipse = Ellipse(segments = 16, pos = (self.x, self.y + self.height / 2), size = (10, 10))

    def update(self):
        """updates the visual position of the connector and curve, if attached."""

        if self.base.out:
            self.ellipse.pos = (self.x + self.width + 5, self.y + self.height / 2 - 5)
            for i in self.attachedTo:
                i.drawCurveAttached()
        else:
            self.ellipse.pos = (self.x - 15, self.y + self.height / 2 - 5)
            if self.base.attachedTo is not None:
                self.drawCurveAttached()
        
    def on_parent(self, instance, parent):
        """calls update() when the connector is initially created"""

        self.update()

    def on_pos(self, instance, pos):
        """calls update() when the position of the connector changes"""

        self.update()

    def on_size(self, instance, size):
        """calls update() when the size of the connector changes"""

        self.update()

    def widget_setter(self, instance, value):
        """applies input changes when the widget is defocused, which indicates the user wants to confirm a text input"""

        self.set(value)
    
    def get(self):
        """getter function for the value of a connector. If the connector is an input, it gets its value through the connection.
        If it is an output, it instead causes its own node to be evaluated if necessary, and catches loops in the node graph."""

        return self.base.get()
    
    def set(self, value):
        """sets the value of the connector, both internally and visually, using the conversion function of its node data type class, and performs error handling."""

        self.base.set(value)
        self.children[0].text = repr(self.base._value)

    def attach(self, target:object) -> None:
        """attaches two connectors to each other, and performs the nexessary checks and callbacks."""

        if self.base.out == target.base.out:
            raise NodeAttachError()
        if self.base.out:
            self.base.attachedTo.append(target.base)
            target.base.attachedTo = self.base
            self.attachedTo.append(target)
            target.attachedTo = self
            self.node.checkStatic()
            target.node.checkStatic()
            self.curve.clear()
            if target.dtype.hasWidget:
                toRemove = target.children[0]
                target.remove_widget(toRemove)
                del toRemove
            target.drawCurveAttached()
        else:
            self.base.attachedTo = target.base
            target.base.attachedTo.append(self.base)
            self.attachedTo = target
            target.attachedTo.append(self)
            target.node.checkStatic()
            self.node.checkStatic()
            self.is_focusable = False
            if self.dtype.hasWidget:
                toRemove = self.children[0]
                self.remove_widget(toRemove)
                del toRemove
            self.drawCurveAttached()
        
    def detach(self) -> None:
        """detaches two connected connectors, and performs the nexessary checks and callbacks."""

        if self.base.out:
            return
        tmpNode = self.base.attachedTo.node
        self.base.attachedTo.attachedTo.remove(self.base)
        self.base.attachedTo = None
        tmpNode = self.attachedTo.node
        self.attachedTo.attachedTo.remove(self)
        self.attachedTo = None
        self.curve.clear()
        tmpNode.checkStatic()
        self.node.checkStatic()
        self.is_focusable = True
        if self.dtype.hasWidget:
            self.dtype.make_widget(self, self.widget_setter)

    def drawCurve(self, touch):
        """updates the visual of the curve attached to the connector during dragging"""

        if "draggedConnFrom" in touch.ud:
            x, y = self.parent.parent.parent.to_local(*touch.pos)
            self.curve.clear()
            self.curve.add(Color(1., 1., 1.))
            if self.base.out:
                self.curve.add(Bezier(points = [self.ellipse.pos[0] + 5, self.ellipse.pos[1] + 5,
                                                self.ellipse.pos[0] + 105, self.ellipse.pos[1] + 5,
                                                x, y]))
            else:
                self.curve.add(Bezier(points = [self.ellipse.pos[0] + 5, self.ellipse.pos[1] + 5,
                                                self.ellipse.pos[0] - 95, self.ellipse.pos[1] + 5,
                                                x, y]))
            return True
        return False
    
    def drawCurveAttached(self):
        """updates the visual of the curve of the connector when it is attached"""
        
        if self.base.attachedTo is None:
            return
        if isinstance(self.base.attachedTo, list) and len(self.base.attachedTo) == 0:
            return
        self.curve.clear()
        self.curve.add(Color(1., 1., 1.))
        self.curve.add(Bezier(points = [self.ellipse.pos[0] + 5, self.ellipse.pos[1] + 5,
                                        self.ellipse.pos[0] - 95, self.ellipse.pos[1] + 5,
                                        self.attachedTo.ellipse.pos[0] + 105, self.attachedTo.ellipse.pos[1] + 5,
                                        self.attachedTo.ellipse.pos[0] + 5, self.attachedTo.ellipse.pos[1] + 5]))
