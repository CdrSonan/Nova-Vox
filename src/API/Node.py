#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.behaviors.drag import DragBehavior
from kivy.properties import ObjectProperty
from kivy.graphics import Color, Ellipse, RoundedRectangle, Bezier, InstructionGroup
from kivy.clock import Clock
from torch import Tensor

class Node(DragBehavior, BoxLayout):
    """base class of an audio processing node. All node classes should inherit from this class."""

    def __init__(self, inputs:dict, outputs:dict, func:object, timed = False, **kwargs) -> None:
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
        self.rectangle = ObjectProperty()
        with self.canvas:
            Color(0.322, 0.259, 0.463, 1.)
            self.rectangle = RoundedRectangle(pos = self.pos, size = self.size)
        self.inputs = dict()
        self.outputs = dict()
        self.add_widget(Label(text = self.name()[-1]))
        for i in inputs.keys():
            conn = Connector(False, self, inputs[i], i)
            self.add_widget(conn)
            self.inputs[i] = ObjectProperty()
            self.inputs[i] = self.children[0]
        for i in outputs.keys():
            conn = Connector(True, self, outputs[i], i)
            self.add_widget(conn)
            self.outputs[i] = ObjectProperty()
            self.outputs[i] = self.children[0]
        self.func = func
        self.timed = timed
        self.static = not self.timed
        self.isUpdated = False
        self.isUpdating = False
        self.isPacked = True

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
            self.outputs[i].updateCurve()

    def on_parent(self, instance, parent):
        """call of recalculateSize() during initial widget creation"""

        self.recalculateSize()

    def on_touch_down(self, touch):
        """processes initial Kivy touch input, and triggers scrolling or dragging accordingly"""

        xx, yy, w, h = self.drag_rectangle
        x, y = self.parent.parent.to_local(*touch.pos)
        if not self.collide_point(x, y):
            touch.ud[self._get_uid('svavoid')] = True
            return super(DragBehavior, self).on_touch_down(touch)
        if self._drag_touch or ('button' in touch.profile and
                                touch.button.startswith('scroll')) or\
                not ((xx < x <= xx + w) and (yy < y <= yy + h)):
            return super(DragBehavior, self).on_touch_down(touch)

        # no mouse scrolling, so the user is going to drag with this touch.
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

        if self.isPacked:
            self.unpack()
            self.isPacked = False
        inputs = dict()
        for i in self.inputs.keys():
            inputs[i] = self.inputs[i].get()
        result = self.func(**inputs)
        for i in result.keys():
            self.outputs[i].set(result[i])
        self.isUpdated = True

    def checkStatic(self):
        """checks whether the node needs to be evaluated every frame, or can be considered static, returning a constant value"""

        #TODO: add recursion
        self.static = not self.timed
        for i in self.inputs.keys():
            if self.inputs[i].attachedTo != None:
                if self.inputs[i].attachedTo.node.static == False:
                    self.static = False
                    break
        if self.static:
            self.calculate()

    def reset(self):
        """resets time-dependent components of the node when evaluation jumps to a different point of the track. Designed to be overloaded by inheriting classes."""

        pass

    def pack(self):
        """packs the node, lowering its memory footprint and simplifying its transfer between processes. Designed to be overloaded by inheriting classes."""

        pass

    def unpack(self):
        """unpacks the node, restoring its full functionality. Designed to be overloaded by inheriting classes."""

        pass


class Connector(BoxLayout):
    """Connector used for processing in- or output for a node. Always instantiated as child of a node."""

    def __init__(self, out:bool, node:Node, dtype:object, name:str, **kwargs) -> None:
        """
        Constructor function.
        
        Arguments:
        out: Boolean flag indicating whether the connector is an output (True) or an input (False)

        node: reference to the node the connector is a part of

        dtype: node datatype class of the data type of the connector

        name: display name and key of the connector
        """


        super().__init__(**kwargs)
        self.multiline = False
        self.out = out
        self.dtype = dtype()
        self.name = name
        self._value = self.dtype.defaultValue
        self.attachedTo = None
        self.node = node
        self.orientation = "horizontal"
        self.add_widget(Label(text = self.name))
        self.add_widget(TextInput(multiline = False, is_focusable = not self.out))
        self.children[0].bind(focus = self.on_focus)
        with self.canvas:
            self.curve = InstructionGroup()
            Color(self.dtype.UIColor)
            self.ellipse = Ellipse(segments = 16, pos = (self.x, self.y + self.height / 2), size = (10, 10))

    def update(self):
        """updates the visual position of the connector"""

        if self.out:
            self.ellipse.pos = (self.x + self.width + 5, self.y + self.height / 2 - 5)
        else:
            self.ellipse.pos = (self.x - 15, self.y + self.height / 2 - 5)
    
    def updateCurve(self):
        """updates the visual position of the connector, and recursively prompts the update of all connectors connected to it"""

        self.update()
        if self.attachedTo is None:
            return
        with self.canvas:
            self.curve = InstructionGroup()
            self.curve.add(Color(1., 1., 1.))
            self.curve.add(Bezier(points = [self.ellipse.pos[0] + 5, self.ellipse.pos[1] + 5, self.attachedTo.ellipse.pos[0] + 5, self.attachedTo.ellipse.pos[1] + 5],
                                  control = [self.ellipse.pos[0] + 5, self.ellipse.pos[1] + 5, self.attachedTo.ellipse.pos[0] + 5, self.ellipse.pos[1] + 5]))
        
    def on_parent(self, instance, parent):
        """calls update() when the connector is initially created"""

        self.update()

    def on_pos(self, instance, pos):
        """calls update() when the position of the connector changes"""

        self.update()

    def on_size(self, instance, size):
        """calls update() when the size of the connector changes"""

        self.update()

    def on_focus(self, instance, focus):
        """applies input changes when the widget is defocused, which indicates the user wants to confirm a text input"""

        if not focus:
            self.set(self.children[0].text)
    
    def get(self):
        """getter function for the value of a connector. If the connector is an input, it gets its value through the connection.
        If it is an output, it instead causes its own node to be evaluated if necessary, and catches loops in the node graph."""

        if self.out:
            if self.node.isUpdating:
                raise NodeLoopError(self.node)
            if self.node.isUpdated == False and self.node.static == False:
                self.isUpdating = True
                self.node.calculate()
                self.isUpdating = False
            return self._value
        else:
            return self.attachedTo.get()
    
    def set(self, value):
        """sets the value of the connector, both internally and visually, using the conversion function of its node data type class, and performs error handling."""

        try:
            value = self.dtype.convert(value)
        except:
            raise NodeTypeMismatchError(self.node)
        else:
            self._value = value
            self.children[0].text = repr(self._value)

    def attach(self, target:object) -> None:
        """attaches two connectors to each other, and performs the nexessary checks and callbacks."""

        if self.out == target.out:
            raise NodeAttachError()
        else:
            self.attachedTo = target
            target.attachedTo = self
        if self.out:
            self.node.checkStatic()
            target.node.checkStatic()
        else:
            target.node.checkStatic()
            self.node.checkStatic()
            self.is_focusable = False
        
    def detach(self) -> None:
        """detaches two connected connectors, and performs the nexessary checks and callbacks."""

        tmpNode = self.attachedTo.node
        self.attachedTo.attachedTo = None
        self.attachedTo = None
        if self.out:
            self.node.checkStatic()
            tmpNode.checkStatic()
        else:
            tmpNode.checkStatic()
            self.node.checkStatic()
            self.is_focusable = True


class NodeTypeMismatchError(Exception):
    def __init__(self, connection, *args: object) -> None:
        super().__init__(*args)
        self.connection = connection


class NodeLoopError(Exception):
    def __init__(self, connection, *args: object) -> None:
        super().__init__(*args)
        self.connection = connection


class NodeAttachError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ClampedFloat():
    """node data type class for a Float confined within the [-1, 1] interval"""

    def __init__(self) -> None:
        self.UIColor = (1., 0.1, 0.1)
        self.defaultValue = 0.5

    @staticmethod
    def convert(*args):
        result = float(*args)
        result = max(-1., result)
        result = min(1., result)
        return result


class Float():
    """node data type class for a standard Float"""

    def __init__(self) -> None:
        self.UIColor = (1., 0.7, 0.7)
        self.defaultValue = 0.5

    @staticmethod
    def convert(*args):
        return float(*args)


class Int():
    """node data type class for a standard integer"""

    def __init__(self) -> None:
        self.UIColor = (0.1, 1., 0.1)
        self.defaultValue = 1

    @staticmethod
    def convert(*args):
        return int(*args)


class Bool():
    """node data type class for a standard boolean"""

    def __init__(self) -> None:
        self.UIColor = (0.5, 0.5, 0.5)
        self.defaultValue = False

    @staticmethod
    def convert(*args):
        return bool(*args)


class ESPERAudio():
    """node data type class for a PyTorch tensor representing a "Specharm", a point in an audio signal encoded using ESPER."""

    #TODO: finish this class
    def __init__(self) -> None:
        self.UIColor = (1., 1., 1.)
        self.defaultValue = 0.5

    @staticmethod
    def convert(*args):
        return Tensor(*args)


class Phoneme():
    """node data type class for a "phoneme state" of a track. Consists of one or two phonemes, and a value between 0 and 1 representing their relative strength in the case of two phonemes"""

    #TODO: finish this class
    def __init__(self) -> None:
        self.UIColor = (0.1, 0.1, 1.)
        self.defaultValue = 0.5

    @staticmethod
    def convert(*args):
        return None

#basic unit test
def testfunc(**kwargs) -> dict:
    return {"testoutput" : 2.}
def testfunc2(testinput, **kwargs) -> dict:
    print(testinput)
    return dict()

nodeA = Node({"testinput":Float}, dict(), testfunc2)
nodeB = Node(dict(), {"testoutput":Float}, testfunc)
nodeA.inputs["testinput"].attach(nodeB.outputs["testoutput"])
print(nodeA.inputs["testinput"].get())
