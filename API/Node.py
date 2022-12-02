from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.behaviors.drag import DragBehavior
from kivy.uix.image import Image
from kivy.properties import ObjectProperty
from kivy.graphics import Color, Ellipse, RoundedRectangle
from kivy.clock import Clock
from torch import Tensor

#Node DType import hook

class Node(DragBehavior, BoxLayout):
    def __init__(self, inputs:dict, outputs:dict, func:object, timed = False, **kwargs) -> None:
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
        self.size = (250 * self.parent.parent.scale, (len(self.inputs) + len(self.outputs) + 1) * 30 * self.parent.parent.scale)
        self.recalculateRectangle()

    def recalculateRectangle(self):
        self.drag_rectangle = (*self.pos, *self.size)
        with self.canvas:
            self.rectangle.pos = self.pos
            self.rectangle.size = self.size

    def on_parent(self, instance, parent):
        self.recalculateSize()

    def on_touch_down(self, touch):
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
        self.recalculateRectangle()
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        self.recalculateRectangle()
        return super().on_touch_up(touch)

    @staticmethod
    def name() -> list:
        return ["",]

    def calculate(self) -> None:
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
        self.static = not self.timed
        for i in self.inputs.keys():
            if self.inputs[i].attachedTo != None:
                if self.inputs[i].attachedTo.node.static == False:
                    self.static = False
                    break
        if self.static:
            self.calculate()

    def reset(self):
        pass

    def pack(self):
        pass

    def unpack(self):
        pass


class Connector(BoxLayout):
    def __init__(self, out:bool, node:Node, dtype:object, name:str, hasinput:bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.multiline = False
        self.out = out
        self.dtype = dtype()
        self.name = name
        self._value = self.dtype.defaultValue
        self.attachedTo = ObjectProperty()
        self.node = ObjectProperty()
        self.node = node
        self.orientation = "horizontal"
        self.add_widget(Label(text = self.name))
        self.add_widget(TextInput(multiline = False, is_focusable = not self.out))
        self.children[0].bind(focus = self.on_focus)
        self.ellipse = ObjectProperty()
        with self.canvas:
            Color(self.dtype.UIColor)
            self.ellipse = Ellipse(segments = 16, pos = (self.x, self.y + self.height / 2), size = (10, 10))

    def update(self):
        if self.out:
            self.ellipse.pos = (self.x + self.width + 5, self.y + self.height / 2 - 5)
        else:
            self.ellipse.pos = (self.x - 15, self.y + self.height / 2 - 5)
        
    def on_parent(self, instance, parent):
        self.update()

    def on_pos(self, instance, pos):
        self.update()

    def on_size(self, instance, size):
        self.update()

    def on_focus(self, instance, focus):
        #self.update()
        if not focus:
            self.set(self.children[0].text)
    
    def get(self):
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
        try:
            value = self.dtype.convert(value)
        except:
            raise NodeTypeMismatchError(self.node)
        else:
            self._value = value
            self.children[0].text = repr(self._value)

    def attach(self, target:object) -> None:
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
    def __init__(self) -> None:
        self.UIColor = (1., 0.7, 0.7)
        self.defaultValue = 0.5

    @staticmethod
    def convert(*args):
        return float(*args)


class Int():
    def __init__(self) -> None:
        self.UIColor = (0.1, 1., 0.1)
        self.defaultValue = 1

    @staticmethod
    def convert(*args):
        return int(*args)


class Bool():
    def __init__(self) -> None:
        self.UIColor = (0.5, 0.5, 0.5)
        self.defaultValue = False

    @staticmethod
    def convert(*args):
        return bool(*args)


class ESPERAudio():
    def __init__(self) -> None:
        self.UIColor = (1., 1., 1.)
        self.defaultValue = 0.5

    @staticmethod
    def convert(*args):
        return Tensor(*args)


class Phoneme():
    def __init__(self) -> None:
        self.UIColor = (0.1, 0.1, 1.)
        self.defaultValue = 0.5

    @staticmethod
    def convert(*args):
        return None

def testfunc(**kwargs) -> dict:
    return {"testoutput" : 2.}
def testfunc2(testinput, **kwargs) -> dict:
    print(testinput)
    return dict()

nodeA = Node({"testinput":Float}, dict(), testfunc2)
nodeB = Node(dict(), {"testoutput":Float}, testfunc)
nodeA.inputs["testinput"].attach(nodeB.outputs["testoutput"])
print(nodeA.inputs["testinput"].get())
