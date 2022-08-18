from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from torch import Tensor

#Node DType import hook

class Node(Widget):
    def __init__(self, inputs:dict, outputs:dict, func:object, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs = dict()
        self.outputs = dict()
        for i in inputs.keys():
            self.inputs[i] = Connector(False, self, inputs[i])
        self.inputs = inputs
        self.outputs = outputs
        self.func = func
        self.isUpdated = False

    def calculate(self) -> None:
        result = self.func(**self.inputs)
        for i in result.keys():
            self.outputs[i] = result[i]
        self.isUpdated = True


class Connector(Widget):
    def __init__(self, out:bool, node:Node, dtype:object, **kwargs) -> None:
        super().__init__(**kwargs)
        self.out = out
        self.dtype = dtype
        self._value = None
        self.attachedTo = ObjectProperty()
        self.node = ObjectProperty()
        self.node = node
    
    def __get__(self, instance, owner):
        if self.out:
            #if owner.isUpdated:
            if self.node.isUpdated:
                raise NodeLoopError(instance)
            owner.calculate()
            self.node.calculate()
            return self._value
        else:
            return self.attachedTo
    
    def __set__(self, instance, value):
        if self.out:
            try:
                value = self.dtype(value)
            except:
                raise NodeTypeMismatchError(instance)
            else:
                self._value = value
        else:
            raise NotImplementedError()

    def attach(self, target:object) -> None:
        if self.out == target.out:
            raise NodeAttachError()
        else:
            self.attachedTo = target


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


class ClampedFloat(float):
    def __init__(self) -> None:
        super().__init__()
        self.UIColor = (1., 0.1, 0.1)


class Float(float):
    def __init__(self) -> None:
        super().__init__()
        self.UIColor = (1., 0.7, 0.7)


class Int(int):
    def __init__(self) -> None:
        super().__init__()
        self.UIColor = (0.1, 1., 0.1)


class Bool(int):
    def __init__(self) -> None:
        super().__init__()
        self.UIColor = (0.5, 0.5, 0.5)


class ESPERAudio(Tensor):
    def __init__(self) -> None:
        super().__init__()
        self.UIColor = (1., 1., 1.)


class Phoneme():
    def __init__(self) -> None:
        self.UIColor = (0.1, 0.1, 1.)
