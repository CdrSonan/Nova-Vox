from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from torch import Tensor

#Node DType import hook

class Node(Widget):
    def __init__(self, inputs:dict, outputs:dict, func:object, timed = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs = dict()
        self.outputs = dict()
        for i in inputs.keys():
            self.inputs[i] = Connector(False, self, inputs[i])
        for i in outputs.keys():
            self.outputs[i] = Connector(True, self, outputs[i])
        self.func = func
        self.timed = timed
        self.static = not self.timed
        self.isUpdated = False
        self.isUpdating = False

    def calculate(self) -> None:
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


class Connector(Widget):
    def __init__(self, out:bool, node:Node, dtype:object, **kwargs) -> None:
        super().__init__(**kwargs)
        self.out = out
        self.dtype = dtype()
        self._value = self.dtype.defaultValue
        self.attachedTo = ObjectProperty()
        self.node = ObjectProperty()
        self.node = node
    
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
        if self.out:
            try:
                value = self.dtype.convert(value)
            except:
                raise NodeTypeMismatchError(self.node)
            else:
                self._value = value
        else:
            raise NotImplementedError()

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
