#Copyright 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from Backend.Node.TypeConverters import getConverter

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


class NodeBase():
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


        self.inputs = dict()
        self.outputs = dict()
        for i in inputs.keys():
            self.inputs[i] = ConnectorBase(False, self, inputs[i], None, i)
        for i in outputs.keys():
            self.outputs[i] = ConnectorBase(True, self, outputs[i], None, i)
        self.func = func
        self.auxData = dict()
        self.timed = timed
        self.static = not self.timed
        self.isUpdated = False
        self.isUpdating = False

    @staticmethod
    def name() -> list:
        """returns the name of the node and its position in the available node browser, akin to a file path and name.
        Designed to be overwritten by classes inheriting from this class."""

        return ["",]

    def calculate(self) -> None:
        """evaluates the node, and recursively prompts evaluation of all nodes connected to its inputs, if required"""

        inputs = dict()
        for i in self.inputs.keys():
            inputs[i] = self.inputs[i].get()
        result = self.func(self, **inputs)
        for i in result.keys():
            self.outputs[i].set(result[i])
        self.isUpdated = True

    def checkStatic(self):
        """checks whether the node needs to be evaluated every frame, or can be considered static, returning a constant value"""

        if self.timed:
            self.static = False
            return
        self.static = True
        for i in self.inputs.keys():
            if self.inputs[i].attachedTo != None:
                self.inputs[i].attachedTo.node.checkStatic()
                if self.inputs[i].attachedTo.node.static == False:
                    self.static = False
                    break
        if self.static:
            self.calculate()

    def reset(self):
        """resets time-dependent components of the node when evaluation jumps to a different point of the track. Designed to be overloaded by inheriting classes."""

        pass

class ConnectorBase():
    """Connector used for processing in- or output for a node. Always instantiated as child of a node."""

    def __init__(self, out:bool, node:NodeBase, type:str, value:object, name:str) -> None:
        """
        Constructor function.
        
        Arguments:
        out: Boolean flag indicating whether the connector is an output (True) or an input (False)

        node: reference to the node the connector is a part of

        dtype: node datatype class of the data type of the connector

        name: display name and key of the connector
        """


        self.out = out
        self.converter = getConverter(type)
        self._type = type
        self.name = name
        self._value = value
        if self.out:
            self.attachedTo = []
        else:
            self.attachedTo = None
        self.node = node
    
    def get(self):
        """getter function for the value of a connector. If the connector is an input, it gets its value through the connection.
        If it is an output, it instead causes its own node to be evaluated if necessary, and catches loops in the node graph."""

        if self.out:
            if self.node.isUpdating:
                raise NodeLoopError(self.node)
            if self.node.isUpdated == False and self.node.static == False:
                self.node.isUpdating = True
                self.node.calculate()
                self.node.isUpdating = False
            return self._value
        else:
            if self.attachedTo is None:
                return self._value
            return self.attachedTo.get()
    
    def set(self, value):
        """sets the value of the connector, both internally and visually, using the conversion function of its node data type class, and performs error handling."""

        try:
            value = self.converter(value)
        except Exception as e:
            print(e)
            raise NodeTypeMismatchError(self.node)
        else:
            self._value = value
