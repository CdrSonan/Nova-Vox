#Copyright 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from inspect import getsource

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


class PackedNode():
    """base class of a packed audio processing node. All packed node classes should inherit from this class."""

    def __init__(self, base) -> None:
        """
        constructs an arbitrary node using the provided parameters. Designed to be called from the __init__ method of a child class using super().__init__(), after the required arguments have been set up.
        """
        self.inputs = dict()
        self.outputs = dict()
        for i in base.inputs.keys():
            self.inputs[i] = PackedConnector(self, base.inputs[i])
        for i in base.outputs.keys():
            self.outputs[i] = PackedConnector(self, base.outputs[i])
        self.func = getsource(base.func)
        self.timed = base.timed
        self.static = base.static
        self.isUpdated = False
        self.isUpdating = False
    
    def calculate(self) -> None:
        """evaluates the node, and recursively prompts evaluation of all nodes connected to its inputs, if required"""

        if isinstance(self.func, str):
            eval(self.func)
            self.func = locals()["func"]
        inputs = dict()
        for i in self.inputs.keys():
            inputs[i] = self.inputs[i].get()
        result = self.func(**inputs)
        for i in result.keys():
            self.outputs[i].set(result[i])
        self.isUpdated = True


class PackedConnector():
    """Packed Connector object, used for transferring node data between processes"""

    def __init__(self, node:PackedNode, base) -> None:
        """
        Constructor function.
        
        Arguments:
        out: Boolean flag indicating whether the connector is an output (True) or an input (False)

        node: reference to the node the connector is a part of

        dtype: node datatype class of the data type of the connector

        name: display name and key of the connector
        """


        self.out = base.out
        self.name = base.name
        self._value = base._value
        self.convert = getsource(base.dtype.convert)
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
                self.isUpdating = True
                self.node.calculate()
                self.isUpdating = False
            return self._value
        else:
            if self.attachedTo is None:
                return self._value
            return self.attachedTo.get()
    
    def set(self, value):
        """sets the value of the connector, using the conversion function of its node data type class, and performs error handling."""

        if isinstance(self.convert, str):
            eval(self.convert)
            self.convert = locals()["convert"]
        try:
            value = self.convert(value)
        except:
            raise NodeTypeMismatchError(self.node)
        else:
            self._value = value
