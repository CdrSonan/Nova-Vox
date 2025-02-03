#Copyright 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

"""Half-finished code for a tree structure. Originally designed for handling notes in the editor, but put on indefinite hold due to the performance improvement being negligible."""

class Node():
    def __init__(self, maxBranches:int = 4, parent:object = None, data:object = None) -> None:
        self.data = data
        self.size = 0
        self.children = []
        self.iterator = None
    
    def isRoot(self) -> bool:
        return self.parent == None
    
    def fwd(self, index:int) -> object:
        if index >= self.size:
            raise IndexError("Index out of range.")
        counter = 0
        for i in self.children:
            if isinstance(i, Node):
                if counter + i.size <= index:
                    counter += i.size
                else:
                    return i.fwd(index - counter)
            else:
                if counter == index:
                    return i
                counter += 1
    
    def attach(self, child:object, index:int) -> None:
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range.")
        if len(self.children) == self.maxBranches:
            if self.parent == None:
                self.parent = Node(self.maxBranches)
                self.parent.children[0] = self
                self.parent.size = 1
            else:
                pass
        self.children[index] = child
        child.parent = self
        self.size += 1


class TreeIterator():
    def __init__(self, root:Node) -> None:
        self.current = root
        self.previous = None
    
    def __next__(self) -> object:
        if not isinstance(self.current, Node):
            self.previous = self.current
            self.current = self.current.parent
        while isinstance(self.current, Node):
            if self.previous == self.current.parent or self.previous == None:
                self.previous = self.current
                self.current = self.current.children[0]
            else:
                index = self.current.children.index(self.previous)
                if index + 1 < self.current.size:
                    self.previous = self.current
                    self.current = self.current.children[index + 1]
                else:
                    self.previous = self.current
                    self.current = self.current.parent
        if self.current == None:
            raise StopIteration
        return self.current


class Tree():
    def __init__(self, maxBranches:int = 4) -> None:
        self.maxBranches = maxBranches
        self.root = None
    
    def __getitem__(self, index:int) -> object:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range.")
        return self.root.fwd(index)
    
    def __setitem__(self, index:int, value:object) -> None:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range.")
        item = self.root.fwd(index)
        parent = item.parent
        if parent != None:
            itemIdx = parent.children.index(item)
        item = value
        if parent != None:
            item.parent = parent
            parent.children[itemIdx] = item
    
    def __delitem__(self, index:int) -> None:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range.")
        item = self.root.fwd(index)
        parent = item.parent
        if parent == None:
            del item
            return
        itemIdx = parent.children.index(item)
        parent.children.pop(itemIdx)
        del item
        while parent != None:
            parent.size -= 1
            if parent.size == 0:
                toDelete = parent
            else:
                toDelete = None
            parent = parent.parent
            if toDelete != None:
                del toDelete
    
    def __len__(self) -> int:
        return self.root.size
    
    def __iter__(self) -> object:
        return TreeIterator(self.root)
    
    def index(self, item:object) -> int:
        counter = 0
        parent = item.parent
        while parent != None:
            index = parent.children.index(item)
            for i in range(index):
                if isinstance(parent.children[i], Node):
                    counter += parent.children[i].size
                else:
                    counter += 1
            item = parent
            parent = item.parent
        return counter
    
    def insert(self, index:int, data:object) -> None:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range.")
        if self.root == None:
            if index != 0:
                raise IndexError("Index out of range.")
            self.root = data
            self.root.parent = None
            return
        if index == len(self):
            sibling = self.root.fwd(index - 1)
            if sibling.parent == None:
                sibling.parent = Node(self.maxBranches)
                sibling.parent.children[0] = sibling
                sibling.parent.size = 1
                self.root = sibling.parent
            index = sibling.parent.children.index(sibling) + 1
            
            
        
    
    def remove(self, data:object) -> None:
        self.__delitem__(self.index(data))
    
    def append(self, data:object) -> None:
        self.insert(-1, data)
