#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

class Node():
    def __init__(self, maxBranches:int = 4, parent:object = None, data:object = None) -> None:
        self.data = data
        self.size = 0
        self.children = [None for _ in range(maxBranches)]
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
    
    def __iter__(self) -> object:
        self.iterator = 0
        return self
    
    def __next__(self) -> object:
        if self.iterator >= self.size:
            self.iterator = None
            raise StopIteration
        item = self.children[self.iterator]
        if isinstance(item, Node):
            try:
                return item.__next__()
            except StopIteration:
                self.iterator += 1
                return self.__next__()
        else:
            self.iterator += 1
            return item

class Tree():
    def __init__(self, maxBranches:int = 4) -> None:
        self.maxBranches = maxBranches
        self.root = None
    
    def __getitem__(self, index:int) -> object:
        return self.root.fwd(index)
    
    def __setitem__(self, index:int, value:object) -> None:
        item = self.root.fwd(index)
        parent = item.parent
        if parent != None:
            itemIdx = parent.children.index(item)
        item = value
        if parent != None:
            item.parent = parent
            parent.children[itemIdx] = item
    
    def __delitem__(self, index:int) -> None:
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
        if self.root == None:
            raise StopIteration
        if isinstance(self.root, Node):
            return self.root.__iter__()
        return self.root
    
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
        if parent == None:
            parent = self.root
        if parent.size < self.maxBranches:
            parent.children[parent.size] = Node(parent = parent, data = data)
            parent.size += 1
        else:
            raise IndexError("Parent node is full.")
    
    def remove(self, data:object) -> None:
        self.__delitem__(self.index(data))
    
    def append(self, data:object) -> None:
        self.insert(-1, data)
