#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch

class DenseCache():
    """Cache holding a Tensor of arbitrary size and dtype, with index- or slice-based accessing and saving"""

    def __init__(self, size:int, device:torch.device, dType:torch.dtype = torch.float32) -> None:
        """Constructor based on cache size, dtype and the device it should be stored on"""

        self.tensor = torch.zeros(size, device = device, dtype = dType)
    def read(self, start:int, end:int = None) -> torch.tensor:
        """reads a slice from the index start to the index end from the cache. If end is not given, returns only the element at index start instead."""

        if end == None:
            return self.tensor[start]
        return self.tensor[start:end]
    def write(self, value:torch.tensor, start:int, end:int = None) -> None:
        """writes value to the cache, starting at index start and ending at index end. Therefore, the length of value should be end - start. If it is 1, the end parameter can be omitted."""

        if end == None:
            self.tensor[start] = value
        else:
            self.tensor[start:end] = value

class SparseCache():
    """Cache holding a Tensor of arbitrary size and dtype, with index- or slice-based accessing and saving. Internally, only elements between the first and the last non-zero element are saved, reducing memory usage."""

    def __init__(self, size:int, device:torch.device, dType:torch.dtype = torch.float32) -> None:
        """Constructor based on cache size, dtype and the device it should be stored on"""

        size2 = []
        for i in size:
            size2.append(0)
        self.tensor = torch.zeros(size2, device = device, dtype = dType)
        self.start = None
        self.end = None
        self.fullSize = size
    def read(self, start:int, end:int = None) -> torch.tensor:
        """reads a slice from the index start to the index end from the cache. If end is not given, returns only the element at index start instead."""

        if start < 0:
            start += self.fullSize[0]
        if end == None:
            if self.start == None:
                return torch.zeros((1,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device)
            elif start < self.start or start >= self.end:
                return torch.zeros((1,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device)
            return self.tensor[start - self.start]
        if end < 0:
            end += self.fullSize[0]
        if self.start == None or self.end == None:
            return torch.zeros((end - start,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device)
        start -= self.start
        end -= self.end
        prepend = 0
        append = 0
        if start < 0:
            prepend -= start
            start = 0
        if end > 0:
            append += end
            end = 0
        end += self.end - self.start
        if end < 0:
            prepend += end
            output = self.tensor[0:0]
        elif start > self.end - self.start:
            append += self.end - self.start - start
            output = self.tensor[0:0]
        else:
            output = self.tensor[start:end]
        output = torch.cat((torch.zeros((prepend,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device), output, torch.zeros((append,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device)), 0)
        return output
    def write(self, value:torch.tensor, start:int, end:int = None) -> None:
        """writes value to the cache, starting at index start and ending at index end. Therefore, the length of value should be end - start. If it is 1, the end parameter can be omitted."""

        if start < 0:
            start += self.fullSize[0]
        if end == None:
            if self.start == None or self.end == None:
                self.start = start
                self.end = start + 1
                self.tensor = torch.zeros((1,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device)
                self.tensor[0] = value
            elif start < self.start:
                self.tensor =  torch.cat((torch.zeros((self.start - start,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device), self.tensor), 0)
                self.start = start
                self.tensor[0] = value
            elif start >= self.end:
                self.tensor =  torch.cat((self.tensor, torch.zeros((start + 1 - self.end,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device)), 0)
                self.end = start + 1
                self.tensor[-1] = value
            else:
                self.tensor[start - self.start] = value
            return
        if end < 0:
            end += self.fullSize[0]
        if self.start == None or self.end == None:
            self.start = start
            self.end = end
            self.tensor = value.to(self.tensor.device)
            return
        start -= self.start
        end -= self.end
        prepend = 0
        append = 0
        if start < 0:
            prepend -= start
            start = 0
        if end > 0:
            append += end
            end = 0
        self.start -= prepend
        self.end += append
        end += (self.end - self.start)
        self.tensor = torch.cat((torch.zeros((prepend,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device), self.tensor, torch.zeros((append,) + self.fullSize[1:], dtype = self.tensor.dtype, device = self.tensor.device)), 0)
        self.tensor[start:end] = value
        