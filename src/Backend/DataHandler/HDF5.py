#Copyright 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import os

import h5py

from Backend.DataHandler.AudioSample import audioCollection, sampleCollection
from API.Addon import override

class AudioStorage:
    """Class for handling audio data in an HDF5 file"""

    def __init__(self, path:str, groups:list = []):
        self.path = path
        self.file = h5py.File(path, "r+")
        self.group = self.file
        for i in groups:
            if i not in self.group:
                self.group.create_group(i)
            self.group = self.group[i]
        self.pendingDeletions = []
        if "audio" not in self.group:
            self.group.create_dataset("audio", (0,), maxshape=(None,), dtype="float32", compression="gzip")
        if "idxs" not in self.group:
            self.group.create_dataset("idxs", (0,), maxshape=(None,), dtype="int64")
        if "filepaths" not in self.group:
            self.group.create_dataset("filepaths", (0,), maxshape=(None,), dtype="S256")
        if "keys" not in self.group:
            self.group.create_dataset("keys", (0,), maxshape=(None,), dtype="S32")
        if "flags" not in self.group:
            self.group.create_dataset("flags", (0, 2), maxshape=(None, 2), dtype="bool")
        if "floatCfg" not in self.group:
            self.group.create_dataset("floatCfg", (0, 3), maxshape=(None, 3), dtype="float32")
        if "intCfg" not in self.group:
            self.group.create_dataset("intCfg", (0, 5), maxshape=(None, 5), dtype="int16")
    
    def fetch(self, idx:int, dtype:str = "full") -> audioCollection:
        