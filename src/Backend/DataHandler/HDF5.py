#Copyright 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import h5py
import torch
import numpy as np
from PIL import Image

from Backend.DataHandler.AudioSample import AudioSample, AISample, LiteAudioSample, AudioSampleCollection, AISampleCollection, LiteSampleCollection
from Backend.VB_Components.VbMetadata import VbMetadata
import global_consts as gc

class SampleStorage:
    """Class for handling audio data in an HDF5 file"""

    def __init__(self, file, groups:list = [], isTransition:bool = False):
        self.group = file
        for i in groups:
            if i not in self.group:
                self.group.create_group(i)
            self.group = self.group[i]
        self.pendingDeletions = []
        if "audio" not in self.group:
            self.group.create_dataset("audio", (0,), maxshape=(None,), dtype="float32", compression="gzip")
        if "audioIdxs" not in self.group:
            self.group.create_dataset("audioIdxs", (0,), maxshape=(None,), dtype="int64")
        if "pitchDeltas" not in self.group:
            self.group.create_dataset("pitchDeltas", (0,), maxshape=(None,), dtype="int32", compression="gzip")
        if "pitchDeltasIdxs" not in self.group:
            self.group.create_dataset("pitchDeltasIdxs", (0,), maxshape=(None,), dtype="int64")
        if "pitchMarkers" not in self.group:
            self.group.create_dataset("pitchMarkers", (0,), maxshape=(None,), dtype="int32", compression="gzip")
        if "pitchMarkerValidity" not in self.group:
            self.group.create_dataset("pitchMarkerValidity", (0,), maxshape=(None,), dtype="int8", compression="gzip")
        if "pitchMarkersIdxs" not in self.group:
            self.group.create_dataset("pitchMarkersIdxs", (0,), maxshape=(None,), dtype="int64")
        if "pitch" not in self.group:
            self.group.create_dataset("pitch", (0,), maxshape=(None,), dtype="int32")
        if "specharm" not in self.group:
            self.group.create_dataset("specharm", (0, gc.frameSize), maxshape=(None, gc.frameSize), dtype="float32", compression="gzip")
        if "specharmIdxs" not in self.group:
            self.group.create_dataset("specharmIdxs", (0,), maxshape=(None,), dtype="int64")
        if "avgSpecharm" not in self.group:
            self.group.create_dataset("avgSpecharm", (0, gc.reducedFrameSize), maxshape=(None, gc.reducedFrameSize), dtype="float32")
        if "filepaths" not in self.group:
            self.group.create_dataset("filepaths", (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8"))
        if "keys" not in self.group:
            self.group.create_dataset("keys", (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8"))
        if "flags" not in self.group:
            self.group.create_dataset("flags", (0, 2), maxshape=(None, 2), dtype="bool")
        if "floatCfg" not in self.group:
            self.group.create_dataset("floatCfg", (0, 2), maxshape=(None, 2), dtype="float32")
        if "intCfg" not in self.group:
            if isTransition:
                self.group.create_dataset("intCfg", (0, 2), maxshape=(None, 2), dtype="int64")
            else:
                self.group.create_dataset("intCfg", (0, 1), maxshape=(None, 1), dtype="int64")
        if "length" not in self.group.attrs:
            self.group.attrs["length"] = 0
        if "isTransition" not in self.group.attrs:
            self.group.attrs["isTransition"] = isTransition
    
    def __len__(self):
        return self.group.attrs["length"]
    
    def fetch(self, idx:int, dtype:str = "full"):
        if idx < 0 or idx >= self.group.attrs["length"]:
            raise IndexError("Index out of range")
        for i in self.pendingDeletions:
            if i <= idx:
                idx += 1
        if dtype == "full":
            return self.fetchFull(idx)
        elif dtype == "AI":
            return self.fetchAI(idx)
        elif dtype == "lite":
            return self.fetchLite(idx)
        else:
            raise ValueError("Invalid sample dtype")
        
    def fetchFull(self, idx:int):
        sample = AudioSample(isTransition = self.group.attrs["isTransition"])
        if idx == 0:
            sample.waveform = torch.tensor(self.group["audio"][:self.group["audioIdxs"][idx]], dtype=torch.float32)
            sample.pitchDeltas = torch.tensor(self.group["pitchDeltas"][:self.group["pitchDeltasIdxs"][idx]], dtype=torch.int)
            sample.pitchMarkers = torch.tensor(self.group["pitchMarkers"][:self.group["pitchMarkersIdxs"][idx]], dtype=torch.int)
            sample.pitchMarkerValidity = torch.tensor(self.group["pitchMarkerValidity"][:self.group["pitchMarkersIdxs"][idx]], dtype=torch.int8)
            sample.specharm = torch.tensor(self.group["specharm"][:self.group["specharmIdxs"][idx]], dtype=torch.float32)
        else:
            sample.waveform = torch.tensor(self.group["audio"][self.group["audioIdxs"][idx - 1]:self.group["audioIdxs"][idx]], dtype=torch.float32)
            sample.pitchDeltas = torch.tensor(self.group["pitchDeltas"][self.group["pitchDeltasIdxs"][idx - 1]:self.group["pitchDeltasIdxs"][idx]], dtype=torch.int)
            sample.pitchMarkers = torch.tensor(self.group["pitchMarkers"][self.group["pitchMarkersIdxs"][idx - 1]:self.group["pitchMarkersIdxs"][idx]], dtype=torch.int)
            sample.specharm = torch.tensor(self.group["specharm"][self.group["specharmIdxs"][idx - 1]:self.group["specharmIdxs"][idx]], dtype=torch.float32)
        sample.pitch = self.group["pitch"][idx].item()
        sample.avgSpecharm = torch.tensor(self.group["avgSpecharm"][idx], dtype=torch.float32)
        sample.filepath = self.group["filepaths"][idx].decode("utf-8")
        sample.key = self.group["keys"][idx].decode("utf-8")
        if sample.key == "_None":
            sample.key = None
        sample.isVoiced = self.group["flags"][idx][0].item()
        sample.isPlosive = self.group["flags"][idx][1].item()
        sample.expectedPitch = self.group["floatCfg"][idx][0].item()
        sample.searchRange = self.group["floatCfg"][idx][1].item()
        if self.group.attrs["isTransition"]:
            sample.embedding = (self.group["intCfg"][idx][0].item(), self.group["intCfg"][idx][1].item())
        else:
            sample.embedding = self.group["intCfg"][idx][0].item()
        return sample
    
    def fetchAI(self, idx:int):
        sample = AISample(isTransition = self.group.attrs["isTransition"])
        if idx == 0:
            sample.waveform = torch.tensor(self.group["audio"][:self.group["audioIdxs"][idx]], dtype=torch.float32)
        else:
            sample.waveform = torch.tensor(self.group["audio"][self.group["audioIdxs"][idx - 1]:self.group["audioIdxs"][idx]], dtype=torch.float32)
        sample.filepath = self.group["filepaths"][idx].decode("utf-8")
        sample.key = self.group["keys"][idx].decode("utf-8")
        if sample.key == "_None":
            sample.key = None
        sample.isVoiced = self.group["flags"][idx][0].item()
        sample.isPlosive = self.group["flags"][idx][1].item()
        sample.expectedPitch = self.group["floatCfg"][idx][0].item()
        sample.searchRange = self.group["floatCfg"][idx][1].item()
        if self.group.attrs["isTransition"]:
            sample.embedding = (self.group["intCfg"][idx][0].item(), self.group["intCfg"][idx][1].item())
        else:
            sample.embedding = self.group["intCfg"][idx][0].item()
        return sample
    
    def fetchLite(self, key:str, byKey:bool = False):
        if byKey:
            idx = self.group["keys"].tolist().index(key)
        else:
            idx = key
        sample = LiteAudioSample()
        if idx == 0:
            sample.pitchDeltas = torch.tensor(self.group["pitchDeltas"][:self.group["pitchDeltasIdxs"][idx]], dtype=torch.int)
            sample.specharm = torch.tensor(self.group["specharm"][:self.group["specharmIdxs"][idx]], dtype=torch.float32)
        else:
            sample.pitchDeltas = torch.tensor(self.group["pitchDeltas"][self.group["pitchDeltasIdxs"][idx - 1]:self.group["pitchDeltasIdxs"][idx]], dtype=torch.int)
            sample.specharm = torch.tensor(self.group["specharm"][self.group["specharmIdxs"][idx - 1]:self.group["specharmIdxs"][idx]], dtype=torch.float32)
        sample.pitch = self.group["pitch"][idx].item()
        sample.avgSpecharm = torch.tensor(self.group["avgSpecharm"][idx], dtype=torch.float32)
        sample.key = self.group["keys"][idx].decode("utf-8")
        if sample.key == "_None":
            sample.key = None
        sample.isVoiced = self.group["flags"][idx][0].item()
        sample.isPlosive = self.group["flags"][idx][1].item()
        if self.group.attrs["isTransition"]:
            sample.embedding = (self.group["intCfg"][idx][0], self.group["intCfg"][idx][1])
        else:
            sample.embedding = self.group["intCfg"][idx][0]
        return sample
    
    def append(self, sample):
        if isinstance(sample, AISample):
            sample = sample.convert()
        if isinstance(sample, LiteAudioSample):
            raise NotImplementedError("LiteAudioSample objects cannot be appended to a SampleStorage")
        if not isinstance(sample, AudioSample):
            raise ValueError("sample has invalid data type")
        self.group["audio"].resize(self.group["audio"].shape[0] + sample.waveform.shape[0], axis=0)
        self.group["audio"][-sample.waveform.shape[0]:] = sample.waveform
        self.group["audioIdxs"].resize(self.group["audioIdxs"].shape[0] + 1, axis=0)
        if self.group["audioIdxs"].shape[0] == 1:
            self.group["audioIdxs"][-1] = sample.waveform.shape[0]
        else:
            self.group["audioIdxs"][-1] = self.group["audioIdxs"][-2] + sample.waveform.shape[0]
        self.group["pitchDeltas"].resize(self.group["pitchDeltas"].shape[0] + sample.pitchDeltas.shape[0], axis=0)
        self.group["pitchDeltas"][-sample.pitchDeltas.shape[0]:] = sample.pitchDeltas
        self.group["pitchDeltasIdxs"].resize(self.group["pitchDeltasIdxs"].shape[0] + 1, axis=0)
        if self.group["pitchDeltasIdxs"].shape[0] == 1:
            self.group["pitchDeltasIdxs"][-1] = self.group["pitchDeltas"].shape[0]
        else:
            self.group["pitchDeltasIdxs"][-1] = self.group["pitchDeltasIdxs"][-2] + sample.pitchDeltas.shape[0]
        self.group["pitchMarkers"].resize(self.group["pitchMarkers"].shape[0] + sample.pitchMarkers.shape[0], axis=0)
        self.group["pitchMarkers"][-sample.pitchMarkers.shape[0]:] = sample.pitchMarkers
        self.group["pitchMarkerValidity"].resize(self.group["pitchMarkerValidity"].shape[0] + sample.pitchMarkerValidity.shape[0], axis=0)
        self.group["pitchMarkerValidity"][-sample.pitchMarkerValidity.shape[0]:] = sample.pitchMarkerValidity
        self.group["pitchMarkersIdxs"].resize(self.group["pitchMarkersIdxs"].shape[0] + 1, axis=0)
        if self.group["pitchMarkersIdxs"].shape[0] == 1:
            self.group["pitchMarkersIdxs"][-1] = self.group["pitchMarkers"].shape[0]
        else:
            self.group["pitchMarkersIdxs"][-1] = self.group["pitchMarkersIdxs"][-2] + sample.pitchMarkers.shape[0]
        self.group["pitch"].resize(self.group["pitch"].shape[0] + 1, axis=0)
        self.group["pitch"][-1] = sample.pitch
        self.group["specharm"].resize(self.group["specharm"].shape[0] + sample.specharm.shape[0], axis=0)
        self.group["specharm"][-sample.specharm.shape[0]:] = sample.specharm
        self.group["specharmIdxs"].resize(self.group["specharmIdxs"].shape[0] + 1, axis=0)
        if self.group["specharmIdxs"].shape[0] == 1:
            self.group["specharmIdxs"][-1] = self.group["specharm"].shape[0]
        else:
            self.group["specharmIdxs"][-1] = self.group["specharmIdxs"][-2] + sample.specharm.shape[0]
        self.group["avgSpecharm"].resize(self.group["avgSpecharm"].shape[0] + 1, axis=0)
        self.group["avgSpecharm"][-1] = sample.avgSpecharm
        self.group["filepaths"].resize(self.group["filepaths"].shape[0] + 1, axis=0)
        self.group["filepaths"][-1] = sample.filepath.encode("utf-8")
        self.group["keys"].resize(self.group["keys"].shape[0] + 1, axis=0)
        if sample.key is None:
            self.group["keys"][-1] = "_None".encode("utf-8")
        else:
            self.group["keys"][-1] = sample.key.encode("utf-8")
        self.group["flags"].resize(self.group["flags"].shape[0] + 1, axis=0)
        self.group["flags"][-1] = [sample.isVoiced, sample.isPlosive]
        self.group["floatCfg"].resize(self.group["floatCfg"].shape[0] + 1, axis=0)
        self.group["floatCfg"][-1] = [sample.expectedPitch, sample.searchRange]
        self.group["intCfg"].resize(self.group["intCfg"].shape[0] + 1, axis=0)
        if self.group.attrs["isTransition"]:
            self.group["intCfg"][-1] = [sample.embedding[0], sample.embedding[1]]
        else:
            self.group["intCfg"][-1] = [sample.embedding,]
        self.group.attrs["length"] += 1
        
    def delete(self, idx:int):
        self.pendingDeletions.append(idx)
        self.group.attrs["length"] -= 1
        
    def commitDeletions(self):
        for i in self.pendingDeletions:
            self.group["audio"] = torch.cat((self.group["audio"][:self.group["audioIdxs"][i]], self.group["audio"][self.group["audioIdxs"][i + 1]:]), dim=0)
            self.group["audioIdxs"] = torch.cat((self.group["audioIdxs"][:i], self.group["audioIdxs"][i + 1:]), dim=0)
            self.group["pitchDeltas"] = torch.cat((self.group["pitchDeltas"][:self.group["pitchDeltasIdxs"][i]], self.group["pitchDeltas"][self.group["pitchDeltasIdxs"][i + 1]:]), dim=0)
            self.group["pitchDeltasIdxs"] = torch.cat((self.group["pitchDeltasIdxs"][:i], self.group["pitchDeltasIdxs"][i + 1:]), dim=0)
            self.group["pitchMarkers"] = torch.cat((self.group["pitchMarkers"][:self.group["pitchMarkersIdxs"][i]], self.group["pitchMarkers"][self.group["pitchMarkersIdxs"][i + 1]:]), dim=0)
            self.group["pitchMarkerValidity"] = torch.cat((self.group["pitchMarkerValidity"][:self.group["pitchMarkersIdxs"][i]], self.group["pitchMarkerValidity"][self.group["pitchMarkersIdxs"][i + 1]:]), dim=0)
            self.group["pitchMarkersIdxs"] = torch.cat((self.group["pitchMarkersIdxs"][:i], self.group["pitchMarkersIdxs"][i + 1:]), dim=0)
            self.group["pitch"] = torch.cat((self.group["pitch"][:i], self.group["pitch"][i + 1:]), dim=0)
            self.group["specharm"] = torch.cat((self.group["specharm"][:self.group["specharmIdxs"][i]], self.group["specharm"][self.group["specharmIdxs"][i + 1]:]), dim=0)
            self.group["specharmIdxs"] = torch.cat((self.group["specharmIdxs"][:i], self.group["specharmIdxs"][i + 1:]), dim=0)
            self.group["avgSpecharm"] = torch.cat((self.group["avgSpecharm"][:i], self.group["avgSpecharm"][i + 1:]), dim=0)
            self.group["filepaths"] = torch.cat((self.group["filepaths"][:i], self.group["filepaths"][i + 1:]), dim=0)
            self.group["keys"] = torch.cat((self.group["keys"][:i], self.group["keys"][i + 1:]), dim=0)
            self.group["flags"] = torch.cat((self.group["flags"][:i], self.group["flags"][i + 1:]), dim=0)
            self.group["floatCfg"] = torch.cat((self.group["floatCfg"][:i], self.group["floatCfg"][i + 1:]), dim=0)
            self.group["intCfg"] = torch.cat((self.group["intCfg"][:i], self.group["intCfg"][i + 1:]), dim=0)
            self.group.attrs["length"] -= 1
        self.pendingDeletions = []
    
    def toCollection(self, dType:str = "full"):
        if dType == "full":
            collection = AudioSampleCollection(isTransition=self.group.attrs["isTransition"])
        elif dType == "AI":
            collection = AISampleCollection(isTransition=self.group.attrs["isTransition"])
        elif dType == "lite":
            collection = LiteSampleCollection(isTransition=self.group.attrs["isTransition"])
        else:
            raise ValueError("Invalid sample dtype")
        for i in range(len(self)):
            collection.append(self.fetch(i, dType))
        return collection
    
    def fromCollection(self, collection, overwrite:bool = False):
        if overwrite:
            for i in range(len(self)):
                self.delete(-1)
            self.commitDeletions()
        if isinstance(collection, LiteSampleCollection):
            raise NotImplementedError("LiteAudioSample objects cannot be appended to a SampleStorage")
        for i in collection:
            self.append(i)
    
    def toDict(self):
        dictionary = {}
        for i in range(len(self)):
            key = self.group["keys"][i].decode("utf-8")
            if key in dictionary.keys():
                dictionary[key].append(self.fetch(i))
            else:
                dictionary[key] = [self.fetch(i),]
        return dictionary
    
    def fromDict(self, dictionary, overwrite:bool = False):
        if overwrite:
            for i in range(len(self)):
                self.delete(-1)
            self.commitDeletions()
        for i in dictionary.keys():
            for j in dictionary[i]:
                self.append(j)


class DictStorage:
    """Class for handling dictionaries in an HDF5 file"""

    def __init__(self, file, groups:list = [], torchDevice:str = "cpu"):
        self.group = file
        for i in groups:
            if i not in self.group:
                self.group.create_group(i)
            self.group = self.group[i]
        self.torchDevice = torchDevice
    
    def fetch(self, keys:list):
        position = self.group
        for i in keys:
            if i not in position:
                raise KeyError("Key not found")
            position = position[i]
        return position[()]
    
    def insert(self, keys:list, value):
        position = self.group
        for i in keys[:-1]:
            if i not in position:
                position.create_group(i)
            position = position[i]
        if keys[-1] in position:
            del position[keys[-1]]
        position.create_dataset(keys[-1], data=value)
        
    def delete(self, keys:list, recursive:bool = False):
        position = self.group
        for i in keys[:-1]:
            if i not in position:
                raise KeyError("Key not found")
            position = position[i]
        if keys[-1] not in position:
            raise KeyError("Key not found")
        del position[keys[-1]]
        if recursive:
            while len(position.keys()) == 0:
                newPosition = position.parent
                del position
                position = newPosition
        
    def toDict(self):
        def unpack(data):
            if data.attrs["type"] == "None":
                return None
            elif data.attrs["type"] in ("float", "int"):
                return data[()].item()
            elif data.attrs["type"] == "bool":
                return bool(data[()].item())
            elif data.attrs["type"] == "str":
                return data[()].decode("utf-8")
            elif data.attrs["type"] == "tensor":
                return torch.tensor(data[()], device=self.torchDevice)
            else:
                print("WARNING: unknown data type hint in loaded file, deserialization may not produce expected results")
                return data[()]
        def recursiveFetch(position):
            if position.attrs["type"] in ("list", "tuple"):
                target = []
                for i in position:
                    if isinstance(position[i], h5py.Group):
                        target.append(recursiveFetch(position[i]))
                    else:
                        target.append(unpack(position[i]))
                if position.attrs["type"] == "tuple":
                    target = tuple(target)
            else:
                target = {}
                for i in position.keys():
                    if i == "__emptyString__":
                        iOut = ""
                    elif i.startswith("__int__"):
                        iOut = int(i[7:])
                    else:
                        iOut = i
                    if isinstance(position[i], h5py.Group):
                        target[iOut] = recursiveFetch(position[i])
                    else:
                        target[iOut] = unpack(position[i])
            
            return target
        return recursiveFetch(self.group)
        
    
    def fromDict(self, dictionary):
        def pack(data):
            if data is None:
                outData = np.array([])
                outDType = "None"
            elif isinstance(data, torch.Tensor):
                outData = data.cpu().numpy()
                outDType = "tensor"
            elif isinstance(data, bool):
                outData = np.array(data)
                outDType = "bool"
            elif isinstance(data, float):
                outData = np.array(data)
                outDType = "float"
            elif isinstance(data, int):
                outData = np.array(data)
                outDType = "int"
            elif isinstance(data, str):
                outData = data.encode("utf-8")
                outDType = "str"
            else:
                raise ValueError("Invalid data type for serialization")
            return outData, outDType
        def recursiveInsert(position, data):
            print(position, data)
            if isinstance(data, dict):
                for key in data.keys():
                    if isinstance(key, int):
                        newKey = "__int__" + str(key)
                    elif isinstance(key, str):
                        if key.startswith("__int__"):
                            raise ValueError("Keys starting with __int__ are reserved for internal use")
                        elif key == "__emptyString__":
                            raise ValueError("__emptyString__ is reserved for internal use")
                        elif key == "":
                            newKey = "__emptyString__"
                        else:
                            newKey = key
                    else:
                        raise ValueError("Keys must be strings or ints")
                    print("key:", key)
                    if isinstance(data[key], dict):
                        if newKey not in position:
                            position.create_group(newKey)
                            position[newKey].attrs["type"] = "dict"
                        recursiveInsert(position[newKey], data[key])
                    elif isinstance(data[key], list):
                        if newKey not in position:
                            position.create_group(newKey)
                            position[newKey].attrs["type"] = "list"
                        recursiveInsert(position[newKey], data[key])
                    elif isinstance(data[key], tuple):
                        if newKey not in position:
                            position.create_group(newKey)
                            position[newKey].attrs["type"] = "tuple"
                        recursiveInsert(position[newKey], data[key])
                    else:
                        outData, outDType = pack(data[key])
                        position.create_dataset(newKey, data=outData)
                        position[newKey].attrs["type"] = outDType
            elif isinstance(data, list) or isinstance(data, tuple):
                for idx, i in enumerate(data):
                    print("key:", str(idx))
                    if isinstance(i, dict):
                        if str(idx) not in position:
                            position.create_group(str(idx))
                            position[str(idx)].attrs["type"] = "dict"
                        recursiveInsert(position[str(idx)], i)
                    elif isinstance(i, list):
                        if str(idx) not in position:
                            position.create_group(str(idx))
                            position[str(idx)].attrs["type"] = "list"
                        recursiveInsert(position[str(idx)], i)
                    elif isinstance(i, tuple):
                        if str(idx) not in position:
                            position.create_group(str(idx))
                            position[str(idx)].attrs["type"] = "tuple"
                        recursiveInsert(position[str(idx)], i)
                    else:
                        outData, outDType = pack(i)
                        position.create_dataset(str(idx), data = outData)
                        position[str(idx)].attrs["type"] = outDType
        if isinstance(dictionary, dict):
            self.group.attrs["type"] = "dict"
        elif isinstance(dictionary, list):
            self.group.attrs["type"] = "list"
        elif isinstance(dictionary, tuple):
            self.group.attrs["type"] = "tuple"
        recursiveInsert(self.group, dictionary)


class WordStorage:
    """Class for handling word data in an HDF5 file"""

    def __init__(self, file, groups:list = ["wordDict",],):
        self.group = file
        for i in groups:
            if i not in self.group:
                self.group.create_group(i)
            self.group = self.group[i]
        if "words" not in self.group:
            self.group.create_group("words")
        if "syllables_keys" not in self.group:
            self.group.create_group("syllables_keys")
        if "syllables_vals" not in self.group:
            self.group.create_group("syllables_vals")
    
    def fetchWord(self, word:str):
        if word not in self.group["words"]:
            raise KeyError("Word not found")
        return self.group[word][()]
    
    def fetchSyllable(self, syllable:str):
        tgtLength = len(syllable)
        for i, key in enumerate(self.group["syllables_keys"][tgtLength][()]):
            if key == syllable:
                return self.group["syllables_vals"][tgtLength][i]
        raise KeyError("Syllable not found")
    
    def insertWord(self, word:str, value:str):
        words = self.group["words"]
        if word in words:
            del words[word]
        words.create_dataset(word, data=[i.encode("utf-8") for i in value], dtype=h5py.string_dtype(encoding="utf-8"))
    
    def insertSyllable(self, syllable:str, value:str):
        tgtLength = len(syllable)
        self.group["syllables_keys"][str(tgtLength)].resize(self.group["syllables_keys"][str(tgtLength)].shape[0] + 1, axis=0)
        self.group["syllables_keys"][str(tgtLength)][-1] = syllable.encode("utf-8")
        self.group["syllables_vals"][str(tgtLength)].resize(self.group["syllables_vals"][str(tgtLength)].shape[0] + 1, axis=0)
        self.group["syllables_vals"][str(tgtLength)][-1] = value.encode("utf-8")
    
    def toDict(self):
        words = self.group["words"]
        wordDict = [{}, []]
        for i in words.keys():
            wordDict[0][i] = [j.decode("utf-8") for j in words[i][()]]
        syllableKeys = self.group["syllables_keys"]
        syllableValues = self.group["syllables_vals"]
        for keyIdx, valueIdx in zip(syllableKeys, syllableValues):
            keys = syllableKeys[keyIdx][()]
            values = syllableValues[valueIdx][()]
            syllableDict = {i.decode("utf-8"): j.decode("utf-8") for i, j in zip(keys, values)}
            wordDict[1].append(syllableDict)
        return wordDict
    
    def fromDict(self, dictionary):
        words = dictionary[0]
        for word, mappings in words.items():
            self.insertWord(word, mappings)
        for idx, syllableDict in enumerate(dictionary[1]):
            self.group["syllables_keys"].create_dataset(str(idx + 1), (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8"))
            self.group["syllables_vals"].create_dataset(str(idx + 1), (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8"))
            for syllable, value in syllableDict.items():
                self.insertSyllable(syllable, value)


class MetadataStorage:
    
    def __init__(self, file, groups:list = ["metadata",]):
        self.group = file
        for i in groups:
            if i not in self.group:
                self.group.create_group(i)
            self.group = self.group[i]
        if "image" not in self.group:
            self.group.create_dataset("image", (200, 200, 4), dtype="uint8")
    
    def fromMetadata(self, metadata):
        self.group.attrs["name"] = metadata.name
        self.group.attrs["sampleRate"] = metadata.sampleRate
        self.group.attrs["version"] = metadata.version
        self.group.attrs["description"] = metadata.description
        self.group.attrs["license"] = metadata.license
        imageArray = np.array(metadata.image)
        if imageArray.shape[-1] == 3:
            imageArray = np.concatenate((imageArray, np.full(imageArray.shape[:-1] + (1,), 255, dtype=np.uint8)), axis=-1)
        self.group["image"][:, :, :] = imageArray
    
    def toMetadata(self):
        metadata = VbMetadata()
        metadata.name = self.group.attrs["name"]
        metadata.sampleRate = self.group.attrs["sampleRate"]
        metadata.version = self.group.attrs["version"]
        metadata.description = self.group.attrs["description"]
        metadata.license = self.group.attrs["license"]
        metadata.image = Image.fromarray(self.group["image"][()])
        return metadata

