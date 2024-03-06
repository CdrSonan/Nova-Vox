#Copyright 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import h5py
import torch

from Backend.DataHandler.AudioSample import AudioSample, AISample, LiteAudioSample, AudioSampleCollection, AISampleCollection, LiteSampleCollection
import global_consts as gc

class SampleStorage:
    """Class for handling audio data in an HDF5 file"""

    def __init__(self, path:str, groups:list = [], isTransition:bool = False):
        self.path = path
        self.groups = groups
        self.file = h5py.File(path, "a")
        self.group = self.file
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
        if "pitch" not in self.group:
            self.group.create_dataset("pitch", (0,), maxshape=(None,), dtype="int32")
        if "specharm" not in self.group:
            self.group.create_dataset("specharm", (0, gc.frameSize), maxshape=(None, gc.frameSize), dtype="float32", compression="gzip")
        if "specharmIdxs" not in self.group:
            self.group.create_dataset("specharmIdxs", (0,), maxshape=(None,), dtype="int64")
        if "avgSpecharm" not in self.group:
            self.group.create_dataset("avgSpecharm", (0, gc.reducedFrameSize), maxshape=(None, gc.reducedFrameSize), dtype="float32")
        if "excitation" not in self.group:
            self.group.create_dataset("excitation", (0, gc.halfTripleBatchSize), maxshape=(None, gc.halfTripleBatchSize), dtype="float32", compression="gzip")
        if "filepaths" not in self.group:
            self.group.create_dataset("filepaths", (0,), maxshape=(None,), dtype="S256")
        if "keys" not in self.group:
            self.group.create_dataset("keys", (0,), maxshape=(None,), dtype="S32")
        if "flags" not in self.group:
            self.group.create_dataset("flags", (0, 2), maxshape=(None, 2), dtype="bool")
        if "floatCfg" not in self.group:
            self.group.create_dataset("floatCfg", (0, 3), maxshape=(None, 3), dtype="float32")
        if "intCfg" not in self.group:
            if isTransition:
                self.group.create_dataset("intCfg", (0, 6), maxshape=(None, 6), dtype="int64")
            else:
                self.group.create_dataset("intCfg", (0, 5), maxshape=(None, 5), dtype="int64")
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
        sample = AudioSample()
        if idx == 0:
            sample.waveform = torch.tensor(self.group["audio"][:self.group["audioIdxs"][idx]], dtype=torch.float32)
            sample.pitchDeltas = torch.tensor(self.group["pitchDeltas"][:self.group["pitchDeltasIdxs"][idx]], dtype=torch.int)
            sample.specharm = torch.tensor(self.group["specharm"][:self.group["specharmIdxs"][idx]], dtype=torch.float32)
            sample.excitation = torch.tensor(self.group["excitation"][:self.group["specharmIdxs"][idx]], dtype=torch.float32)
        else:
            sample.waveform = torch.tensor(self.group["audio"][self.group["audioIdxs"][idx - 1]:self.group["audioIdxs"][idx]], dtype=torch.float32)
            sample.pitchDeltas = torch.tensor(self.group["pitchDeltas"][self.group["pitchDeltasIdxs"][idx - 1]:self.group["pitchDeltasIdxs"][idx]], dtype=torch.int)
            sample.specharm = torch.tensor(self.group["specharm"][self.group["specharmIdxs"][idx - 1]:self.group["specharmIdxs"][idx]], dtype=torch.float32)
            sample.excitation = torch.tensor(self.group["excitation"][self.group["specharmIdxs"][idx - 1]:self.group["specharmIdxs"][idx]], dtype=torch.float32)
        sample.pitch = self.group["pitch"][idx]
        sample.avgSpecharm = torch.tensor(self.group["avgSpecharm"][idx], dtype=torch.float32)
        sample.filepath = self.group["filepaths"][idx].decode("utf-8")
        sample.key = self.group["keys"][idx].decode("utf-8")
        if sample.key == "_None":
            sample.key = None
        sample.isVoiced = self.group["flags"][idx][0]
        sample.isPlosive = self.group["flags"][idx][1]
        sample.expectedPitch = self.group["floatCfg"][idx][0]
        sample.searchRange = self.group["floatCfg"][idx][1]
        sample.voicedThrh = self.group["floatCfg"][idx][2]
        sample.specWidth = self.group["intCfg"][idx][0]
        sample.specDepth = self.group["intCfg"][idx][1]
        sample.tempWidth = self.group["intCfg"][idx][2]
        sample.tempDepth = self.group["intCfg"][idx][3]
        if self.group.attrs["isTransition"]:
            sample.embedding = (self.group["intCfg"][idx][4], self.group["intCfg"][idx][5])
        else:
            sample.embedding = self.group["intCfg"][idx][4]
    
    def fetchAI(self, idx:int):
        sample = AISample()
        if idx == 0:
            sample.waveform = torch.tensor(self.group["audio"][:self.group["audioIdxs"][idx]], dtype=torch.float32)
        else:
            sample.waveform = torch.tensor(self.group["audio"][self.group["audioIdxs"][idx - 1]:self.group["audioIdxs"][idx]], dtype=torch.float32)
        sample.filepath = self.group["filepaths"][idx].decode("utf-8")
        sample.key = self.group["keys"][idx].decode("utf-8")
        if sample.key == "_None":
            sample.key = None
        sample.isVoiced = self.group["flags"][idx][0]
        sample.isPlosive = self.group["flags"][idx][1]
        sample.expectedPitch = self.group["floatCfg"][idx][0]
        sample.searchRange = self.group["floatCfg"][idx][1]
        sample.voicedThrh = self.group["floatCfg"][idx][2]
        sample.specWidth = self.group["intCfg"][idx][0]
        sample.specDepth = self.group["intCfg"][idx][1]
        sample.tempWidth = self.group["intCfg"][idx][2]
        sample.tempDepth = self.group["intCfg"][idx][3]
        if self.group.attrs["isTransition"]:
            sample.embedding = (self.group["intCfg"][idx][4], self.group["intCfg"][idx][5])
        else:
            sample.embedding = self.group["intCfg"][idx][4]
    
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
        sample.pitch = self.group["pitch"][idx]
        sample.avgSpecharm = torch.tensor(self.group["avgSpecharm"][idx], dtype=torch.float32)
        sample.key = self.group["keys"][idx].decode("utf-8")
        if sample.key == "_None":
            sample.key = None
        sample.isVoiced = self.group["flags"][idx][0]
        sample.isPlosive = self.group["flags"][idx][1]
        if self.group.attrs["isTransition"]:
            sample.embedding = (self.group["intCfg"][idx][4], self.group["intCfg"][idx][5])
        else:
            sample.embedding = self.group["intCfg"][idx][4]
    
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
        self.group["audioIdxs"][-1] = self.group["audioIdxs"][-2] + self.group["audio"].shape[0]
        self.group["pitchDeltas"].resize(self.group["pitchDeltas"].shape[0] + sample.pitchDeltas.shape[0], axis=0)
        self.group["pitchDeltas"][-sample.pitchDeltas.shape[0]:] = sample.pitchDeltas
        self.group["pitchDeltasIdxs"].resize(self.group["pitchDeltasIdxs"].shape[0] + 1, axis=0)
        self.group["pitchDeltasIdxs"][-1] = self.group["pitchDeltasIdxs"][-2] + self.group["pitchDeltas"].shape[0]
        self.group["pitch"].resize(self.group["pitch"].shape[0] + 1, axis=0)
        self.group["pitch"][-1] = sample.pitch
        self.group["specharm"].resize(self.group["specharm"].shape[0] + sample.specharm.shape[0], axis=0)
        self.group["specharm"][-sample.specharm.shape[0]:] = sample.specharm
        self.group["specharmIdxs"].resize(self.group["specharmIdxs"].shape[0] + 1, axis=0)
        self.group["specharmIdxs"][-1] = self.group["specharmIdxs"][-2] + self.group["specharm"].shape[0]
        self.group["avgSpecharm"].resize(self.group["avgSpecharm"].shape[0] + 1, axis=0)
        self.group["avgSpecharm"][-1] = sample.avgSpecharm
        self.group["excitation"].resize(self.group["excitation"].shape[0] + sample.excitation.shape[0], axis=0)
        self.group["excitation"][-sample.excitation.shape[0]:] = sample.excitation
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
        self.group["floatCfg"][-1] = [sample.expectedPitch, sample.searchRange, sample.voicedThrh]
        self.group["intCfg"].resize(self.group["intCfg"].shape[0] + 1, axis=0)
        if self.group.attrs["isTransition"]:
            self.group["intCfg"][-1] = [sample.specWidth, sample.specDepth, sample.tempWidth, sample.tempDepth, sample.embedding[0], sample.embedding[1]]
        else:
            self.group["intCfg"][-1] = [sample.specWidth, sample.specDepth, sample.tempWidth, sample.tempDepth, sample.embedding]
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
            self.group["pitch"] = torch.cat((self.group["pitch"][:i], self.group["pitch"][i + 1:]), dim=0)
            self.group["specharm"] = torch.cat((self.group["specharm"][:self.group["specharmIdxs"][i]], self.group["specharm"][self.group["specharmIdxs"][i + 1]:]), dim=0)
            self.group["specharmIdxs"] = torch.cat((self.group["specharmIdxs"][:i], self.group["specharmIdxs"][i + 1:]), dim=0)
            self.group["avgSpecharm"] = torch.cat((self.group["avgSpecharm"][:i], self.group["avgSpecharm"][i + 1:]), dim=0)
            self.group["excitation"] = torch.cat((self.group["excitation"][:i], self.group["excitation"][i + 1:]), dim=0)
            self.group["filepaths"] = torch.cat((self.group["filepaths"][:i], self.group["filepaths"][i + 1:]), dim=0)
            self.group["keys"] = torch.cat((self.group["keys"][:i], self.group["keys"][i + 1:]), dim=0)
            self.group["flags"] = torch.cat((self.group["flags"][:i], self.group["flags"][i + 1:]), dim=0)
            self.group["floatCfg"] = torch.cat((self.group["floatCfg"][:i], self.group["floatCfg"][i + 1:]), dim=0)
            self.group["intCfg"] = torch.cat((self.group["intCfg"][:i], self.group["intCfg"][i + 1:]), dim=0)
            self.group.attrs["length"] -= 1
        self.pendingDeletions = []
    
    def toCollection(self, dType:str = "full"):
        if dType == "full":
            collection = AudioSampleCollection()
        elif dType == "AI":
            collection = AISampleCollection()
        elif dType == "lite":
            collection = LiteSampleCollection()
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
    
    def close(self):
        self.file.close()
