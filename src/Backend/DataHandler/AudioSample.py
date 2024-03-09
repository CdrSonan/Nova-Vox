#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torchaudio
import torch
import torch.utils.data as torch_data

import global_consts

class AudioSample():
    """class for holding a single recording audio sample and processing it for usage in a Voicebank.
    
    Attributes:
        filepath: original filepath of the audio sample. used for reloading files.
        
        waveform: original audio waveform
        
        pitchDeltas: duration of each vocal chord vibration in data points
        
        pitch: average pitch of the sample given as wavelength in data points
        
        specharm: deviations of the audio spectrum and harmonics from the average during each engine tick
        
        avgSpecharm: audio spectrum and harmonics averaged across the entire sample
        
        excitation: unvoiced excitation signal
        
        voicedExcitation: voiced excitation signal

        isVoiced: flag indicating whether the sample is voiced (voicedExcitation is muted for unvoiced samples during ESPER processing)

        isPlosive: flag indicating whether the sample is considered a plosive sound (if possible, plosives retain their original length after border creation during synthesis)
        
        expectedPitch: estimated pitch of the sample in Hz. Must be manually filled and is used during ESPER.calculatePitch() calls.
        
        searchRange: pitch frequency search range relative to expectedPitch. Should be a value between 0 and 1. Must be manually filled and is used during ESPER.calculatePitch() calls.
        
        specWidth, specDepth, tempWidth, tempDepth: width and depth used for spectral smoothing when calling ESPER.calculateSpectra(), along the frequency dimension of each stft batch, and the time dimension between the batches, respectively.
    
    Methods:
        __init__: Constructor for initialising an AudioSample based on an audio file"""
        
        
    def __init__(self, filepath:str = None, isTransition:bool = False) -> None:
        """Constructor for initialising an AudioSample based on an audio file and desired sample rate.
        
        Arguments:
            filepath: Expects a String that can be interpreted as a filepath to a .wav audio file. Determines the audio file to be loaded into the object.
            
        Returns:
            None
        
        This method initializes the properties used for spectral and excitation calculation to the default values and initialises all attributes with empty objects.
        Loads the selected audio file (based on filepath string) into the waveform property and resamples it to the desired sample rate"""
        
        
        if filepath not in (None, ""):
            loadedData = torchaudio.load(filepath)
            self.filepath = filepath
            self.waveform = loadedData[0][0]
            sampleRate = loadedData[1]
            del loadedData
            transform = torchaudio.transforms.Resample(sampleRate, global_consts.sampleRate)
            self.waveform = transform(self.waveform)
            del transform
            del sampleRate
        else:
            self.filepath = ""
            self.waveform = torch.tensor([], dtype = torch.float)
        self.pitchDeltas = torch.tensor([], dtype = torch.int)
        self.pitch = torch.tensor([global_consts.defaultExpectedPitch,], dtype = torch.int)
        self.specharm = torch.empty((0, global_consts.frameSize), dtype = torch.float)
        self.avgSpecharm = torch.empty((0, global_consts.reducedFrameSize), dtype = torch.float)
        self.excitation = torch.empty((0, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64)
        self.isVoiced = True
        self.isPlosive = False
        
        if isTransition:
            self.embedding = (0, 0)
        else:
            self.embedding = 0
        self.key = None
        
        self.expectedPitch = global_consts.defaultExpectedPitch
        self.searchRange = global_consts.defaultSearchRange
        self.voicedThrh = global_consts.defaultVoicedThrh
        self.specWidth = global_consts.defaultSpecWidth
        self.specDepth = global_consts.defaultSpecDepth
        self.tempWidth = global_consts.defaultTempWidth
        self.tempDepth = global_consts.defaultTempDepth

class AISample():
    """class for holding a single recording audio sample and relevant settings. Saves memory compared to an AudioSample, but must be converted to one before use. Mainly used as a container for AI training samples.
    
    Attributes:
        filepath: original filepath of the audio sample. used for reloading files.
        
        waveform: original audio waveform

        isVoiced: flag indicating whether the sample is voiced (voicedExcitation is muted for unvoiced samples during ESPER processing)

        isPlosive: flag indicating whether the sample is considered a plosive sound (if possible, plosives retain their original length after border creation during synthesis)
        
        expectedPitch: estimated pitch of the sample in Hz. Must be manually filled and is used during ESPER.calculatePitch() calls.
        
        searchRange: pitch frequency search range relative to expectedPitch. Should be a value between 0 and 1. Must be manually filled and is used during ESPER.calculatePitch() calls.
        
        specWidth, specDepth, tempWidth, tempDepth: width and depth used for spectral smoothing when calling ESPER.calculateSpectra(), along the frequency dimension of each stft batch, and the time dimension between the batches, respectively.
    
    Methods:
        __init__: Constructor for initialising an AudioSample based on an audio file"""
        
        
    def __init__(self, filepath:str = None, isTransition:bool = False) -> None:
        """Constructor for initialising an AISample based on an audio file and desired sample rate.
        
        Arguments:
            filepath: Expects a String that can be interpreted as a filepath to a .wav audio file. Determines the audio file to be loaded into the object.
            
        Returns:
            None
        
        This method initializes the properties used for spectral and excitation calculation to the default values, loads the selected audio file (based on filepath string) into the waveform property, and resamples it to the desired sample rate"""

        
        if filepath not in (None, ""):
            loadedData = torchaudio.load(filepath)
            self.filepath = filepath
            self.waveform = loadedData[0][0]
            sampleRate = loadedData[1]
            del loadedData
            transform = torchaudio.transforms.Resample(sampleRate, global_consts.sampleRate)
            self.waveform = transform(self.waveform)
            del transform
            del sampleRate
        else:
            self.filepath = ""
            self.waveform = torch.tensor([], dtype = torch.float)
        self.isVoiced = True
        self.isPlosive = False
        
        if isTransition:
            self.embedding = (0, 0)
        else:
            self.embedding = 0
        self.key = ""
        
        self.expectedPitch = global_consts.defaultExpectedPitch
        self.searchRange = global_consts.defaultSearchRange
        self.voicedThrh = global_consts.defaultVoicedThrh
        self.specWidth = global_consts.defaultSpecWidth
        self.specDepth = global_consts.defaultSpecDepth
        self.tempWidth = global_consts.defaultTempWidth
        self.tempDepth = global_consts.defaultTempDepth

    def convert(self, batch:bool = False):
        """converts the AISample sample to an AudioSample instance. If the batch flag is set, the waveform is divided and the parts are wrapped into individual AUdioSample instances, which are then returned as a list."""
        def createAudioSample() -> AudioSample:
            audioSample = AudioSample(None)
            audioSample.waveform = self.waveform
            audioSample.isVoiced = self.isVoiced
            audioSample.isPlosive = self.isPlosive
            audioSample.embedding = self.embedding
            audioSample.key = self.key
            audioSample.expectedPitch = self.expectedPitch
            audioSample.searchRange = self.searchRange
            audioSample.voicedThrh = self.voicedThrh
            audioSample.specWidth = self.specWidth
            audioSample.specDepth = self.specDepth
            audioSample.tempWidth = self.tempWidth
            audioSample.tempDepth = self.tempDepth
            return audioSample
        if batch:
            audioSamples = []
            while self.waveform.size()[0] > global_consts.LSTMBatchSize:
                audioSample = createAudioSample()
                audioSample.waveform = audioSample.waveform[:global_consts.LSTMBatchSize]
                self.waveform = self.waveform[global_consts.LSTMBatchSize:]
                audioSamples.append(audioSample)
            audioSamples.append(createAudioSample())
            return audioSamples
        return createAudioSample()
        
class LiteAudioSample():
    """A stripped down version of AudioSample only holding the data required for synthesis.
    
    Attributes:
        pitchDeltas: duration of each vocal chord vibration in data points
        
        pitch: average pitch of the sample given as wavelength in data points
        
        specharm: deviations of the audio spectrum and harmonics from the average during each engine tick
        
        avgSpecharm: audio spectrum and harmonics averaged across the entire sample
        
        excitation: unvoiced excitation signal
        
        voicedExcitation: voiced excitation signal
        
    Methods:
        __init__: Constructor for initialising the class based on an AudioSample object, discarding all extraneous data."""
    
    
    def __init__(self, audioSample:AudioSample = None) -> None:
        """Constructor for initialising the class based on an AudioSample object, discarding all extraneous data.
        
        Arguments:
            audioSample: AudioSample base object
            
        Returns:
            None"""
            
            
        if audioSample is None:
            self.pitchDeltas = torch.tensor([], dtype = torch.int)
            self.pitch = torch.tensor([], dtype = torch.int)
            self.specharm = torch.tensor([], dtype = torch.float)
            self.avgSpecharm = torch.tensor([], dtype = torch.float)
            self.excitation = torch.tensor([], dtype = torch.complex64)
            self.key = None
            self.isVoiced = False
            self.isPlosive = False
            self.embedding = 0
        else:
            self.pitchDeltas = audioSample.pitchDeltas
            self.pitch = audioSample.pitch
            self.specharm = audioSample.specharm
            self.avgSpecharm = audioSample.avgSpecharm
            self.excitation = audioSample.excitation
            self.key = audioSample.key
            self.isVoiced = audioSample.isVoiced
            self.isPlosive = audioSample.isPlosive
            self.embedding = audioSample.embedding

def addElement(tensor:torch.Tensor, element:torch.Tensor, length:int) -> torch.Tensor:
    if length == 0:
        if isinstance(element, torch.Tensor):
            return element.unsqueeze(0)
        elif isinstance(element, float):
            return torch.tensor([element,], dtype = torch.float)
        else:
            return torch.tensor([element,], dtype = torch.int64)
    if tensor.size()[0] <= length:
        tensor = torch.cat([tensor, torch.empty_like(tensor)], 0).contiguous()
    tensor[length] = element
    return tensor

def addIdx(tensor:torch.Tensor, idx:int, length:int) -> torch.Tensor:
    if length == 0:
        return torch.tensor([idx,], dtype = torch.int64)
    if tensor.size()[0] <= length:
        tensor = torch.cat([tensor, torch.empty_like(tensor)], 0).contiguous()
    tensor[length] = tensor[length - 1] + idx
    return tensor

def addElementWithIdxs(tensor:torch.Tensor, element:torch.Tensor, idxs:torch.Tensor, length:int) -> torch.Tensor:
    if length == 0:
        return element
    while tensor.size()[0] <= idxs[length - 1] + element.size()[0]:
        tensor = torch.cat([tensor, torch.empty_like(tensor)], 0).contiguous()
    tensor[idxs[length - 1]:idxs[length - 1] + element.size()[0]] = element
    return tensor

class AISampleCollection(torch_data.Dataset):
    """Class for holding a collection of AISample instances."""
    
    def __init__(self, source:list = None, isTransition:bool = False) -> None:
        if source is None:
            self.audio = torch.empty([0,], dtype = torch.float)
            self.audioIdxs = torch.empty([0,], dtype = torch.int64)
            self.filepaths = []
            self.keys = []
            self.flags = torch.empty([0, 2], dtype = torch.bool)
            self.floatCfg = torch.empty([0, 3], dtype = torch.float32)
            if isTransition:
                self.intCfg = torch.empty([0, 6], dtype = torch.int16)
            else:
                self.intCfg = torch.empty([0, 5], dtype = torch.int16)
            self.length = 0
        else:
            self.audio = torch.cat([i.waveform for i in source], 0)
            self.audioIdxs = torch.tensor([i.waveform.size()[0] for i in source], dtype = torch.int64)
            self.audioIdxs = torch.cumsum(self.audioIdxs, 0)
            self.filepaths = [i.filepath for i in source]
            self.keys = [i.key for i in source]
            self.flags = torch.tensor([[i.isVoiced, i.isPlosive] for i in source], dtype = torch.bool)
            self.floatCfg = torch.tensor([[i.expectedPitch, i.searchRange, i.voicedThrh] for i in source], dtype = torch.float32)
            if isTransition:
                self.intCfg = torch.tensor([[i.specWidth, i.specDepth, i.tempWidth, i.tempDepth, i.embedding[0], i.embedding[1]] for i in source], dtype = torch.int64)
            else:
                self.intCfg = torch.tensor([[i.specWidth, i.specDepth, i.tempWidth, i.tempDepth, i.embedding] for i in source], dtype = torch.int64)
            self.length = len(source)
        self.pendingDeletions = []
        self.isTransition = isTransition
        self.iterIdx = 0
    
    def __getitem__(self, idx:int) -> AISample:
        if idx < 0:
            idx += self.length
        for i in self.pendingDeletions:
            if i <= idx:
                idx += 1
        if idx >= self.length:
            raise IndexError("Index out of range")
        sample = AISample()
        if idx == 0:
            sample.waveform = self.audio.narrow(0, 0, self.audioIdxs[idx])
        else:
            sample.waveform = self.audio.narrow(0, self.audioIdxs[idx - 1], self.audioIdxs[idx] - self.audioIdxs[idx - 1])
        sample.filepath = self.filepaths[idx]
        sample.key = self.keys[idx]
        sample.isVoiced = self.flags.select(0, idx).select(0, 0)
        sample.isPlosive = self.flags.select(0, idx).select(0, 1)
        sample.expectedPitch = self.floatCfg.select(0, idx).select(0, 0)
        sample.searchRange = self.floatCfg.select(0, idx).select(0, 1)
        sample.voicedThrh = self.floatCfg.select(0, idx).select(0, 2)
        sample.specWidth = self.intCfg.select(0, idx).select(0, 0)
        sample.specDepth = self.intCfg.select(0, idx).select(0, 1)
        sample.tempWidth = self.intCfg.select(0, idx).select(0, 2)
        sample.tempDepth = self.intCfg.select(0, idx).select(0, 3)
        if self.isTransition:
            sample.embedding = (self.intCfg.select(0, idx).select(0, 4), self.intCfg.select(0, idx).select(0, 5))
        else:
            sample.embedding = self.intCfg.select(0, idx).select(0, 4)
        return sample
    
    def __len__(self) -> int:
        return self.length
    
    def __iter__(self):
        self.iterIdx = 0
        return self
    
    def __next__(self) -> LiteAudioSample:
        if self.iterIdx >= self.length:
            raise StopIteration
        sample = self.__getitem__(self.iterIdx)
        self.iterIdx += 1
        return sample
    
    def append(self, sample:AISample) -> None:
        """Appends an AISample instance to the collection."""
        self.audio = addElementWithIdxs(self.audio, sample.waveform, self.audioIdxs, self.length)
        self.audioIdxs = addIdx(self.audioIdxs, sample.waveform.size()[0], self.length)
        self.filepaths.append(sample.filepath)
        self.keys.append(sample.key)
        self.flags = addElement(self.flags, torch.tensor([sample.isVoiced, sample.isPlosive], dtype = torch.bool), self.length)
        self.floatCfg = addElement(self.floatCfg, torch.tensor([sample.expectedPitch, sample.searchRange, sample.voicedThrh], dtype = torch.float32), self.length)
        if self.isTransition:
            self.intCfg = addElement(self.intCfg, torch.tensor([sample.specWidth, sample.specDepth, sample.tempWidth, sample.tempDepth, sample.embedding[0], sample.embedding[1]], dtype = torch.int64), self.length)
        else:
            self.intCfg = addElement(self.intCfg, torch.tensor([sample.specWidth, sample.specDepth, sample.tempWidth, sample.tempDepth, sample.embedding], dtype = torch.int64), self.length)
        self.length += 1
    
    def delete(self, idx:int) -> None:
        if idx < 0:
            idx += self.length
        self.pendingDeletions.append(idx)
    
    def commitDeletions(self) -> None:
        if len(self.pendingDeletions) == 0:
            return
        if len(self.pendingDeletions) == self.length:
            self.__init__(None, self.isTransition)
        if len(self.pendingDeletions) > self.length / 2:
            idxs = list(range(self.length))
            for i in self.pendingDeletions:
                idxs.pop(i)
            self.audio = torch.cat([self.audio[:self.audioIdxs[i]] if i == 0 else self.audio[self.audioIdxs[i - 1]:self.audioIdxs[i]] for i in idxs], 0)
            self.audioIdxs = torch.cat([self.audioIdxs[i] for i in idxs], 0)
            self.filepaths = [self.filepaths[i] for i in idxs]
            self.keys = [self.keys[i] for i in idxs]
            self.flags = torch.cat([self.flags[i] for i in idxs], 0)
            self.floatCfg = torch.cat([self.floatCfg[i] for i in idxs], 0)
            self.intCfg = torch.cat([self.intCfg[i] for i in idxs], 0)
            self.length = len(idxs)
        else:
            for i in self.pendingDeletions:
                if i == 0:
                    self.audio = self.audio[self.audioIdxs[i]:]
                else:
                    self.audio = torch.cat([self.audio[:self.audioIdxs[i - 1]], self.audio[self.audioIdxs[i]:]], 0)
                self.audioIdxs = torch.cat([self.audioIdxs[:i], self.audioIdxs[i + 1:]], 0)
                self.filepaths.pop(i)
                self.keys.pop(i)
                self.flags = torch.cat([self.flags[:i], self.flags[i + 1:]], 0)
                self.floatCfg = torch.cat([self.floatCfg[:i], self.floatCfg[i + 1:]], 0)
                self.intCfg = torch.cat([self.intCfg[:i], self.intCfg[i + 1:]], 0)
                self.length -= 1
        self.pendingDeletions = []

class AudioSampleCollection(torch_data.Dataset):
    """Class for holding a collection of AudioSample instances."""
    
    def __init__(self, source:list = None, isTransition:bool = False) -> None:
        if source is None:
            self.audio = torch.empty([0,], dtype = torch.float)
            self.audioIdxs = torch.empty([0,], dtype = torch.int64)
            self.pitchDeltas = torch.empty([0,], dtype = torch.int)
            self.pitchDeltaIdxs = torch.empty([0,], dtype = torch.int64)
            self.pitch = torch.empty([0,], dtype = torch.int)
            self.specharm = torch.empty([0, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3], dtype = torch.float)
            self.specharmIdxs = torch.empty([0,], dtype = torch.int64)
            self.avgSpecharm = torch.empty([0, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3], dtype = torch.float)
            self.excitation = torch.empty([0, global_consts.halfTripleBatchSize + 1], dtype = torch.complex64)
            self.filepaths = []
            self.keys = []
            self.flags = torch.empty([0, 2], dtype = torch.bool)
            self.floatCfg = torch.empty([0, 3], dtype = torch.float32)
            if isTransition:
                self.intCfg = torch.empty([0, 6], dtype = torch.int16)
            else:
                self.intCfg = torch.empty([0, 5], dtype = torch.int16)
            self.length = 0
        else:
            self.audio = torch.cat([i.waveform for i in source], 0)
            self.audioIdxs = torch.tensor([i.waveform.size()[0] for i in source], dtype = torch.int64)
            self.audioIdxs = torch.cumsum(self.audioIdxs, 0)
            self.pitchDeltas = torch.cat([i.pitchDeltas for i in source], 0)
            self.pitchDeltaIdxs = torch.tensor([i.pitchDeltas.size()[0] for i in source], dtype = torch.int64)
            self.pitchDeltaIdxs = torch.cumsum(self.pitchDeltaIdxs, 0)
            self.pitch = torch.cat([i.pitch for i in source], 0)
            self.specharm = torch.cat([i.specharm for i in source], 0)
            self.specharmIdxs = torch.tensor([i.specharm.size()[0] for i in source], dtype = torch.int64)
            self.specharmIdxs = torch.cumsum(self.specharmIdxs, 0)
            self.avgSpecharm = torch.cat([i.avgSpecharm for i in source], 0)
            self.excitation = torch.cat([i.excitation for i in source], 0)
            self.filepaths = [i.filepath for i in source]
            self.keys = [i.key for i in source]
            self.flags = torch.tensor([[i.isVoiced, i.isPlosive] for i in source], dtype = torch.bool)
            self.floatCfg = torch.tensor([[i.expectedPitch, i.searchRange, i.voicedThrh] for i in source], dtype = torch.float32)
            if isTransition:
                self.intCfg = torch.tensor([[i.specWidth, i.specDepth, i.tempWidth, i.tempDepth, i.embedding[0], i.embedding[1]] for i in source], dtype = torch.int64)
            else:
                self.intCfg = torch.tensor([[i.specWidth, i.specDepth, i.tempWidth, i.tempDepth, i.embedding] for i in source], dtype = torch.int64)
            self.length = len(source)
        self.pendingDeletions = []
        self.isTransition = isTransition
        self.iterIdx = 0
    
    def __getitem__(self, idx:int) -> AudioSample:
        if idx < 0:
            idx += self.length
        for i in self.pendingDeletions:
            if i <= idx:
                idx += 1
        if idx >= self.length:
            raise IndexError("Index out of range")
        sample = AudioSample()
        if idx == 0:
            sample.waveform = self.audio.narrow(0, 0, self.audioIdxs[idx])
        else:
            sample.waveform = self.audio.narrow(0, self.audioIdxs[idx - 1], self.audioIdxs[idx] - self.audioIdxs[idx - 1])
        sample.filepath = self.filepaths[idx]
        if idx == 0:
            sample.pitchDeltas = self.pitchDeltas.narrow(0, 0, self.pitchDeltaIdxs[idx])
        else:
            sample.pitchDeltas = self.pitchDeltas.narrow(0, self.pitchDeltaIdxs[idx - 1], self.pitchDeltaIdxs[idx] - self.pitchDeltaIdxs[idx - 1])
        sample.pitch = self.pitch.select(0, idx)
        if idx == 0:
            sample.specharm = self.specharm.narrow(0, 0, self.specharmIdxs[idx])
        else:
            sample.specharm = self.specharm.narrow(0, self.specharmIdxs[idx - 1], self.specharmIdxs[idx] - self.specharmIdxs[idx - 1])
        sample.avgSpecharm = self.avgSpecharm.select(0, idx)
        if idx == 0:
            sample.excitation = self.excitation.narrow(0, 0, self.specharmIdxs[idx])
        else:
            sample.excitation = self.excitation.narrow(0, self.specharmIdxs[idx - 1], self.specharmIdxs[idx] - self.specharmIdxs[idx - 1])
        sample.key = self.keys[idx]
        sample.isVoiced = self.flags.select(0, idx).select(0, 0)
        sample.isPlosive = self.flags.select(0, idx).select(0, 1)
        sample.expectedPitch = self.floatCfg.select(0, idx).select(0, 0)
        sample.searchRange = self.floatCfg.select(0, idx).select(0, 1)
        sample.voicedThrh = self.floatCfg.select(0, idx).select(0, 2)
        sample.specWidth = self.intCfg.select(0, idx).select(0, 0)
        sample.specDepth = self.intCfg.select(0, idx).select(0, 1)
        sample.tempWidth = self.intCfg.select(0, idx).select(0, 2)
        sample.tempDepth = self.intCfg.select(0, idx).select(0, 3)
        if self.isTransition:
            sample.embedding = (self.intCfg.select(0, idx).select(0, 4), self.intCfg.select(0, idx).select(0, 5))
        else:
            sample.embedding = self.intCfg.select(0, idx).select(0, 4)
        return sample
    
    def __len__(self) -> int:
        return self.length
    
    def __iter__(self):
        self.iterIdx = 0
        return self
    
    def __next__(self) -> LiteAudioSample:
        if self.iterIdx >= self.length:
            raise StopIteration
        sample = self.__getitem__(self.iterIdx)
        self.iterIdx += 1
        return sample
    
    def append(self, sample:AudioSample) -> None:
        """Appends an AISample instance to the collection."""
        self.audio = addElementWithIdxs(self.audio, sample.waveform, self.audioIdxs, self.length)
        self.audioIdxs = addIdx(self.audioIdxs, sample.waveform.size()[0], self.length)
        self.pitchDeltas = addElementWithIdxs(self.pitchDeltas, sample.pitchDeltas, self.pitchDeltaIdxs, self.length)
        self.pitchDeltaIdxs = addIdx(self.pitchDeltaIdxs, sample.pitchDeltas.size()[0], self.length)
        self.pitch = addElement(self.pitch, sample.pitch, self.length)
        self.specharm = addElementWithIdxs(self.specharm, sample.specharm, self.specharmIdxs, self.length)
        self.specharmIdxs = addIdx(self.specharmIdxs, sample.specharm.size()[0], self.length)
        self.avgSpecharm = addElement(self.avgSpecharm, sample.avgSpecharm, self.length)
        self.excitation = addElementWithIdxs(self.excitation, sample.excitation, self.specharmIdxs, self.length)
        self.filepaths.append(sample.filepath)
        self.keys.append(sample.key)
        self.flags = addElement(self.flags, torch.tensor([sample.isVoiced, sample.isPlosive], dtype = torch.bool), self.length)
        self.floatCfg = addElement(self.floatCfg, torch.tensor([sample.expectedPitch, sample.searchRange, sample.voicedThrh], dtype = torch.float32), self.length)
        if self.isTransition:
            self.intCfg = addElement(self.intCfg, torch.tensor([sample.specWidth, sample.specDepth, sample.tempWidth, sample.tempDepth, sample.embedding[0], sample.embedding[1]], dtype = torch.int64), self.length)
        else:
            self.intCfg = addElement(self.intCfg, torch.tensor([sample.specWidth, sample.specDepth, sample.tempWidth, sample.tempDepth, sample.embedding], dtype = torch.int64), self.length)
        self.length += 1
    
    def delete(self, idx:int) -> None:
        if idx < 0:
            idx += self.length
        self.pendingDeletions.append(idx)
    
    def commitDeletions(self) -> None:
        if len(self.pendingDeletions) == self.length:
            self.__init__(None, self.isTransition)
        if len(self.pendingDeletions) == 0:
            return
        if len(self.pendingDeletions) > self.length / 2:
            idxs = list(range(self.length))
            for i in self.pendingDeletions:
                idxs.pop(i)
            self.audio = torch.cat([self.audio[:self.audioIdxs[i]] if i == 0 else self.audio[self.audioIdxs[i - 1]:self.audioIdxs[i]] for i in idxs], 0)
            self.audioIdxs = torch.cat([self.audioIdxs[i] for i in idxs], 0)
            self.pitchDeltas = torch.cat([self.pitchDeltas[:self.pitchDeltaIdxs[i]] if i == 0 else self.pitchDeltas[self.pitchDeltaIdxs[i - 1]:self.pitchDeltaIdxs[i]] for i in idxs], 0)
            self.pitchDeltaIdxs = torch.cat([self.pitchDeltaIdxs[i] for i in idxs], 0)
            self.pitch = torch.cat([self.pitch[i] for i in idxs], 0)
            self.specharm = torch.cat([self.specharm[:self.specharmIdxs[i]] if i == 0 else self.specharm[self.specharmIdxs[i - 1]:self.specharmIdxs[i]] for i in idxs], 0)
            self.specharmIdxs = torch.cat([self.specharmIdxs[i] for i in idxs], 0)
            self.avgSpecharm = torch.cat([self.avgSpecharm[i] for i in idxs], 0)
            self.excitation = torch.cat([self.excitation[:self.specharmIdxs[i]] if i == 0 else self.excitation[self.specharmIdxs[i - 1]:self.specharmIdxs[i]] for i in idxs], 0)
            self.filepaths = [self.filepaths[i] for i in idxs]
            self.keys = [self.keys[i] for i in idxs]
            self.flags = torch.cat([self.flags[i] for i in idxs], 0)
            self.floatCfg = torch.cat([self.floatCfg[i] for i in idxs], 0)
            self.intCfg = torch.cat([self.intCfg[i] for i in idxs], 0)
            self.length = len(idxs)
        else:
            for i in self.pendingDeletions:
                if i == 0:
                    self.audio = self.audio[self.audioIdxs[i]:]
                else:
                    self.audio = torch.cat([self.audio[:self.audioIdxs[i - 1]], self.audio[self.audioIdxs[i]:]], 0)
                self.audioIdxs = torch.cat([self.audioIdxs[:i], self.audioIdxs[i + 1:]], 0)
                self.pitchDeltas = self.pitchDeltas[self.pitchDeltaIdxs[i]:] if i == 0 else torch.cat([self.pitchDeltas[:self.pitchDeltaIdxs[i - 1]], self.pitchDeltas[self.pitchDeltaIdxs[i]:]], 0)
                self.pitchDeltaIdxs = torch.cat([self.pitchDeltaIdxs[:i], self.pitchDeltaIdxs[i + 1:]], 0)
                self.pitch = torch.cat([self.pitch[:i], self.pitch[i + 1:]], 0)
                self.specharm = self.specharm[self.specharmIdxs[i]:] if i == 0 else torch.cat([self.specharm[:self.specharmIdxs[i - 1]], self.specharm[self.specharmIdxs[i]:]], 0)
                self.specharmIdxs = torch.cat([self.specharmIdxs[:i], self.specharmIdxs[i + 1:]], 0)
                self.avgSpecharm = torch.cat([self.avgSpecharm[:i], self.avgSpecharm[i + 1:]], 0)
                self.excitation = self.excitation[self.specharmIdxs[i]:] if i == 0 else torch.cat([self.excitation[:self.specharmIdxs[i - 1]], self.excitation[self.specharmIdxs[i]:]], 0)
                self.filepaths.pop(i)
                self.keys.pop(i)
                self.flags = torch.cat([self.flags[:i], self.flags[i + 1:]], 0)
                self.floatCfg = torch.cat([self.floatCfg[:i], self.floatCfg[i + 1:]], 0)
                self.intCfg = torch.cat([self.intCfg[:i], self.intCfg[i + 1:]], 0)
                self.length -= 1
        self.pendingDeletions = []

class LiteSampleCollection(torch_data.Dataset):
    """Class for holding a collection of AudioSample instances."""
    
    def __init__(self, source:list = None, isTransition:bool = False) -> None:
        if source is None:
            self.pitchDeltas = torch.empty([0,], dtype = torch.int)
            self.pitchDeltaIdxs = torch.empty([0,], dtype = torch.int64)
            self.pitch = torch.empty([0,], dtype = torch.int)
            self.specharm = torch.empty([0, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3], dtype = torch.float)
            self.specharmIdxs = torch.empty([0,], dtype = torch.int64)
            self.avgSpecharm = torch.empty([0, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3], dtype = torch.float)
            self.excitation = torch.empty([0, global_consts.halfTripleBatchSize + 1], dtype = torch.complex64)
            self.keys = []
            self.flags = torch.empty([0, 2], dtype = torch.bool)
            if isTransition:
                self.embedding = torch.empty([0, 2], dtype = torch.int64)
            else:
                self.embedding = torch.empty([0,], dtype = torch.int64)
            self.length = 0
        else:
            self.pitchDeltas = torch.cat([i.pitchDeltas for i in source], 0)
            self.pitchDeltaIdxs = torch.tensor([i.pitchDeltas.size()[0] for i in source], dtype = torch.int64)
            self.pitchDeltaIdxs = torch.cumsum(self.pitchDeltaIdxs, 0)
            self.pitch = torch.cat([i.pitch for i in source], 0)
            self.specharm = torch.cat([i.specharm for i in source], 0)
            self.specharmIdxs = torch.tensor([i.specharm.size()[0] for i in source], dtype = torch.int64)
            self.specharmIdxs = torch.cumsum(self.specharmIdxs, 0)
            self.avgSpecharm = torch.cat([i.avgSpecharm for i in source], 0)
            self.excitation = torch.cat([i.excitation for i in source], 0)
            self.keys = [i.key for i in source]
            self.flags = torch.tensor([[i.isVoiced, i.isPlosive] for i in source], dtype = torch.bool)
            if isTransition:
                self.embedding = torch.tensor([[i.embedding[0], i.embedding[1]] for i in source], dtype = torch.int64)
            else:
                self.embedding = torch.tensor([i.embedding for i in source], dtype = torch.int64)
            self.length = len(source)
        self.pendingDeletions = []
        self.isTransition = isTransition
        self.iterIdx = 0
    
    def fetch(self, key:str, byKey:bool = False) -> LiteAudioSample:
        if byKey:
            idx = [i for i in range(self.length) if self.keys[i] == key]
            return [self.fetch(i, False) for i in idx]
        else:
            idx = key
        if idx < 0:
            idx += self.length
        for i in self.pendingDeletions:
            if i <= idx:
                idx += 1
        if idx >= self.length:
            raise IndexError("Index out of range")
        sample = LiteAudioSample()
        if idx == 0:
            sample.pitchDeltas = self.pitchDeltas.narrow(0, 0, self.pitchDeltaIdxs[idx])
        else:
            sample.pitchDeltas = self.pitchDeltas.narrow(0, self.pitchDeltaIdxs[idx - 1], self.pitchDeltaIdxs[idx] - self.pitchDeltaIdxs[idx - 1])
        sample.pitch = self.pitch.select(0, idx)
        if idx == 0:
            sample.specharm = self.specharm.narrow(0, 0, self.specharmIdxs[idx])
        else:
            sample.specharm = self.specharm.narrow(0, self.specharmIdxs[idx - 1], self.specharmIdxs[idx] - self.specharmIdxs[idx - 1])
        sample.avgSpecharm = self.avgSpecharm.select(0, idx)
        if idx == 0:
            sample.excitation = self.excitation.narrow(0, 0, self.specharmIdxs[idx])
        else:
            sample.excitation = self.excitation.narrow(0, self.specharmIdxs[idx - 1], self.specharmIdxs[idx] - self.specharmIdxs[idx - 1])
        sample.key = self.keys[idx]
        sample.isVoiced = self.flags.select(0, idx).select(0, 0)
        sample.isPlosive = self.flags.select(0, idx).select(0, 1)
        sample.embedding = self.embedding.select(0, idx)
        return sample
    
    def __getitem__(self, index) -> LiteAudioSample:
        return self.fetch(index, False)
    
    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        self.iterIdx = 0
        return self
    
    def __next__(self) -> LiteAudioSample:
        if self.iterIdx >= self.length:
            raise StopIteration
        sample = self.__getitem__(self.iterIdx)
        self.iterIdx += 1
        return sample

    def append(self, sample:AudioSample) -> None:
        """Appends an AISample instance to the collection."""
        self.pitchDeltas = addElementWithIdxs(self.pitchDeltas, sample.pitchDeltas, self.pitchDeltaIdxs, self.length)
        self.pitchDeltaIdxs = addIdx(self.pitchDeltaIdxs, sample.pitchDeltas.size()[0], self.length)
        self.pitch = addElement(self.pitch, sample.pitch, self.length)
        self.specharm = addElementWithIdxs(self.specharm, sample.specharm, self.specharmIdxs, self.length)
        self.specharmIdxs = addIdx(self.specharmIdxs, sample.specharm.size()[0], self.length)
        self.avgSpecharm = addElement(self.avgSpecharm, sample.avgSpecharm, self.length)
        self.excitation = addElementWithIdxs(self.excitation, sample.excitation, self.specharmIdxs, self.length)
        self.keys.append(sample.key)
        self.flags = addElement(self.flags, torch.tensor([sample.isVoiced, sample.isPlosive], dtype = torch.bool), self.length)
        if self.isTransition:
            self.embedding = addElement(self.embedding, torch.tensor([sample.embedding[0], sample.embedding[1]], dtype = torch.int64), self.length)
        else:
            self.embedding = addElement(self.embedding, torch.tensor([sample.embedding,], dtype = torch.int64), self.length)
        self.length += 1
