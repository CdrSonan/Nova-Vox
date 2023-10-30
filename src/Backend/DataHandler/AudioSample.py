#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torchaudio
import torch
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
        
        
    def __init__(self, filepath:str) -> None:
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
            self.waveform = torch.tensor([], dtype = float)
        self.pitchDeltas = torch.tensor([], dtype = int)
        self.pitch = torch.tensor([global_consts.defaultExpectedPitch], dtype = int)
        self.specharm = torch.tensor([[]], dtype = float)
        self.avgSpecharm = torch.tensor([], dtype = float)
        self.excitation = torch.tensor([], dtype = float)
        self.isVoiced = True
        self.isPlosive = False
        
        self.embedding = 0
        
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
        
        
    def __init__(self, filepath:str) -> None:
        """Constructor for initialising an AISample based on an audio file and desired sample rate.
        
        Arguments:
            filepath: Expects a String that can be interpreted as a filepath to a .wav audio file. Determines the audio file to be loaded into the object.
            
        Returns:
            None
        
        This method initializes the properties used for spectral and excitation calculation to the default values, loads the selected audio file (based on filepath string) into the waveform property, and resamples it to the desired sample rate"""

        
        loadedData = torchaudio.load(filepath)
        self.filepath = filepath
        self.waveform = loadedData[0][0]
        sampleRate = loadedData[1]
        del loadedData
        transform = torchaudio.transforms.Resample(sampleRate, global_consts.sampleRate)
        self.waveform = transform(self.waveform)
        del transform
        del sampleRate
        self.isVoiced = True
        self.isPlosive = False
        
        self.embedding = 0
        
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
    
    
    def __init__(self, audioSample:AudioSample) -> None:
        """Constructor for initialising the class based on an AudioSample object, discarding all extraneous data.
        
        Arguments:
            audioSample: AudioSample base object
            
        Returns:
            None"""
            
            
        self.pitchDeltas = audioSample.pitchDeltas
        self.pitch = audioSample.pitch
        self.specharm = audioSample.specharm
        self.avgSpecharm = audioSample.avgSpecharm
        self.excitation = audioSample.excitation
        self.isVoiced = audioSample.isVoiced
        self.isPlosive = audioSample.isPlosive
        self.embedding = audioSample.embedding
