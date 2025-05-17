#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import torch
from tqdm.auto import tqdm
import h5py
import random

from Backend.VB_Components.VbMetadata import VbMetadata
from Backend.VB_Components.Ai.Wrapper import AIWrapper
from Backend.DataHandler.AudioSample import AudioSample, LiteAudioSample, AISample, AISampleCollection, LiteSampleCollection
from Backend.ESPER.PitchCalculator import calculatePitch
from Backend.ESPER.SpectralCalculator import calculateSpectra, asyncProcess
from Backend.DataHandler.UtauSample import UtauSample
from Backend.DataHandler.HDF5 import SampleStorage, DictStorage, WordStorage, MetadataStorage
import global_consts

class Voicebank():
    """Class for holding a Voicebank as handled by the devkit.
    
    Attributes:
        metadata: VbMetadata object containing the Voicebank's metadata
        
        filepath: The filepath to the Voicebank's file
        
        phonemeDict: a Dictionary object containing the samples for the individual phonemes
        
        ai: Wrapper for all mandatory AI-driven components of the Voicebank
        
        parameters: currently a placeholder. Will contain the Voicebank's Ai-driven parameters.
        
        wordDict: a Dictionary containing overrides for Nova-Vox's default dictionary
        
        stagedCrfTrainSamples: a List object containing the samples staged to be used in phoneme crossfade Ai training

        stagedPredTrainSamples: a List object containing the samples staged to be used in prediction Ai training
        
    Functions:
        __init__: Universal constructor for initialisation both from a Voicebank file, and of an empty/new Voicebank
        
        save: saves the loaded Voicebank to a file
        
        loadMetadata: loads Voicebank Metadata from a Voicebank file
        
        loadPhonemeDict: loads Phoneme data from a Voicebank file
        
        loadCrfWeights: loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai

        loadPredWeights: loads the Ai state saved in a Voicebank file into the loadedVoicebank's prediction Ai
        
        loadParameters: currently placeholder
        
        loadWordDict: currently placeholder
        
        addPHoneme: adds a phoneme to the Voicebank's PhonemeDict
        
        delPhoneme: deletes a Phoneme from the vVoicebank's PhonemeDict
        
        changePhonemeKey: changes the key by which a phoneme in the Voicebank's phonemeDict can be accessed, but leaves the rest of its data unchanged
        
        changePhonemeFile: changes the audio file used for a phoneme in the Voicebank's PhonemeDict
        
        finalizePhoneme: finalizes a Phoneme, discarding any data related to it that's not strictly required for synthesis
        
        addCrfTrainSample: stages an audio sample the phoneme crossfade Ai is to be trained with
        
        delCrfTrainSampled: removes an audio sample from the list of staged crossfade Ai training phonemes
        
        changeCrfTrainSampleFile: currently unused method that changes the file of a staged phoneme crossfade Ai training sample

        addPredTrainSample: stages an audio sample the prediction Ai is to be trained with
        
        delPredTrainSampled: removes an audio sample from the list of staged prediction Ai training phonemes
        
        changePredTrainSampleFile: currently unused method that changes the file of a staged prediction Ai training sample
        
        trainTrAi: initiates the training of the Voicebank's phoneme crossfade Ai using all staged training samples and the Ai's settings
        
        trainPredAi: initiates the training of the Voicebank's prediction Ai using all staged training samples and the Ai's settings"""
        
        
    def __init__(self, filepath:str, device:torch.device = None) -> None:
        """ Universal constructor for initialisation both from a Voicebank file, and of an empty/new Voicebank.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to initialize the object with. If NONE is passed instead,
            it will be initialised with an empty Voicebank.
            
        Returns:
            None"""
            
            
        self.metadata = VbMetadata()
        self.filepath = filepath
        self.phonemeDict = dict()
        self.ai = AIWrapper(self, device)
        self.parameters = []
        self.wordDict = (dict(), [])
        self.stagedTrTrainSamples = AISampleCollection(isTransition = True)
        self.stagedMainTrainSamples = AISampleCollection()
        self.device = device
        if filepath != None:
            self.loadMetadata(self.filepath)
            self.loadPhonemeDict(self.filepath, False)
            self.loadMainWeights(self.filepath)
            self.loadParameters(self.filepath, False)
            self.loadWordDict(self.filepath, False)
        
    def save(self, filepath:str) -> None:
        """saves the loaded Voicebank to a file"""

        with h5py.File(filepath, "w") as f:
            aiStorage = DictStorage(f, ["aiState",], self.ai.device)
            aiStorage.fromDict(self.ai.getState())
            hparamStorage = DictStorage(f, ["hparams",], self.ai.device)
            hparamStorage.fromDict(self.ai.hparams)
            phonemeStorage = SampleStorage(f, ["phonemeDict",], False)
            phonemeStorage.fromDict(self.phonemeDict, True)
            parameterStorage = DictStorage(f, ["Parameters",], self.ai.device)
            parameterStorage.fromDict(self.parameters)
            wordDictStorage = WordStorage(f, ["wordDict",])
            wordDictStorage.fromDict(self.wordDict)
            metadataStorage = MetadataStorage(f, ["metadata",])
            metadataStorage.fromMetadata(self.metadata)
        
    def loadMetadata(self, filepath:str) -> None:
        """loads Voicebank Metadata from a Voicebank file"""
        with h5py.File(filepath, "r") as f:
            storage = MetadataStorage(f, ["metadata",])
            self.metadata = storage.toMetadata()
    
    def loadPhonemeDict(self, filepath:str, additive:bool) -> None:
        """loads Phoneme data from a Voicebank file.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to load the phoneme dictionary from.
            additive: a Bool defining whether the existing dictionary should be overwritten (False) or expanded (True) in the case of duplicate phoneme keys
            
        Returns:
            None"""
            
        with h5py.File(filepath, "r") as f:
            storage = SampleStorage(f, ["phonemeDict",], False)
            data = storage.toDict()
        if additive:
            for i in data.keys():
                if i in self.phonemeDict.keys():
                    self.phonemeDict[i].extend(data[i])
                else:
                    self.phonemeDict[i] = data[i]
        else:
            self.phonemeDict = data

    def loadMainWeights(self, filepath:str) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's prediction Ai"""

        with h5py.File(filepath, "r") as f:
            #hparamStorage = DictStorage(f, ["hparams",], self.ai.device)
            #self.ai.hparams = hparamStorage.toDict()
            aiStorage = DictStorage(f, ["aiState",], self.ai.device)
            aiState = aiStorage.toDict()
        self.ai.loadState(aiState, "main", True)
        
    def loadParameters(self, filepath:str, additive:bool) -> None:
        """currently placeholder"""

        if additive:
            pass
        else:
            pass
        
    def loadWordDict(self, filepath:str, additive:bool) -> None:
        """currently placeholder"""
        with h5py.File(filepath, "r") as f:
            wordStorage = WordStorage(f, ["wordDict"])
            self.wordDict = wordStorage.toDict()
        
    def addPhoneme(self, key:str, filepath:str) -> None:
        """adds a phoneme to the Voicebank's PhonemeDict.
        
        Arguments:
            key: The key the phoneme is going to be accessed by
            
            filepath: a String representing the filepath to the .wav audio file of the phoneme sample"""
            
            
        self.phonemeDict[key] = [AudioSample(filepath),]
        self.phonemeDict[key][0].key = key
        if filepath == None:
            return
        calculatePitch(self.phonemeDict[key][0])
        calculateSpectra(self.phonemeDict[key][0])
    
    def addSample(self, key:str, filepath:str) -> None:
        """adds a sample to an existing phoneme.
        
        Arguments:
            key: The key of the phoneme the sample should be added to
            
            filepath: a String representing the filepath to the .wav audio file of the sample"""
            
            
        self.phonemeDict[key].append(AudioSample(filepath))
        self.phonemeDict[key][-1].key = key
        calculatePitch(self.phonemeDict[key][-1])
        calculateSpectra(self.phonemeDict[key][-1])

    def addPhonemeUtau(self, sample:UtauSample) -> None:
        if (sample.end - sample.start) * global_consts.sampleRate / 1000 > 3 * global_consts.tripleBatchSize:
            if sample.key in self.phonemeDict.keys():
                self.phonemeDict[sample.key].append(sample.convert(False))
            else:
                self.phonemeDict[sample.key] = [sample.convert(False),]
            self.phonemeDict[sample.key][-1].key = sample.key
            calculatePitch(self.phonemeDict[sample.key][-1])
            calculateSpectra(self.phonemeDict[sample.key][-1])
        else:
            logging.warning("skipped one or several samples below the size threshold")
    
    def delPhoneme(self, key:str) -> None:
        """deletes a phoneme from the Voicebank's PhonemeDict"""

        self.phonemeDict.pop(key)
    
    def delSample(self, key:str, index:int) -> None:
        """deletes a sample from an existing phoneme.
        
        Arguments:
            key: The key of the phoneme the sample should be deleted from
            
            index: The index of the sample in the phoneme's sample list"""
            
            
        self.phonemeDict[key].pop(index)
    
    def changePhonemeKey(self, key:str, newKey:str) -> None:
        """changes the key of a Phoneme to a new key without changing its underlaying data"""

        self.phonemeDict[newKey] = self.phonemeDict.pop(key)
        for i in range(len(self.phonemeDict[newKey])):
            self.phonemeDict[newKey][i].key = newKey
    
    def changePhonemeFile(self, key:str, filepath:str) -> None:
        """changes the file of a Phoneme"""

        self.phonemeDict[key] = [AudioSample(filepath),]
        self.phonemeDict[key][0].key = key
    
    def finalizePhoneme(self, key:str) -> None:
        """finalizes a Phoneme, discarding any data related to it that's not strictly required for synthesis"""

        for i in range(len(self.phonemeDict[key])):
            self.phonemeDict[key][i] = LiteAudioSample(self.phonemeDict[key][i])
        print("staged phoneme " + key + " finalized")

    def addMainTrainSample(self, filepath:str) -> None:
        """stages an audio sample the prediction Ai is to be trained with"""

        self.stagedMainTrainSamples.append(AISample(filepath))

    def addMainTrainSampleUtau(self, sample:UtauSample) -> None:
        """stages an audio sample the prediction Ai is to be trained with"""

        if (sample.end - sample.start) * global_consts.sampleRate / 1000 > 4 * global_consts.tripleBatchSize:
            self.stagedMainTrainSamples.append(sample.convert(True))
        else:
            logging.warning("skipped one or several samples below the size threshold")
    
    def delMainTrainSample(self, index:int) -> None:
        """removes an audio sample from the list of staged training samples"""

        self.stagedMainTrainSamples.delete(index)
    
    def changeMainTrainSampleFile(self, index:int, filepath:str) -> None:
        """currently unused method that changes the file of a staged prediction Ai training sample"""

        self.delMainTrainSample(index)
        self.addMainTrainSample(filepath)

    def trainMainAi(self, epochs:int, additive:bool, generatorMode:str = "reclist", logging:bool = False) -> None:
        """initiates the training of the Voicebank's prediction Ai using all staged training samples and the Ai's settings.
        
        Arguments:
            epochs: Integer, the number of epochs the training is to be conducted with
            
            additive: Bool, whether the training should be conducted in addition to any existing training (True), or replaye it (False)
            
            logging: flag indicationg whether to write telemetry data to a .csv log"""
            

        print("sample preprocessing started")
        sampleCount = len(self.stagedMainTrainSamples)
        mainTrainSamples = LiteSampleCollection()
        
        for i in tqdm(range(sampleCount), desc = "preprocessing", unit = "samples"):
            samples = self.stagedMainTrainSamples[i].convert(True)
            for sample in samples:
                try:
                    calculatePitch(sample)
                    calculateSpectra(sample, False, True)
                    mainTrainSamples.append(sample)
                except Exception as e:
                    print(e)
        self.stagedMainTrainSamples.__init__()
        
        print("sample preprocessing complete")
        print("AI training started")
        self.ai.trainMain(mainTrainSamples, epochs = epochs, logging = logging, reset = not additive, generatorMode = generatorMode)
        print("AI training complete")

class LiteVoicebank():
    """Class for holding a Voicebank with only the features required for synthesis.
    
    Attributes:
        metadata: VbMetadata object containing the Voicebank's metadata
        
        filepath: The filepath to the Voicebank's file
        
        phonemeDict: a Dictionary object containing the samples for the individual phonemes
        
        ai: Wrapper for all mandatory AI-driven components of the Voicebank
        
        parameters: currently a placeholder. Will contain the Voicebank's Ai-driven parameters.
        
        wordDict: a tuple of a Dictionary containing overrides for the pronunciation of individual words,
                  and a list containing dictionaries of syllable mappings used when no override is present.
                  Within this list, the dictionaries are ordered by syllable length.
        
        stagedCrfTrainSamples: a List object containing the samples staged to be used in phoneme crossfade Ai training

        stagedPredTrainSamples: a List object containing the samples staged to be used in prediction Ai training
        
    Functions:
        __init__: Universal constructor for initialisation both from a Voicebank file, and of an empty/new Voicebank
        
        loadMetadata: loads Voicebank Metadata from a Voicebank file
        
        loadPhonemeDict: loads Phoneme data from a Voicebank file
        
        loadCrfWeights: loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai

        loadPredWeights: loads the Ai state saved in a Voicebank file into the loadedVoicebank's prediction Ai
        
        loadParameters: currently placeholder
        
        loadWordDict: currently placeholder"""
        
        
    def __init__(self, filepath:str, device:torch.device = torch.device("cpu")) -> None:
        """ Universal constructor for initialisation both from a Voicebank file, and of an empty/new Voicebank.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to initialize the object with. If NONE is passed instead,
            it will be initialised with an empty Voicebank.
            
        Returns:
            None"""
            
            
        self.metadata = VbMetadata()
        self.filepath = filepath
        self.phonemeDict = LiteSampleCollection()
        self.device = device
        self.ai = AIWrapper(self, device, inferOnly = True)
        self.parameters = []
        self.wordDict = (dict(), [])
        self.stagedTrTrainSamples = []
        self.stagedMainTrainSamples = []
        if filepath != None:
            self.loadMetadata(self.filepath)
            self.loadPhonemeDict(self.filepath, False)
            self.loadMainWeights(self.filepath)
            self.loadParameters(self.filepath, False)
            self.loadWordDict(self.filepath, False)
        
    def loadMetadata(self, filepath:str) -> None:
        """loads Voicebank Metadata from a Voicebank file"""
        with h5py.File(filepath, "r") as f:
            storage = MetadataStorage(f, ["metadata",])
            self.metadata = storage.toMetadata()
    
    def loadPhonemeDict(self, filepath:str, additive:bool) -> None:
        """loads Phoneme data from a Voicebank file.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to load the phoneme dictionary from.
            additive: a Bool defining whether the existing dictionary should be overwritten (False) or expanded (True) in the case of duplicate phoneme keys
            
        Returns:
            None"""
        
        with h5py.File(filepath, "r") as f:
            storage = SampleStorage(f, ["phonemeDict",], False)
            data = storage.toCollection("lite")
        if additive:
            for i in data.keys():
                if i in self.phonemeDict.keys():
                    self.phonemeDict[i + "#"] = data[i]
                    print("phoneme " + i + " is already present in voicebank; its key has been changed to " + i + "#")
                else:
                    self.phonemeDict[i] = data[i]
        else:
            self.phonemeDict = data

    def loadMainWeights(self, filepath:str) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's prediction Ai"""

        with h5py.File(filepath, "r") as f:
            #hparamStorage = DictStorage(f, ["hparams",], self.ai.device)
            #self.ai.hparams = hparamStorage.toDict()
            aiStorage = DictStorage(f, ["aiState",], self.ai.device)
            aiState = aiStorage.toDict()
        self.ai.loadState(aiState, "main", True)
        
    def loadParameters(self, filepath:str, additive:bool) -> None:
        """currently placeholder"""

        if additive:
            pass
        else:
            pass
        
    def loadWordDict(self, filepath:str, additive:bool) -> None:
        """currently placeholder"""
        with h5py.File(filepath, "r") as f:
            wordStorage = WordStorage(f, ["wordDict"])
            self.wordDict = wordStorage.toDict()
