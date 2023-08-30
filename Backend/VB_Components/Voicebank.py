#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import torch
from tqdm.auto import tqdm

from Backend.VB_Components.VbMetadata import VbMetadata
from Backend.VB_Components.Ai.Wrapper import AIWrapper
from Backend.DataHandler.AudioSample import AudioSample, LiteAudioSample, AISample
from Backend.ESPER.PitchCalculator import calculatePitch
from Backend.ESPER.SpectralCalculator import calculateSpectra
from Backend.DataHandler.UtauSample import UtauSample
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
        
        trainCrfAi: initiates the training of the Voicebank's phoneme crossfade Ai using all staged training samples and the Ai's settings
        
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
        self.stagedTrTrainSamples = []
        self.stagedMainTrainSamples = []
        self.device = device
        if filepath != None:
            data = torch.load(filepath, map_location = torch.device("cpu"))
            if self.device == torch.device("cpu"):
                data_ai = data
            else:
                data_ai = torch.load(filepath, map_location = self.device)
            self.loadMetadata(self.filepath, data)
            self.loadPhonemeDict(self.filepath, False, data)
            self.loadTrWeights(self.filepath, data_ai)
            self.loadMainWeights(self.filepath, data_ai)
            self.loadParameters(self.filepath, False, data)
            self.loadWordDict(self.filepath, False, data)
        
    def save(self, filepath:str) -> None:
        """saves the loaded Voicebank to a file"""

        torch.save({
            "metadata":self.metadata,
            "aiState":self.ai.getState(),
            "hparams":self.ai.hparams,
            "phonemeDict":self.phonemeDict,
            "Parameters":self.parameters,
            "wordDict":self.wordDict
            }, filepath)
        
    def loadMetadata(self, filepath:str, data:dict=None) -> None:
        """loads Voicebank Metadata from a Voicebank file"""
        if not data:
            data = torch.load(filepath, map_location = torch.device("cpu"))
        self.metadata = data["metadata"]
    
    def loadPhonemeDict(self, filepath:str, additive:bool, data:dict=None) -> None:
        """loads Phoneme data from a Voicebank file.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to load the phoneme dictionary from.
            additive: a Bool defining whether the existing dictionary should be overwritten (False) or expanded (True) in the case of duplicate phoneme keys
            
        Returns:
            None"""
            
        if not data:
            data = torch.load(filepath, map_location = torch.device("cpu"))
        if additive:
            for i in data["phonemeDict"].keys():
                if i in self.phonemeDict.keys():
                    self.phonemeDict[i + "#"] = data["phonemeDict"][i]
                    print("phoneme " + i + " is already present in voicebank; its key has been changed to " + i + "#")
                else:
                    self.phonemeDict[i] = data["phonemeDict"][i]
        else:
            self.phonemeDict = data["phonemeDict"]
    
    def loadTrWeights(self, filepath:str, data:dict=None) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai"""

        if data:
            aiState = data["aiState"]
            hparams = data["hparams"]
        else:
            aiState = torch.load(filepath, map_location = self.device)["aiState"]
            hparams = torch.load(filepath, map_location = self.device)["hparams"]
        self.ai.hparams["tr_lr"] = hparams["tr_lr"]
        self.ai.hparams["tr_reg"] = hparams["tr_reg"]
        self.ai.hparams["tr_hlc"] = hparams["tr_hlc"]
        self.ai.hparams["tr_hls"] = hparams["tr_hls"]
        self.ai.hparams["tr_def_thrh"] = hparams["tr_def_thrh"]
        self.ai.loadState(aiState, "tr", True)

    def loadMainWeights(self, filepath:str, data:dict=None) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's prediction Ai"""

        if data:
            aiState = data["aiState"]
            hparams = data["hparams"]
        else:
            aiState = torch.load(filepath, map_location = self.device)["aiState"]
            hparams = torch.load(filepath, map_location = self.device)["hparams"]
        self.ai.hparams["latent_dim"] = hparams["latent_dim"]
        self.ai.hparams["main_blkA"] = hparams["main_blkA"]
        self.ai.hparams["main_blkB"] = hparams["main_blkB"]
        self.ai.hparams["main_blkC"] = hparams["main_blkC"]
        self.ai.hparams["main_lr"] = hparams["main_lr"]
        self.ai.hparams["main_reg"] = hparams["main_reg"]
        self.ai.hparams["main_drp"] = hparams["main_drp"]
        self.ai.hparams["crt_blkA"] = hparams["crt_blkA"]
        self.ai.hparams["crt_blkB"] = hparams["crt_blkB"]
        self.ai.hparams["crt_blkC"] = hparams["crt_blkC"]
        self.ai.hparams["crt_lr"] = hparams["crt_lr"]
        self.ai.hparams["crt_reg"] = hparams["crt_reg"]
        self.ai.hparams["crt_drp"] = hparams["crt_drp"]
        self.ai.hparams["vae_lr"] = hparams["vae_lr"]
        self.ai.hparams["gan_guide_wgt"] = hparams["gan_guide_wgt"]
        self.ai.hparams["gan_train_asym"] = hparams["gan_train_asym"]
        self.ai.loadState(aiState, "main", True)
        
    def loadParameters(self, filepath:str, additive:bool, data:dict=None) -> None:
        """currently placeholder"""

        if not data:
            pass
        if additive:
            pass
        else:
            pass
        
    def loadWordDict(self, filepath:str, additive:bool, data:dict=None) -> None:
        """currently placeholder"""
        if data:
            self.wordDict = data["wordDict"]
        else:
            self.wordDict = torch.load(filepath, map_location = torch.device("cpu"))["wordDict"]
        
    def addPhoneme(self, key:str, filepath:str) -> None:
        """adds a phoneme to the Voicebank's PhonemeDict.
        
        Arguments:
            key: The key the phoneme is going to be accessed by
            
            filepath: a String representing the filepath to the .wav audio file of the phoneme sample"""
            
            
        self.phonemeDict[key] = [AudioSample(filepath),]
        calculatePitch(self.phonemeDict[key][0])
        calculateSpectra(self.phonemeDict[key][0])
    
    def addSample(self, key:str, filepath:str) -> None:
        """adds a sample to an existing phoneme.
        
        Arguments:
            key: The key of the phoneme the sample should be added to
            
            filepath: a String representing the filepath to the .wav audio file of the sample"""
            
            
        self.phonemeDict[key].append(AudioSample(filepath))
        calculatePitch(self.phonemeDict[key][-1])
        calculateSpectra(self.phonemeDict[key][-1])

    def addPhonemeUtau(self, sample:UtauSample) -> None:
        if (sample.end - sample.start) * global_consts.sampleRate / 1000 > 3 * global_consts.tripleBatchSize:
            if sample.key in self.phonemeDict.keys():
                self.phonemeDict[sample.key].append(sample.convert(False))
            else:
                self.phonemeDict[sample.key] = [sample.convert(False),]
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
    
    def changePhonemeFile(self, key:str, filepath:str) -> None:
        """changes the file of a Phoneme"""

        self.phonemeDict[key] = [AudioSample(filepath),]
    
    def finalizePhoneme(self, key:str) -> None:
        """finalizes a Phoneme, discarding any data related to it that's not strictly required for synthesis"""

        self.phonemeDict[key][0] = LiteAudioSample(self.phonemeDict[key][0])
        print("staged phoneme " + key + " finalized")
    
    def addTrTrainSample(self, filepath:str) -> None:
        """stages an audio sample the phoneme crossfade Ai is to be trained with"""

        self.stagedTrTrainSamples.append(AISample(filepath))
        self.stagedTrTrainSamples[-1].embedding = (0, 0)

    def addTrTrainSampleUtau(self, sample:UtauSample) -> None:
        """stages an audio sample the phoneme crossfade Ai is to be trained with"""

        if (sample.end - sample.start) * global_consts.sampleRate / 1000 > 3 * global_consts.tripleBatchSize:
            self.stagedTrTrainSamples.append(sample.convert(True))
        else:
            logging.warning("skipped one or several samples below the size threshold")
    
    def delTrTrainSample(self, index:int) -> None:
        """removes an audio sample from the list of staged training phonemes"""

        del self.stagedTrTrainSamples[index]
    
    def changeTrTrainSampleFile(self, index:int, filepath:str) -> None:
        """currently unused method that changes the file of a staged phoneme crossfade Ai training sample"""

        self.stagedTrTrainSamples[index] = AISample(filepath)

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

        del self.stagedMainTrainSamples[index]
    
    def changeMainTrainSampleFile(self, index:int, filepath:str) -> None:
        """currently unused method that changes the file of a staged prediction Ai training sample"""

        self.stagedMainTrainSamples[index] = AISample(filepath)
    
    def trainTrAi(self, epochs:int, additive:bool, logging:bool = False) -> None:
        """initiates the training of the Voicebank's phoneme crossfade Ai using all staged training samples and the Ai's settings.
        
        Arguments:
            epochs: Integer, the number of epochs the training is to be conducted with
            
            additive: Bool, whether the training should be conducted in addition to any existing training (True), or replaye it (False)
            
            logging: flag indicationg whether to write telemetry data to a .csv log"""
            

        print("sample preprocessing started")
        sampleCount = len(self.stagedTrTrainSamples)
        stagedTrTrainSamples = []
        for i in tqdm(range(sampleCount), desc = "preprocessing", unit = "samples"):
            sample = self.stagedTrTrainSamples.pop().convert(False)
            calculatePitch(sample, True)
            calculateSpectra(sample, False)
            avgSpecharm = torch.cat((sample.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1], torch.zeros([int(global_consts.nHarmonics / 2) + 1]), sample.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]), 0)
            stagedTrTrainSamples.append(((avgSpecharm + sample.specharm).to(device = self.device), sample.embedding))
        print("sample preprocessing complete")
        print("AI training started")
        self.ai.trainCrf(stagedTrTrainSamples, epochs = epochs, logging = logging, reset = not additive)
        print("AI training complete")

    def trainMainAi(self, epochs:int, additive:bool, logging:bool = False) -> None:
        """initiates the training of the Voicebank's prediction Ai using all staged training samples and the Ai's settings.
        
        Arguments:
            epochs: Integer, the number of epochs the training is to be conducted with
            
            additive: Bool, whether the training should be conducted in addition to any existing training (True), or replaye it (False)
            
            logging: flag indicationg whether to write telemetry data to a .csv log"""
            

        print("sample preprocessing started")
        sampleCount = len(self.stagedMainTrainSamples)
        stagedMainTrainSamples = []
        for i in tqdm(range(sampleCount), desc = "preprocessing", unit = "samples"):
            samples = self.stagedMainTrainSamples.pop().convert(True)
            for j in samples:
                calculatePitch(j, False)
                calculateSpectra(j, False)
                avgSpecharm = torch.cat((j.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1], torch.zeros([int(global_consts.nHarmonics / 2) + 1]), j.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]), 0)
                stagedMainTrainSamples.append((avgSpecharm + j.specharm).to(device = self.device))
        print("sample preprocessing complete")
        print("AI training started")
        self.ai.trainPred(stagedMainTrainSamples, epochs = epochs, logging = logging, reset = not additive)
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
        self.device = device
        data = torch.load(filepath, map_location = torch.device("cpu"))
        if self.device == torch.device("cpu"):
            data_ai = data
        else:
            data_ai = torch.load(filepath, map_location = self.device)
        self.ai = AIWrapper(self, device, data["hparams"], inferOnly = True)
        self.parameters = []
        self.wordDict = (dict(), [])
        self.stagedTrTrainSamples = []
        self.stagedMainTrainSamples = []
        if filepath != None:
            self.loadMetadata(self.filepath, data)
            self.loadPhonemeDict(self.filepath, False, data)
            self.loadTrWeights(self.filepath, data_ai)
            self.loadMainWeights(self.filepath, data_ai)
            self.loadParameters(self.filepath, False, data)
            self.loadWordDict(self.filepath, False, data)
        
    def loadMetadata(self, filepath:str, data:dict=None) -> None:
        """loads Voicebank Metadata from a Voicebank file"""
        
        if not data:
            data = torch.load(filepath, map_location = torch.device("cpu"))
        self.metadata = data["metadata"]
    
    def loadPhonemeDict(self, filepath:str, additive:bool, data:dict=None) -> None:
        """loads Phoneme data from a Voicebank file.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to load the phoneme dictionary from.
            additive: a Bool defining whether the existing dictionary should be overwritten (False) or expanded (True) in the case of duplicate phoneme keys
            
        Returns:
            None"""
            
        if not data:
            data = torch.load(filepath, map_location = torch.device("cpu"))
        if additive:
            for i in data["phonemeDict"].keys():
                if i in self.phonemeDict.keys():
                    self.phonemeDict[i + "#"] = data["phonemeDict"][i]
                    print("phoneme " + i + " is already present in voicebank; its key has been changed to " + i + "#")
                else:
                    self.phonemeDict[i] = data["phonemeDict"][i]
        else:
            self.phonemeDict = data["phonemeDict"]
    
    def loadTrWeights(self, filepath:str, data:dict=None) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai"""

        if data:
            aiState = data["aiState"]
            hparams = data["hparams"]
        else:
            aiState = torch.load(filepath, map_location = self.device)["aiState"]
            hparams = torch.load(filepath, map_location = self.device)["hparams"]
        self.ai.hparams["tr_hlc"] = hparams["tr_hlc"]
        self.ai.hparams["tr_hls"] = hparams["tr_hls"]
        self.ai.loadState(aiState, "tr", True)

    def loadMainWeights(self, filepath:str, data:dict=None) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's prediction Ai"""

        if data:
            aiState = data["aiState"]
            hparams = data["hparams"]
        else:
            aiState = torch.load(filepath, map_location = self.device)["aiState"]
            hparams = torch.load(filepath, map_location = self.device)["hparams"]
        self.ai.hparams["latent_dim"] = hparams["latent_dim"]
        self.ai.hparams["main_blkA"] = hparams["main_blkA"]
        self.ai.hparams["main_blkB"] = hparams["main_blkB"]
        self.ai.hparams["main_blkC"] = hparams["main_blkC"]
        self.ai.loadState(aiState, "main", True)
        
    def loadParameters(self, filepath:str, additive:bool, data:dict=None) -> None:
        """currently placeholder"""

        if data:
            pass
        if additive:
            pass
        else:
            pass
        
    def loadWordDict(self, filepath:str, additive:bool, data:dict=None) -> None:
        """currently placeholder"""

        if data:
            self.wordDict = data["wordDict"]
        else:
            self.wordDict = torch.load(filepath, map_location = self.device)["wordDict"]
