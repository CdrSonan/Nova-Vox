#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import torch

from Backend.VB_Components.VbMetadata import VbMetadata
from Backend.VB_Components.Ai.Wrapper import AIWrapper
from Backend.DataHandler.AudioSample import AudioSample, LiteAudioSample
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
        self.ai = AIWrapper(device)
        self.parameters = []
        self.wordDict = dict()
        self.stagedCrfTrainSamples = []
        self.stagedPredTrainSamples = []
        self.device = device
        if filepath != None:
            self.loadMetadata(self.filepath)
            self.loadPhonemeDict(self.filepath, False)
            self.loadCrfWeights(self.filepath)
            self.loadPredWeights(self.filepath)
            self.loadParameters(self.filepath, False)
            self.loadWordDict(self.filepath, False)
        
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
        
    def loadMetadata(self, filepath:str) -> None:
        """loads Voicebank Metadata from a Voicebank file"""
        data = torch.load(filepath)
        self.metadata = data["metadata"]
    
    def loadPhonemeDict(self, filepath:str, additive:bool) -> None:
        """loads Phoneme data from a Voicebank file.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to load the phoneme dictionary from.
            additive: a Bool defining whether the existing dictionary should be overwritten (False) or expanded (True) in the case of duplicate phoneme keys
            
        Returns:
            None"""
            
            
        data = torch.load(filepath)
        if additive:
            for i in data["phonemeDict"].keys():
                if i in self.phonemeDict.keys():
                    self.phonemeDict[i + "#"] = data["phonemeDict"][i]
                    print("phoneme " + i + " is already present in voicebank; its key has been changed to " + i + "#")
                else:
                    self.phonemeDict[i] = data["phonemeDict"][i]
        else:
            self.phonemeDict = data["phonemeDict"]
    
    def loadCrfWeights(self, filepath:str) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai"""

        aiState = torch.load(filepath)["aiState"]
        hparams = torch.load(filepath)["hparams"]
        self.ai.hparams["crf_lr"] = hparams["crf_lr"]
        self.ai.hparams["crf_reg"] = hparams["crf_reg"]
        self.ai.hparams["crf_hls"] = hparams["crf_hls"]
        self.ai.hparams["crf_hlc"] = hparams["crf_hlc"]
        self.ai.loadState(aiState, "crf", True)

    def loadPredWeights(self, filepath:str) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's prediction Ai"""

        aiState = torch.load(filepath)["aiState"]
        hparams = torch.load(filepath)["hparams"]
        self.ai.hparams["pred_lr"] = hparams["pred_lr"]
        self.ai.hparams["pred_reg"] = hparams["pred_reg"]
        self.ai.hparams["pred_rs"] = hparams["pred_rs"]
        self.ai.loadState(aiState, "pred", True)
        
    def loadParameters(self, filepath:str, additive:bool) -> None:
        """currently placeholder"""

        if additive:
            pass
        else:
            pass
        
    def loadWordDict(self, filepath:str, additive:bool) -> None:
        """currently placeholder"""

        if additive:
            pass
        else:
            pass
        
    def addPhoneme(self, key:str, filepath:str) -> None:
        """adds a phoneme to the Voicebank's PhonemeDict.
        
        Arguments:
            key: The key the phoneme is going to be accessed by
            
            filepath: a String representing the filepath to the .wav audio file of the phoneme sample"""
            
            
        self.phonemeDict[key] = AudioSample(filepath)
        calculatePitch(self.phonemeDict[key])
        calculateSpectra(self.phonemeDict[key])

    def addPhonemeUtau(self, sample:UtauSample) -> None:
        if (sample.end - sample.start) * global_consts.sampleRate / 1000 > 3 * global_consts.tripleBatchSize:
            self.phonemeDict[sample.key] = sample.convert()
            calculatePitch(self.phonemeDict[sample.key])
            calculateSpectra(self.phonemeDict[sample.key])
        else:
            logging.warning("skipped one or several samples below the size threshold")
    
    def delPhoneme(self, key:str) -> None:
        """deletes a phoneme from the Voicebank's PhonemeDict"""

        self.phonemeDict.pop(key)
    
    def changePhonemeKey(self, key:str, newKey:str) -> None:
        """changes the key of a Phoneme to a new key without changing its underlaying data"""

        self.phonemeDict[newKey] = self.phonemeDict.pop(key)
    
    def changePhonemeFile(self, key:str, filepath:str) -> None:
        """changes the file of a Phoneme"""

        self.phonemeDict[key] = AudioSample(filepath)
    
    def finalizePhoneme(self, key:str) -> None:
        """finalizes a Phoneme, discarding any data related to it that's not strictly required for synthesis"""

        self.phonemeDict[key] = LiteAudioSample(self.phonemeDict[key])
        print("staged phoneme " + key + " finalized")
    
    def addCrfTrainSample(self, filepath:str) -> None:
        """stages an audio sample the phoneme crossfade Ai is to be trained with"""

        self.stagedCrfTrainSamples.append(AudioSample(filepath))

    def addCrfTrainSampleUtau(self, sample:UtauSample) -> None:
        """stages an audio sample the phoneme crossfade Ai is to be trained with"""

        if (sample.end - sample.start) * global_consts.sampleRate / 1000 > 3 * global_consts.tripleBatchSize:
            self.stagedCrfTrainSamples.append(sample.convert())
        else:
            logging.warning("skipped one or several samples below the size threshold")
    
    def delCrfTrainSample(self, index:int) -> None:
        """removes an audio sample from the list of staged training phonemes"""

        del self.stagedCrfTrainSamples[index]
    
    def changeCrfTrainSampleFile(self, index:int, filepath:str) -> None:
        """currently unused method that changes the file of a staged phoneme crossfade Ai training sample"""

        self.stagedCrfTrainSamples[index] = AudioSample(filepath)

    def addPredTrainSample(self, filepath:str) -> None:
        """stages an audio sample the prediction Ai is to be trained with"""

        self.stagedPredTrainSamples.append(AudioSample(filepath))

    def addPredTrainSampleUtau(self, sample:UtauSample) -> None:
        """stages an audio sample the prediction Ai is to be trained with"""

        if (sample.end - sample.start) * global_consts.sampleRate / 1000 > 4 * global_consts.tripleBatchSize:
            self.stagedPredTrainSamples.append(sample.convert())
        else:
            logging.warning("skipped one or several samples below the size threshold")
    
    def delPredTrainSample(self, index:int) -> None:
        """removes an audio sample from the list of staged training samples"""

        del self.stagedPredTrainSamples[index]
    
    def changePredTrainSampleFile(self, index:int, filepath:str) -> None:
        """currently unused method that changes the file of a staged prediction Ai training sample"""

        self.stagedPredTrainSamples[index] = AudioSample(filepath)
    
    def trainCrfAi(self, epochs:int, additive:bool, specWidth:int, specDepth:int, tempWidth:int, tempDepth:int, expPitch:float, pSearchRange:float, logging:bool = False) -> None:
        """initiates the training of the Voicebank's phoneme crossfade Ai using all staged training samples and the Ai's settings.
        
        Arguments:
            epochs: Integer, the number of epochs the training is to be conducted with
            
            additive: Bool, whether the training should be conducted in addition to any existing training (True), or replaye it (False)

            specWidth, specDepth, tempWidth, tempDepth: parameters used for spectral smoothing of the training samples

            expPitch, pSearchRange: parameters used for pitch calculation of the training samples
            
            logging: flag indicationg whether to write telemetry data to a Tensorboard log"""
            

        print("sample preprocessing started")
        sampleCount = len(self.stagedCrfTrainSamples)
        for i in range(sampleCount):
            print("processing sample [", i + 1, "/", sampleCount, "]")
            self.stagedCrfTrainSamples[i].expectedPitch = expPitch
            self.stagedCrfTrainSamples[i].searchRange = pSearchRange
            self.stagedCrfTrainSamples[i].specWidth = specWidth
            self.stagedCrfTrainSamples[i].specDepth = specDepth
            self.stagedCrfTrainSamples[i].tempWidth = tempWidth
            self.stagedCrfTrainSamples[i].tempDepth = tempDepth
            calculatePitch(self.stagedCrfTrainSamples[i], True)
            calculateSpectra(self.stagedCrfTrainSamples[i], False)
            avgSpecharm = torch.cat((self.stagedCrfTrainSamples[i].avgSpecharm[:int(global_consts.nHarmonics / 2) + 1], torch.zeros([int(global_consts.nHarmonics / 2) + 1]), self.stagedCrfTrainSamples[i].avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]), 0)
            self.stagedCrfTrainSamples[i] = (avgSpecharm + self.stagedCrfTrainSamples[i].specharm).to(device = self.device)
        print("sample preprocessing complete")
        print("AI training started")
        self.ai.trainCrf(self.stagedCrfTrainSamples, epochs = epochs, logging = logging, reset = not additive)
        print("AI training complete")

    def trainPredAi(self, epochs:int, additive:bool, voicedThrh:float, specWidth:int, specDepth:int, tempWidth:int, tempDepth:int, expPitch:float, pSearchRange:float, logging:bool = False) -> None:
        """initiates the training of the Voicebank's prediction Ai using all staged training samples and the Ai's settings.
        
        Arguments:
            epochs: Integer, the number of epochs the training is to be conducted with
            
            additive: Bool, whether the training should be conducted in addition to any existing training (True), or replaye it (False)

            specWidth, specDepth, tempWidth, tempDepth: parameters used for spectral smoothing of the training samples

            expPitch, pSearchRange: parameters used for pitch calculation of the training samples
            
            logging: flag indicationg whether to write telemetry data to a Tensorboard log"""
            

        print("sample preprocessing started")
        sampleCount = len(self.stagedPredTrainSamples)
        for i in range(sampleCount):
            print("processing sample [", i + 1, "/", sampleCount, "]")
            self.stagedPredTrainSamples[i].expectedPitch = expPitch
            self.stagedPredTrainSamples[i].searchRange = pSearchRange
            self.stagedPredTrainSamples[i].voicedThrh = voicedThrh
            self.stagedPredTrainSamples[i].specWidth = specWidth
            self.stagedPredTrainSamples[i].specDepth = specDepth
            self.stagedPredTrainSamples[i].tempWidth = tempWidth
            self.stagedPredTrainSamples[i].tempDepth = tempDepth
            calculatePitch(self.stagedPredTrainSamples[i], False)
            calculateSpectra(self.stagedPredTrainSamples[i], False)
            avgSpecharm = torch.cat((self.stagedPredTrainSamples[i].avgSpecharm[:int(global_consts.nHarmonics / 2) + 1], torch.zeros([int(global_consts.nHarmonics / 2) + 1]), self.stagedPredTrainSamples[i].avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]), 0)
            self.stagedPredTrainSamples[i] = (avgSpecharm + self.stagedPredTrainSamples[i].specharm).to(device = self.device)
        print("sample preprocessing complete")
        print("AI training started")
        self.ai.trainPred(self.stagedPredTrainSamples, epochs = epochs, logging = logging, reset = not additive)
        print("AI training complete")

class LiteVoicebank():
    """Class for holding a Voicebank with only the features required for synthesis.
    
    Attributes:
        metadata: VbMetadata object containing the Voicebank's metadata
        
        filepath: The filepath to the Voicebank's file
        
        phonemeDict: a Dictionary object containing the samples for the individual phonemes
        
        ai: Wrapper for all mandatory AI-driven components of the Voicebank
        
        parameters: currently a placeholder. Will contain the Voicebank's Ai-driven parameters.
        
        wordDict: a Dictionary containing overrides for NovaVox's default dictionary
        
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
        self.ai = AIWrapper(device, torch.load(filepath)["hparams"])
        self.parameters = []
        self.wordDict = dict()
        self.stagedCrfTrainSamples = []
        self.stagedPredTrainSamples = []
        self.device = device
        if filepath != None:
            self.loadMetadata(self.filepath)
            self.loadPhonemeDict(self.filepath, False)
            self.loadCrfWeights(self.filepath)
            self.loadPredWeights(self.filepath)
            self.loadParameters(self.filepath, False)
            self.loadWordDict(self.filepath, False)
        
    def loadMetadata(self, filepath:str) -> None:
        """loads Voicebank Metadata from a Voicebank file"""
        data = torch.load(filepath)
        self.metadata = data["metadata"]
    
    def loadPhonemeDict(self, filepath:str, additive:bool) -> None:
        """loads Phoneme data from a Voicebank file.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to load the phoneme dictionary from.
            additive: a Bool defining whether the existing dictionary should be overwritten (False) or expanded (True) in the case of duplicate phoneme keys
            
        Returns:
            None"""
            
            
        data = torch.load(filepath)
        if additive:
            for i in data["phonemeDict"].keys():
                if i in self.phonemeDict.keys():
                    self.phonemeDict[i + "#"] = data["phonemeDict"][i]
                    print("phoneme " + i + " is already present in voicebank; its key has been changed to " + i + "#")
                else:
                    self.phonemeDict[i] = data["phonemeDict"][i]
        else:
            self.phonemeDict = data["phonemeDict"]
    
    def loadCrfWeights(self, filepath:str) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai"""

        aiState = torch.load(filepath)["aiState"]
        hparams = torch.load(filepath)["hparams"]
        self.ai.hparams["crf_hls"] = hparams["crf_hls"]
        self.ai.hparams["crf_hlc"] = hparams["crf_hlc"]
        self.ai.loadState(aiState, "crf", True)

    def loadPredWeights(self, filepath:str) -> None:
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's prediction Ai"""

        aiState = torch.load(filepath)["aiState"]
        hparams = torch.load(filepath)["hparams"]
        self.ai.hparams["pred_rs"] = hparams["pred_rs"]
        self.ai.loadState(aiState, "pred", True)
        
    def loadParameters(self, filepath:str, additive:bool) -> None:
        """currently placeholder"""

        if additive:
            pass
        else:
            pass
        
    def loadWordDict(self, filepath:str, additive:bool) -> None:
        """currently placeholder"""

        if additive:
            pass
        else:
            pass