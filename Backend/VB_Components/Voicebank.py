class Voicebank:
    """Class for holding a Voicebank as handled by the devkit.
    
    Attributes:
        metadata: VbMetadata object containing the Voicebank's metadata
        
        filepath: The filepath to the Voicebank's file
        
        phonemeDict: a Dictionary object containing the samples for the individual phonemes
        
        crfAi: The phoneme crossfade Ai of the Voicebank, including its training
        
        parameters: currently a placeholder. Will contain the Voicebank's Ai-driven parameters.
        
        wordDict: a Dictionary containing overrides for Nova-Vox's default dictionary
        
        stagedTrainSamples: a List object containing the samples staged to be used in Ai training
        
    Functions:
        __init__: Universal constructor for initialisation both from a Voicebank file, and of an empty/new Voicebank
        
        save: saves the loaded Voicebank to a file
        
        loadMetadata: loads Voicebank Metadata from a Voicebank file
        
        loadPhonemeDict: loads Phoneme data from a Voicebank file
        
        loadCrfWeights: loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai
        
        loadParameters: currently placeholder
        
        loadWordDict: currently placeholder
        
        addPHoneme: adds a phoneme to the Voicebank's PhonemeDict
        
        delPhoneme: deletes a Phoneme from the vVoicebank's PhonemeDict
        
        changePhonemeKey: changes the key by which a phoneme in the Voicebank's phonemeDict can be accessed, but leaves the rest of its data unchanged
        
        changePhonemeFile: changes the audio file used for a phoneme in the Voicebank's PhonemeDict
        
        finalizePhoneme: finalizes a Phoneme, discarding any data related to it that's not strictly required for synthesis
        
        addTrainSample: stages an audio sample the phoneme crossfade Ai is to be trained with
        
        delTrainSampled: removes an audio sample from the list of staged training phonemes
        
        changeTrainSampleFile: currently unused method that changes the file of a staged phoneme crossfade Ai training sample
        
        trainCrfAi: initiates the training of the Voicebank's phoneme crossfade Ai using all staged training samples and the Ai's settings
        
        finalizeCrfAi: finalized the Voicebank's phoneme crossfade Ai, discarding all data related to it that's not strictly required for synthesis"""
        
        
    def __init__(self, filepath):
        """ Universal constructor for initialisation both from a Voicebank file, and of an empty/new Voicebank.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to initialize the object with. If NONE is passed instead,
            it will be initialised with an empty Voicebank.
            
        Returns:
            None"""
            
            
        self.metadata = VbMetadata()
        self.filepath = filepath
        self.phonemeDict = dict()
        self.crfAi = SpecCrfAi()
        self.parameters = []
        self.wordDict = dict()
        self.stagedTrainSamples = []
        if filepath != None:
            self.loadMetadata(self.filepath)
            self.loadPhonemeDict(self.filepath, False)
            self.loadCrfWeights(self.filepath)
            self.loadParameters(self.filepath, False)
            self.loadWordDict(self.filepath, False)
        
    def save(self, filepath):
        """saves the loaded Voicebank to a file"""
        torch.save({
            "metadata":self.metadata,
            "crfAiState":self.crfAi.getState(),
            "phonemeDict":self.phonemeDict,
            "Parameters":self.parameters,
            "wordDict":self.wordDict
            }, filepath)
        
    def loadMetadata(self, filepath):
        """loads Voicebank Metadata from a Voicebank file"""
        data = torch.load(filepath)
        self.metadata = data["metadata"]
    
    def loadPhonemeDict(self, filepath, additive):
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
    
    def loadCrfWeights(self, filepath):
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai"""
        data = torch.load(filepath)
        self.crfAi = SpecCrfAi()
        self.crfAi.epoch = data["crfAiState"]['epoch']
        self.crfAi.load_state_dict(data["crfAiState"]['model_state_dict'])
        if "loss" in data["crfAiState"].keys():
            self.crfAi.optimizer.load_state_dict(data["crfAiState"]['optimizer_state_dict'])
            self.crfAi.loss = data["crfAiState"]['loss']
        else:
            self.crfAi = SavedSpecCrfAi(self.crfAi)
        self.crfAi.eval()
        
    def loadParameters(self, filepath, additive):
        """currently placeholder"""
        if additive:
            pass
        else:
            pass
        
    def loadWordDict(self, filepath, additive):
        """currently placeholder"""
        if additive:
            pass
        else:
            pass
        
    def addPhoneme(self, key, filepath):
        """adds a phoneme to the Voicebank's PhonemeDict.
        
        Arguments:
            key: The key the phoneme is going to be accessed by
            
            filepath: a String representing the filepath to the .wav audio file of the phoneme sample"""
            
            
        self.phonemeDict[key] = AudioSample(filepath)
    
    def delPhoneme(self, key):
        """deletes a phoneme from the Voicebank's PhonemeDict"""
        self.phonemeDict.pop(key)
    
    def changePhonemeKey(self, key, newKey):
        """changes the key of a Phoneme to a new key without changing its underlaying data"""
        self.phonemeDict[newKey] = self.phonemeDict.pop(key)
    
    def changePhonemeFile(self, key, filepath):
        """changes the file of a Phoneme"""
        self.phonemeDict[key] = AudioSample(filepath)
    
    def finalizePhoneme(self, key):
        """finalizes a Phoneme, discarding any data related to it that's not strictly required for synthesis"""
        self.phonemeDict[key] = loadedAudioSample(self.phonemeDict[key])
        print("staged phoneme " + key + " finalized")
    
    def addTrainSample(self, filepath):
        """stages an audio sample the phoneme crossfade Ai is to be trained with"""
        self.stagedTrainSamples.append(AudioSample(filepath))
    
    def delTrainSample(self, index):
        """removes an audio sample from the list of staged training phonemes"""
        del self.stagedTrainSamples[index]
    
    def changeTrainSampleFile(self, index, filepath):
        """currently unused method that changes the file of a staged phoneme crossfade Ai training sample"""
        self.stagedTrainSamples[index] = AudioSample(filepath)
    
    def trainCrfAi(self, epochs, additive, filterWidth, voicedIterations, unvoicedIterations):
        """initiates the training of the Voicebank's phoneme crossfade Ai using all staged training samples and the Ai's settings.
        
        Arguments:
            epochs: Integer, the number of epochs the training is to be conducted with
            
            additive: Bool, whether the training should be conducted in addition to any existing training (True), or replaye it (False)
            
            filterWidth: Integer, the width of the spectral filter applied to the training samples
            
            voicedIterations, unvoicedIterations: Integer, the number of filtering iterations for the voiced and unvoiced voice components respectively.
            In the context of this function, only the sum of both is relevant."""
            
            
        if additive == False:
            self.crfAi = SpecCrfAi()
        print("sample preprocessing started")
        for i in range(len(self.stagedTrainSamples)):
            self.stagedTrainSamples[i].voicedIterations = voicedIterations
            self.stagedTrainSamples[i].unvoicedIterations = unvoicedIterations
            self.stagedTrainSamples[i].calculateSpectra()
            self.stagedTrainSamples[i] = self.stagedTrainSamples[i].spectrum + self.stagedTrainSamples[i].spectra
        print("sample preprocessing complete")
        print("AI training started")
        self.crfAi.train(self.stagedTrainSamples, epochs = epochs)
        print("AI training complete")
        
    def finalizCrfAi(self):
        """finalized the Voicebank's phoneme crossfade Ai, discarding all data related to it that's not strictly required for synthesis"""
        self.crfAi = SavedSpecCrfAi(self.crfAi)

class Voicebank:
    """Class for holding a Voicebank as handled by the devkit.
    
    Attributes:
        metadata: VbMetadata object containing the Voicebank's metadata
        
        filepath: The filepath to the Voicebank's file
        
        phonemeDict: a Dictionary object containing the samples for the individual phonemes
        
        crfAi: The phoneme crossfade Ai of the Voicebank, including its training
        
        parameters: currently a placeholder. Will contain the Voicebank's Ai-driven parameters.
        
        wordDict: a Dictionary containing overrides for NovaVox's default dictionary
        
        stagedTrainSamples: a List object containing the samples staged to be used in Ai training
        
    Functions:
        __init__: Universal constructor for initialisation both from a Voicebank file, and of an empty/new Voicebank
        
        loadMetadata: loads Voicebank Metadata from a Voicebank file
        
        loadPhonemeDict: loads Phoneme data from a Voicebank file
        
        loadCrfWeights: loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai
        
        loadParameters: currently placeholder
        
        loadWordDict: currently placeholder"""
        
        
    def __init__(self, filepath):
        """ Universal constructor for initialisation both from a Voicebank file, and of an empty/new Voicebank.
        
        Arguments:
            filepath: a String representing the filepath of the Voicebank file to initialize the object with. If NONE is passed instead,
            it will be initialised with an empty Voicebank.
            
        Returns:
            None"""
            
            
        self.metadata = VbMetadata()
        self.filepath = filepath
        self.phonemeDict = dict()
        self.crfAi = SavedSpecCrfAi()
        self.parameters = []
        self.wordDict = dict()
        self.stagedTrainSamples = []
        if filepath != None:
            self.loadMetadata(self.filepath)
            self.loadPhonemeDict(self.filepath, False)
            self.loadCrfWeights(self.filepath)
            self.loadParameters(self.filepath, False)
            self.loadWordDict(self.filepath, False)
        self.sampleRate = self.metadata.sampleRate
        
    def loadMetadata(self, filepath):
        """loads Voicebank Metadata from a Voicebank file"""
        data = torch.load(filepath)
        self.metadata = data["metadata"]
    
    def loadPhonemeDict(self, filepath, additive):
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
    
    def loadCrfWeights(self, filepath):
        """loads the Ai state saved in a Voicebank file into the loadedVoicebank's phoneme crossfade Ai"""
        data = torch.load(filepath)
        self.crfAi.epoch = data["crfAiState"]['epoch']
        self.crfAi.load_state_dict(data["crfAiState"]['model_state_dict'])
        self.crfAi.eval()
        
    def loadParameters(self, filepath, additive):
        """currently placeholder"""
        if additive:
            pass
        else:
            pass
        
    def loadWordDict(self, filepath, additive):
        """currently placeholder"""
        if additive:
            pass
        else:
            pass