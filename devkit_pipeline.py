# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 22:46:54 2021

@author: CdrSonan
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torchaudio
torchaudio.set_audio_backend("soundfile")

import global_consts

class AudioSample:
    """class for holding a single recording audio sample and processing it for usage in a Voicebank.
    
    Attributes:
        filepath: original filepath of the audio sample. used for reloading files.
        
        waveform: original audio waveform
        
        pitchDeltas: duration of each vocal chord vibration in data points
        
        pitchBorders: borders between each vocal chord vibration in data points measured from sample start
        
        pitch: average pitch of the sample given as wavelength in data points
        
        spectra: deviations of the audio spectrum from the average for each stft batch
        
        spectrum: audio spectrum averaged across the entire sample
        
        excitation: unvoiced excitation signal
        
        voicedExcitation: voiced excitation signal
        
        _voicedExcitations: please do not access. Temp container used during excitation signal calculation
        
        expectedPitch: estimated pitch of the sample in Hz. Must be manually filled and is used during calculatePitch() calls.
        
        searchRange: pitch frequency search range relative to expectedPitch. Should be a value between 0 and 1. Must be manually filled and is used during calculatePitch() calls.
        
        filterWidth: filter width used for spectral calculation. Must be manually filled with a positive Integer and is used during calculateSpectra() calls.
        
        voicedIterations: spectral filtering iterations used for voiced component. Must be manually filled with a positive Integer and is used during calculateSpectra() calls.
        
        unvoicedIterations: spectral filtering iterations used for unvoiced component. Must be manually filled with a positive Integer and is used during calculateSpectra() calls.
    
    Methods:
        __init__: Constructor for initialising an AudioSample based on an audio file
        
        calculatePitch(): Method for calculating pitch data based on previously set attributes
        
        calculateSpectra(): Method for calculating spectral data based on previously set attributes
        
        calculateExcitation(): Method for calculating excitation signal data based on previously set attributes"""
        
        
    def __init__(self, filepath):
        """Constructor for initialising an AudioSample based on an audio file and desired sample rate.
        
        Arguments:
            filepath: Expects a String that can be interpreted as a filepath to a .wav audio file. Determines the audio file to be loaded into the object.
            
        Returns:
            None
        
        This method initializes the properties used for spectral and excitation calculation to the default values and initialises all attributes with empty objects.
        Loads the selected audio file (based on filepath string) into the waveform property and resamples it to the desired sample rate"""
        
        
        loadedData = torchaudio.load(filepath)
        self.filepath = filepath
        self.waveform = loadedData[0][0]
        self.sampleRate = loadedData[1]
        del loadedData
        transform = torchaudio.transforms.Resample(self.sampleRate, global_consts.sampleRate)
        self.waveform = transform(self.waveform)
        del transform
        del self.sampleRate
        self.pitchDeltas = torch.tensor([], dtype = int)
        self.pitchBorders = torch.tensor([], dtype = int)
        self.pitch = torch.tensor([0], dtype = int)
        self.spectra = torch.tensor([[]], dtype = float)
        self.spectrum = torch.tensor([], dtype = float)
        self.excitation = torch.tensor([], dtype = float)
        self.voicedExcitation = torch.tensor([], dtype = float)
        self._voicedExcitations = torch.tensor([], dtype = float)
        
        self.expectedPitch = 249.
        self.searchRange = 0.2
        self.voicedIterations = 2
        self.unvoicedIterations = 10
        
    def calculatePitch(self):
        """Method for calculating pitch data based on previously set attributes expectedPitch and searchRange.
        
        Arguments:
            None
            
        Returns:
            None
            
        The pitch calculation uses 0-transitions to determine the borders between vocal chord vibrations. The algorithm searches for such transitions around expectedPitch (should be a value in Hz),
        with the range around it being defined by searchRange (should be a value between 0 and 1), which is interpreted as a percentage of the wavelength of expectedPitch.
        The function fills the pitchDeltas, pitchBorders and pitch properties."""
        
        
        batchSize = math.floor((1. + self.searchRange) * global_consts.sampleRate / self.expectedPitch)
        lowerSearchLimit = math.floor((1. - self.searchRange) * global_consts.sampleRate / self.expectedPitch)
        batchStart = 0
        while batchStart + batchSize <= self.waveform.size()[0] - batchSize:
            sample = torch.index_select(self.waveform, 0, torch.linspace(batchStart, batchStart + batchSize, batchSize, dtype = int))
            zeroTransitions = torch.tensor([], dtype = int)
            for i in range(lowerSearchLimit, batchSize):
                if (sample[i-1] < 0) and (sample[i] > 0):
                    zeroTransitions = torch.cat([zeroTransitions, torch.tensor([i])], 0)
            error = math.inf
            delta = math.floor(global_consts.sampleRate / self.expectedPitch)
            for i in zeroTransitions:
                shiftedSample = torch.index_select(self.waveform, 0, torch.linspace(batchStart + i.item(), batchStart + batchSize + i.item(), batchSize, dtype = int))
                newError = torch.sum(torch.pow(sample - shiftedSample, 2))
                if error > newError:
                    delta = i.item()
                    error = newError
            self.pitchDeltas = torch.cat([self.pitchDeltas, torch.tensor([delta])])
            batchStart += delta
        nBatches = self.pitchDeltas.size()[0]
        self.pitchBorders = torch.zeros(nBatches + 1, dtype = int)
        for i in range(nBatches):
            self.pitchBorders[i+1] = self.pitchBorders[i] + self.pitchDeltas[i]
        self.pitch = torch.mean(self.pitchDeltas.float()).int()
        del batchSize
        del lowerSearchLimit
        del batchStart
        del sample
        del zeroTransitions
        del error
        del delta
        del shiftedSample
        del newError
        del nBatches
        
    def calculateSpectra(self):
        """Method for calculating spectral data based on previously set attributes filterWidth, voicedIterations and unvoicedIterations.
        
        Arguments:
            None
            
        Returns:
            None
            
        The spectral calculation uses an adaptation of the True Envelope Estimator. It works with a fixed smoothing range determined by filterWidth (inta fourier space data points).
        The algorithm first runs an amount of filtering iterations determined by voicedIterations, selectively saves the peaking frequencies of the signal into _voicedExcitations, 
        then runs the filtering algorithm again a number of iterations determined by unvoicedIterations.
        The function fills the spectrum, spectra and _voicedExcitations properties."""
        
        
        Window = torch.hann_window(global_consts.tripleBatchSize)
        signals = torch.stft(self.waveform, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        signals = torch.transpose(signals, 0, 1)
        signalsAbs = signals.abs()
        
        workingSpectra = torch.sqrt(signalsAbs)
        
        workingSpectra = torch.max(workingSpectra, torch.tensor([-100]))
        self.spectra = torch.full_like(workingSpectra, -float("inf"), dtype=torch.float)
        
        for j in range(self.voicedIterations):
            workingSpectra = torch.max(workingSpectra, self.spectra)
            self.spectra = workingSpectra
            for i in range(global_consts.spectralFilterWidth):
                self.spectra = torch.roll(workingSpectra, -i, dims = 1) + self.spectra + torch.roll(workingSpectra, i, dims = 1)
            self.spectra = self.spectra / (2 * global_consts.spectralFilterWidth + 1)
        
        self._voicedExcitations = torch.zeros_like(signals)
        for i in range(signals.size()[0]):
            for j in range(signals.size()[1]):
                if torch.sqrt(signalsAbs[i][j]) > self.spectra[i][j]:
                    self._voicedExcitations[i][j] = signals[i][j]
                
        for j in range(self.unvoicedIterations):
            workingSpectra = torch.max(workingSpectra, self.spectra)
            self.spectra = workingSpectra
            for i in range(global_consts.spectralFilterWidth):
                self.spectra = torch.roll(workingSpectra, -i, dims = 1) + self.spectra + torch.roll(workingSpectra, i, dims = 1)
            self.spectra = self.spectra / (2 * global_consts.spectralFilterWidth + 1)
        
        self.spectrum = torch.mean(self.spectra, 0)
        for i in range(self.spectra.size()[0]):
            self.spectra[i] = self.spectra[i] - self.spectrum
        del Window
        del signals
        del workingSpectra
        
    def calculateExcitation(self):
        """Method for calculating excitation signal data.
        
        Arguments:
            None
            
        Returns:
            None
            
        The excitation is calculated from the original waveform, calculated spectra and _voicedExcitations property only.
        The function fills the excitation and voicedExcitation properties."""
        
        
        Window = torch.hann_window(global_consts.tripleBatchSize)
        signals = torch.stft(self.waveform, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        signals = torch.transpose(signals, 0, 1)
        excitations = torch.empty_like(signals)
        for i in range(excitations.size()[0]):
            excitations[i] = signals[i] / torch.square(self.spectrum + self.spectra[i])
            self._voicedExcitations[i] = self._voicedExcitations[i] / torch.square(self.spectrum + self.spectra[i])
        
        voicedExcitations = torch.transpose(self._voicedExcitations, 0, 1)
        excitations = torch.transpose(excitations, 0, 1)
        self.excitation = torch.istft(excitations, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided = True)
        self.voicedExcitation = torch.istft(voicedExcitations, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided = True)
        
        self.excitation = self.excitation - self.voicedExcitation
        self.excitation = torch.stft(self.excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        self.voicedExcitation = torch.stft(self.voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        self.excitation = torch.transpose(self.excitation, 0, 1)
        
        del Window
        del signals
        del excitations
        
class loadedAudioSample:
    """A stripped down version of AudioSample only holding the data required for synthesis.
    
    Attributes:
        pitchDeltas: duration of each vocal chord vibration in data points
        
        pitch: average pitch of the sample given as wavelength in data points
        
        spectra: deviations of the audio spectrum from the average for each stft batch
        
        spectrum: audio spectrum averaged across the entire sample
        
        excitation: unvoiced excitation signal
        
        voicedExcitation: voiced excitation signal
        
    Methods:
        __init__: Constructor for initialising the class based on an AudioSample object, discarding all extraneous data."""
    
    
    def __init__(self, audioSample):
        """Constructor for initialising the class based on an AudioSample object, discarding all extraneous data.
        
        Arguments:
            audioSample: AudioSample base object
            
        Returns:
            None"""
            
            
        self.pitchDeltas = audioSample.pitchDeltas
        self.pitch = audioSample.pitch
        self.spectra = audioSample.spectra
        self.spectrum = audioSample.spectrum
        self.excitation = audioSample.excitation
        self.voicedExcitation = audioSample.voicedExcitation
        
class RelLoss(nn.Module):
    """function for calculating relative loss values between target and actual Tensor objects. Designed to be used with AI optimizers.
    
    Attributes:
        None
        
    Methods:
        __init__: basic class constructor
        
        forward: calculates relative loss based on input and target tensors after successful initialisation."""
    
    
    def __init__(self, weight=None, size_average=True):
        """basic class constructor.
        
        Arguments:
            weight: required by PyTorch in some situations. Unused.
            
            size_average: required by PyTorch in some situations. Unused.
            
        Returns:
            None"""
        
        
        super(RelLoss, self).__init__()
 
    def forward(self, inputs, targets):  
        """calculates relative loss based on input and target tensors after successful initialisation.
        
        Arguments:
            inputs: AI-generated input Tensor
            
            targets: target Tensor
            
        Returns:
            Relative error value calculated from the difference between input and target Tensor as Float"""
        
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        differences = torch.abs(inputs - targets)
        refs = torch.abs(targets)
        out = (differences / refs).sum() / inputs.size()[0]
        return out
    
class SpecCrfAi(nn.Module):
    """class for generating crossphades between the spectra of different phonemes using AI.
    
    Attributes:
        layer1-4, ReLu1-4: FC and Nonlinear layers of the NN.
        
        learningRate: Learning Rate of the NN
        
        optimizer: Optimization algorithm to use during training. Changes not advised.
        
        criterion: Loss criterion to be used during AI training. Changes not advised.
        
        epoch: training epoch counter displayed in Metadata panels
        
        loss: Torch.Loss object holding Loss data from AI training
        
    Methods:
        __init__: Constructor initialising NN layers and prerequisite properties
        
        forward: Forward NN pass with unprocessed in-and outputs
        
        processData: forward NN pass with data pre-and postprocessing as expected by other classes
        
        train: NN training with forward and backward passes, Loss criterion and optimizer runs based on a dataset of spectral transition samples
        
        dataLoader: helper method for shuffled data loading from an arbitrary dataset
        
        getState: returns the state of the NN, its optimizer and their prerequisites in a Dictionary
        
    The structure of the NN is a forward-feed fully connected NN with ReLU nonlinear activation functions.
    It is designed to process non-negative data. Negative data can still be processed, but may negatively impact performance.
    The size of the NN layers is set to match the batch size and tick rate of the rest of the engine.
    Since network performance deteriorates with skewed data, it internally passes the input through a square root function and squares the output."""
        
        
    def __init__(self, learningRate=1e-4):
        """Constructor initialising NN layers and prerequisite attributes.
        
        Arguments:
            learningRate: desired learning rate of the NN as float. supports scientific format.
            
        Returns:
            None"""
            
            
        super(SpecCrfAi, self).__init__()
        
        self.layer1 = torch.nn.Linear(global_consts.tripleBatchSize + 3, global_consts.tripleBatchSize + 3)
        self.ReLu1 = nn.ReLU()
        self.layer2 = torch.nn.Linear(global_consts.tripleBatchSize + 3, 2 * global_consts.tripleBatchSize)
        self.ReLu2 = nn.ReLU()
        self.layer3 = torch.nn.Linear(2 * global_consts.tripleBatchSize, global_consts.tripleBatchSize + 3)
        self.ReLu3 = nn.ReLU()
        self.layer4 = torch.nn.Linear(global_consts.tripleBatchSize + 3, global_consts.halfTripleBatchSize + 1)
        
        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=0.)
        #self.criterion = nn.L1Loss()
        self.criterion = RelLoss()
        self.epoch = 0
        self.loss = None
        
    def forward(self, spectrum1, spectrum2, factor):
        """Forward NN pass with unprocessed in-and outputs.
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""
        
        
        fac = torch.tensor([factor])
        x = torch.cat((spectrum1, spectrum2, fac), dim = 0)
        x = x.float()
        x = self.layer1(x)
        x = self.ReLu1(x)
        x = self.layer2(x)
        x = self.ReLu2(x)
        x = self.layer3(x)
        x = self.ReLu3(x)
        x = self.layer4(x)
        return x
    
    def processData(self, spectrum1, spectrum2, factor):
        """forward NN pass with data pre-and postprocessing as expected by other classes
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode and applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(spectrum1), torch.sqrt(spectrum2), factor)))
        return output
    
    def train(self, indata, epochs=1):
        """NN training with forward and backward passes, Loss criterion and optimizer runs based on a dataset of spectral transition samples.
        
        Arguments:
            indata: Tensor, List or other Iterable containing sets of stft-like spectra. Each element should represent a phoneme transition.
            
            epochs: number of epochs to use for training as Integer.
            
        Returns:
            None
            
        Like processData(), train() also takes the square root of the input internally before using the data for inference."""
        
        
        if indata != False:
            if (self.epoch == 0) or self.epoch == epochs:
                self.epoch = epochs
            else:
                self.epoch = None
            for epoch in range(epochs):
                for data in self.dataLoader(indata):
                    print('epoch [{}/{}], switching to next sample'.format(epoch + 1, epochs))
                    #data = torch.sqrt(data)
                    data = torch.squeeze(data)
                    spectrum1 = data[0]
                    spectrum2 = data[-1]
                    indexList = np.arange(0, data.size()[0], 1)
                    np.random.shuffle(indexList)
                    for i in indexList:
                        factor = i / float(data.size()[0])
                        spectrumTarget = data[i]
                        output = torch.squeeze(self(spectrum1, spectrum2, factor))
                        loss = self.criterion(output, spectrumTarget)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        print('epoch [{}/{}], sub-sample index {}, loss:{:.4f}'.format(epoch + 1, epochs, i, loss.data))
            
            self.loss = loss
            
    def dataLoader(self, data):
        """helper method for shuffled data loading from an arbitrary dataset.
        
        Arguments:
            data: Tensor, List or Iterable representing a dataset with several elements
            
        Returns:
            Iterable representing the same dataset, with the order of its elements shuffled"""
        
        
        return torch.utils.data.DataLoader(dataset=data, shuffle=True)
    
    def getState(self):
        """returns the state of the NN, its optimizer and their prerequisites as well as its epoch attribute in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's epoch attribute (epoch), weights (state dict), optimizer state (optimizer state dict) and loss object (loss)"""
            
            
        AiState = {'epoch': self.epoch,
                 'model_state_dict': self.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'loss': self.loss
                 }
        return AiState

class SavedSpecCrfAi(nn.Module):
    """A stripped down version of SpecCrfAi only holding the data required for synthesis.
    
    Attributes:
        layer1-4, ReLu1-4: FC and Nonlinear layers of the NN.
        
        epoch: training epoch counter displayed in Metadata panels
        
    Methods:
        __init__: Constructor initialising NN layers and prerequisite properties
        
        forward: Forward NN pass with unprocessed in-and outputs
        
        processData: forward NN pass with data pre-and postprocessing as expected by other classes
        
        getState: returns the state of the NN and its epoch attribute in a Dictionary
        
    This version of the AI can only run data through the NN forward, backpropagation and, by extension, training, are not possible."""
    
    
    def __init__(self, specCrfAi, learningRate=1e-4):
        """Constructor initialising NN layers and other attributes based on SpecCrfAi base object.
        
        Arguments:
            specCrfAi: SpecCrfAi base object
            
        Returns:
            None"""
            
            
        super(SavedSpecCrfAi, self).__init__()
        
        self.layer1 = torch.nn.Linear(global_consts.tripleBatchSize + 3, global_consts.tripleBatchSize + 3)
        self.ReLu1 = nn.ReLU()
        self.layer2 = torch.nn.Linear(global_consts.tripleBatchSize + 3, 2 * global_consts.tripleBatchSize)
        self.ReLu2 = nn.ReLU()
        self.layer3 = torch.nn.Linear(2 * global_consts.tripleBatchSize, global_consts.tripleBatchSize + 3)
        self.ReLu3 = nn.ReLU()
        self.layer4 = torch.nn.Linear(global_consts.tripleBatchSize + 3, global_consts.halfTripleBatchSize + 1)
        
        self.epoch = specCrfAi.getState()['epoch']
        self.load_state_dict(specCrfAi.getState()['model_state_dict'])
        self.eval()
        
    def forward(self, spectrum1, spectrum2, factor):
        """Forward NN pass with unprocessed in-and outputs.
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""
        
        
        fac = torch.tensor([factor])
        x = torch.cat((spectrum1, spectrum2, fac), dim = 0)
        x = x.float()
        x = self.layer1(x)
        x = self.ReLu1(x)
        x = self.layer2(x)
        x = self.ReLu2(x)
        x = self.layer3(x)
        x = self.ReLu3(x)
        x = self.layer4(x)
        return x
    
    def processData(self, spectrum1, spectrum2, factor):
        """forward NN pass with data pre-and postprocessing as expected by other classes
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode and applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(spectrum1), torch.sqrt(spectrum2), factor)))
        return output
    
    def getState(self):
        """returns the state of the NN and its epoch attribute in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's weights (state dict) and its epoch attribute (epoch)"""
            
            
        AiState = {'epoch': self.epoch,
                 'model_state_dict': self.state_dict()
                 }
        return AiState
    
class VbMetadata:
    """Helper class for holding Voicebank metadata. To be expanded.
    
    Attributes:
        name: The name of the Voicebank
        
    Methods:
        __init__: basic class constructor"""
        
        
    def __init__(self):
        """basic class constructor.
        
        Arguments:
            None
            
        Returns:
            None"""
            
            
        self.name = ""
    
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
