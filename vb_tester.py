# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 22:14:32 2021

@author: CdrSonan
"""

import math
import numpy as np
import tkinter.filedialog
import torch
import torch.nn as nn
import torchaudio
torchaudio.set_audio_backend("soundfile")

import devkit_pipeline
import global_consts

import matplotlib.pyplot as plt

class VocalSegment:
    """Class representing the segment covered by a single phoneme within a VocalSequence.
    
    Attributes:
        start1-3, end1-3: timing borders of the segment
        
        startCap, endCap: Whether there is a transition from the previous, and to the next phoneme
        
        phonemeKey: The key of the phoneme of the sequence
        
        vb: The Voicebank to use data from
        
        offset: The offset applied to the audio before sampling. Non-zero values discard the beginning of the audio.
        
        repetititionSpacing: The amout of overlap applied when looping the sample
        
        pitch: relevant part of the pitch parameter curve
        
        steadiness: relevant part of the steadiness parameter curve
        
        breathiness: relevant part of the breathiness parameter curve
        
    Methods:
        phaseShift: helper function for phase shifting audio by a certain phase at a certain pitch
        
        loopSamplerVoicedExcitation: helper function for looping the voiced excitation signal
        
        loopSamplerSpectrum: helper function for looping time sequences of spectra
        
        getSpectrum: samples the time sequence of spectra for the segment
        
        getExcitation: samples the unvoiced excitation signal of the segment
        
        getVoicedExcitation: samples the voiced excitation signal of the segment"""


    def __init__(self, start1, start2, start3, end1, end2, end3, startCap, endCap, phonemeKey, vb, offset, repetititionSpacing, pitch, steadiness):
        self.start1 = start1
        self.start2 = start2
        self.start3 = start3
        self.end1 = end1
        self.end2 = end2
        self.end3 = end3
        self.startCap = startCap
        self.endCap = endCap
        self.phonemeKey = phonemeKey
        self.vb = vb
        self.offset = offset
        self.repetititionSpacing = repetititionSpacing
        self.pitch = pitch
        self.steadiness = steadiness
        
    def phaseShift(self, inputTensor, pitch, phase):
        absolutes = inputTensor.abs()
        phases = inputTensor.angle()
        phaseOffsets = torch.full(phases.size(), phase / pitch)
        phases += phaseOffsets
        return torch.polar(absolutes, phases)
        
    def loopSamplerVoicedExcitation(self, inputTensor, targetSize, repetititionSpacing, pitch):
        batchRS = math.ceil(repetititionSpacing * inputTensor.size()[1] / 2)
        repetititionSpacing = int(repetititionSpacing * global_consts.batchSize * math.ceil(inputTensor.size()[1] / 2))
        window = torch.hann_window(global_consts.tripleBatchSize)
        alignPhase = inputTensor[batchRS][pitch].angle()
        finalPhase = inputTensor[1][pitch].angle()
        phaseDiff = (finalPhase - alignPhase)
        requiredTensors = max(math.ceil((targetSize/global_consts.batchSize - batchRS) / (inputTensor.size()[1] - batchRS)), 1)
        if requiredTensors <= 1:
            outputTensor = inputTensor.clone()
            outputTensor = torch.istft(outputTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
        else:
            outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing) + repetititionSpacing)
            
            workingTensor = inputTensor.clone()
            workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing)
            outputTensor[0:inputTensor.size()[1] * global_consts.batchSize] += workingTensor

            for i in range(1, requiredTensors - 1):
                workingTensor = inputTensor.clone()
                workingTensor = self.phaseShift(workingTensor, pitch, i * phaseDiff)
                workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
                workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing)
                workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing)
                outputTensor[i * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing):i * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing) + inputTensor.size()[1] * global_consts.batchSize] += workingTensor

            workingTensor = inputTensor.clone()
            workingTensor = self.phaseShift(workingTensor, pitch, (requiredTensors - 1) * phaseDiff)
            workingTensor = torch.istft(workingTensor, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*global_consts.batchSize)
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing)
            outputTensor[(requiredTensors - 1) * (inputTensor.size()[1] * global_consts.batchSize - repetititionSpacing):] += workingTensor
        return outputTensor[0:targetSize * global_consts.batchSize]
    
    def loopSamplerSpectrum(self, inputTensor, targetSize, repetititionSpacing):
        repetititionSpacing = math.ceil(repetititionSpacing * inputTensor.size()[0] / 2)
        requiredTensors = math.ceil((targetSize - repetititionSpacing) / (inputTensor.size()[0] - repetititionSpacing))
        if requiredTensors <= 1:
            outputTensor = inputTensor.clone()
        else:
            outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[0] - repetititionSpacing) + repetititionSpacing, inputTensor.size()[1])
            workingTensor = inputTensor.clone()
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing), 1)
            outputTensor[0:inputTensor.size()[0]] += workingTensor
            del workingTensor

            for i in range(1, requiredTensors - 1):
                workingTensor = inputTensor.clone()
                workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing), 1)
                workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing), 1)
                outputTensor[i * (inputTensor.size()[0] - repetititionSpacing):i * (inputTensor.size()[0] - repetititionSpacing) + inputTensor.size()[0]] += workingTensor
                del workingTensor

            workingTensor = inputTensor.clone()
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing), 1)
            outputTensor[(requiredTensors - 1) * (inputTensor.size()[0] - repetititionSpacing):] += workingTensor
            del workingTensor
        return outputTensor[0:targetSize]
    
    def getSpectrum(self):
        if self.startCap:
            windowStart = self.offset
        else:
            windowStart = self.start3 - self.start1 + self.offset
        if self.endCap:
            windowEnd = self.end3 - self.start1 + self.offset
        else:
            windowEnd = self.end1 - self.start1 + self.offset
        spectrum =  self.vb.phonemeDict[self.phonemeKey].spectrum
        spectra = self.loopSamplerSpectrum(self.vb.phonemeDict[self.phonemeKey].spectra, windowEnd, self.repetititionSpacing)[windowStart:windowEnd]
        
        return torch.square(spectrum + (torch.pow(1 - torch.unsqueeze(self.steadiness[windowStart-self.offset:windowEnd-self.offset], 1), 2) * spectra))
    
    def getExcitation(self):
        premul = self.vb.phonemeDict[self.phonemeKey].excitation.size()[0] / (self.end3 - self.start1 + 1)
        if self.startCap:
            windowStart = 0
            brStart = 0
            length = -self.start1
        else:
            windowStart = math.floor((self.start2 - self.start1) * premul)
            brStart = self.start2 - self.start1
            length = -self.start2
        if self.endCap:
            windowEnd = math.ceil((self.end3 - self.start1) * premul)
            brEnd = self.end3 - self.start1
            length += self.end3
        else:
            windowEnd = math.ceil((self.end2 - self.start1) * premul)
            brEnd = self.end2 - self.start1
            length += self.end2
        excitation = self.vb.phonemeDict[self.phonemeKey].excitation[windowStart:windowEnd]
        excitation = torch.transpose(excitation, 0, 1)
        transform = torchaudio.transforms.TimeStretch(hop_length = global_consts.batchSize,
                                                      n_freq = global_consts.halfTripleBatchSize + 1, 
                                                      fixed_rate = premul)
        excitation = transform(excitation)[:, 0:length]
        #phaseAdvance = torch.linspace(0, math.pi * global_consts.batchSize,  global_consts.halfTripleBatchSize + 1)[..., None]
        #excitation = torchaudio.functional.phase_vocoder(excitation, premul, phaseAdvance)[:, 0:length]
        window = torch.hann_window(global_consts.tripleBatchSize)
        excitation = torch.istft(excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = length*global_consts.batchSize)
        return excitation[0:length*global_consts.batchSize]
    
    def getVoicedExcitation(self):
        nativePitch = self.vb.phonemeDict[self.phonemeKey].pitch
        requiredSize = math.ceil(torch.max(self.vb.phonemeDict[self.phonemeKey].pitchDeltas) / torch.min(self.pitch) * (self.end3 - self.start1) * global_consts.batchSize)
        voicedExcitation = self.loopSamplerVoicedExcitation(self.vb.phonemeDict[self.phonemeKey].voicedExcitation, requiredSize, self.repetititionSpacing, math.ceil(nativePitch / global_consts.tickRate))
        cursor = 0
        cursor2 = 0
        pitchDeltas = torch.empty(math.ceil(self.vb.phonemeDict[self.phonemeKey].pitchDeltas.sum() / global_consts.batchSize))
        for i in range(math.floor(self.vb.phonemeDict[self.phonemeKey].pitchDeltas.sum() / global_consts.batchSize)):
            while cursor2 >= self.vb.phonemeDict[self.phonemeKey].pitchDeltas[cursor]:
                cursor += 1
                cursor2 -= self.vb.phonemeDict[self.phonemeKey].pitchDeltas[cursor]
            cursor2 += global_consts.batchSize
            pitchDeltas[i] = self.vb.phonemeDict[self.phonemeKey].pitchDeltas[cursor]
        pitchDeltas = torch.squeeze(self.loopSamplerSpectrum(torch.unsqueeze(pitchDeltas, 1), requiredSize, self.repetititionSpacing))

        cursor = 0
        voicedExcitationFourier = torch.empty(self.end3 - self.start1, global_consts.halfTripleBatchSize + 1, dtype = torch.cdouble)
        window = torch.hann_window(global_consts.tripleBatchSize)
        for i in range(self.end3 - self.start1):
            precisePitch = pitchDeltas[i]
            nativePitchMod = math.ceil(nativePitch + ((precisePitch - nativePitch) * (1. - self.steadiness[i])))
            transform = torchaudio.transforms.Resample(orig_freq = nativePitchMod,
                                                       new_freq = int(self.pitch[i]),
                                                       resampling_method = 'sinc_interpolation')
            buffer = 1000 #this is a terrible idea, but it seems to work
            if cursor < math.ceil(global_consts.batchSize*nativePitchMod/self.pitch[i]):
                voicedExcitationPart = torch.cat((torch.zeros(math.ceil(global_consts.batchSize*nativePitchMod/self.pitch[i]) - cursor), voicedExcitation), 0)
                voicedExcitationPart = transform(voicedExcitationPart[self.offset:(3*math.ceil(global_consts.batchSize*nativePitchMod/self.pitch[i])) + self.offset + buffer])[0:global_consts.tripleBatchSize]
            else:
                voicedExcitationPart = transform(voicedExcitation[cursor - math.ceil(global_consts.batchSize*nativePitchMod/self.pitch[i]) + self.offset:cursor + (2*math.ceil(global_consts.batchSize*nativePitchMod/self.pitch[i])) + self.offset + buffer])[0:global_consts.tripleBatchSize]

            voicedExcitationFourier[i] = torch.fft.rfft(voicedExcitationPart * window)
            cursor += math.ceil(global_consts.batchSize * (nativePitchMod/self.pitch[i]))
        voicedExcitationFourier = voicedExcitationFourier.transpose(0, 1)
        voicedExcitation = torch.istft(voicedExcitationFourier, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = (self.end3 - self.start1)*global_consts.batchSize)

        if self.startCap == False:
            factor = math.log(0.5, (self.start2 - self.start1) / (self.start3 - self.start1))
            slope = torch.linspace(0, 1, (self.start3 - self.start1) * global_consts.batchSize)
            slope = torch.pow(slope, factor)
            voicedExcitation[0:(self.start3 - self.start1) * global_consts.batchSize] *= slope
        if self.endCap == False:
            factor = math.log(0.5, (self.end3 - self.end2) / (self.end3 - self.end1))
            slope = torch.linspace(1, 0, (self.end3 - self.end1) * global_consts.batchSize)
            slope = torch.pow(slope, factor)
            voicedExcitation[(self.end1 - self.start1) * global_consts.batchSize:(self.end3 - self.start1) * global_consts.batchSize] *= slope
        return voicedExcitation[0:(self.end3 - self.start1) * global_consts.batchSize]

class VocalSequence:
    """temporary class for combining several VocalSegments into a sequence. Currently no acceleration structure"""
    def __init__(self, start, end, vb, borders, phonemes, offsets, repetititionSpacing, pitch, steadiness, breathiness):
        self.start = start
        self.end = end
        self.vb = vb
        self.synth = Synthesizer(self.vb.sampleRate)
        
        self.spectrum = torch.zeros((self.end - self.start, global_consts.halfTripleBatchSize + 1))
        self.unvoicedSpectrum = torch.zeros((self.end - self.start, global_consts.halfTripleBatchSize + 1))
        self.excitation = torch.zeros((self.end - self.start) * global_consts.batchSize)
        self.voicedExcitation = torch.zeros((self.end - self.start) * global_consts.batchSize)

        self.breathiness = breathiness
        
        self.segments = []
        if len(phonemes)== 1:#rewrite border system to use tensor
            self.segments.append(VocalSegment(borders[0], borders[1], borders[2], borders[3], borders[4], borders[5], True, True, phonemes[0], vb, offsets[0], repetititionSpacing[0], pitch[borders[0]:borders[5]], steadiness[borders[0]:borders[5]]))
        else:
            self.segments.append(VocalSegment(borders[0], borders[1], borders[2], borders[3], borders[4], borders[5], True, False, phonemes[0], vb, offsets[0], repetititionSpacing[0], pitch[borders[0]:borders[5]], steadiness[borders[0]:borders[5]]))
            for i in range(1, len(phonemes)-1):
                self.segments.append(VocalSegment(borders[3*i], borders[3*i+1], borders[3*i+2], borders[3*i+3], borders[3*i+4], borders[3*i+5], False, False, phonemes[i], vb, offsets[i], repetititionSpacing[i], pitch[borders[3*i]:borders[3*i+5]], steadiness[borders[3*i]:borders[3*i+5]]))
            endpoint = len(phonemes)-1
            self.segments.append(VocalSegment(borders[3*endpoint], borders[3*endpoint+1], borders[3*endpoint+2], borders[3*endpoint+3], borders[3*endpoint+4], borders[3*endpoint+5], False, True, phonemes[-1], vb, offsets[-1], repetititionSpacing[-1], pitch[borders[3*endpoint]:borders[3*endpoint+5]], steadiness[borders[3*endpoint]:borders[3*endpoint+5]]))

        self.requiresUpdate = np.ones(len(phonemes))
        self.update()
    def update(self):
        for i in range(self.requiresUpdate.size):
            if self.requiresUpdate[i] == 1:
                print(i)
                segment = self.segments[i]
                spectrum = torch.zeros((segment.end3 - segment.start1, global_consts.halfTripleBatchSize + 1))
                excitation = torch.zeros((segment.end3 - segment.start1) * global_consts.batchSize)
                voicedExcitation = torch.zeros((segment.end3 - segment.start1) * global_consts.batchSize)
                if segment.startCap:
                    windowStart = 0
                else:
                    windowStart = segment.start3 - segment.start1
                    previousSpectrum = self.segments[i-1].getSpectrum()[-1]
                    previousVoicedExcitation = self.segments[i-1].getVoicedExcitation()[(self.segments[i-1].end1-self.segments[i-1].end3)*global_consts.batchSize:]
                if segment.endCap:
                    windowEnd = segment.end3 - segment.start1
                else:
                    windowEnd = segment.end1 - segment.start1
                    nextSpectrum = self.segments[i+1].getSpectrum()[0]
                    nextVoicedExcitation = self.segments[i+1].getVoicedExcitation()[0:(self.segments[i+1].start3-self.segments[i+1].start1)*global_consts.batchSize]
                spectrum[windowStart:windowEnd] = segment.getSpectrum()
                voicedExcitation = segment.getVoicedExcitation()
                if segment.startCap == False:
                    for j in range(segment.start3 - segment.start1):
                        spectrum[j] = self.vb.crfAi.processData(previousSpectrum, spectrum[windowStart], j / (segment.start3 - segment.start1))
                    voicedExcitation[0:(segment.start3-segment.start1)*global_consts.batchSize] += previousVoicedExcitation
                if segment.endCap == False:
                    for j in range(segment.end1 - segment.start1, segment.end3 - segment.start1):
                        spectrum[j] = self.vb.crfAi.processData(spectrum[windowEnd], nextSpectrum, (j - segment.start1) / (segment.end3 - segment.end1))
                    voicedExcitation[(segment.end1-segment.end3)*global_consts.batchSize:] += nextVoicedExcitation
                if segment.startCap:
                    windowStart = 0
                else:
                    windowStart = (segment.start2 - segment.start1) * global_consts.batchSize
                    previousExcitation = self.segments[i-1].getExcitation()[(segment.start1-segment.start2)*global_consts.batchSize:]
                    excitation[0:windowStart] = previousExcitation
                if segment.endCap:
                    windowEnd = (segment.end3 - segment.start1) * global_consts.batchSize
                else:
                    windowEnd = (segment.end2 - segment.start1) * global_consts.batchSize
                    nextExcitation = self.segments[i+1].getExcitation()[0:(segment.end3-segment.end2)*global_consts.batchSize]
                    excitation[windowEnd:] = nextExcitation
                excitation[windowStart:windowEnd] = segment.getExcitation()
                self.spectrum[segment.start1:segment.end3] = spectrum
                pitchSlopeLength = 5
                for i in range(segment.end3 - segment.start1):
                    pitchBorder = math.ceil(global_consts.tripleBatchSize / segment.pitch[i])
                    fourierPitchShift = math.ceil(global_consts.tripleBatchSize / self.vb.phonemeDict[segment.phonemeKey].pitch) - pitchBorder
                    shiftedSpectrum = torch.roll(spectrum[i], fourierPitchShift)
                    slope = torch.zeros(global_consts.halfTripleBatchSize + 1)
                    slope[pitchBorder:pitchBorder + pitchSlopeLength] = torch.linspace(0, 1, pitchSlopeLength)
                    slope[pitchBorder + pitchSlopeLength:] = 1
                    self.spectrum[i + segment.start1] = (slope * spectrum[i]) + ((1 - slope) * shiftedSpectrum)
                self.excitation[segment.start1*global_consts.batchSize:segment.end3*global_consts.batchSize] = excitation
                self.voicedExcitation[segment.start1*global_consts.batchSize:segment.end3*global_consts.batchSize] = voicedExcitation
                skipPrevious = True#implement skipPrevious
            else:
                skipPrevious = False
            
        self.synth.synthesize(self.breathiness, self.spectrum, self.excitation, self.voicedExcitation)
    def save(self, name):
        self.synth.save(name)

class VbMetadata:
    """Helper class for holding Voicebank metadata. To be expanded.
    
    Attributes:
        name: The name of the Voicebank
        
    Methods:
        __init__: basic class constructor"""
        
        
    def __init__(self):
        """basic class constructor.
        
        Attributes:
            None
            
        Returns:
            None"""
            
            
        self.name = ""

class SavedSpecCrfAi(nn.Module):
    """A stripped down version of SpecCrfAi only holding the data required for synthesis.
    
    Attributes:
        layer1-4, ReLu1-4: FC and Nonlinear layers of the NN.
        
        epoch: training epoch counter displayed in Metadata panels
        
    Methods:
        __init__: Constructor initialising NN layers and prerequisite properties
        
        forward: Forward NN pass with unprocessed in-and outputs
        
        processData: forward NN pass with data pre-and postprocessing as expected by other classes
        
        getState: returns the state of the NN, its optimizer and their prerequisites in a Dictionary
        
    This version of the AI can only run data through the NN forward, backpropagation and, by extension, training, are not possible."""
    
    
    def __init__(self, learningRate=1e-4):
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
        
        self.epoch = 0
        
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
            
            
        Other than when using forward() directly, this method applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(spectrum1), torch.sqrt(spectrum2), factor)))
        return output

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

class Synthesizer:
    def __init__(self, sampleRate):
        self.sampleRate = sampleRate
        self.returnSignal = torch.tensor([], dtype = float)
        
    def synthesize(self, breathiness, spectrum, excitation, voicedExcitation):
        Window = torch.hann_window(global_consts.tripleBatchSize)
        
        self.returnSignal = torch.stft(voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        unvoicedSignal = torch.stft(excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        
        breathinessCompensation = torch.sum(torch.abs(self.returnSignal), 0) / torch.sum(torch.abs(unvoicedSignal), 0) * global_consts.breCompPremul
        breathinessUnvoiced = 1. + breathiness * breathinessCompensation[0:-1] * torch.gt(breathiness, 0) + breathiness * torch.logical_not(torch.gt(breathiness, 0))
        breathinessVoiced = 1. - (breathiness * torch.gt(breathiness, 0))
        self.returnSignal = self.returnSignal[:, 0:-1] * torch.transpose(spectrum, 0, 1) * breathinessVoiced
        unvoicedSignal = unvoicedSignal[:, 0:-1] * torch.transpose(spectrum, 0, 1) * breathinessUnvoiced

        self.returnSignal = torch.istft(self.returnSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided=True)
        unvoicedSignal = torch.istft(unvoicedSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided=True)
        self.returnSignal += unvoicedSignal

        del Window
        
    def save(self, filepath):
        torchaudio.save(filepath, torch.unsqueeze(self.returnSignal.detach(), 0), global_consts.sampleRate, format="wav", encoding="PCM_S", bits_per_sample=32)


filepath = tkinter.filedialog.askopenfilename(filetypes = ((".nvvb Voicebanks", ".nvvb"), ("all_files", "*")))
if filepath != "":
    vb = Voicebank(filepath)
    filepath = tkinter.filedialog.askopenfilename(filetypes = (("text files", ".txt"), ("all_files", "*")))
    if filepath != "":
        with open(filepath, 'r') as f:
            exec(f.read())

    """
    borders = [0, 2, 4,
               65, 70, 75,
               90, 102, 104,
               150, 152, 156,
               164, 166, 172,
               656,657, 658
              ]
    phonemes = ["A", "N", "A", "T", "A"]
    #offsets = [0, 5, 1, 1, 1]
    offsets = [0, 40, 20, 0, 13]

    repetititionSpacing = torch.full([700], 0.8)

    pitch = torch.full([700], 193)

    steadiness = torch.full([700], 0)

    breathiness = torch.full([700], 1)

    sequence = VocalSequence(0, 700, vb, borders, phonemes, offsets, repetititionSpacing, pitch, steadiness, breathiness)

    sequence.save("Anata test.wav")
    
    borders = [0, 2, 4,
               70, 72, 74,
               80, 102, 104,
               150, 152, 156,
               164, 166, 172,
               656,657, 658
              ]
    phonemes = ["O", "K", "S", "G", "N"]
    #offsets = [0, 5, 1, 1, 1]
    offsets = [0, 20, 20, 0, 13]

    repetititionSpacing = torch.full([700], 0.8)

    pitch = torch.full([700], 193)
    #pitch = torch.linspace(250, 100, 700)

    steadiness = torch.full([700], 0)

    breathiness = torch.linspace(-1, 1, 700)

    sequence = VocalSequence(0, 700, vb, borders, phonemes, offsets, repetititionSpacing, pitch, steadiness, breathiness)

    sequence.save("Consonant Breathiness test.wav")

    borders = [0, 2, 4,
               70, 72, 74,
               80, 102, 104,
               150, 152, 156,
               164, 166, 172,
               656,657, 658
              ]
    phonemes = ["A", "E", "I", "O", "U"]
    #offsets = [0, 5, 1, 1, 1]
    offsets = [0, 20, 20, 0, 13]

    repetititionSpacing = torch.full([700], 0.8)

    #pitch = torch.full([700], 193)
    pitch = torch.linspace(250, 100, 700)

    steadiness = torch.full([700], 0)

    breathiness = torch.full([700], 0)

    sequence = VocalSequence(0, 700, vb, borders, phonemes, offsets, repetititionSpacing, pitch, steadiness, breathiness)

    sequence.save("short Vowel Pitch test.wav")

    borders = [0, 2, 4,
               70, 72, 74,
               80, 82, 104,
               150, 152, 156,
               164, 171, 172,
               656,657, 658
              ]
    phonemes = ["E", "K", "I", "K", "U"]
    #offsets = [0, 5, 1, 1, 1]
    offsets = [0, 20, 20, 0, 13]

    repetititionSpacing = torch.full([700], 0.8)

    pitch = torch.full([700], 193)
    #pitch = torch.linspace(250, 100, 700)

    steadiness = torch.linspace(1, -1, 700)

    breathiness =torch.full([700], 0)

    sequence = VocalSequence(0, 700, vb, borders, phonemes, offsets, repetititionSpacing, pitch, steadiness, breathiness)

    sequence.save("Steadiness test.wav")
    """