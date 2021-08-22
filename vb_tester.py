# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 22:14:32 2021

@author: CdrSonan
"""

import math
import numpy as np
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")

class VocalSegment:
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
        #factor = torch.exp(-j * phase / pitch * torch.unsqueeze(torch.arange(inputTensor.size()[0]), 1))
        #return factor * inputTensor
        absolutes = inputTensor.abs()
        phases = inputTensor.angle()
        phaseOffsets = torch.full(phases.size(), phase / pitch)
        #phaseOffsets *= torch.unsqueeze(torch.arange(phases.size()[0]), 1)
        phases += phaseOffsets
        #phases = torch.fmod(phases, 2 * math.pi)
        return torch.polar(absolutes, phases)
        
    def loopSamplerVoicedExcitation(self, inputTensor, targetSize, repetititionSpacing, pitch):
        batchRS = math.ceil(repetititionSpacing * inputTensor.size()[1] / 2)
        BatchSize = int(self.vb.sampleRate / 75)
        tripleBatchSize = int(self.vb.sampleRate / 25)
        repetititionSpacing = int(repetititionSpacing * BatchSize * math.ceil(inputTensor.size()[1] / 2))
        window = torch.hann_window(tripleBatchSize)
        alignPhase = inputTensor[batchRS][pitch].angle()
        finalPhase = inputTensor[1][pitch].angle()
        phaseDiff = (finalPhase - alignPhase)
        requiredTensors = math.ceil((targetSize/BatchSize - batchRS) / (inputTensor.size()[1] - batchRS))
        
        if requiredTensors == 1:
            outputTensor = inputTensor.clone()
            outputTensor = torch.istft(outputTensor, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*BatchSize)
        else:
            outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[1] * BatchSize - repetititionSpacing) + repetititionSpacing)
            
            workingTensor = inputTensor.clone()
            workingTensor = torch.istft(workingTensor, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*BatchSize)
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing)
            outputTensor[0:inputTensor.size()[1] * BatchSize] += workingTensor
            
            for i in range(1, requiredTensors - 1):
                workingTensor = inputTensor.clone()
                workingTensor = self.phaseShift(workingTensor, pitch, i * phaseDiff)
                workingTensor = torch.istft(workingTensor, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*BatchSize)
                workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing)
                workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.linspace(1, 0, repetititionSpacing)
                outputTensor[i * (inputTensor.size()[1] * BatchSize - repetititionSpacing):i * (inputTensor.size()[1] * BatchSize - repetititionSpacing) + inputTensor.size()[1] * BatchSize] += workingTensor
            
            workingTensor = inputTensor.clone()
            workingTensor = self.phaseShift(workingTensor, pitch, (requiredTensors - 1) * phaseDiff)
            workingTensor = torch.istft(workingTensor, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = window, onesided = True, length = inputTensor.size()[1]*BatchSize)
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.linspace(0, 1, repetititionSpacing)
            outputTensor[(requiredTensors - 1) * (inputTensor.size()[1] * BatchSize - repetititionSpacing):] += workingTensor
        return outputTensor[0:targetSize * BatchSize]
    
    def loopSamplerSpectrum(self, inputTensor, targetSize, repetititionSpacing):
        repetititionSpacing = math.ceil(repetititionSpacing * inputTensor.size()[0] / 2)
        requiredTensors = math.ceil((targetSize - repetititionSpacing) / (inputTensor.size()[0] - repetititionSpacing))
        if requiredTensors == 1:
            outputTensor = inputTensor.clone()
        else:
            outputTensor = torch.zeros(requiredTensors * (inputTensor.size()[0] - repetititionSpacing) + repetititionSpacing, inputTensor.size()[1])
            workingTensor = inputTensor.clone()
            workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing), 1)
            outputTensor[0:inputTensor.size()[0]] += workingTensor
            for i in range(1, requiredTensors - 1):
                workingTensor = inputTensor.clone()
                workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing), 1)
                workingTensor[-repetititionSpacing:] = workingTensor[-repetititionSpacing:] * torch.unsqueeze(torch.linspace(1, 0, repetititionSpacing), 1)
                outputTensor[i * (inputTensor.size()[0] - repetititionSpacing):i * (inputTensor.size()[0] - repetititionSpacing) + inputTensor.size()[0]] += workingTensor
            
            workingTensor = inputTensor.clone()
            workingTensor[0:repetititionSpacing] = workingTensor[0:repetititionSpacing] * torch.unsqueeze(torch.linspace(0, 1, repetititionSpacing), 1)
            outputTensor[(requiredTensors - 1) * (inputTensor.size()[0] - repetititionSpacing):] += workingTensor
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
        spectrum =  self.vb.phonemeDict[self.phonemeKey].spectrum#implement looping
        #spectra =  self.vb.phonemeDict[self.phonemeKey].spectra[windowStart:windowEnd]
        spectra = self.loopSamplerSpectrum(self.vb.phonemeDict[self.phonemeKey].spectra, windowEnd, self.repetititionSpacing)[windowStart:windowEnd]
        return torch.square(spectrum + (math.pow(1 - self.steadiness, 2) * spectra))
    
    def getExcitation(self):
        BatchSize = int(self.vb.sampleRate / 75)
        tripleBatchSize = int(self.vb.sampleRate / 25)
        premul = self.vb.phonemeDict[self.phonemeKey].excitation.size()[0] / (self.end3 - self.start1 + 1)
        #premul = 1
        if self.startCap:
            windowStart = 0
            length = -self.start1
        else:
            windowStart = math.floor((self.start2 - self.start1) * premul)
            length = -self.start2
        if self.endCap:
            windowEnd = math.ceil((self.end3 - self.start1) * premul)
            length += self.end3
        else:
            windowEnd = math.ceil((self.end2 - self.start1) * premul)
            length += self.end2
        excitation = self.vb.phonemeDict[self.phonemeKey].excitation[windowStart:windowEnd]
        excitation = torch.transpose(excitation, 0, 1)
        transform = torchaudio.transforms.TimeStretch(hop_length = int(self.vb.sampleRate / 75),
                                                      n_freq = int(self.vb.sampleRate / 25 / 2) + 1, 
                                                      fixed_rate = premul)
        excitation = transform(torch.view_as_real(excitation))
        excitation = torch.view_as_complex(excitation)
        window = torch.hann_window(int(self.vb.sampleRate / 25))
        excitation = torch.istft(excitation, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = window, onesided = True, length = length*int(self.vb.sampleRate / 75))
        
        return excitation[0:length*int(self.vb.sampleRate / 75)]
    
    def getVoicedExcitation(self):
        BatchSize = int(self.vb.sampleRate / 75)
        nativePitch = self.vb.phonemeDict[self.phonemeKey].pitch
        #nativePitch = self.vb.phonemeDict[self.phonemeKey].pitches[...]
        #pitch = nativePitch + self.vb.phonemeDict[phonemeKey].pitches...
        premul = self.pitch / nativePitch * self.vb.sampleRate / 75
        windowStart = math.floor(self.offset * self.vb.sampleRate / 75)
        windowEnd = math.ceil((self.end3 - self.start1) * premul + (self.offset * self.vb.sampleRate / 75))
        #windowStart = math.floor(self.offset)
        #windowEnd = math.ceil((self.end3 - self.start1) * premul + self.offset)
        #voicedExcitation = self.vb.phonemeDict[self.phonemeKey].voicedExcitation[windowStart:windowEnd]
        voicedExcitation = self.loopSamplerVoicedExcitation(self.vb.phonemeDict[self.phonemeKey].voicedExcitation, windowEnd, self.repetititionSpacing, math.ceil(nativePitch / 75.))[windowStart:windowEnd]
        transform = torchaudio.transforms.Resample(orig_freq = nativePitch,
                                                   new_freq = self.pitch,
                                                   resampling_method = 'sinc_interpolation')
        voicedExcitation = transform(voicedExcitation)
        if self.startCap == False:
            slope = torch.linspace(0, 1, (self.start3 - self.start1) * int(self.vb.sampleRate / 75))
            voicedExcitation[0:(self.start3 - self.start1) * int(self.vb.sampleRate / 75)] *= slope
        if self.endCap == False:
            slope = torch.linspace(1, 0, (self.end3 - self.end1) * int(self.vb.sampleRate / 75))
            voicedExcitation[(self.end1 - self.start1) * int(self.vb.sampleRate / 75):(self.end3 - self.start1) * int(self.vb.sampleRate / 75)] *= slope
        return voicedExcitation[0:(self.end3 - self.start1) * int(self.vb.sampleRate / 75)]
        #resample segments
        #individual fourier transform
        #istft
        #windowing adaptive to borders

class VocalSequence:
    def __init__(self, start, end, vb, borders, phonemes, offsets):
        self.start = start
        self.end = end
        self.vb = vb
        self.synth = Synthesizer(self.vb.sampleRate)
        
        self.spectrum = torch.zeros((self.end - self.start, int(self.vb.sampleRate / 25 / 2) + 1))
        self.excitation = torch.zeros((self.end - self.start) * int(self.vb.sampleRate / 75))
        self.voicedExcitation = torch.zeros((self.end - self.start) * int(self.vb.sampleRate / 75))
        
        self.segments = []
        if len(phonemes)== 1:#rewrite border system to use tensor, implement pitch and steadiness
            self.segments.append(VocalSegment(borders[0], borders[1], borders[2], borders[3], borders[4], borders[5],
                                             True, True, phonemes[0], vb, offsets[0], 0.8, 386, 0))
        else:
            self.segments.append(VocalSegment(borders[0], borders[1], borders[2], borders[3], borders[4], borders[5],
                                             True, False, phonemes[0], vb, offsets[0], 0.8, 386, 0))
            for i in range(1, len(phonemes)-1):
                self.segments.append(VocalSegment(borders[3*i], borders[3*i+1], borders[3*i+2], borders[3*i+3], borders[3*i+4], borders[3*i+5],
                                                  False, False, phonemes[i], vb, offsets[i], 0.8, 386, 0))
            endpoint = len(phonemes)-1
            self.segments.append(VocalSegment(borders[3*endpoint], borders[3*endpoint+1], borders[3*endpoint+2], borders[3*endpoint+3], borders[3*endpoint+4], borders[3*endpoint+5],
                                             False, True, phonemes[-1], vb, offsets[-1], 0.8, 386, 0))

        self.requiresUpdate = np.ones(len(phonemes))
        self.update()
    def update(self):
        for i in range(self.requiresUpdate.size):
            if self.requiresUpdate[i] == 1:
                print(i)
                segment = self.segments[i]
                spectrum = torch.zeros((segment.end3 - segment.start1, int(self.vb.sampleRate / 25 / 2) + 1))
                excitation = torch.zeros((segment.end3 - segment.start1) * int(self.vb.sampleRate / 75))
                voicedExcitation = torch.zeros((segment.end3 - segment.start1) * int(self.vb.sampleRate / 75))
                if segment.startCap:
                    windowStart = 0
                else:
                    windowStart = segment.start3 - segment.start1
                    previousSpectrum = self.segments[i-1].getSpectrum()[-1]
                    previousVoicedExcitation = self.segments[i-1].getVoicedExcitation()[(self.segments[i-1].end1-self.segments[i-1].end3)*int(self.vb.sampleRate/75):]
                if segment.endCap:
                    windowEnd = segment.end3 - segment.start1
                else:
                    windowEnd = segment.end1 - segment.start1
                    nextSpectrum = self.segments[i+1].getSpectrum()[0]
                    nextVoicedExcitation = self.segments[i+1].getVoicedExcitation()[0:(self.segments[i+1].start3-self.segments[i+1].start1)*int(self.vb.sampleRate/75)]
                
                spectrum[windowStart:windowEnd] = segment.getSpectrum()
                voicedExcitation = segment.getVoicedExcitation()
                if segment.startCap == False:
                    for j in range(segment.start3 - segment.start1):
                        spectrum[j] = self.vb.crfAi.processData(previousSpectrum, spectrum[windowStart], j / (segment.start3 - segment.start1))
                    voicedExcitation[0:(segment.start3-segment.start1)*int(self.vb.sampleRate/75)] += previousVoicedExcitation
                if segment.endCap == False:
                    for j in range(segment.end1 - segment.start1, segment.end3 - segment.start1):
                        spectrum[j] = self.vb.crfAi.processData(spectrum[windowEnd], nextSpectrum, (j - segment.start1) / (segment.end3 - segment.end1))
                    voicedExcitation[(segment.end1-segment.end3)*int(self.vb.sampleRate/75):] += nextVoicedExcitation
                if segment.startCap:
                    windowStart = 0
                else:
                    windowStart = (segment.start2 - segment.start1) * int(self.vb.sampleRate / 75)
                    previousExcitation = self.segments[i-1].getExcitation()[(segment.start1-segment.start2)*int(self.vb.sampleRate/75):]
                    excitation[0:windowStart] = previousExcitation
                if segment.endCap:
                    windowEnd = (segment.end3 - segment.start1) * int(self.vb.sampleRate / 75)
                else:
                    windowEnd = (segment.end2 - segment.start1) * int(self.vb.sampleRate / 75)
                    nextExcitation = self.segments[i+1].getExcitation()[0:(segment.end3-segment.end2)*int(self.vb.sampleRate/75)]
                    excitation[windowEnd:] = nextExcitation
                excitation[windowStart:windowEnd] = segment.getExcitation()
                
                self.spectrum[segment.start1:segment.end3] = spectrum
                self.excitation[segment.start1*int(self.vb.sampleRate/75):segment.end3*int(self.vb.sampleRate/75)] = excitation
                self.voicedExcitation[segment.start1*int(self.vb.sampleRate/75):segment.end3*int(self.vb.sampleRate/75)] = voicedExcitation
                
                skipPrevious = True#implement skipPrevious
            else:
                skipPrevious = False
            
        self.synth.Synthesize(0, self.spectrum, self.excitation, self.voicedExcitation)
    def save(self):
        self.synth.save("Output_Demo.wav")
    
class TempVB:
    def __init__(self):
        self.sampleRate = 48000
        self.phonemeDict = dict([])
        phonemeKeys = ["A", "E", "I", "O", "U", "G", "K", "N", "S", "T"]
        for key in phonemeKeys:
            self.phonemeDict[key] = AudioSample("Samples_rip/"+key+".wav")
            self.sampleRate = self.phonemeDict[key].sampleRate
            self.phonemeDict[key].CalculatePitch(249.)
            self.phonemeDict[key].CalculateSpectra(iterations = 15)
            self.phonemeDict[key].CalculateExcitation()
        self.crfAi = SpecCrfAi(learningRate=1e-4)

class Synthesizer:
    def __init__(self, sampleRate):
        self.sampleRate = sampleRate
        self.returnSignal = torch.tensor([], dtype = float)
        
    def Synthesize(self, steadiness, spectrum, Excitation, VoicedExcitation):
        tripleBatchSize = int(self.sampleRate / 25)
        BatchSize = int(self.sampleRate / 75)
        Window = torch.hann_window(tripleBatchSize)
        
        #HERE + VoicedExcitation
        
        self.returnSignal = torch.stft(Excitation + VoicedExcitation , tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = Window, return_complex = True, onesided = True)
        self.returnSignal = torch.transpose(self.returnSignal, 0, 1)[0:-1]
        self.returnSignal = self.returnSignal * spectrum
        self.returnSignal = torch.transpose(self.returnSignal, 0, 1)
        self.returnSignal = torch.istft(self.returnSignal, tripleBatchSize, hop_length = BatchSize, win_length = tripleBatchSize, window = Window, onesided=True)
        del Window
        
    def save(self, filepath):
        torchaudio.save(filepath, torch.unsqueeze(self.returnSignal.detach(), 0), self.sampleRate, format="wav", encoding="PCM_S", bits_per_sample=32)

borders = [0, 1, 2,
           35, 36, 37,
           40, 51, 52,
           75, 76, 79,
           82, 83, 86,
           328,329, 330
          ]
phonemes = ["A", "N", "A", "T", "A"]
#offsets = [0, 5, 1, 1, 1]
offsets = [0, 20, 20, 0, 13]

sequence = VocalSequence(0, 400, vb, borders, phonemes, offsets)