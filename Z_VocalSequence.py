from Backend.Resampler.Resamplers import getExcitation, getSpectrum, getVoicedExcitation
import torch
import math
import numpy as np
import global_consts

import Backend.DataHandler.VocalSegment
VocalSegment = Backend.DataHandler.VocalSegment.VocalSegment
import Backend.ESPER.ParametricSynthesizer
Synthesizer = Backend.ESPER.ParametricSynthesizer.Synthesizer
import Backend.Resampler.Resamplers
getSpectrum = Backend.Resampler.Resamplers.getSpectrum
getExcitation = Backend.Resampler.Resamplers.getExcitation
getVoicedExcitation = Backend.Resampler.Resamplers.getVoicedExcitation

class VocalSequence:
    """temporary class for combining several VocalSegments into a sequence. Currently no acceleration structure"""
    def __init__(self, start, end, vb, borders, phonemes, offsets, repetititionSpacing, pitch, steadiness, breathiness):
        self.start = start
        self.end = end
        self.vb = vb
        self.synth = Synthesizer()
        
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
                    previousSpectrum = getSpectrum(self.segments[i-1])[-1]
                    previousVoicedExcitation = getVoicedExcitation(self.segments[i-1])[(self.segments[i-1].end1-self.segments[i-1].end3)*global_consts.batchSize:]
                if segment.endCap:
                    windowEnd = segment.end3 - segment.start1
                else:
                    windowEnd = segment.end1 - segment.start1
                    nextSpectrum = getSpectrum(self.segments[i+1])[0]
                    nextVoicedExcitation = getVoicedExcitation(self.segments[i+1])[0:(self.segments[i+1].start3-self.segments[i+1].start1)*global_consts.batchSize]
                spectrum[windowStart:windowEnd] = getSpectrum(segment)
                voicedExcitation = getVoicedExcitation(segment)
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
                    previousExcitation = getExcitation(self.segments[i-1])[(segment.start1-segment.start2)*global_consts.batchSize:]
                    excitation[0:windowStart] = previousExcitation
                if segment.endCap:
                    windowEnd = (segment.end3 - segment.start1) * global_consts.batchSize
                else:
                    windowEnd = (segment.end2 - segment.start1) * global_consts.batchSize
                    nextExcitation = getExcitation(self.segments[i+1])[0:(segment.end3-segment.end2)*global_consts.batchSize]
                    excitation[windowEnd:] = nextExcitation
                excitation[windowStart:windowEnd] = getExcitation(segment)
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