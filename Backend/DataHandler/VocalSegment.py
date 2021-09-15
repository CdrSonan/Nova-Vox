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