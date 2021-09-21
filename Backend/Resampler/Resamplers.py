import math
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
import global_consts
import Backend.Resampler.Loop as Loop

def getSpectrum(vocalSegment):
    if vocalSegment.startCap:
        windowStart = vocalSegment.offset
    else:
        windowStart = vocalSegment.start3 - vocalSegment.start1 + vocalSegment.offset
    if vocalSegment.endCap:
        windowEnd = vocalSegment.end3 - vocalSegment.start1 + vocalSegment.offset
    else:
        windowEnd = vocalSegment.end1 - vocalSegment.start1 + vocalSegment.offset
    spectrum =  vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].spectrum
    spectra = Loop.loopSamplerSpectrum(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].spectra, windowEnd, vocalSegment.repetititionSpacing)[windowStart:windowEnd]
        
    return torch.square(spectrum + (torch.pow(1 - torch.unsqueeze(vocalSegment.steadiness[windowStart-vocalSegment.offset:windowEnd-vocalSegment.offset], 1), 2) * spectra))
    
def getExcitation(vocalSegment):
    premul = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].excitation.size()[0] / (vocalSegment.end3 - vocalSegment.start1 + 1)
    if vocalSegment.startCap:
        windowStart = 0
        brStart = 0
        length = -vocalSegment.start1
    else:
        windowStart = math.floor((vocalSegment.start2 - vocalSegment.start1) * premul)
        brStart = vocalSegment.start2 - vocalSegment.start1
        length = -vocalSegment.start2
    if vocalSegment.endCap:
        windowEnd = math.ceil((vocalSegment.end3 - vocalSegment.start1) * premul)
        brEnd = vocalSegment.end3 - vocalSegment.start1
        length += vocalSegment.end3
    else:
        windowEnd = math.ceil((vocalSegment.end2 - vocalSegment.start1) * premul)
        brEnd = vocalSegment.end2 - vocalSegment.start1
        length += vocalSegment.end2
    excitation = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].excitation[windowStart:windowEnd]
    excitation = torch.transpose(excitation, 0, 1)
    transform = torchaudio.transforms.TimeStretch(hop_length = global_consts.batchSize,
                                                  n_freq = global_consts.halfTripleBatchSize + 1, 
                                                  fixed_rate = premul)
    excitation = transform(excitation)[:, 0:length]
    #phaseAdvance = torch.linspace(0, math.pi * global_consts.batchSize,  global_consts.halfTripleBatchSize + 1)[..., None]
    #excitation = torchaudio.functional.phase_vocoder(excitation, premul, phaseAdvance)[:, 0:length]
    #window = torch.hann_window(global_consts.tripleBatchSize)
    #excitation = torch.istft(excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = length*global_consts.batchSize)
    return excitation.transpose(0, 1)
    
def getVoicedExcitation(vocalSegment):
    nativePitch = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitch
    requiredSize = math.ceil(torch.max(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas) / torch.min(vocalSegment.pitch) * (vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize)
    voicedExcitation = Loop.loopSamplerVoicedExcitation(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].voicedExcitation, requiredSize, vocalSegment.repetititionSpacing, math.ceil(nativePitch / global_consts.tickRate))
    cursor = 0
    cursor2 = 0
    pitchDeltas = torch.empty(math.ceil(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas.sum() / global_consts.batchSize))
    for i in range(math.floor(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas.sum() / global_consts.batchSize)):
        while cursor2 >= vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas[cursor]:
            cursor += 1
            cursor2 -= vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas[cursor]
        cursor2 += global_consts.batchSize
        pitchDeltas[i] = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas[cursor]
    pitchDeltas = torch.squeeze(Loop.loopSamplerSpectrum(torch.unsqueeze(pitchDeltas, 1), requiredSize, vocalSegment.repetititionSpacing))

    cursor = 0
    voicedExcitationFourier = torch.empty(vocalSegment.end3 - vocalSegment.start1, global_consts.halfTripleBatchSize + 1, dtype = torch.cdouble)
    window = torch.hann_window(global_consts.tripleBatchSize)
    for i in range(vocalSegment.end3 - vocalSegment.start1):
        precisePitch = pitchDeltas[i]
        nativePitchMod = math.ceil(nativePitch + ((precisePitch - nativePitch) * (1. - vocalSegment.steadiness[i])))
        transform = torchaudio.transforms.Resample(orig_freq = nativePitchMod,
                                                   new_freq = int(vocalSegment.pitch[i]),
                                                   resampling_method = 'sinc_interpolation')
        buffer = 1000 #this is a terrible idea, but it seems to work
        if cursor < math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i]):
            voicedExcitationPart = torch.cat((torch.zeros(math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i]) - cursor), voicedExcitation), 0)
            voicedExcitationPart = transform(voicedExcitationPart[vocalSegment.offset:(3*math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i])) + vocalSegment.offset + buffer])[0:global_consts.tripleBatchSize]
        else:
            voicedExcitationPart = transform(voicedExcitation[cursor - math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i]) + vocalSegment.offset:cursor + (2*math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i])) + vocalSegment.offset + buffer])[0:global_consts.tripleBatchSize]

        voicedExcitationFourier[i] = torch.fft.rfft(voicedExcitationPart * window)
        cursor += math.ceil(global_consts.batchSize * (nativePitchMod/vocalSegment.pitch[i]))
    voicedExcitationFourier = voicedExcitationFourier.transpose(0, 1)
    voicedExcitation = torch.istft(voicedExcitationFourier, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = (vocalSegment.end3 - vocalSegment.start1)*global_consts.batchSize)

    if vocalSegment.startCap == False:
        factor = math.log(0.5, (vocalSegment.start2 - vocalSegment.start1) / (vocalSegment.start3 - vocalSegment.start1))
        slope = torch.linspace(0, 1, (vocalSegment.start3 - vocalSegment.start1) * global_consts.batchSize)
        slope = torch.pow(slope, factor)
        voicedExcitation[0:(vocalSegment.start3 - vocalSegment.start1) * global_consts.batchSize] *= slope
    if vocalSegment.endCap == False:
        factor = math.log(0.5, (vocalSegment.end3 - vocalSegment.end2) / (vocalSegment.end3 - vocalSegment.end1))
        slope = torch.linspace(1, 0, (vocalSegment.end3 - vocalSegment.end1) * global_consts.batchSize)
        slope = torch.pow(slope, factor)
        voicedExcitation[(vocalSegment.end1 - vocalSegment.start1) * global_consts.batchSize:(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize] *= slope
    return voicedExcitation[0:(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize]