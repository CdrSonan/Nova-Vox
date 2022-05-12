import math
import torch
import torch.nn.functional
import torchaudio
torchaudio.set_audio_backend("soundfile")
import global_consts
import Backend.Resampler.Loop as Loop
import Backend.Resampler.CubicSplineInter as CubicSplineInter
interpolate = CubicSplineInter.interp

def getSpectrum(vocalSegment, device):
    if vocalSegment.phonemeKey == "_autopause":
        offset = 0
    else:
        offset = math.ceil(vocalSegment.offset * vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].spectra.size()[0] / 2)
    if vocalSegment.startCap:
        windowStart = offset
    else:
        windowStart = vocalSegment.start3 - vocalSegment.start1 + offset
    if vocalSegment.endCap:
        windowEnd = vocalSegment.end3 - vocalSegment.start1 + offset
    else:
        windowEnd = vocalSegment.end1 - vocalSegment.start1 + offset
    if vocalSegment.phonemeKey == "_autopause":
        return torch.full([windowEnd - windowStart, global_consts.halfTripleBatchSize + 1], 0.001)
    spectrum = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].spectrum.to(device = device)
    spectra = Loop.loopSamplerSpectrum(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].spectra, windowEnd, vocalSegment.repetititionSpacing, device)[windowStart:windowEnd]
    return spectrum + (torch.pow(1 - torch.unsqueeze(vocalSegment.steadiness[windowStart-offset:windowEnd-offset], 1), 2) * spectra)
    
def getExcitation(vocalSegment, device):
    if vocalSegment.phonemeKey == "_autopause":
        premul = 1
    else:
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
    if vocalSegment.phonemeKey == "_autopause":
        return torch.zeros([windowEnd - windowStart, global_consts.halfTripleBatchSize + 1])
    excitation = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].excitation.to(device = device)[windowStart:windowEnd]
    excitation = torch.transpose(excitation, 0, 1)
    transform = torchaudio.transforms.TimeStretch(hop_length = global_consts.batchSize,
                                                  n_freq = global_consts.halfTripleBatchSize + 1, 
                                                  fixed_rate = premul).to(device = device)
    excitation = transform(excitation)[:, 0:length]
    return excitation.transpose(0, 1)
    
def getVoicedExcitation(vocalSegment, device):
    import matplotlib.pyplot as plt
    if vocalSegment.phonemeKey == "_autopause":
        return torch.zeros([(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize,])
    offset = math.ceil(vocalSegment.offset * vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].spectra.size()[0] / 2)
    nativePitch = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitch.to(device = device)
    requiredSize = math.ceil(torch.max(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas) / torch.min(vocalSegment.pitch)) * (vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize
    pitchDeltas = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas
    pitch = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitch
    voicedExcitation = Loop.loopSamplerVoicedExcitation(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].voicedExcitation, requiredSize, vocalSegment.repetititionSpacing, pitchDeltas, pitch, vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].phases, device)
    pitchDeltas = torch.squeeze(Loop.loopSamplerSpectrum(torch.unsqueeze(pitchDeltas, 1), requiredSize, vocalSegment.repetititionSpacing, device = device))
    cursor = 0
    voicedExcitationFourier = torch.empty(vocalSegment.end3 - vocalSegment.start1, global_consts.halfTripleBatchSize + 1, dtype = torch.cdouble, device = device)
    window = torch.hann_window(global_consts.tripleBatchSize, device = device)
    for i in range(vocalSegment.end3 - vocalSegment.start1):
        precisePitch = pitchDeltas[i]
        nativePitchMod = math.ceil(nativePitch + ((precisePitch - nativePitch) * (1. - vocalSegment.steadiness[i])))
        rescale_factor = float(vocalSegment.pitch[i] / nativePitchMod)
        buffer = 10 #this is a terrible idea, but it seems to work
        if cursor < math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i]):#case for first sample, where no padding can be provided to acommodate stft overlap space
            voicedExcitationPart = torch.cat((voicedExcitation, torch.zeros(math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i]) - cursor).to(device = device)), 0)
            length = (3*math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i])) + buffer
            voicedExcitationPart = voicedExcitationPart[offset:length + offset]
            x = torch.linspace(0, 1, length, device = device)
            xs = torch.linspace(0, 1, int(length * rescale_factor), device = device)
            voicedExcitationPart = interpolate(x, voicedExcitationPart, xs)[0:global_consts.tripleBatchSize]
        else:
            length = math.ceil(global_consts.batchSize*nativePitchMod/vocalSegment.pitch[i])
            voicedExcitationPart = voicedExcitation[cursor - length + offset:cursor + (2*length) + offset + buffer]
            x = torch.linspace(0, 1, 3 * length + buffer, device = device)
            xs = torch.linspace(0, 1, int((3 * length + buffer) * rescale_factor), device = device)
            voicedExcitationPart = interpolate(x, voicedExcitationPart, xs)[0:global_consts.tripleBatchSize]
        voicedExcitationFourier[i] = torch.fft.rfft(voicedExcitationPart * window)
        cursor += math.ceil(global_consts.batchSize * (nativePitchMod/vocalSegment.pitch[i]))
    voicedExcitationFourier = voicedExcitationFourier.transpose(0, 1)
    voicedExcitation = torch.istft(voicedExcitationFourier, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True, length = (vocalSegment.end3 - vocalSegment.start1)*global_consts.batchSize)

    if vocalSegment.startCap == False:
        factor = math.log(0.5, (vocalSegment.start2 - vocalSegment.start1) / (vocalSegment.start3 - vocalSegment.start1))
        slope = torch.linspace(0, 1, (vocalSegment.start3 - vocalSegment.start1) * global_consts.batchSize, device = device)
        slope = torch.pow(slope, factor)
        voicedExcitation[0:(vocalSegment.start3 - vocalSegment.start1) * global_consts.batchSize] *= slope
    if vocalSegment.endCap == False:
        factor = math.log(0.5, (vocalSegment.end3 - vocalSegment.end2) / (vocalSegment.end3 - vocalSegment.end1))
        slope = torch.linspace(1, 0, (vocalSegment.end3 - vocalSegment.end1) * global_consts.batchSize, device = device)
        slope = torch.pow(slope, factor)
        voicedExcitation[(vocalSegment.end1 - vocalSegment.start1) * global_consts.batchSize:(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize] *= slope
    return voicedExcitation[0:(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize]
