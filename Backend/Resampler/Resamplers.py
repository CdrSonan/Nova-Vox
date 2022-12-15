import math
import torch
import torch.nn.functional
import torchaudio
torchaudio.set_audio_backend("soundfile")
import global_consts
import Backend.Resampler.Loop as Loop
from Backend.Resampler.CubicSplineInter import interp as interpolate
from Backend.DataHandler.VocalSegment import VocalSegment

def getExcitation(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:
    """resampler function for aquiring the unvoiced excitation of a VocalSegment according to the settings stored in it. Also requires a device argument specifying where the calculations are to be performed."""

    if vocalSegment.phonemeKey == "_autopause":
        premul = 1
    else:
        premul = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].excitation.size()[0] / (vocalSegment.end3 - vocalSegment.start1 + 1)
    if vocalSegment.startCap:
        windowStart = 0
        length = -vocalSegment.start1
    else:
        windowStart = math.floor((vocalSegment.start2 - vocalSegment.start1) * premul)
        length = -vocalSegment.start2
    if vocalSegment.endCap:
        windowEnd = math.ceil((vocalSegment.end3 - vocalSegment.start1) * premul)
        length += vocalSegment.end3
    else:
        windowEnd = math.ceil((vocalSegment.end2 - vocalSegment.start1) * premul)
        length += vocalSegment.end2
    if vocalSegment.phonemeKey == "_autopause":
        return torch.zeros([windowEnd - windowStart, global_consts.halfTripleBatchSize + 1], dtype = torch.complex64)
    excitation = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].excitation.to(device = device)[windowStart:windowEnd]
    excitation = torch.transpose(excitation, 0, 1)
    transform = torchaudio.transforms.TimeStretch(hop_length = global_consts.batchSize,
                                                  n_freq = global_consts.halfTripleBatchSize + 1, 
                                                  fixed_rate = premul).to(device = device)
    excitation = transform(excitation)[:, 0:length]
    return excitation.transpose(0, 1)
    
def getSpecharm(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:
    """resampler function for aquiring the voiced excitation of a VocalSegment according to the settings stored in it. Also requires a device argument specifying where the calculations are to be performed."""

    if vocalSegment.phonemeKey == "_autopause":
        return torch.full([windowEnd - windowStart, global_consts.halfTripleBatchSize + 1], 0.001)
    offset = math.ceil(vocalSegment.offset * vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].specharm.size()[0] / 2)
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
    spectrum = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].avgSpecharm.to(device = device)
    specharm = Loop.loopSamplerSpecharm(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].specharm, windowEnd, vocalSegment.repetititionSpacing, device)[windowStart:windowEnd]
    specharm[:, :int(global_consts.nHarmonics / 2) + 1] *= torch.pow(1 - torch.unsqueeze(vocalSegment.steadiness[windowStart-offset:windowEnd-offset], 1), 2)
    specharm[:, global_consts.nHarmonics + 2:] *= torch.pow(1 - torch.unsqueeze(vocalSegment.steadiness[windowStart-offset:windowEnd-offset], 1), 2)
    specharm[:, :int(global_consts.nHarmonics / 2) + 1] += spectrum[:int(global_consts.nHarmonics / 2) + 1]
    specharm[:, global_consts.nHarmonics + 2:] += spectrum[int(global_consts.nHarmonics / 2) + 1:]
    return specharm

def getPitch(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:

    if vocalSegment.phonemeKey == "_autopause":
        return torch.zeros([(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize,])
    pitchDeltas = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitchDeltas
    requiredSize = math.ceil(torch.max(pitchDeltas) / torch.min(vocalSegment.pitch)) * (vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize
    pitch = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey].pitch.to(device = device)
    pitchDeltas = Loop.loopSamplerPitch(pitchDeltas, requiredSize, vocalSegment.repetititionSpacing, device = device)
    pitchDeltas -= pitch
    if vocalSegment.startCap == False:
        factor = math.log(0.5, (vocalSegment.start2 - vocalSegment.start1) / (vocalSegment.start3 - vocalSegment.start1))
        slope = torch.linspace(0, 1, (vocalSegment.start3 - vocalSegment.start1), device = device)
        slope = torch.pow(slope, factor)
        pitchDeltas[:(vocalSegment.start3 - vocalSegment.start1)] *= slope
    if vocalSegment.endCap == False:
        factor = math.log(0.5, (vocalSegment.end3 - vocalSegment.end2) / (vocalSegment.end3 - vocalSegment.end1))
        slope = torch.linspace(1, 0, (vocalSegment.end3 - vocalSegment.end1), device = device)
        slope = torch.pow(slope, factor)
        pitchDeltas[(vocalSegment.end1 - vocalSegment.start1):(vocalSegment.end3 - vocalSegment.start1)] *= slope
    return pitchDeltas