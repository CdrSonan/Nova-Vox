#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import math
import torch
import torch.nn.functional
import torchaudio
torchaudio.set_audio_backend("soundfile")
import ctypes

import C_Bridge
import global_consts
import Backend.Resampler.Loop as Loop
from Backend.DataHandler.VocalSegment import VocalSegment

def getExcitation(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:
    """resampler function for aquiring the unvoiced excitation of a VocalSegment according to the settings stored in it. Also requires a device argument specifying where the calculations are to be performed."""

    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        premul = 1
    else:
        premul = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0].excitation.size()[0] / (vocalSegment.end3 - vocalSegment.start1 + 1)
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
    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        return torch.zeros([windowEnd - windowStart, global_consts.halfTripleBatchSize + 1], dtype = torch.complex64, device = device)
    excitation = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0].excitation.to(device = device)[windowStart:windowEnd]
    excitation = torch.transpose(excitation, 0, 1)
    transform = torchaudio.transforms.TimeStretch(hop_length = global_consts.batchSize,
                                                  n_freq = global_consts.halfTripleBatchSize + 1, 
                                                  fixed_rate = premul).to(device = device)
    excitation = transform(excitation)[:, 0:length]
    return excitation.transpose(0, 1)

def getSpecharm(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:
    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        offset = 0
    else:
        offset = math.ceil(vocalSegment.offset * vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0].specharm.size()[0] / 2)
    if vocalSegment.startCap:
        windowStart = offset
    else:
        windowStart = vocalSegment.start3 - vocalSegment.start1 + offset
    if vocalSegment.endCap:
        windowEnd = vocalSegment.end3 - vocalSegment.start1 + offset
    else:
        windowEnd = vocalSegment.end1 - vocalSegment.start1 + offset
    output = torch.full([windowEnd - windowStart, global_consts.halfTripleBatchSize + global_consts.nHarmonics + 3], 0.001, device = device)
    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        return output
    vocalSegment.steadiness = vocalSegment.steadiness.contiguous()
    timings = C_Bridge.segmentTiming(start1 = vocalSegment.start1,
                                     start2 = vocalSegment.start2,
                                     start3 = vocalSegment.start3,
                                     end1 = vocalSegment.end1,
                                     end2 = vocalSegment.end2,
                                     end3 = vocalSegment.end3,
                                     windowStart = windowStart,
                                     windowEnd = windowEnd,
                                     offset = offset)
    phoneme = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0]
    C_Bridge.resampler.resampleSpecharm(ctypes.cast(phoneme.avgSpecharm.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               ctypes.cast(phoneme.specharm.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               int(phoneme.specharm.size()[0]),
                               ctypes.cast(vocalSegment.steadiness.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               ctypes.c_float(vocalSegment.repetititionSpacing.item()),
                               int(vocalSegment.startCap),
                               int(vocalSegment.endCap),
                               ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               timings,
                               global_consts.config)
    return output
    
def getSpecharm_legacy(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:
    """resampler function for aquiring the specharm of a VocalSegment according to the settings stored in it. Also requires a device argument specifying where the calculations are to be performed."""

    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        offset = 0
    else:
        offset = math.ceil(vocalSegment.offset * vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0].specharm.size()[0] / 2)
    if vocalSegment.startCap:
        windowStart = offset
    else:
        windowStart = vocalSegment.start3 - vocalSegment.start1 + offset
    if vocalSegment.endCap:
        windowEnd = vocalSegment.end3 - vocalSegment.start1 + offset
    else:
        windowEnd = vocalSegment.end1 - vocalSegment.start1 + offset
    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        return torch.full([windowEnd - windowStart, global_consts.halfTripleBatchSize + global_consts.nHarmonics + 3], 0.001, device = device)
    spectrum = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0].avgSpecharm.to(device = device)
    specharm = Loop.loopSamplerSpecharm(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0].specharm, windowEnd, vocalSegment.repetititionSpacing, device)[windowStart:windowEnd]
    specharm[:, :int(global_consts.nHarmonics / 2) + 1] *= torch.pow(1 - torch.unsqueeze(vocalSegment.steadiness[windowStart-offset:windowEnd-offset], 1), 2)
    specharm[:, global_consts.nHarmonics + 2:] *= torch.pow(1 - torch.unsqueeze(vocalSegment.steadiness[windowStart-offset:windowEnd-offset], 1), 2)
    specharm[:, :int(global_consts.nHarmonics / 2) + 1] += spectrum[:int(global_consts.nHarmonics / 2) + 1]
    specharm[:, global_consts.nHarmonics + 2:] += spectrum[int(global_consts.nHarmonics / 2) + 1:]
    if vocalSegment.startCap:
        factor = math.log(0.5, (vocalSegment.start2 - vocalSegment.start1) / (vocalSegment.start3 - vocalSegment.start1))
        slope = torch.linspace(0, 1, (vocalSegment.start3 - vocalSegment.start1), device = device)
        slope = torch.pow(slope, factor)
        specharm[:vocalSegment.start3 - vocalSegment.start1, global_consts.nHarmonics + 2:] *= slope.unsqueeze(1)
        specharm[:vocalSegment.start3 - vocalSegment.start1, :int(global_consts.nHarmonics / 2) + 1] *= slope.unsqueeze(1)
    if vocalSegment.endCap:
        factor = math.log(0.5, (vocalSegment.end3 - vocalSegment.end2) / (vocalSegment.end3 - vocalSegment.end1))
        slope = torch.linspace(1, 0, (vocalSegment.end3 - vocalSegment.end1), device = device)
        slope = torch.pow(slope, factor)
        specharm[vocalSegment.end1 - vocalSegment.end3:, global_consts.nHarmonics + 2:] *= slope.unsqueeze(1)
        specharm[vocalSegment.end1 - vocalSegment.end3:, :int(global_consts.nHarmonics / 2) + 1] *= slope.unsqueeze(1)
    return specharm

def getPitch(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:
    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        return torch.zeros([(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize,], device = device)
    phoneme = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0]
    requiredSize = math.ceil(torch.max(phoneme.pitchDeltas) / torch.min(vocalSegment.pitch)) * (vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize
    output = torch.zeros([requiredSize,], device = device, dtype = torch.float32)
    timings = C_Bridge.segmentTiming(start1 = vocalSegment.start1,
                                     start2 = vocalSegment.start2,
                                     start3 = vocalSegment.start3,
                                     end1 = vocalSegment.end1,
                                     end2 = vocalSegment.end2,
                                     end3 = vocalSegment.end3,
                                     windowStart = 0,
                                     windowEnd = 0,
                                     offset = 0)
    phoneme = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0]
    C_Bridge.resampler.resamplePitch(ctypes.cast(phoneme.pitchDeltas.data_ptr(), ctypes.POINTER(ctypes.c_short)),
                               int(phoneme.pitchDeltas.size()[0]),
                               phoneme.pitch.item(),
                               ctypes.c_float(vocalSegment.repetititionSpacing.item()),
                               int(vocalSegment.startCap),
                               int(vocalSegment.endCap),
                               ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               requiredSize,
                               timings)
    return output
    

def getPitch_legacy(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:
    """resampler function for aquiring the pitch curve of a VocalSegment according to the settings stored in it. Also requires a device argument specifying where the calculations are to be performed."""

    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        return torch.zeros([(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize,], device = device)
    pitchDeltas = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0].pitchDeltas
    requiredSize = math.ceil(torch.max(pitchDeltas) / torch.min(vocalSegment.pitch)) * (vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize
    pitch = vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][0].pitch.to(device = device)
    pitchDeltas = Loop.loopSamplerPitch(pitchDeltas, requiredSize, vocalSegment.repetititionSpacing, device = device)
    pitchDeltas -= pitch
    if vocalSegment.startCap == False:
        factor = math.log(0.5, (vocalSegment.start2 - vocalSegment.start1) / (vocalSegment.start3 - vocalSegment.start1))
        slope = torch.linspace(0, 1, (vocalSegment.start3 - vocalSegment.start1), device = device)
        slope = torch.pow(slope, factor)
        pitchDeltas[:vocalSegment.start3 - vocalSegment.start1] *= slope
    if vocalSegment.endCap == False:
        factor = math.log(0.5, (vocalSegment.end3 - vocalSegment.end2) / (vocalSegment.end3 - vocalSegment.end1))
        slope = torch.linspace(1, 0, (vocalSegment.end3 - vocalSegment.end1), device = device)
        slope = torch.pow(slope, factor)
        pitchDeltas[vocalSegment.end1 - vocalSegment.start1:vocalSegment.end3 - vocalSegment.start1] *= slope
    return pitchDeltas
