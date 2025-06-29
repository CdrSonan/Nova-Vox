#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import math
import torch
import torch.nn.functional
import torchaudio
import ctypes

import C_Bridge
import global_consts
import Backend.Resampler.Loop as Loop
from Backend.DataHandler.VocalSegment import VocalSegment

def getClosestSample(samples:list, pitch:float):
    closestSample = None
    closestDistance = math.inf
    for sample in samples:
        if abs(sample.pitch - pitch) < closestDistance:
            closestSample = sample
            closestDistance = abs(sample.pitch - pitch)
    return closestSample

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
    phoneme = getClosestSample(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey], torch.mean(vocalSegment.pitch))
    pitches = [i.pitch for i in vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey]]
    lowerPitchIndices = torch.full([vocalSegment.pitch.size()[0],], pitches.index(min(pitches)), device = device)
    upperPitchIndices = torch.full([vocalSegment.pitch.size()[0],], pitches.index(max(pitches)), device = device)
    pitches = torch.tensor(pitches, device = device)
    for i, pitch in enumerate(pitches):
        lowerPitchIndices = torch.where(torch.logical_and(torch.lt(pitches[lowerPitchIndices], pitch), torch.lt(pitch, vocalSegment.pitch)), torch.tensor(i, device = device), lowerPitchIndices)
        upperPitchIndices = torch.where(torch.logical_and(torch.gt(pitches[upperPitchIndices], pitch), torch.gt(pitch, vocalSegment.pitch)), torch.tensor(i, device = device), upperPitchIndices)
    ratio = torch.where(upperPitchIndices == lowerPitchIndices,
                        torch.zeros([vocalSegment.pitch.size()[0],], device = device),
                        torch.pow(torch.sin(abs(vocalSegment.pitch - pitches[lowerPitchIndices]) / abs(pitches[upperPitchIndices] - pitches[lowerPitchIndices]) * math.pi / 2), 2))
    avgSpecharm = torch.zeros([global_consts.halfTripleBatchSize + global_consts.halfHarms + 1], device = device)
    for i in range(vocalSegment.pitch.size()[0]):
        avgSpecharm += (ratio[i] * vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][upperPitchIndices[i]].avgSpecharm + (1 - ratio[i]) * vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey][lowerPitchIndices[i]].avgSpecharm).to(device)
    avgSpecharm /= vocalSegment.pitch.size()[0]
    avgSpecharm = avgSpecharm.contiguous()
    phoneme.specharm = phoneme.specharm.contiguous()
    vocalSegment.steadiness = vocalSegment.steadiness.contiguous()
    output = output.contiguous()
    C_Bridge.esper.resampleSpecharm(ctypes.cast(avgSpecharm.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               ctypes.cast(phoneme.specharm.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               int(phoneme.specharm.size()[0]),
                               ctypes.cast(vocalSegment.steadiness.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               ctypes.c_float(vocalSegment.repetititionSpacing),
                               int(vocalSegment.startCap),
                               int(vocalSegment.endCap),
                               ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               timings,
                               global_consts.config)
    return output

def getPitch(vocalSegment:VocalSegment, device:torch.device) -> torch.Tensor:
    if vocalSegment.phonemeKey == "_autopause" or vocalSegment.phonemeKey == "pau":
        return torch.zeros([(vocalSegment.end3 - vocalSegment.start1) * global_consts.batchSize,], device = device)
    phoneme = getClosestSample(vocalSegment.vb.phonemeDict[vocalSegment.phonemeKey], torch.mean(vocalSegment.pitch))
    requiredSize = math.ceil(torch.max(phoneme.pitchDeltas) / torch.min(vocalSegment.pitch)) * (vocalSegment.end3 - vocalSegment.start1)
    output = torch.zeros([requiredSize,], device = device, dtype = torch.float32).contiguous()
    phoneme.pitchDeltas = phoneme.pitchDeltas.contiguous()
    timings = C_Bridge.segmentTiming(start1 = vocalSegment.start1,
                                     start2 = vocalSegment.start2,
                                     start3 = vocalSegment.start3,
                                     end1 = vocalSegment.end1,
                                     end2 = vocalSegment.end2,
                                     end3 = vocalSegment.end3,
                                     windowStart = 0,
                                     windowEnd = 0,
                                     offset = 0)
    C_Bridge.esper.resamplePitch(ctypes.cast(phoneme.pitchDeltas.data_ptr(), ctypes.POINTER(ctypes.c_short)),
                               int(phoneme.pitchDeltas.size()[0]),
                               ctypes.c_float(phoneme.pitch.item()),
                               ctypes.c_float(vocalSegment.repetititionSpacing),
                               int(vocalSegment.startCap),
                               int(vocalSegment.endCap),
                               ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                               requiredSize,
                               timings)
    return output
