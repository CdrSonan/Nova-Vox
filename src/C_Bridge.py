# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import ctypes
from math import floor
import torch

esper = ctypes.PyDLL("bin/esper.dll")

class engineCfg(ctypes.Structure):
    _fields_ = [("sampleRate", ctypes.c_uint),
                ("tickRate", ctypes.c_uint16),
                ("batchSize", ctypes.c_uint),
                ("tripleBatchSize", ctypes.c_uint),
                ("halfTripleBatchSize", ctypes.c_uint),
                ("nHarmonics", ctypes.c_uint),
                ("halfHarmonics", ctypes.c_uint),
                ("frameSize", ctypes.c_uint),
                ("breCompPremul", ctypes.c_float),]

class cSampleCfg(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint),
                ("batches", ctypes.c_uint),
                ("pitchLength", ctypes.c_uint),
                ("markerLength", ctypes.c_uint),
                ("pitch", ctypes.c_uint),
                ("isVoiced", ctypes.c_int),
                ("isPlosive", ctypes.c_int),
                ("useVariance", ctypes.c_int),
                ("expectedPitch", ctypes.c_float),
                ("searchRange", ctypes.c_float),
                ("tempWidth", ctypes.c_uint16)]

class cSample(ctypes.Structure):
    _fields_ = [("waveform", ctypes.POINTER(ctypes.c_float)),
                ("pitchDeltas", ctypes.POINTER(ctypes.c_int)),
                ("pitchMarkers", ctypes.POINTER(ctypes.c_int)),
                ("pitchMarkerValidity", ctypes.POINTER(ctypes.c_char)),
                ("specharm", ctypes.POINTER(ctypes.c_float)),
                ("avgSpecharm", ctypes.POINTER(ctypes.c_float)),
                ("config", cSampleCfg)]

class segmentTiming(ctypes.Structure):
    _fields_ = [("start1", ctypes.c_uint),
                ("start2", ctypes.c_uint),
                ("start3", ctypes.c_uint),
                ("end1", ctypes.c_uint),
                ("end2", ctypes.c_uint),
                ("end3", ctypes.c_uint),
                ("windowStart", ctypes.c_uint),
                ("windowEnd", ctypes.c_uint),
                ("offset", ctypes.c_uint)]

def makeCSample(sample, useVariance:bool, allow_oop:bool = False) -> cSample:
    import global_consts
    config = cSampleCfg(length = sample.waveform.size()[0],
                        batches = floor(sample.waveform.size()[0] / global_consts.batchSize),
                        pitchLength = sample.pitchDeltas.size()[0],
                        markerLength = sample.pitchMarkers.size()[0],
                        pitch = sample.pitch,
                        isVoiced = int(sample.isVoiced),
                        isPlosive = int(sample.isPlosive),
                        useVariance = int(useVariance),
                        expectedPitch = sample.expectedPitch,
                        searchRange = sample.searchRange,
                        tempWidth = global_consts.defaultTempWidth)
    if allow_oop:
        sample.waveform = sample.waveform.to(torch.float).contiguous()
        sample.pitchDeltas = sample.pitchDeltas.to(torch.int).contiguous()
        sample.pitchMarkerValidity = sample.pitchMarkerValidity.to(torch.int8).contiguous()
        sample.pitchMarkers = sample.pitchMarkers.to(torch.int).contiguous()
        sample.specharm = sample.specharm.to(torch.float).contiguous()
        sample.avgSpecharm = sample.avgSpecharm.to(torch.float).contiguous()
    else:
        assert sample.waveform.is_contiguous() & (sample.waveform.dtype == torch.float)
        assert sample.pitchDeltas.is_contiguous() & (sample.pitchDeltas.dtype == torch.int)
        assert sample.pitchMarkers.is_contiguous() & (sample.pitchMarkers.dtype == torch.int)
        assert sample.pitchMarkerValidity.is_contiguous() & (sample.pitchMarkerValidity.dtype == torch.int8)
        assert sample.specharm.is_contiguous() & (sample.specharm.dtype == torch.float)
        assert sample.avgSpecharm.is_contiguous() & (sample.avgSpecharm.dtype == torch.float)
    return cSample(waveform = ctypes.cast(sample.waveform.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                   pitchDeltas = ctypes.cast(sample.pitchDeltas.data_ptr(), ctypes.POINTER(ctypes.c_int)),
                   pitchMarkers = ctypes.cast(sample.pitchMarkers.data_ptr(), ctypes.POINTER(ctypes.c_int)),
                   pitchMarkerValidity = ctypes.cast(sample.pitchMarkerValidity.data_ptr(), ctypes.POINTER(ctypes.c_char)),
                   specharm = ctypes.cast(sample.specharm.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                   avgSpecharm = ctypes.cast(sample.avgSpecharm.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                   config = config)
