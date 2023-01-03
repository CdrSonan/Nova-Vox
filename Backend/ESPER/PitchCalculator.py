#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import math
import torch
from torchaudio.functional import detect_pitch_frequency
from Backend.DataHandler.AudioSample import AudioSample
import global_consts
from Backend.Resampler.CubicSplineInter import interp

def calculatePitch(audioSample:AudioSample, limiter:bool = True) -> None:
    """current method for calculating pitch data for an AudioSample object based on the previously set attributes expectedPitch and searchRange.
    
    Arguments:
        audioSample: The AudioSample object the operation is to be performed on
        
    Returns:
        None
        
    This method of pitch calculation uses the detect_pitch_frequency method implemented in TorchAudio. It does not reliably work for values of searchRange below 0.33. This means a rather large search range is required, 
    introducing the risk of the pitch being detected as a whole multiple or fraction of its real value. For this reason, the pitch is multiplied or divided by a compensation factor if it is too far off the expected value.
    In case the pitch detection fails in spite of the correctly set search range, it uses a fallback."""

    try:
        audioSample.pitchDeltas = global_consts.sampleRate / detect_pitch_frequency(audioSample.waveform, global_consts.sampleRate, 1. / global_consts.tickRate, 30, audioSample.expectedPitch * (1 - audioSample.searchRange), audioSample.expectedPitch * (1 + audioSample.searchRange))
    except Exception as e:
        print("nonfatal_pitch_calc_err")
        print(e)
        calculatePitchFallback(audioSample)
    if audioSample.pitchDeltas.size()[0] < 2:
        calculatePitchFallback(audioSample)
    if limiter:
        mul1 = torch.maximum(torch.floor(audioSample.pitchDeltas * audioSample.expectedPitch / global_consts.sampleRate + 0.5), torch.ones([1,]))
        mul2 = torch.maximum(torch.floor(global_consts.sampleRate / audioSample.expectedPitch / audioSample.pitchDeltas + 0.5), torch.ones([1,]))
        audioSample.pitchDeltas /= mul1
        audioSample.pitchDeltas *= mul2
    audioSample.pitch = torch.mean(audioSample.pitchDeltas).int()
    length = math.floor(audioSample.waveform.size()[0] / global_consts.batchSize)
    audioSample.pitchDeltas = interp(torch.linspace(0., 1., audioSample.pitchDeltas.size()[0]), audioSample.pitchDeltas, torch.linspace(0., 1., length))
    audioSample.pitchDeltas = audioSample.pitchDeltas.to(torch.int16)

def calculatePitchFallback(audioSample:AudioSample) -> None:
    """Fallback method for calculating pitch data for an AudioSample object based on the previously set attributes expectedPitch and searchRange.
    
    Arguments:
        audioSample: The AudioSample object the operation is to be performed on
    
    Returns:
        None
    
    This method for pitch calculation uses 0-transitions to determine the borders between vocal chord vibrations. The algorithm searches for such transitions around expectedPitch (should be a value in Hz),
    with the range around it being defined by searchRange (should be a value between 0 and 1), which is interpreted as a percentage of the wavelength of expectedPitch.
    The function fills the pitchDeltas and pitch properties. Compared to the non-legacy version, it can be applied to smaller search ranges without the risk of failure, but suffers from a worse
    signal-to-noise ratio."""
    
    
    batchSize = math.floor((1. + audioSample.searchRange) * global_consts.sampleRate / audioSample.expectedPitch)
    lowerSearchLimit = math.floor((1. - audioSample.searchRange) * global_consts.sampleRate / audioSample.expectedPitch)
    batchStart = 0
    while batchStart + batchSize <= audioSample.waveform.size()[0] - batchSize:#TODO: check for end batchSize compensation possibly being doubled
        sample = torch.index_select(audioSample.waveform, 0, torch.linspace(batchStart, batchStart + batchSize, batchSize, dtype = int))
        zeroTransitions = torch.tensor([], dtype = int)
        for i in range(lowerSearchLimit, batchSize):
            if (sample[i-1] < 0) and (sample[i] > 0):
                zeroTransitions = torch.cat([zeroTransitions, torch.tensor([i])], 0)
        error = math.inf
        delta = math.floor(global_consts.sampleRate / audioSample.expectedPitch)
        for i in zeroTransitions:
            shiftedSample = torch.index_select(audioSample.waveform, 0, torch.linspace(batchStart + i.item(), batchStart + batchSize + i.item(), batchSize, dtype = int))
            bias = torch.abs(i - math.floor(global_consts.sampleRate / audioSample.expectedPitch))
            newError = torch.sum(torch.pow(sample - shiftedSample, 2)) * bias + 1
            if error > newError:
                delta = i.item()
                error = newError
        audioSample.pitchDeltas = torch.cat([audioSample.pitchDeltas, torch.tensor([delta])])
        batchStart += delta
    audioSample.pitch = torch.mean(audioSample.pitchDeltas.float()).int()

    #map sequence of pitchDeltas to sampling interval TODO: check for error causing double wavelength to be locked on
    cursor = 0
    cursor2 = 0
    pitchDeltas = torch.empty(math.floor(audioSample.pitchDeltas.sum() / global_consts.batchSize))
    for i in range(math.floor(audioSample.pitchDeltas.sum() / global_consts.batchSize)):
        while cursor2 >= audioSample.pitchDeltas[cursor]:
            if cursor < audioSample.pitchDeltas.size()[0] - 1:
                cursor += 1
            cursor2 -= audioSample.pitchDeltas[cursor]
        cursor2 += global_consts.batchSize
        pitchDeltas[i] = audioSample.pitchDeltas[cursor]
    audioSample.pitchDeltas = pitchDeltas

def calculatePhases(audioSample:AudioSample) -> None:
    """legacy method for calculating phase information for an AudioSample object. Was originally called after pitch calculation.
    
    Arguments:
        audioSample: The AudioSample object the operation is to be performed on
        
    Returns:
        None
        
    This method fits a sine and cosine curve of the f0 frequency to the waveform. The phase is then determined by reinterpreting the premul factors of the two curves as real and imaginary part of a complex exponential function, and extracting its phase.
    The result was then written to a phase attribute of the AudioSample class, which is now deprecated."""


    audioSample.phases = torch.empty_like(audioSample.pitchDeltas)
    previousLimit = 0
    previousPhase = 0
    for i in range(audioSample.pitchDeltas.size()[0]):
        limit = math.floor(global_consts.batchSize / audioSample.pitchDeltas[i])
        func = audioSample.waveform[i * global_consts.batchSize:i * global_consts.batchSize + limit * audioSample.pitchDeltas[i].to(torch.int64)]
        funcspace = torch.linspace(0, (limit * audioSample.pitchDeltas[i] - 1) * 2 * math.pi / audioSample.pitchDeltas[i], limit * audioSample.pitchDeltas[i])
        #TODO: Test new func and funcspace
        func = audioSample.waveform[i * global_consts.batchSize:(i + 1) * global_consts.batchSize]
        funcspace = torch.linspace(0, global_consts.batchSize * 2 * math.pi / audioSample.pitchDeltas[i], global_consts.batchSize)

        sine = torch.sin(funcspace)
        cosine = torch.cos(funcspace)
        sine *= func
        cosine *= func
        sine = torch.sum(sine)# / pi (would be required for normalization, but amplitude is irrelevant here, so normalization is not required)
        cosine = torch.sum(cosine)# / pi
        phase = torch.complex(sine, cosine).angle()
        if phase < 0:
            phase += 2 * math.pi
        offset = previousLimit
        if phase < previousPhase % (2 * math.pi):
            offset += 1
        phase += 2 * math.pi * offset
        audioSample.phases[i] = phase
        previousLimit = limit + offset
        previousPhase = phase