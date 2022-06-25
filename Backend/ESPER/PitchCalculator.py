import math
import torch
from torchaudio.functional import detect_pitch_frequency
from Backend.DataHandler.AudioSample import AudioSample
from Locale.devkit_locale import getLocale
import global_consts

def calculatePitch(audioSample:AudioSample) -> None:
    """current method for calculating pitch data for an AudioSample object  based on the previously set attributes expectedPitch and searchRange.
    
    Arguments:
        audioSample: The AudioSample object the operation is to be performed on
        
    Returns:
        None
        
    This method of pitch calculation uses the detect_pitch_frequency method implemented in TorchAudio. It does not reliably work for values of searchRange below 0.33. This means a rather large search range is required, 
    introducing the risk of the pitch being detected as a whole multiple or fraction of its real value. For this reason, the pitch is multiplied or divided by a compensation factor if it is too far off the expected value.
    In case the pitch detection fails in spite of the correctly set search range, it uses a fallback."""

    
    try:
        audioSample.pitchDeltas = global_consts.sampleRate / detect_pitch_frequency(audioSample.waveform, global_consts.sampleRate, 1. / global_consts.tickRate, 30, audioSample.expectedPitch * (1 - audioSample.searchRange), audioSample.expectedPitch * (1 + audioSample.searchRange))
        mul1 = torch.maximum(torch.floor(audioSample.pitchDeltas / audioSample.expectedPitch), torch.ones([1,]))
        mul2 = torch.maximum(torch.floor(audioSample.expectedPitch / audioSample.pitchDeltas), torch.ones([1,]))
        audioSample.pitchDeltas /= mul1
        audioSample.pitchDeltas *= mul2
    except Exception as e:
        print(getLocale()["pitch_calc_err"], e)
        calculatePitchFallback(audioSample)
    audioSample.pitch = torch.mean(audioSample.pitchDeltas).int()
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
    while batchStart + batchSize <= audioSample.waveform.size()[0] - batchSize:
        sample = torch.index_select(audioSample.waveform, 0, torch.linspace(batchStart, batchStart + batchSize, batchSize, dtype = int))
        zeroTransitions = torch.tensor([], dtype = int)
        for i in range(lowerSearchLimit, batchSize):
            if (sample[i-1] < 0) and (sample[i] > 0):
                zeroTransitions = torch.cat([zeroTransitions, torch.tensor([i])], 0)
        error = math.inf
        delta = math.floor(global_consts.sampleRate / audioSample.expectedPitch)
        for i in zeroTransitions:
            shiftedSample = torch.index_select(audioSample.waveform, 0, torch.linspace(batchStart + i.item(), batchStart + batchSize + i.item(), batchSize, dtype = int))
            newError = torch.sum(torch.pow(sample - shiftedSample, 2))
            if error > newError:
                delta = i.item()
                error = newError
        audioSample.pitchDeltas = torch.cat([audioSample.pitchDeltas, torch.tensor([delta])])
        batchStart += delta
    audioSample.pitch = torch.mean(audioSample.pitchDeltas.float()).int()

    #map sequence of pitchDeltas to sampling interval
    cursor = 0
    cursor2 = 0
    pitchDeltas = torch.empty(math.ceil(audioSample.pitchDeltas.sum() / global_consts.batchSize))
    for i in range(math.floor(audioSample.pitchDeltas.sum() / global_consts.batchSize)):
        while cursor2 >= audioSample.pitchDeltas[cursor]:
            cursor += 1
            cursor2 -= audioSample.pitchDeltas[cursor]
        cursor2 += global_consts.batchSize
        pitchDeltas[i] = audioSample.pitchDeltas[cursor]
    audioSample.pitchDeltas = pitchDeltas
