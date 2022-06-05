from math import ceil
import torch
from torchaudio.functional import detect_pitch_frequency
import global_consts

def calculatePitch(audioSample):
    """Method for calculating pitch data based on previously set attributes expectedPitch and searchRange.
    
    Arguments:
        None
    
    Returns:
        None
    
    The pitch calculation uses 0-transitions to determine the borders between vocal chord vibrations. The algorithm searches for such transitions around expectedPitch (should be a value in Hz),
    with the range around it being defined by searchRange (should be a value between 0 and 1), which is interpreted as a percentage of the wavelength of expectedPitch.
    The function fills the pitchDeltas and pitch properties."""
    
    
    """batchSize = math.floor((1. + audioSample.searchRange) * global_consts.sampleRate / audioSample.expectedPitch)
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
    nBatches = audioSample.pitchDeltas.size()[0]
    audioSample.pitchBorders = torch.zeros(nBatches + 1, dtype = int)
    for i in range(nBatches):
        audioSample.pitchBorders[i+1] = audioSample.pitchBorders[i] + audioSample.pitchDeltas[i]
    audioSample.pitch = torch.mean(audioSample.pitchDeltas.float()).int()"""

    #cursor = 0
    #cursor2 = 0
    #pitchDeltas = torch.empty(math.ceil(audioSample.pitchDeltas.sum() / global_consts.batchSize))
    #for i in range(math.floor(audioSample.pitchDeltas.sum() / global_consts.batchSize)):
    #    while cursor2 >= audioSample.pitchDeltas[cursor]:
    #        cursor += 1
    #        cursor2 -= audioSample.pitchDeltas[cursor]
    #    cursor2 += global_consts.batchSize
    #    pitchDeltas[i] = audioSample.pitchDeltas[cursor]
    #audioSample.pitchDeltasFull = pitchDeltas
    try:
        audioSample.pitchDeltas = global_consts.sampleRate / detect_pitch_frequency(audioSample.waveform, global_consts.sampleRate, 1. / global_consts.tickRate, 30, audioSample.expectedPitch * (1 - audioSample.searchRange), audioSample.expectedPitch * (1 + audioSample.searchRange))
        mul1 = torch.maximum(torch.floor(audioSample.pitchDeltas / audioSample.expectedPitch), torch.ones([1,]))
        mul2 = torch.maximum(torch.floor(audioSample.expectedPitch / audioSample.pitchDeltas), torch.ones([1,]))
        audioSample.pitchDeltas /= mul1
        audioSample.pitchDeltas *= mul2
        #mask = audioSample.pitchDeltas.less(torch.tensor([global_consts.sampleRate / ,]))
    except Exception as e:
        print("error during pitch detection; falling back to default values", e)
        audioSample.pitchDeltas = torch.full([ceil(audioSample.waveform.size()[0] / global_consts.batchSize),], audioSample.expectedPitch)
    #print("pitch: ", audioSample.pitchDeltas)
    audioSample.pitch = torch.mean(audioSample.pitchDeltas).int()
    audioSample.pitchDeltas = audioSample.pitchDeltas.to(torch.int16)