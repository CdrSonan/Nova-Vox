import torch
from math import floor

import global_consts
from Backend.DataHandler.AudioSample import AudioSample
from Backend.ESPER.SpecCalcComponents import finalizeSpectra, separateVoicedUnvoiced, lowRangeSmooth, highRangeSmooth, averageSpectra

def calculateSpectra(audioSample:AudioSample, useVariance:bool = True) -> None:
    """Method for calculating spectral data based on the previously set attributes filterWidth, voicedFilter and unvoicedIterations.
        
    Arguments:
        None
            
    Returns:
        None
            
    This function separates voiced and unvoiced excitation, and calculates the spectral envelope for the unvoiced part of an AudioSample object.
    The voiced/unvoiced separation is performed by calculating per-utterance phase and amplitude continuity functions from the windowed input signal.
    Based on these functions, the voiced signal is generated, and the unvoiced signal is calculated as the difference between the voiced signal and the input.
    The result is an STFT sequence of the unvoiced signal, and a sequence harmonic spectra of the voiced signal.
    Subsequently, the spectral envelope of the unvoiced signal is calculated, and the unvoiced excitation is obtained as the unvoiced signal divided by this spectral envelope.
    The spectral calculation uses two methods for low and high frequencies respectively:
    -an adaptation of the True Envelope Estimator
    -fourier space running mean smoothing
    The two results are then combined, forming the final spectrum.
    """


    length = floor(audioSample.waveform.size()[0] / global_consts.batchSize)
    audioSample.excitation = torch.empty((length, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64)
    audioSample.specharm = torch.empty((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3))
    signalsAbs = torch.stft(audioSample.waveform, global_consts.tripleBatchSize, global_consts.batchSize, global_consts.tripleBatchSize, torch.hann_window(global_consts.tripleBatchSize), return_complex = True)
    signalsAbs = torch.sqrt(signalsAbs.transpose(0, 1)[:audioSample.excitation.size()[0]].abs())
    lowSpectra = lowRangeSmooth(audioSample, signalsAbs)
    highSpectra = highRangeSmooth(audioSample, signalsAbs)
    audioSample = finalizeSpectra(audioSample, lowSpectra, highSpectra)
    audioSample = separateVoicedUnvoiced(audioSample)
    audioSample = averageSpectra(audioSample, useVariance)