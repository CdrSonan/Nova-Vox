import torch

from Backend.DataHandler.AudioSample import AudioSample
from Backend.ESPER.SpecCalcComponents import finalizeSpectra, separateVoicedUnvoiced, lowRangeSmooth, highRangeSmooth

def calculateSpectra(audioSample:AudioSample) -> None:
    """Method for calculating spectral data based on the previously set attributes filterWidth, voicedFilter and unvoicedIterations.
        
    Arguments:
        None
            
    Returns:
        None
            
    This function separates voiced and unvoiced excitation, and calculates the spectral envelope for an AudioSample object.
    The voiced/unvoiced separation is performed in an stft space with higher spectral, but lower time resolution that is used during synthesis.
    It is based on three metrics:
    -stft frequency amplitude compared to the surrounding frequencies
    -resonance with f0 frequency
    -phase continuity between stft bins
    Based on these metrics, each frequency is flagged as either voiced or unvoiced. There are currently no partly voiced frequencies.
    The spectral calculation uses two methods for low and high frequencies respectively:
    -an adaptation of the True Envelope Estimator
    -fourier space running mean smoothing
    The two results are then combined, forming the final spectrum.
    """


    #signals, audioSample = calculateHighResSpectra(audioSample)
    #resonanceFunction, audioSample = calculateResonance(audioSample)
    #phaseContinuity = calculatePhaseContinuity(signals)
    #audioSample = separateVoicedUnvoiced(audioSample, signals, resonanceFunction, phaseContinuity) !!!replace!!!
    #signalsAbs, audioSample = transformHighRestoStdRes(audioSample)
    audioSample = separateVoicedUnvoiced(audioSample)
    signalsAbs = audioSample.excitation.abs()
    signalsAbs = torch.sqrt(torch.transpose(signalsAbs, 0, 1))
    lowSpectra = lowRangeSmooth(audioSample, signalsAbs)
    highSpectra = highRangeSmooth(audioSample, signalsAbs)
    audioSample = finalizeSpectra(audioSample, lowSpectra, highSpectra)
