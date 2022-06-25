from math import pi, floor, ceil
import torch
import torchaudio
from Backend.DataHandler.AudioSample import AudioSample
import global_consts

def calculateSpectra(audioSample:AudioSample):
    """Method for calculating spectral data based on the previously set attributes filterWidth, voicedFilter and unvoicedIterations.
        
    Arguments:
        None
            
    Returns:
        None
            
    The spectral calculation uses an adaptation of the True Envelope Estimator. It works with a fixed smoothing range determined by filterWidth (inta fourier space data points).
    The algorithm first runs an amount of filtering iterations determined by voicedIterations, selectively saves the peaking frequencies of the signal into _voicedExcitations, 
    then runs the filtering algorithm again a number of iterations determined by unvoicedIterations.
    The function fills the spectrum, spectra and _voicedExcitations properties."""


    threshold = torch.nn.Threshold(0.001, 0.001)
    window = torch.hann_window(global_consts.tripleBatchSize * global_consts.filterBSMult)
    signals = torch.stft(audioSample.waveform, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, return_complex = True, onesided = True)
    signals = torch.transpose(signals, 0, 1)
    signalsAbs = signals.abs()
    
    #spectrum approximation for voiced/unvoiced separation
    spectralFilterWidth = torch.max(torch.floor(audioSample.pitch / global_consts.tripleBatchSize * global_consts.filterBSMult * global_consts.filterHRSSMult), torch.Tensor([1])).int().item()
    workingSpectra = torch.sqrt(signalsAbs)
    audioSample.spectra = workingSpectra.clone()
    for j in range(audioSample.voicedFilter):
        for i in range(spectralFilterWidth):
            audioSample.spectra = torch.roll(workingSpectra, -i, dims = 1) + audioSample.spectra + torch.roll(workingSpectra, i, dims = 1)
        audioSample.spectra = audioSample.spectra / (2 * spectralFilterWidth + 1)
        workingSpectra = torch.min(workingSpectra, audioSample.spectra)
        audioSample.spectra = workingSpectra
    
    #resonance calculations
    resonanceFunction = torch.zeros_like(audioSample.spectra)
    for i in range(resonanceFunction.size()[0]):
        #sin(2*pi*a)/(a^2-1) with a = f/f0 = l0/l
        for j in range(global_consts.nFormants):
            freqspace = torch.linspace(0, audioSample.pitch / (j + 1), global_consts.halfTripleBatchSize * global_consts.filterBSMult + 1)
            freqspace = torch.sin(2 * pi * freqspace) / (torch.pow(freqspace, torch.full([1,], 2.)) - torch.ones([1,]))
            freqspace = torch.pow(freqspace / pi, torch.full([1,], 2.))# / (j + 1)
            resonanceFunction[i] = torch.max(resonanceFunction[i], freqspace)
        transitionPoint = min(int(global_consts.halfTripleBatchSize * global_consts.filterBSMult / audioSample.pitch * global_consts.nFormants), resonanceFunction.size()[1])
        resonanceFunction[i][transitionPoint - global_consts.nFormants:transitionPoint] *= torch.linspace(1, 0, global_consts.nFormants)
        resonanceFunction[i][transitionPoint - global_consts.nFormants:transitionPoint] += torch.linspace(0, 1, global_consts.nFormants)
        resonanceFunction[i][transitionPoint:] = torch.ones_like(resonanceFunction[i][transitionPoint:])

    #phase continuity calculations
    phaseContinuity = torch.empty_like(signals, dtype = torch.float32)
    for i in range(phaseContinuity.size()[0] - 1):
        phaseContinuity[i] = signals[i].angle() - signals[i + 1].angle()
    phaseContinuity[-1] = phaseContinuity[-2]
    phaseContinuity = torch.remainder(phaseContinuity, torch.full([1,], 2 * pi))
    phaseContinuity -= pi
    phaseContinuity -= 2 * phaseContinuity * torch.heaviside(phaseContinuity, torch.zeros([1,]))
    phaseContinuity += pi
    phaseContinuity /= pi

    #voiced/unvoiced separation
    audioSample.voicedExcitation = signals.clone()
    audioSample.voicedExcitation *= torch.gt(signalsAbs * resonanceFunction * (1. - 0.2 * torch.pow(phaseContinuity, torch.tensor([2.,]))), audioSample.spectra * audioSample.voicedFilter)
    audioSample.excitation = signals.clone()
    audioSample.excitation *= torch.less_equal(signalsAbs * resonanceFunction * (1. - 0.2 * torch.pow(phaseContinuity, torch.tensor([2.,]))), audioSample.spectra * audioSample.voicedFilter)
    if audioSample.isVoiced == False:
        audioSample.voicedExcitation *= 0

    audioSample.voicedExcitation = torch.transpose(audioSample.voicedExcitation, 0, 1)
    audioSample.excitation = torch.transpose(audioSample.excitation, 0, 1)

    #fourier space transforms
    audioSample.excitation = torch.istft(audioSample.excitation, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, onesided = True)
    audioSample.voicedExcitation = torch.istft(audioSample.voicedExcitation, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, onesided = True)

    window = torch.hann_window(global_consts.tripleBatchSize)

    audioSample.excitation = torch.stft(audioSample.excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
    audioSample.voicedExcitation = torch.stft(audioSample.voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)

    #signals = torch.stft(audioSample.waveform, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
    signals = audioSample.excitation.abs() + audioSample.voicedExcitation.abs()
    signals = torch.transpose(signals, 0, 1)
    #signalsAbs = signals.abs()
    signalsAbs = signals
    signalsAbs = torch.sqrt(signalsAbs)

    #low-range quefrency-space lowpass smoothing
    audioSample.spectra = signalsAbs.clone()
    spectralFilterWidth = torch.max(torch.floor(global_consts.tripleBatchSize * global_consts.filterTEEMult / audioSample.pitch), torch.Tensor([1])).int().item()
    spectralFilterWidth = min(spectralFilterWidth, floor(audioSample.spectra.size()[1] / 2))
    for j in range(audioSample.unvoicedIterations):
        audioSample.spectra = torch.maximum(audioSample.spectra, signalsAbs)
        audioSample.spectra = torch.fft.rfft(audioSample.spectra, dim = 1)
        cutoffWindow = torch.zeros(audioSample.spectra.size()[1])
        cutoffWindow[0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        audioSample.spectra = torch.fft.irfft(cutoffWindow * audioSample.spectra, dim = 1, n = global_consts.halfTripleBatchSize + 1)

    #high-range frequency-space running mean smoothing
    spectralFilterWidth = torch.max(torch.floor(audioSample.pitch / global_consts.tripleBatchSize * global_consts.filterHRSSMult), torch.Tensor([1])).int()
    workingSpectra = signalsAbs.clone()
    workingSpectra = torch.cat((workingSpectra, torch.tile(torch.unsqueeze(workingSpectra[:, -1], 1), (1, audioSample.unvoicedIterations))), 1)
    spectra = workingSpectra.clone()
    for j in range(audioSample.unvoicedIterations):
        for i in range(spectralFilterWidth):
            spectra = torch.roll(workingSpectra, -i, dims = 1) + spectra + torch.roll(workingSpectra, i, dims = 1)
        spectra = spectra / (2 * spectralFilterWidth + 1)
        workingSpectra = torch.max(workingSpectra, spectra)
        spectra = workingSpectra
    spectra = spectra[:, 0:global_consts.halfTripleBatchSize + 1]

    #finalisation
    slope = torch.ones_like(spectra)
    slope[:, global_consts.spectralRolloff2:] = 0.
    slope[:, global_consts.spectralRolloff1:global_consts.spectralRolloff2] = torch.linspace(1, 0, global_consts.spectralRolloff2 - global_consts.spectralRolloff1)
    audioSample.spectra = threshold(audioSample.spectra)
    spectra = threshold(spectra)
    audioSample.spectra = slope * audioSample.spectra + ((1. - slope) * spectra)

    audioSample.spectrum = torch.mean(audioSample.spectra, 0)
    for i in range(audioSample.spectra.size()[0]):
        audioSample.spectra[i] = audioSample.spectra[i] - audioSample.spectrum

    audioSample.voicedExcitation = audioSample.voicedExcitation / torch.transpose(torch.square(audioSample.spectrum + audioSample.spectra)[0:audioSample.voicedExcitation.size()[1]], 0, 1)
    audioSample.excitation = torch.transpose(audioSample.excitation, 0, 1) / torch.square(audioSample.spectrum + audioSample.spectra)[0:audioSample.excitation.size()[1]]

    audioSample.voicedExcitation = torch.istft(audioSample.voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True)

    #phase calculations
    audioSample.phases = torch.empty_like(audioSample.pitchDeltas)
    """
        for i in range(audioSample.pitchDeltas.size()[0]):
            position = global_consts.tripleBatchSize / audioSample.pitchDeltas[i].item()
            lowerBin = floor(position)
            upperBin = ceil(position)
            lowerFactor = signals[i][lowerBin].abs() / (signals[i][lowerBin].abs() + signals[i][upperBin].abs())
            upperFactor = signals[i][upperBin].abs() / (signals[i][lowerBin].abs() + signals[i][upperBin].abs())
            lowerFactor += 1 - position + lowerBin
            upperFactor += 1 - upperBin + position
            lowerFactor = int(lowerFactor / 2)
            upperFactor = int(upperFactor / 2)
            audioSample.phases[i] = lowerFactor * signals[i][lowerBin].angle() + upperFactor * signals[i][upperBin].angle()
            func = audioSample.waveform[i * global_consts.batchSize:(i + 1) * global_consts.batchSize]
            arange = torch.arange(0, global_consts.batchSize, 2 * pi / audioSample.pitchDeltas[i])
            sine = torch.sin(arange)
            cosine = torch.cos(arange)
            sine *= func
            cosine *= func
            sine = torch.sum(sine)# / pi
            cosine = torch.sum(cosine)# / pi
            audioSample.phases[i] = torch.complex(sine, cosine).angle()
            if i > 0:
                audioSample.phases[i] += floor(global_consts.batchSize / audioSample.pitchDeltas[i - 1])"""
    previousLimit = 0
    previousPhase = 0
    for i in range(audioSample.pitchDeltas.size()[0]):
        limit = floor(global_consts.batchSize / audioSample.pitchDeltas[i])
        func = audioSample.waveform[i * global_consts.batchSize:i * global_consts.batchSize + limit * audioSample.pitchDeltas[i].to(torch.int64)]
        funcspace = torch.linspace(0, (limit * audioSample.pitchDeltas[i] - 1) * 2 * pi / audioSample.pitchDeltas[i], limit * audioSample.pitchDeltas[i])
#TODO: Test this, move to dedicated function called from calculatePitch()
        func = audioSample.waveform[i * global_consts.batchSize:(i + 1) * global_consts.batchSize.to(torch.int64)]
        funcspace = torch.linspace(0, global_consts.batchSize * 2 * pi / audioSample.pitchDeltas[i], global_consts.batchSize)

        sine = torch.sin(funcspace)
        cosine = torch.cos(funcspace)
        sine *= func
        cosine *= func
        sine = torch.sum(sine)# / pi (would be required for normalization, but amplitude is irrelevant here, so normalization is not required)
        cosine = torch.sum(cosine)# / pi
        phase = torch.complex(sine, cosine).angle()
        if phase < 0:
            phase += 2 * pi
        offset = previousLimit
        if phase < previousPhase % (2 * pi):
            offset += 1
        phase += 2 * pi * offset
        audioSample.phases[i] = phase
        previousLimit = limit + offset
        previousPhase = phase
        