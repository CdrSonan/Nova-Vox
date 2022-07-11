from Backend.DataHandler.AudioSample import AudioSample
from Backend.Resampler.CubicSplineInter import interp, extrap
import torch
import global_consts
import math

def calculateHighResSpectra(audioSample:AudioSample) -> tuple([torch.Tensor, AudioSample]):
    """calculates high spectral resolution, but low time resolution approximate spectra for an AudioSample object. This data can then be used to separate voiced and unvoiced excitation."""

    window = torch.hann_window(global_consts.tripleBatchSize * global_consts.filterBSMult)
    signals = torch.stft(audioSample.waveform, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, return_complex = True, onesided = True)
    signals = torch.transpose(signals, 0, 1)
    signalsAbs = signals.abs()
    spectralFilterWidth = torch.max(torch.floor(audioSample.pitch / global_consts.tripleBatchSize * global_consts.filterBSMult * global_consts.filterHRSSMult), torch.Tensor([1])).int().item()
    workingSpectra = torch.sqrt(signalsAbs)
    audioSample.spectra = workingSpectra.clone()
    for j in range(audioSample.voicedFilter):
        for i in range(spectralFilterWidth):
            audioSample.spectra = torch.roll(workingSpectra, -i, dims = 1) + audioSample.spectra + torch.roll(workingSpectra, i, dims = 1)
        audioSample.spectra = audioSample.spectra / (2 * spectralFilterWidth + 1)
        workingSpectra = torch.min(workingSpectra, audioSample.spectra)
        audioSample.spectra = workingSpectra
    return signals, audioSample

def calculateResonance(audioSample:AudioSample) -> tuple([torch.Tensor, AudioSample]):
    """calculates a resonance curve relative to the pitch of an audio sample in high-res fourier space. Frequencies with a higher resonance are more likely to be voiced."""

    resonanceFunction = torch.zeros_like(audioSample.spectra)
    for i in range(resonanceFunction.size()[0]):
        #sin(2*pi*a)/(a^2-1) with a = f/f0 = l0/l
        for j in range(global_consts.nFormants):
            freqspace = torch.linspace(0, audioSample.pitch / (j + 1), global_consts.halfTripleBatchSize * global_consts.filterBSMult + 1)
            freqspace = torch.sin(2 * math.pi * freqspace) / (torch.pow(freqspace, torch.full([1,], 2.)) - torch.ones([1,]))
            freqspace = torch.pow(freqspace / math.pi, torch.full([1,], 2.))# / (j + 1)
            resonanceFunction[i] = torch.max(resonanceFunction[i], freqspace)
        transitionPoint = min(int(global_consts.halfTripleBatchSize * global_consts.filterBSMult / audioSample.pitch * global_consts.nFormants), resonanceFunction.size()[1])
        resonanceFunction[i][transitionPoint - global_consts.nFormants:transitionPoint] *= torch.linspace(1, 0, global_consts.nFormants)
        resonanceFunction[i][transitionPoint - global_consts.nFormants:transitionPoint] += torch.linspace(0, 1, global_consts.nFormants)
        resonanceFunction[i][transitionPoint:] = torch.ones_like(resonanceFunction[i][transitionPoint:])
    return resonanceFunction, audioSample

def calculatePhaseContinuity(signals:torch.Tensor) -> torch.Tensor:
    """calculates phase continuity function of an stft sequence. Frequencies with a high phase continuity are more likely to be voiced."""

    phaseContinuity = torch.empty_like(signals, dtype = torch.float32)
    for i in range(phaseContinuity.size()[0] - 1):
        phaseContinuity[i] = signals[i].angle() - signals[i + 1].angle()
    phaseContinuity[-1] = phaseContinuity[-2]
    phaseContinuity = torch.remainder(phaseContinuity, torch.full([1,], 2 * math.pi))
    phaseContinuity -= math.pi
    phaseContinuity -= 2 * phaseContinuity * torch.heaviside(phaseContinuity, torch.zeros([1,]))
    phaseContinuity += math.pi
    phaseContinuity /= math.pi
    return phaseContinuity

def separateVoicedUnvoiced(audioSample:AudioSample, signals:torch.Tensor, resonanceFunction:torch.Tensor, phaseContinuity:torch.Tensor) -> AudioSample:
    """uses high-res spectrum magnitude in relation to stft magnitude, resonance and phase continuity to determine which parts of an AudioSample object are voiced"""

    signalsAbs = signals.abs()
    audioSample.voicedExcitation = signals.clone()
    audioSample.voicedExcitation *= torch.gt(signalsAbs * resonanceFunction * (1. - 0.2 * torch.pow(phaseContinuity, torch.tensor([2.,]))), audioSample.spectra * audioSample.voicedFilter)
    audioSample.excitation = signals.clone()
    audioSample.excitation *= torch.less_equal(signalsAbs * resonanceFunction * (1. - 0.2 * torch.pow(phaseContinuity, torch.tensor([2.,]))), audioSample.spectra * audioSample.voicedFilter)
    if audioSample.isVoiced == False:
        audioSample.voicedExcitation *= 0
    audioSample.voicedExcitation = torch.transpose(audioSample.voicedExcitation, 0, 1)
    audioSample.excitation = torch.transpose(audioSample.excitation, 0, 1)
    return audioSample

def transformHighRestoStdRes(audioSample:AudioSample) -> tuple([torch.Tensor, AudioSample]):
    """istft followed by stft. used to map the unvoiced and voiced excitation signals of an AudioSample object to the spectral resolution used during synthesis"""

    audioSample.excitation = torch.istft(audioSample.excitation, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, onesided = True)
    audioSample.voicedExcitation = torch.istft(audioSample.voicedExcitation, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, onesided = True)
    window = torch.hann_window(global_consts.tripleBatchSize)
    audioSample.excitation = torch.stft(audioSample.excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
    audioSample.voicedExcitation = torch.stft(audioSample.voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
    signalsAbs = audioSample.excitation.abs() + audioSample.voicedExcitation.abs()
    signalsAbs = torch.sqrt(torch.transpose(signalsAbs, 0, 1))
    return signalsAbs, audioSample

def lowRangeSmooth(audioSample: AudioSample, signalsAbs:torch.Tensor) -> torch.Tensor:
    """calculates a spectrum based on an adaptation of the True Envelope Estimator algorithm. Used for low-frequency area, as it can produce artifacting in high-frequency area"""

    spectra = signalsAbs.clone()
    spectralFilterWidth = torch.max(torch.floor(global_consts.tripleBatchSize * global_consts.filterTEEMult / audioSample.pitch), torch.Tensor([1])).int().item()
    spectralFilterWidth = min(spectralFilterWidth, math.floor(spectra.size()[1] / 2))
    for j in range(audioSample.unvoicedIterations):
        spectra = torch.maximum(spectra, signalsAbs)
        spectra = torch.fft.rfft(spectra, dim = 1)
        cutoffWindow = torch.zeros(spectra.size()[1])
        cutoffWindow[0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        spectra = torch.fft.irfft(cutoffWindow * spectra, dim = 1, n = global_consts.halfTripleBatchSize + 1)
    return spectra

def highRangeSmooth(audioSample:AudioSample, signalsAbs:torch.Tensor) -> torch.Tensor:
    """calculates a spectrum based on fourier space running mean smoothing. Used for high-frequency area, as it can produce oversmoothing in low-frequency area"""

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
    return spectra

def finalizeSpectra(audioSample:AudioSample, lowSpectra:torch.Tensor, highSpectra:torch.Tensor) -> AudioSample:
    """calculates final spectra of an AudioSample object based on low frequency range and high frequency range spectra"""

    threshold = torch.nn.Threshold(0.001, 0.001)
    slope = torch.ones_like(highSpectra)
    slope[:, global_consts.spectralRolloff2:] = 0.
    slope[:, global_consts.spectralRolloff1:global_consts.spectralRolloff2] = torch.linspace(1, 0, global_consts.spectralRolloff2 - global_consts.spectralRolloff1)
    lowSpectra = threshold(lowSpectra)
    highSpectra = threshold(highSpectra)
    audioSample.specharm[:, global_consts.nHarmonics + 2:] = slope * lowSpectra + ((1. - slope) * highSpectra)

    audioSample.spectrum = torch.mean(audioSample.specharm[:, global_consts.nHarmonics + 2:], 0)
    for i in range(audioSample.specharm.size()[0]):
        audioSample.specharm[i, global_consts.nHarmonics + 2:] = audioSample.specharm[i, global_consts.nHarmonics + 2:] - audioSample.spectrum

    #audioSample.voicedExcitation = audioSample.voicedExcitation / torch.transpose(torch.square(audioSample.spectrum + audioSample.spectra)[0:audioSample.voicedExcitation.size()[1]], 0, 1)
    audioSample.excitation = torch.transpose(audioSample.excitation, 0, 1) / torch.square(audioSample.spectrum + audioSample.specharm[0:audioSample.excitation.size()[1], 2 * global_consts.nHarmonics:])

    #window = torch.hann_window(global_consts.tripleBatchSize)
    #audioSample.voicedExcitation = torch.istft(audioSample.voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True)

def dioPitchMarkers(audioSample:AudioSample) -> list:
    """calculates pitch markers using the DIO algorithm, for later use in spectral processing"""

    wave = torch.cat((torch.zeros([global_consts.halfTripleBatchSize * global_consts.filterBSMult,]), audioSample.waveform, torch.zeros([global_consts.halfTripleBatchSize * global_consts.filterBSMult,])), 0)
    length = math.floor(audioSample.waveform.size()[0] / global_consts.batchSize)# + 1 ???
    windows = torch.empty((length, global_consts.tripleBatchSize * global_consts.filterBSMult))
    audioSample.excitation = torch.empty((length, global_consts.halfTripleBatchSize * global_consts.filterBSMult + 1))
    audioSample.specharm = torch.empty((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3))
    for i in range(length):
        windows[i] = wave[i * global_consts.batchSize:i * global_consts.batchSize + global_consts.tripleBatchSize * global_consts.filterBSMult]
    counter = 0
    for i in windows:
        pitch = audioSample.pitchDeltas[counter]
        upTransitionMarkers = torch.tensor([], dtype = torch.int16)
        downTransitionMarkers = torch.tensor([], dtype = torch.int16)
        maximumMarkers = torch.tensor([], dtype = torch.int16)
        minimumMarkers = torch.tensor([], dtype = torch.int16)

        zeroTransitions = torch.tensor([], dtype = int)
        for j in range(1, i.size()[0]):
            if (i[j-1] < 0) and (i[j] >= 0):
                zeroTransitions = torch.cat([zeroTransitions, torch.tensor([j])], 0)
        error = math.inf
        base = 0
        while base < global_consts.tripleBatchSize * global_consts.filterBSMult - pitch:
            validTransitions = []
            transition = base + pitch
            for j in zeroTransitions:
                if j > base + 1.5 * pitch:
                    break
                if j > base + 0.5 * pitch:
                    validTransitions.append(j)
            for j in validTransitions:
                if j + pitch > global_consts.tripleBatchSize * global_consts.filterBSMult:
                    sample = i[base - pitch:base]
                    shiftedSample = i[j - pitch:j]
                else:
                    sample = i[base:base + pitch]
                    shiftedSample = i[j:j + pitch]
                newError = torch.sum(torch.pow(sample - shiftedSample, 2))
                if error > newError:
                    transition = j.item()
                    error = newError
            upTransitionMarkers = torch.cat((upTransitionMarkers, torch.tensor([transition], dtype = torch.int16)), 0)
            base = transition

        zeroTransitions = torch.tensor([], dtype = int)
        for j in range(1, i.size()[0]):
            if (i[j-1] >= 0) and (i[j] < 0):
                zeroTransitions = torch.cat([zeroTransitions, torch.tensor([j])], 0)
        error = math.inf
        base = 0
        while base < global_consts.tripleBatchSize * global_consts.filterBSMult - pitch:
            validTransitions = []
            transition = base + pitch
            for j in zeroTransitions:
                if j > base + 1.5 * pitch:
                    break
                if j > base + 0.5 * pitch:
                    validTransitions.append(j)
            for j in validTransitions:
                if j + pitch > global_consts.tripleBatchSize * global_consts.filterBSMult:
                    sample = i[base - pitch:base]
                    shiftedSample = i[j - pitch:j]
                else:
                    sample = i[base:base + pitch]
                    shiftedSample = i[j:j + pitch]
                newError = torch.sum(torch.pow(sample - shiftedSample, 2))
                if error > newError:
                    transition = j.item()
                    error = newError
            downTransitionMarkers = torch.cat((downTransitionMarkers, torch.tensor([transition], dtype = torch.int16)), 0)
            base = transition

        length = min(len(upTransitionMarkers), len(downTransitionMarkers))
        upTransitionMarkers = upTransitionMarkers[:length]
        downTransitionMarkers = downTransitionMarkers[:length]

        for j in range(length - 1):
            window = i[upTransitionMarkers[j] - 1:upTransitionMarkers[j + 1] + 1]
            convKernel = torch.tensor([1., 1.5, 1.])
            scores = torch.tensor([])
            for k in range(window.size()[0] - 3):
                scores = torch.cat((scores, torch.unsqueeze(torch.sum(window[k:k+3] * convKernel), 0)), 0)
            maximumMarkers = torch.cat((maximumMarkers, torch.unsqueeze(torch.argmax(scores) + upTransitionMarkers[j], 0)), 0)

        window = i[upTransitionMarkers[-1] - 1:-1]
        convKernel = torch.tensor([1., 1.5, 1.])
        scores = torch.tensor([])
        for k in range(window.size()[0] - 3):
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(window[k:k+3] * convKernel), 0)), 0)
        maximumMarkers = torch.cat((maximumMarkers, torch.unsqueeze(torch.argmax(scores) + upTransitionMarkers[-1], 0)), 0)

        for j in range(length - 1):
            window = i[downTransitionMarkers[j] - 1:downTransitionMarkers[j + 1] + 1]
            convKernel = torch.tensor([-1., -1.5, -1.])
            scores = torch.tensor([])
            for k in range(window.size()[0] - 3):
                scores = torch.cat((scores, torch.unsqueeze(torch.sum(window[k:k+3] * convKernel), 0)), 0)
            minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(torch.argmax(scores) + downTransitionMarkers[j], 0)), 0)

        window = i[downTransitionMarkers[-1] - 1:-1]
        convKernel = torch.tensor([-1., -1.5, -1.])
        scores = torch.tensor([])
        for k in range(window.size()[0] - 3):
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(window[k:k+3] * convKernel), 0)), 0)
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(torch.argmax(scores) + downTransitionMarkers[-1], 0)), 0)

        markers = torch.tensor([])
        for j in range(length):
            markers = torch.cat((markers, torch.unsqueeze((upTransitionMarkers[j] + downTransitionMarkers[j] + maximumMarkers[j] + minimumMarkers[j]) / 4, 0)), 0)

        interpolationPoints = interp(torch.linspace(0, len(markers) - 1, len(markers)), markers, torch.linspace(0, len(markers) - 1, len(markers) * global_consts.nHarmonics))
        interpolatedWave = interp(torch.linspace(0, i.size()[0] - 1, i.size()[0]), i, interpolationPoints)
        if length == 1:
            #fallback
            pass
        else:
            harmFunction = torch.tensor([])
            for j in range(global_consts.nHarmonics):
                harm = 0
                harm += interpolatedWave[j] * (j / global_consts.nHarmonics)
                for k in range(1, length - 1):
                    harm += interpolatedWave[j + k * global_consts.nHarmonics]
                harm += interpolatedWave[length + j] * (1 - (j / global_consts.nHarmonics))
                harmFunction = torch.cat((harmFunction, torch.unsqueeze(harm, 0)), 0)
            harmFunctionFull = torch.tile(harmFunction, (length,))
            harmFunctionFull = extrap(interpolationPoints, harmFunctionFull, torch.linspace(0, i.size()[0] - 1, i.size()[0]))
            offharm = i - harmFunctionFull
            offharm *= torch.hann_window(global_consts.tripleBatchSize * global_consts.filterBSMult)
            audioSample.excitation[counter] = torch.fft.rfft(offharm)
            harmFunction = torch.roll(harmFunction, int(markers[0].item()))
            harmFunction = torch.fft.rfft(harmFunction)
            harmFunction = torch.cat((harmFunction.abs(), harmFunction.angle()), 0)
            audioSample.specharm[counter, :global_consts.nHarmonics + 2] = harmFunction
            audioSample.phases[counter] = audioSample.specharm[counter, int(global_consts.nHarmonics / 2) + 1]
        counter += 1
    audioSample.excitation = torch.istft(audioSample.excitation, global_consts.halfTripleBatchSize * global_consts.filterBSMult, global_consts.batchSize, global_consts.halfTripleBatchSize * global_consts.filterBSMult)
    audioSample.excitation = torch.stft(audioSample.excitation, global_consts.tripleBatchSize, global_consts.batchSize, global_consts.tripleBatchSize, torch.hann_window(global_consts.tripleBatchSize))
    return audioSample
