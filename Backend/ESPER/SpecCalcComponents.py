import torchaudio
from Backend.DataHandler.AudioSample import AudioSample
from Backend.Resampler.CubicSplineInter import interp, extrap
from Backend.Resampler.PhaseShift import phaseShiftFourier
import torch
import global_consts
import math

import matplotlib.pyplot as plt

def calculatePhaseContinuity(signals:torch.Tensor) -> torch.Tensor:
    """calculates phase continuity function of an stft sequence. Frequencies with a high phase continuity are more likely to be voiced."""

    diff = torch.zeros_like(signals, dtype = torch.float32)
    if diff.size()[0] == 1:
        return diff
    diff = diff[:-1]
    for i in range(diff.size()[0]):
        diff[i] = signals[i + 1].angle() - signals[i].angle()
    diff = torch.remainder(diff, 2 * math.pi)
    diffB = diff - 2 * math.pi
    mask = torch.ge(diff.abs(), diffB.abs())
    diff -= mask.to(torch.float) * 2 * math.pi
    diff = torch.abs(diff)
    return 1. - (diff / math.pi)

def calculateAmplitudeContinuity(amplitudes:torch.Tensor) -> torch.Tensor:
    if amplitudes.size()[1] == 1:
        return torch.ones_like(amplitudes)
    amplitudeContinuity = amplitudes.clone()
    amplitudeContinuity[:,1:] += amplitudes[:, :-1]
    amplitudeContinuity[:,:-1] += amplitudes[:, 1:]
    amplitudeContinuity[:,1:-1] = amplitudeContinuity[:,1:-1] / 3.
    amplitudeContinuity[:, 0] = amplitudeContinuity[:, 0] / 2.
    amplitudeContinuity[:, -1] = amplitudeContinuity[:, -1] / 2.
    amplitudeContinuity /= amplitudes
    #amplitudeContinuity = torch.cos((torch.min(amplitudeContinuity / amplitudes, amplitudes / amplitudeContinuity) - 1) * math.pi / 2.)
    return amplitudeContinuity

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

    audioSample.avgSpecharm = torch.mean(torch.cat((audioSample.specharm[:, :int(global_consts.nHarmonics / 2) + 1], audioSample.specharm[:, global_consts.nHarmonics + 2:]), 1), 0)
    for i in range(audioSample.specharm.size()[0]):
        audioSample.specharm[i, global_consts.nHarmonics + 2:] -= audioSample.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]
        audioSample.specharm[i, :int(global_consts.nHarmonics / 2) + 1] -= audioSample.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1]

    audioSample.excitation = audioSample.excitation / torch.square(audioSample.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:] + audioSample.specharm[0:audioSample.excitation.size()[1], global_consts.nHarmonics + 2:])

def DIOPitchMarkers(audioSample:AudioSample, window:torch.Tensor, counter:int) -> list:
    pitch = audioSample.pitchDeltas[counter]
    maximumMarkers = torch.tensor([], dtype = torch.int16)
    minimumMarkers = torch.tensor([], dtype = torch.int16)

    zeroTransitionsUp = torch.tensor([], dtype = int)
    for j in range(2, window.size()[0]):
        if (window[j-1] < 0) and (window[j] >= 0):
            zeroTransitionsUp = torch.cat([zeroTransitionsUp, torch.tensor([j])], 0)
    zeroTransitionsDown = torch.tensor([], dtype = int)
    for j in range(2, window.size()[0]):
        if (window[j-1] >= 0) and (window[j] < 0):
            zeroTransitionsDown = torch.cat([zeroTransitionsDown, torch.tensor([j])], 0)

    upTransitionCandidates = zeroTransitionsUp[0:torch.searchsorted(zeroTransitionsUp, zeroTransitionsUp[0] + pitch)]
    derrs = torch.index_select(window, 0, upTransitionCandidates) - torch.index_select(window, 0, upTransitionCandidates - 1)
    upTransitionMarkers = torch.unsqueeze(upTransitionCandidates[torch.argmax(derrs)], 0)
    downTransitionCandidates = zeroTransitionsDown[torch.searchsorted(zeroTransitionsDown, upTransitionMarkers[0]):torch.searchsorted(zeroTransitionsDown, upTransitionMarkers[0] + pitch)]
    derrs = torch.index_select(window, 0, downTransitionCandidates) - torch.index_select(window, 0, downTransitionCandidates - 1)
    downTransitionMarkers = torch.unsqueeze(downTransitionCandidates[torch.argmin(derrs)], 0)

    print(zeroTransitionsUp)
    print(zeroTransitionsDown)

    while downTransitionMarkers[-1] < global_consts.tripleBatchSize * global_consts.filterBSMult - pitch * global_consts.DIOLastWinTolerance:
        error = math.inf
        validTransitions = []
        if upTransitionMarkers.size()[0] > 1:
            transition = upTransitionMarkers[-1] + downTransitionMarkers[-1] - downTransitionMarkers[-2]
        else:
            transition = upTransitionMarkers[-1] + pitch
        for j in zeroTransitionsUp:
            if j > min(upTransitionMarkers[-1] + (1 + global_consts.DIOTolerance) * pitch, global_consts.tripleBatchSize * global_consts.filterBSMult):
                break
            if j > max(downTransitionMarkers[-1], upTransitionMarkers[-1] + (1 - global_consts.DIOTolerance) * pitch):
                validTransitions.append(j)
        for j in validTransitions:
            if j + pitch > global_consts.tripleBatchSize * global_consts.filterBSMult:
                if pitch > upTransitionMarkers[-1]:
                    continue
                sample = window[upTransitionMarkers[-1] - pitch:upTransitionMarkers[-1]]
                shiftedSample = window[j - pitch:j]
            else:
                sample = window[upTransitionMarkers[-1]:upTransitionMarkers[-1] + pitch]
                shiftedSample = window[j:j + pitch]
            newError = torch.sum(torch.pow(sample - shiftedSample, 2)) * ((j - upTransitionMarkers[-1]) / pitch * global_consts.DIOBias + 1. - global_consts.DIOBias)
            if error > newError:
                transition = j.item()
                error = newError
        upTransitionMarkers = torch.cat((upTransitionMarkers, torch.tensor([transition], dtype = torch.int16)), 0)

        print(upTransitionMarkers, downTransitionMarkers, validTransitions)

        error = math.inf
        validTransitions = []
        transition = downTransitionMarkers[-1] + upTransitionMarkers[-1] - upTransitionMarkers[-2]
        for j in zeroTransitionsDown:
            if j > min(downTransitionMarkers[-1] + (1 + global_consts.DIOTolerance) * pitch, global_consts.tripleBatchSize * global_consts.filterBSMult):
                break
            if j > max(upTransitionMarkers[-1], downTransitionMarkers[-1] + (1 - global_consts.DIOTolerance) * pitch):
                validTransitions.append(j)
        for j in validTransitions:
            if j + pitch > global_consts.tripleBatchSize * global_consts.filterBSMult:
                if pitch > downTransitionMarkers[-1]:
                    continue
                sample = window[downTransitionMarkers[-1] - pitch:downTransitionMarkers[-1]]
                shiftedSample = window[j - pitch:j]
            else:
                sample = window[downTransitionMarkers[-1]:downTransitionMarkers[-1] + pitch]
                shiftedSample = window[j:j + pitch]
            newError = torch.sum(torch.pow(sample - shiftedSample, 2)) * ((j - downTransitionMarkers[-1]) / pitch * global_consts.DIOBias + 1. - global_consts.DIOBias)
            if error > newError:
                transition = j.item()
                error = newError
        downTransitionMarkers = torch.cat((downTransitionMarkers, torch.tensor([transition], dtype = torch.int16)), 0)

        print(upTransitionMarkers, downTransitionMarkers, validTransitions)

    if downTransitionMarkers[-1] >= global_consts.tripleBatchSize * global_consts.filterBSMult:
        upTransitionMarkers = upTransitionMarkers[:-1]
        downTransitionMarkers = downTransitionMarkers[:-1]
    length = len(upTransitionMarkers)

    for j in range(length):
        workingWindow = window[upTransitionMarkers[j] - 2:downTransitionMarkers[j] + 1]
        convKernel = torch.tensor([1., 1.1, 1.])
        scores = torch.tensor([])
        for k in range(workingWindow.size()[0] - 2):
            bias = 1. - (global_consts.DIOBias * k / (workingWindow.size()[0] - 2))
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        maximumMarkers = torch.cat((maximumMarkers, torch.unsqueeze(torch.argmax(scores) + upTransitionMarkers[j], 0)), 0)
    for j in range(length - 1):
        workingWindow = window[downTransitionMarkers[j] - 2:upTransitionMarkers[j + 1] + 1]
        convKernel = torch.tensor([-1., -1.1, -1.])
        scores = torch.tensor([])
        for k in range(workingWindow.size()[0] - 2):
            bias = 1. - (global_consts.DIOBias * k / (workingWindow.size()[0] - 2))
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(torch.argmax(scores) + downTransitionMarkers[j], 0)), 0)
    if window.size()[0] - downTransitionMarkers[-1] > pitch * global_consts.DIOLastWinTolerance:
        workingWindow = window[downTransitionMarkers[-1] - 2:-1]
        convKernel = torch.tensor([-1., -1.1, -1.])
        scores = torch.tensor([])
        for k in range(workingWindow.size()[0] - 2):
            bias = 1. - (global_consts.DIOBias * k / (workingWindow.size()[0] - 2))
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(torch.argmax(scores) + downTransitionMarkers[-1], 0)), 0)
    else:
        marker = minimumMarkers[-1] + torch.mean(torch.tensor((upTransitionMarkers[-1] - upTransitionMarkers[-2], downTransitionMarkers[-1] - downTransitionMarkers[-2], maximumMarkers[-1] - maximumMarkers[-2]), dtype = torch.float32))
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(marker, 0)), 0)

    markers = torch.tensor([])
    for j in range(length):
        markers = torch.cat((markers, torch.unsqueeze((upTransitionMarkers[j] + downTransitionMarkers[j] + maximumMarkers[j] + minimumMarkers[j]) / 4, 0)), 0)
    if markers[-1] >= global_consts.tripleBatchSize * global_consts.filterBSMult - 1:
        markers = markers[:-1]
    return markers

def separateVoicedUnvoiced(audioSample:AudioSample) -> AudioSample:
    """calculates pitch markers using the DIO algorithm, for later use in spectral processing"""

    wave = torch.cat((torch.zeros([global_consts.halfTripleBatchSize * global_consts.filterBSMult,]), audioSample.waveform, torch.zeros([global_consts.halfTripleBatchSize * global_consts.filterBSMult,])), 0)
    length = math.floor(audioSample.waveform.size()[0] / global_consts.batchSize)
    windows = torch.empty((length, global_consts.tripleBatchSize * global_consts.filterBSMult))
    audioSample.excitation = torch.empty((length, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64)
    audioSample.specharm = torch.empty((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3))
    for i in range(length):
        windows[i] = wave[i * global_consts.batchSize:i * global_consts.batchSize + global_consts.tripleBatchSize * global_consts.filterBSMult]
    counter = 0
    for i in windows:
        print(counter, "/", windows.size()[0])
        markers = DIOPitchMarkers(audioSample, i, counter)
        length = len(markers)
        if length == 1:
            #fallback
            print("case")
            counter += 1
            continue
        interpolationPoints = interp(torch.linspace(0, len(markers) - 1, len(markers)), markers, torch.linspace(0, len(markers) - 1, (len(markers) - 1) * global_consts.nHarmonics + 1))
        interpolatedWave = interp(torch.linspace(0, i.size()[0] - 1, i.size()[0]), i, interpolationPoints)

        harmFunction = torch.empty((int(global_consts.nHarmonics / 2) + 1, 0))

        for j in range(length - 1):
            harmFunction = torch.cat((harmFunction, torch.unsqueeze(torch.fft.rfft(interpolatedWave[j * global_consts.nHarmonics:(j + 1) * global_consts.nHarmonics]), -1)), 1)

        amplitudes = harmFunction.abs()
        amplitudeContinuity = calculateAmplitudeContinuity(amplitudes)
        phaseContinuity = calculatePhaseContinuity(harmFunction.transpose(0, 1)).transpose(0, 1)
        phaseContinuity = torch.pow(phaseContinuity, 2)
        amplitudes *= amplitudeContinuity * torch.unsqueeze(torch.max(phaseContinuity, dim = 1)[0], -1)
        harmFunction = torch.polar(amplitudes, harmFunction.angle())
        harmFunctionFull = torch.istft(harmFunction, global_consts.nHarmonics, global_consts.nHarmonics, global_consts.nHarmonics, length = (length - 1) * global_consts.nHarmonics, center = False, onesided = True, return_complex = False)
        harmFunction = harmFunction[:, math.floor((length - 1) / 2)]

        offharm = (interpolatedWave[:-1] - harmFunctionFull)
        offharm = extrap(interpolationPoints[:-1], offharm, torch.linspace(0, i.size()[0] - 1, i.size()[0]))
        offharm = offharm[int(global_consts.tripleBatchSize * (global_consts.filterBSMult - 1) / 2):int(global_consts.tripleBatchSize * (global_consts.filterBSMult + 1) / 2)]
        offharm *= torch.hann_window(global_consts.tripleBatchSize)
        audioSample.excitation[counter] = torch.fft.rfft(offharm)
        harmFunction = phaseShiftFourier(harmFunction, markers[0].item() / global_consts.nHarmonics, torch.device("cpu"))
        harmFunction = torch.cat((harmFunction.abs(), harmFunction.angle()), 0)
        audioSample.specharm[counter, :global_consts.nHarmonics + 2] = harmFunction
        #audioSample.phases[counter] = audioSample.specharm[counter, int(global_consts.nHarmonics / 2) + 1]
        counter += 1
    torchaudio.save("test.wav", torch.istft(audioSample.excitation.transpose(0, 1), n_fft = global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = torch.hann_window(global_consts.tripleBatchSize)).unsqueeze(-1), global_consts.sampleRate, False, format = "wav")
    plt.plot(torch.istft(audioSample.excitation.transpose(0, 1), n_fft = global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = torch.hann_window(global_consts.tripleBatchSize)))
    plt.show()
    return audioSample
