from Backend.DataHandler.AudioSample import AudioSample
from Backend.Resampler.CubicSplineInter import interp, extrap
from Backend.Resampler.PhaseShift import phaseShiftFourier, phaseShift
import torch
import global_consts
import math
from collections import deque

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
    """calculates amplitude continuity function based on the absolute values of an stft sequence. Frequencies with a high amplitude continuity are more likely to be voiced."""

    if amplitudes.size()[1] == 1:
        return torch.ones_like(amplitudes)
    amplitudeContinuity = amplitudes.clone()
    amplitudeContinuity[:,1:] += amplitudes[:, :-1]
    amplitudeContinuity[:,:-1] += amplitudes[:, 1:]
    amplitudeContinuity[:,1:-1] = amplitudeContinuity[:,1:-1] / 3.
    amplitudeContinuity[:, 0] = amplitudeContinuity[:, 0] / 2.
    amplitudeContinuity[:, -1] = amplitudeContinuity[:, -1] / 2.
    amplitudeContinuity = abs(amplitudeContinuity - amplitudes)
    amplitudeContinuity /= amplitudes
    amplitudeContinuity = torch.nan_to_num(amplitudeContinuity, 0.)
    amplitudeContinuity = 1. - amplitudeContinuity
    amplitudeContinuity *= torch.heaviside(amplitudeContinuity, torch.zeros((1,)))
    amplitudeContinuity[:5] = 1.
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

def finalizeSpectra(audioSample:AudioSample, lowSpectra:torch.Tensor, highSpectra:torch.Tensor, useVariance:bool = True) -> AudioSample:
    """calculates final spectra of an AudioSample object based on low frequency range and high frequency range spectra"""

    threshold = torch.nn.Threshold(0.001, 0.001)
    slope = torch.ones_like(highSpectra)
    slope[:, global_consts.spectralRolloff2:] = 0.
    slope[:, global_consts.spectralRolloff1:global_consts.spectralRolloff2] = torch.linspace(1, 0, global_consts.spectralRolloff2 - global_consts.spectralRolloff1)
    lowSpectra = threshold(lowSpectra)
    highSpectra = threshold(highSpectra)
    audioSample.specharm[:, global_consts.nHarmonics + 2:] = slope * lowSpectra + ((1. - slope) * highSpectra)

    audioSample.avgSpecharm = torch.mean(torch.cat((audioSample.specharm[:, :int(global_consts.nHarmonics / 2) + 1], audioSample.specharm[:, global_consts.nHarmonics + 2:]), 1), 0)
    if useVariance:
        variance = 0.
        variances = torch.zeros(audioSample.specharm.size()[0])
        for i in range(audioSample.specharm.size()[0]):
            audioSample.specharm[i, global_consts.nHarmonics + 2:] -= audioSample.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]
            audioSample.specharm[i, :int(global_consts.nHarmonics / 2) + 1] -= audioSample.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1]
            variances[i] = torch.sum(torch.pow(audioSample.specharm[i, global_consts.nHarmonics + 2:], 2))
            variances[i] += torch.sum(torch.pow(audioSample.specharm[i, :int(global_consts.nHarmonics / 2) + 1], 2))
        variances = torch.sqrt(variances)
        variance = torch.sum(variances)
        variance /= audioSample.specharm.size()[0]
        limiter = torch.max(variances / variance - 1., torch.ones([1,])).unsqueeze(1)
        audioSample.specharm[:, global_consts.nHarmonics + 2:] /= limiter
        audioSample.specharm[:, :int(global_consts.nHarmonics / 2) + 1] /= limiter
    audioSample.excitation = audioSample.excitation / torch.square(audioSample.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:] + audioSample.specharm[0:audioSample.excitation.size()[0], global_consts.nHarmonics + 2:])
    return audioSample

def DIOPitchMarkers(audioSample:AudioSample, wave:torch.Tensor) -> list:
    "calculates DIO Pitch markers for a single window of an AudioSample. They can then be used for voiced/unvoiced signal separation."

    def pitch(pos:int) -> float:
        return audioSample.pitchDeltas[min(int((pos - global_consts.halfTripleBatchSize * global_consts.filterBSMult) / global_consts.batchSize), audioSample.pitchDeltas.size()[0] - 1)].to(torch.int64)
    
    maximumMarkers = torch.tensor([], dtype = torch.int16)
    minimumMarkers = torch.tensor([], dtype = torch.int16)

    #get full list of zero transitions
    zeroTransitionsUp = torch.tensor([], dtype = int)
    for j in range(2, wave.size()[0]):
        if (wave[j-1] < 0) and (wave[j] >= 0):
            zeroTransitionsUp = torch.cat([zeroTransitionsUp, torch.tensor([j])], 0)
    zeroTransitionsDown = torch.tensor([], dtype = int)
    for j in range(2, wave.size()[0]):
        if (wave[j-1] >= 0) and (wave[j] < 0):
            zeroTransitionsDown = torch.cat([zeroTransitionsDown, torch.tensor([j])], 0)

    if zeroTransitionsUp.size()[0] <= 1 or zeroTransitionsDown.size()[0] <= 1:
        return torch.tensor([0, audioSample.pitch], dtype = torch.float32)

    #determine first relevant transition
    offset = 0
    skip = False
    while True:
        #fallback if no match is found using any offset
        if offset == min(zeroTransitionsUp.size()[0], zeroTransitionsDown.size()[0]):
            upTransitionMarkers = torch.unsqueeze(zeroTransitionsUp[0], 0)
            downTransitionMarkers = torch.unsqueeze(zeroTransitionsUp[0] + int(audioSample.pitch / 2), 0)
            skip = True
            break
        #increase offset until a valid list of upTransitionCandidates for the first upwards transition is obtained
        upTransitionCandidates = zeroTransitionsUp[torch.searchsorted(zeroTransitionsUp, zeroTransitionsUp[offset]):torch.searchsorted(zeroTransitionsUp, min(zeroTransitionsUp[offset] + pitch(zeroTransitionsDown[-1]), zeroTransitionsDown[-1]))]
        if upTransitionCandidates.size()[0] == 0:
            offset += 1
            continue
        #select upwards transition candidate with the highest derivative
        derrs = torch.index_select(wave, 0, upTransitionCandidates) - torch.index_select(wave, 0, upTransitionCandidates - 1)
        upTransitionMarkers = deque([upTransitionCandidates[torch.argmax(derrs)].item(),])
        #construct list of downwards transition candidates
        downTransitionCandidates = zeroTransitionsDown[torch.searchsorted(zeroTransitionsDown, upTransitionMarkers[0]):torch.searchsorted(zeroTransitionsDown, upTransitionMarkers[0] + pitch(upTransitionMarkers[0]))]
        #abort and increase offset if no valid downwards transition candidates can be obtained
        if downTransitionCandidates.size()[0] > 0:
            break
        offset += 1
    if skip == False:
        #select the downwards transition candidate with the lowest derivative
        derrs = torch.index_select(wave, 0, downTransitionCandidates) - torch.index_select(wave, 0, downTransitionCandidates - 1)
        downTransitionMarkers = deque([downTransitionCandidates[torch.argmin(derrs)].item(),])

    while downTransitionMarkers[-1] < wave.size()[0] - audioSample.pitchDeltas[-1] * global_consts.DIOLastWinTolerance:
        print("1", downTransitionMarkers[-1], "/", (wave.size()[0] - audioSample.pitchDeltas[-1] * global_consts.DIOLastWinTolerance).item())
        lastPitch = pitch(upTransitionMarkers[-1])
        lastDown = downTransitionMarkers[-1]
        lastUp = upTransitionMarkers[-1]
        error = math.inf
        validTransitions = []
        if len(upTransitionMarkers) > 1:
            transition = lastUp + lastDown - downTransitionMarkers[-2]
        else:
            transition = lastUp + lastPitch
        if transition < lastDown:
            transition = math.ceil(lastDown * 1.5 - downTransitionMarkers[-2] * 0.5)
        start = torch.searchsorted(zeroTransitionsUp, torch.tensor([max(lastDown, lastUp + (1 - global_consts.DIOTolerance) * lastPitch),])).item()
        while start < zeroTransitionsUp.size()[0] and zeroTransitionsUp[start] <= min(lastUp + (1 + global_consts.DIOTolerance) * lastPitch, wave.size()[0]):
            validTransitions.append(zeroTransitionsUp[start].item())
            start += 1
        for j in validTransitions:
            localPitch = pitch(j)
            if j + localPitch > wave.size()[0]:
                if localPitch > lastUp:
                    continue
                sample = wave[lastUp - localPitch:lastUp]
                shiftedSample = wave[j - localPitch:j]
            else:
                sample = wave[lastUp:lastUp + localPitch]
                shiftedSample = wave[j:j + localPitch]
            newError = torch.sum(torch.pow(sample - shiftedSample, 2)) * (torch.abs(j - lastUp - localPitch) / localPitch + global_consts.DIOBias2)
            if error > newError:
                transition = j
                error = newError
        upTransitionMarkers.append(transition)

        lastUp = transition

        error = math.inf
        validTransitions = []
        transition = lastDown + lastUp - upTransitionMarkers[-2]
        if transition < lastUp:
            transition = math.ceil(lastUp * 1.5 - upTransitionMarkers[-2] * 0.5)
        start = torch.searchsorted(zeroTransitionsDown, torch.tensor([max(lastUp, lastDown + (1 - global_consts.DIOTolerance) * lastPitch),])).item()
        while start < zeroTransitionsDown.size()[0] and zeroTransitionsDown[start] <= min(lastDown + (1 + global_consts.DIOTolerance) * lastPitch, wave.size()[0]):
            validTransitions.append(zeroTransitionsDown[start].item())
            start += 1
        for j in validTransitions:
            localPitch = pitch(j)
            if j + localPitch > wave.size()[0]:
                if localPitch > lastDown:
                    continue
                sample = wave[lastDown - localPitch:lastDown]
                shiftedSample = wave[j - localPitch:j]
            else:
                sample = wave[lastDown:lastDown + localPitch]
                shiftedSample = wave[j:j + localPitch]
            newError = torch.sum(torch.pow(sample - shiftedSample, 2)) * (torch.abs(j - lastDown - localPitch) / localPitch + global_consts.DIOBias2)
            if error > newError:
                transition = j
                error = newError
        downTransitionMarkers.append(transition)

    upTransitionMarkers = torch.tensor(upTransitionMarkers, dtype = torch.int64)
    downTransitionMarkers = torch.tensor(downTransitionMarkers, dtype = torch.int64)

    if downTransitionMarkers[-1] >= wave.size()[0]:
        upTransitionMarkers = upTransitionMarkers[:-1]
        downTransitionMarkers = downTransitionMarkers[:-1]
    length = len(upTransitionMarkers)
    if length <= 1:
        return torch.tensor([0, audioSample.pitch], dtype = torch.float32)

    for j in range(length):
        print("2", j, "/", length)
        workingWindow = wave[upTransitionMarkers[j] - 2:downTransitionMarkers[j] + 1]
        convKernel = torch.tensor([1., 1.1, 1.])
        scores = torch.tensor([])
        for k in range(workingWindow.size()[0] - 2):
            bias = 1. - (global_consts.DIOBias * k / (workingWindow.size()[0] - 2))
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        maximumMarkers = torch.cat((maximumMarkers, torch.unsqueeze(torch.argmax(scores) + upTransitionMarkers[j], 0)), 0)
    for j in range(length - 1):
        print("3", j, "/", length)
        workingWindow = wave[downTransitionMarkers[j] - 2:upTransitionMarkers[j + 1] + 1]
        convKernel = torch.tensor([-1., -1.1, -1.])
        scores = torch.tensor([])
        for k in range(workingWindow.size()[0] - 2):
            bias = 1. - (global_consts.DIOBias * k / (workingWindow.size()[0] - 2))
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(torch.argmax(scores) + downTransitionMarkers[j], 0)), 0)
    if wave.size()[0] - downTransitionMarkers[-1] > audioSample.pitchDeltas[-1] * global_consts.DIOLastWinTolerance:
        workingWindow = wave[downTransitionMarkers[-1] - 2:-1]
        convKernel = torch.tensor([-1., -1.1, -1.])
        scores = torch.tensor([])
        for k in range(workingWindow.size()[0] - 2):
            bias = 1. - (global_consts.DIOBias * k / (workingWindow.size()[0] - 2))
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(torch.argmax(scores) + downTransitionMarkers[-1], 0)), 0)
    elif downTransitionMarkers.size()[0] == 1:
        marker = downTransitionMarkers[-1] + 1
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(marker, 0)), 0)
    else:
        marker = minimumMarkers[-1] + torch.mean(torch.tensor((upTransitionMarkers[-1] - upTransitionMarkers[-2], downTransitionMarkers[-1] - downTransitionMarkers[-2], maximumMarkers[-1] - maximumMarkers[-2]), dtype = torch.float32))
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(marker, 0)), 0)

    markers = torch.tensor([])
    for j in range(length):
        markers = torch.cat((markers, torch.unsqueeze((upTransitionMarkers[j] + downTransitionMarkers[j] + maximumMarkers[j] + minimumMarkers[j]) / 4, 0)), 0)
    if markers[-1] >= wave.size()[0] - 1:
        markers = markers[:-1]
    return markers

def separateVoicedUnvoiced(audioSample:AudioSample) -> AudioSample:
    """separates the voiced and unvoiced parts of an AudioSample using DIO pitch markers of a sequence of windows, and phase and amplitude continuity functions."""

    wave = torch.cat((torch.zeros([global_consts.halfTripleBatchSize * global_consts.filterBSMult,]), audioSample.waveform, torch.zeros([global_consts.halfTripleBatchSize * global_consts.filterBSMult,])), 0)
    length = math.floor(audioSample.waveform.size()[0] / global_consts.batchSize)
    audioSample.excitation = torch.empty((length, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64)
    highResExcitation = torch.empty((length, global_consts.halfTripleBatchSize * global_consts.filterBSMult + 1), dtype = torch.complex64)
    audioSample.specharm = torch.full((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), 1234.)#TODO: Debugging tool; remove
    counter = 0
    markers = DIOPitchMarkers(audioSample, wave)
    hannWindow = torch.hann_window(global_consts.tripleBatchSize * global_consts.filterBSMult)
    for i in range(length):
        print("4", counter, "/", length)
        window = wave[i * global_consts.batchSize:i * global_consts.batchSize + global_consts.tripleBatchSize * global_consts.filterBSMult]
        localMarkers = markers[torch.searchsorted(markers, i * global_consts.batchSize, right = True):torch.searchsorted(markers, i * global_consts.batchSize + global_consts.tripleBatchSize * global_consts.filterBSMult - 1, right = False)] - i * global_consts.batchSize
        markerLength = len(localMarkers)
        if markerLength <= 1:
            harmFunction = torch.stft(window, global_consts.nHarmonics, global_consts.nHarmonics, global_consts.nHarmonics, center = False, return_complex = True)
            amplitudes = harmFunction.abs()
            #amplitudes *= calculateAmplitudeContinuity(amplitudes)

            amplitudeContinuity = calculateAmplitudeContinuity(amplitudes)
            phaseContinuity = calculatePhaseContinuity(harmFunction.transpose(0, 1)).transpose(0, 1)
            phaseContinuity = torch.pow(phaseContinuity, 2)
            amplitudes *= torch.max(amplitudeContinuity, torch.unsqueeze(torch.max(phaseContinuity, dim = 1)[0], -1))
            
            harmFunction = torch.polar(amplitudes, harmFunction.angle())
            harmFunctionFull = torch.istft(harmFunction, global_consts.nHarmonics, global_consts.nHarmonics, global_consts.nHarmonics, length = global_consts.tripleBatchSize * global_consts.filterBSMult, center = False, onesided = True, return_complex = False)
            offharm = (window - harmFunctionFull)
            offharm = offharm[int(global_consts.tripleBatchSize * (global_consts.filterBSMult - 1) / 2):int(global_consts.tripleBatchSize * (global_consts.filterBSMult + 1) / 2)]
            audioSample.excitation[counter] = torch.fft.rfft(offharm)
            harmFunction = torch.cat((harmFunction.abs(), harmFunction.angle()), 0)
            harmFunction = torch.mean(harmFunction, 1)
            audioSample.specharm[counter, :global_consts.nHarmonics + 2] = harmFunction
            audioSample.specharm[counter, int(global_consts.nHarmonics / 2) + 1:global_consts.nHarmonics + 2] = phaseShift(audioSample.specharm[counter, int(global_consts.nHarmonics / 2) + 1:global_consts.nHarmonics + 2], -audioSample.specharm[counter, int(global_consts.nHarmonics / 2) + 2], torch.device("cpu"))
            counter += 1
            continue
        interpolationPoints = interp(torch.linspace(0, markerLength - 1, markerLength), localMarkers, torch.linspace(0, markerLength - 1, (markerLength - 1) * global_consts.nHarmonics + 1))
        interpolatedWave = interp(torch.linspace(0, global_consts.tripleBatchSize * global_consts.filterBSMult - 1, global_consts.tripleBatchSize * global_consts.filterBSMult), window, interpolationPoints)

        harmFunction = torch.empty((int(global_consts.nHarmonics / 2) + 1, 0))

        for j in range(markerLength - 1):
            harmFunction = torch.cat((harmFunction, torch.unsqueeze(torch.fft.rfft(interpolatedWave[j * global_consts.nHarmonics:(j + 1) * global_consts.nHarmonics]), -1)), 1)

        amplitudes = harmFunction.abs()
        amplitudeContinuity = calculateAmplitudeContinuity(amplitudes)
        phaseContinuity = calculatePhaseContinuity(harmFunction.transpose(0, 1)).transpose(0, 1)
        phaseContinuity = torch.pow(phaseContinuity, 2)
        amplitudes *= torch.max(amplitudeContinuity, torch.unsqueeze(torch.max(phaseContinuity, dim = 1)[0], -1))
        harmFunction = torch.polar(amplitudes, harmFunction.angle())
        harmFunctionFull = torch.istft(harmFunction, global_consts.nHarmonics, global_consts.nHarmonics, global_consts.nHarmonics, length = (markerLength - 1) * global_consts.nHarmonics + 1, center = False, onesided = True, return_complex = False)
        harmFunction = harmFunction[:, math.floor((markerLength - 1) / 2)]#TODO: switch to average instead of mid sample

        if audioSample.isVoiced == False:
            harmFunction *= 0.
            harmFunctionFull *= 0.

        offharm = extrap(interpolationPoints, harmFunctionFull, torch.linspace(0, global_consts.tripleBatchSize * global_consts.filterBSMult - 1, global_consts.tripleBatchSize * global_consts.filterBSMult))
        #offharm = (window - harmFunctionFull)
        #offharm = offharm[int(global_consts.tripleBatchSize * (global_consts.filterBSMult - 1) / 2):int(global_consts.tripleBatchSize * (global_consts.filterBSMult + 1) / 2)]
        offharm *= hannWindow
        highResExcitation[counter] = torch.fft.rfft(offharm)
        #audioSample.excitation[counter] = torch.fft.rfft(offharm)
        harmFunction = phaseShiftFourier(harmFunction, localMarkers[0].item() / global_consts.nHarmonics, torch.device("cpu"))
        harmFunction = torch.cat((harmFunction.abs(), harmFunction.angle()), 0)
        audioSample.specharm[counter, :global_consts.nHarmonics + 2] = harmFunction
        audioSample.specharm[counter, int(global_consts.nHarmonics / 2) + 1:global_consts.nHarmonics + 2] = phaseShift(audioSample.specharm[counter, int(global_consts.nHarmonics / 2) + 1:global_consts.nHarmonics + 2], -audioSample.specharm[counter, int(global_consts.nHarmonics / 2) + 2], torch.device("cpu"))
        counter += 1
    highResExcitation = highResExcitation.transpose(0, 1)
    highResExcitation = torch.istft(highResExcitation, global_consts.tripleBatchSize * global_consts.filterBSMult, global_consts.batchSize, global_consts.tripleBatchSize * global_consts.filterBSMult, hannWindow, length = audioSample.waveform.size()[0], onesided = True)
    highResExcitation = audioSample.waveform - highResExcitation
    audioSample.excitation = torch.stft(highResExcitation, global_consts.tripleBatchSize, global_consts.batchSize, global_consts.tripleBatchSize, torch.hann_window(global_consts.tripleBatchSize), onesided = True, return_complex = True)
    audioSample.excitation = audioSample.excitation.transpose(0, 1)[:length]#TODO: remove length flooring for whole function
    return audioSample
