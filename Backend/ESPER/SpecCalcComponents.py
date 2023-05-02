#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

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
    diff = torch.cat((diff, torch.zeros(1, diff.size()[1])), 0)
    diff += torch.roll(diff, 1, 0)
    diff[1:-1] /= 2
    return 1. - (diff / math.pi)

def calculateAmplitudeContinuity(amplitudes:torch.Tensor, spectrum:torch.Tensor, pitch:int) -> torch.Tensor:
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
    amplitudeContinuity *= 1.5
    amplitudeContinuity /= amplitudes
    amplitudeContinuity = torch.nan_to_num(amplitudeContinuity, 0.)
    amplitudeContinuity = 1. - amplitudeContinuity
    amplitudeContinuity *= torch.heaviside(amplitudeContinuity, torch.zeros((1,)))
    amplitudeContinuity *= amplitudes

    effectiveSpectrum = torch.linspace(0, global_consts.nHarmonics / 2 * (global_consts.halfTripleBatchSize + 1) / pitch, int(global_consts.nHarmonics / 2) + 1)
    effectiveSpectrum = torch.min(effectiveSpectrum, torch.full_like(effectiveSpectrum, global_consts.halfTripleBatchSize + 1))
    effectiveSpectrum = spectrum[effectiveSpectrum.to(torch.long)]
    amplitudeDelta = torch.sqrt(amplitudes)
    amplitudeDelta /= effectiveSpectrum.unsqueeze(1)
    amplitudeDelta = torch.minimum(amplitudeDelta * 1.5, torch.tensor([1,]))
    amplitudeDelta *= amplitudes

    amplitudeMax = torch.maximum(amplitudeContinuity, amplitudeDelta)
    amplitudeMin = torch.minimum(amplitudeContinuity, amplitudeDelta)
    slope = torch.zeros_like(amplitudeContinuity)
    slope[20:] = 1.
    slope[10:20] = torch.linspace(0, 1, 10).unsqueeze(1)
    amplitudeContinuity = amplitudeMin * slope
    amplitudeContinuity += (1. - slope) * amplitudeMax
    return amplitudeContinuity

def lowRangeSmooth(audioSample: AudioSample, signalsAbs:torch.Tensor) -> torch.Tensor:
    """calculates a spectrum based on an adaptation of the True Envelope Estimator algorithm. Used for low-frequency area, as it can produce artifacting in high-frequency area"""

    specWidth = int(global_consts.tripleBatchSize / (audioSample.specWidth + 3))
    spectra = signalsAbs.clone()
    for i in range(audioSample.specDepth):
        spectra = torch.maximum(spectra, signalsAbs)
        spectra = torch.fft.rfft(spectra, dim = 1)
        cutoffWindow = torch.zeros(spectra.size()[1])
        cutoffWindow[0:int(specWidth / 2)] = 1.
        cutoffWindow[int(specWidth / 2):specWidth] = torch.linspace(1, 0, specWidth - int(specWidth / 2))
        spectra = torch.fft.irfft(cutoffWindow * spectra, dim = 1, n = global_consts.halfTripleBatchSize + 1)
    return spectra

def highRangeSmooth(audioSample:AudioSample, signalsAbs:torch.Tensor) -> torch.Tensor:
    """calculates a spectrum based on fourier space running mean smoothing. Used for high-frequency area, as it can produce oversmoothing in low-frequency area"""

    workingSpectra = signalsAbs.clone()
    workingSpectra = torch.cat((workingSpectra, torch.tile(torch.unsqueeze(workingSpectra[:, -1], 1), (1, audioSample.specDepth))), 1)
    spectra = workingSpectra.clone()
    for i in range(audioSample.specDepth):
        for j in range(1, audioSample.specWidth + 1):
            spectra = torch.roll(workingSpectra, -j, dims = 1) + spectra + torch.roll(workingSpectra, j, dims = 1)
        spectra = spectra / (2 * audioSample.specWidth + 1)
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
    workingSpectra = slope * lowSpectra + ((1. - slope) * highSpectra)
    workingSpectra = torch.cat((torch.tile(torch.unsqueeze(workingSpectra[0], 0), (audioSample.tempDepth, 1)), workingSpectra, torch.tile(torch.unsqueeze(workingSpectra[-1], 0), (audioSample.tempDepth, 1))), 0)
    spectra = workingSpectra.clone()
    if audioSample.tempDepth > 0:
        for i in range(audioSample.tempDepth):
            for j in range(1, audioSample.tempWidth + 1):
                spectra = torch.roll(workingSpectra, -j, dims = 0) + spectra + torch.roll(workingSpectra, j, dims = 0)
            spectra = spectra / (2 * audioSample.tempWidth + 1)
            workingSpectra = torch.max(workingSpectra, spectra)
            spectra = workingSpectra
        spectra = spectra[audioSample.tempDepth:-audioSample.tempDepth]
    spectra = threshold(spectra)
    audioSample.specharm[:, global_consts.nHarmonics + 2:] = spectra
    return audioSample

def averageSpectra(audioSample:AudioSample, useVariance:bool = True) -> AudioSample:
    """averages the specharm signal after all other calculations, reduces outliers, and writes the result into avgSpecharm"""

    audioSample.avgSpecharm = torch.mean(torch.cat((audioSample.specharm[:, :int(global_consts.nHarmonics / 2) + 1], audioSample.specharm[:, global_consts.nHarmonics + 2:]), 1), 0)
    audioSample.specharm[:, global_consts.nHarmonics + 2:] -= audioSample.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]
    audioSample.specharm[:, :int(global_consts.nHarmonics / 2) + 1] -= audioSample.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1]
    if useVariance:
        variance = 0.
        variances = torch.zeros(audioSample.specharm.size()[0])
        for i in range(audioSample.specharm.size()[0]):
            variances[i] = torch.sum(torch.pow(audioSample.specharm[i, global_consts.nHarmonics + 2:], 2))
            variances[i] += torch.sum(torch.pow(audioSample.specharm[i, :int(global_consts.nHarmonics / 2) + 1], 2))
        variances = torch.sqrt(variances)
        variance = torch.sum(variances)
        variance /= audioSample.specharm.size()[0]
        limiter = torch.max(variances / variance - 1., torch.ones([1,])).unsqueeze(1)
        audioSample.specharm[:, global_consts.nHarmonics + 2:] /= limiter
        audioSample.specharm[:, :int(global_consts.nHarmonics / 2) + 1] /= limiter
    audioSample.excitation = audioSample.excitation / (audioSample.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:] + audioSample.specharm[0:audioSample.excitation.size()[0], global_consts.nHarmonics + 2:])
    return audioSample

def DIOPitchMarkers(audioSample:AudioSample, wave:torch.Tensor) -> list:
    """calculates precise Pitch markers for a given waveform using an algorithm inspired by DIO.
    
    Arguments:
        audioSample: expected to be an AudioSample instance already contain a valid pitch estimation in its pitch and pitchDeltas attributes.

        wave: expected to be the waveform or the same sample padded by global_consts.halfTripleBatchSize * global_consts.filterBSMult on both sides.
        """

    def pitch(pos:int) -> float:
        effectivePos = int((pos - global_consts.halfTripleBatchSize * global_consts.filterBSMult) / global_consts.batchSize)
        return audioSample.pitchDeltas[min(max(effectivePos, 0), audioSample.pitchDeltas.size()[0] - 1)].to(torch.int64)

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
            #upTransitionMarkers = torch.unsqueeze(zeroTransitionsUp[0], 0)
            #downTransitionMarkers = torch.unsqueeze(zeroTransitionsUp[0] + int(audioSample.pitch / 2), 0)
            upTransitionMarkers = [zeroTransitionsUp[0],]
            downTransitionMarkers = [zeroTransitionsUp[0] + int(audioSample.pitch / 2),]
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
        #print("1", downTransitionMarkers[-1], "/", (wave.size()[0] - audioSample.pitchDeltas[-1] * global_consts.DIOLastWinTolerance).item())
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

    markers = torch.tensor([])
    """maximumMarkers = torch.tensor([], dtype = torch.int16)
    minimumMarkers = torch.tensor([], dtype = torch.int16)
    
    for j in range(length):
        #print("2", j, "/", length)
        workingWindow = wave[upTransitionMarkers[j] - 2:downTransitionMarkers[j] + 1]
        convKernel = torch.tensor([-0.5, 1., -0.5])
        scores = torch.tensor([])
        localPitch = pitch(upTransitionMarkers[j])
        for k in range(workingWindow.size()[0] - 2):
            if maximumMarkers.size()[0] == 0:
                bias = 1.
            else:
                bias = global_consts.DIOBias2 + 1. - torch.abs(upTransitionMarkers[j] + k - 2 - maximumMarkers[-1] - localPitch) / localPitch
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        maximumMarkers = torch.cat((maximumMarkers, torch.unsqueeze(torch.argmax(scores) + upTransitionMarkers[j], 0)), 0)
    for j in range(length - 1):
        #print("3", j, "/", length)
        workingWindow = wave[downTransitionMarkers[j] - 2:upTransitionMarkers[j + 1] + 1]
        convKernel = torch.tensor([0.5, -1., 0.5])
        scores = torch.tensor([])
        localPitch = pitch(downTransitionMarkers[j])
        for k in range(workingWindow.size()[0] - 2):
            if maximumMarkers.size()[0] == 0:
                bias = 1.
            else:
                bias = global_consts.DIOBias2 + 1. - torch.abs(downTransitionMarkers[j] + k - 2 - maximumMarkers[-1] - localPitch) / localPitch
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(torch.argmax(scores) + downTransitionMarkers[j], 0)), 0)
    if wave.size()[0] - downTransitionMarkers[-1] > audioSample.pitchDeltas[-1] * global_consts.DIOLastWinTolerance:
        workingWindow = wave[downTransitionMarkers[-1] - 2:-1]
        convKernel = torch.tensor([0.5, -1., 0.5])
        scores = torch.tensor([])
        for k in range(workingWindow.size()[0] - 2):
            bias = global_consts.DIOBias2 - torch.abs(downTransitionMarkers[-1] + k - 2 - maximumMarkers[-1] - localPitch) / localPitch
            scores = torch.cat((scores, torch.unsqueeze(torch.sum(workingWindow[k:k+3] * convKernel * bias), 0)), 0)
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(torch.argmax(scores) + downTransitionMarkers[-1], 0)), 0)
    elif downTransitionMarkers.size()[0] == 1:
        marker = downTransitionMarkers[-1] + 1
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(marker, 0)), 0)
    else:
        marker = minimumMarkers[-1] + torch.mean(torch.tensor((upTransitionMarkers[-1] - upTransitionMarkers[-2], downTransitionMarkers[-1] - downTransitionMarkers[-2], maximumMarkers[-1] - maximumMarkers[-2]), dtype = torch.float32))
        minimumMarkers = torch.cat((minimumMarkers, torch.unsqueeze(marker, 0)), 0)"""

    for j in range(length):
        #markers = torch.cat((markers, torch.unsqueeze((upTransitionMarkers[j] + downTransitionMarkers[j] + maximumMarkers[j] + minimumMarkers[j]) / 4, 0)), 0)
        markers = torch.cat((markers, torch.unsqueeze((upTransitionMarkers[j] + downTransitionMarkers[j]) / 2., 0)), 0)
    if markers[-1] >= wave.size()[0] - 1:
        markers = markers[:-1]
    #import matplotlib.pyplot as plt
    #plt.plot(wave)
    #plt.vlines(upTransitionMarkers, torch.full_like(minimumMarkers, -1), torch.full_like(minimumMarkers, 1), color = "red")
    #plt.show()
    return markers

def separateVoicedUnvoiced(audioSample:AudioSample) -> AudioSample:
    """separates the voiced and unvoiced parts of an AudioSample using DIO pitch markers of a sequence of windows."""

    padLength = global_consts.halfTripleBatchSize * global_consts.filterBSMult
    wave = torch.cat((torch.flip(audioSample.waveform[:padLength], (0,)), audioSample.waveform, torch.flip(audioSample.waveform[-padLength:], (0,))), 0)
    length = math.floor(audioSample.waveform.size()[0] / global_consts.batchSize)#TODO: remove length flooring for whole function
    globalHarmFunction = torch.zeros((length, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64)
    counter = 0
    markers = DIOPitchMarkers(audioSample, wave)
    for i in range(length):
        #print("4", counter, "/", length)
        window = wave[i * global_consts.batchSize:i * global_consts.batchSize + global_consts.tripleBatchSize * global_consts.filterBSMult]
        localMarkers = markers[torch.searchsorted(markers, i * global_consts.batchSize, right = True):torch.searchsorted(markers, i * global_consts.batchSize + global_consts.tripleBatchSize * global_consts.filterBSMult - 1, right = False)] - i * global_consts.batchSize
        markerLength = len(localMarkers)
        if markerLength <= 1:
            harmFunction = torch.stft(window, global_consts.nHarmonics, global_consts.nHarmonics, global_consts.nHarmonics, center = False, return_complex = True)
            amplitudes = harmFunction.abs()
            amplitudes = calculateAmplitudeContinuity(amplitudes, audioSample.specharm[counter, global_consts.nHarmonics + 2:], audioSample.pitchDeltas[counter])
            amplitudes = torch.mean(amplitudes, dim = 1)
            phases = torch.mean(harmFunction.angle(), dim = 1)#TODO: vector-based phase mean
            harmFunction = torch.polar(amplitudes, phases)
            if audioSample.isVoiced == False:
                harmFunction *= 0.
            phases = phaseShift(phases, -phases[1], torch.device("cpu"))
            harmFunction = torch.cat((amplitudes, phases), 0)
            audioSample.specharm[counter, :global_consts.nHarmonics + 2] = harmFunction
            harmFunction = torch.stft(window, global_consts.tripleBatchSize, global_consts.batchSize, global_consts.tripleBatchSize, torch.hann_window(global_consts.tripleBatchSize), onesided = True, return_complex = True)
            amplitudes = harmFunction.abs()
            #amplitudes = calculateAmplitudeContinuity(amplitudes, audioSample.specharm[counter, global_consts.nHarmonics + 2:], audioSample.pitchDeltas[counter])
            amplitudes = torch.mean(amplitudes, dim = 1)
            phases = torch.mean(harmFunction.angle(), dim = 1)#TODO: vector-based phase mean
            harmFunction = torch.polar(amplitudes, phases)
            globalHarmFunction[i] = harmFunction
            counter += 1
            continue
        interpolationPoints = interp(torch.linspace(0, markerLength - 1, markerLength), localMarkers, torch.linspace(0, markerLength - 1, (markerLength - 1) * global_consts.nHarmonics + 1))
        interpolatedWave = interp(torch.linspace(0, global_consts.tripleBatchSize * global_consts.filterBSMult - 1, global_consts.tripleBatchSize * global_consts.filterBSMult), window, interpolationPoints)
        if i == 0:
            print("interp. wave:", interpolatedWave)
        """if markerLength > 2:
            interpolatedWave[:global_consts.nHarmonics + 1] *= torch.linspace(0, 1, global_consts.nHarmonics + 1)
            interpolatedWave[-global_consts.nHarmonics - 1:] *= torch.linspace(1, 0, global_consts.nHarmonics + 1)
            harmFunction = torch.reshape(interpolatedWave[:-1], (markerLength - 1, global_consts.nHarmonics))
            harmFunction = torch.sum(harmFunction, dim = 0) / (markerLength - 2)
        else:
            harmFunction = interpolatedWave"""
        harmFunction = torch.reshape(interpolatedWave[:-1], (markerLength - 1, global_consts.nHarmonics))
        if audioSample.isVoiced:
            harmFunction = torch.fft.rfft(harmFunction, dim = 1)
            if i == 0:
                print("fft:", harmFunction)
            amplitudes = torch.sum(harmFunction.abs(), dim = 0) / (markerLength - 1)
            phases = torch.sum(harmFunction, dim = 0).angle()
            if i == 0:
                print("amplitudes:", amplitudes)
                print("phases:", phases)
            harmFunction = torch.polar(amplitudes, phases)
            harmFunction = torch.fft.irfft(harmFunction, n = global_consts.nHarmonics)
            harmFunction = torch.tile(harmFunction, (markerLength,))[:(markerLength - 1) * global_consts.nHarmonics + 1]
            harmFunction *= audioSample.voicedThrh
            harmFunction += interpolatedWave * (1. - audioSample.voicedThrh)
            if i == 0:
                print("harmFunction:", harmFunction)
            space = torch.linspace(global_consts.halfTripleBatchSize * (global_consts.filterBSMult - 1), global_consts.halfTripleBatchSize * (global_consts.filterBSMult + 1) - 1, global_consts.tripleBatchSize)
            harmFunction = extrap(interpolationPoints, harmFunction, space)
            globalHarmFunction[i] = torch.fft.rfft(harmFunction)
            phases = phaseShift(phases, -phases[1], torch.device("cpu"))
            audioSample.specharm[counter, :global_consts.nHarmonics + 2] = torch.cat((amplitudes, phases), 0)
        else:
            audioSample.specharm[counter, :global_consts.nHarmonics + 2] *= 0.
        counter += 1
    globalHarmFunction = torch.istft(globalHarmFunction.transpose(0, 1), global_consts.tripleBatchSize , global_consts.batchSize, global_consts.tripleBatchSize, length = audioSample.waveform.size()[0], onesided = True)
    audioSample.excitation = audioSample.waveform - globalHarmFunction
    print(audioSample.excitation[:100])
    audioSample.excitation = torch.stft(audioSample.excitation, global_consts.tripleBatchSize, global_consts.batchSize, global_consts.tripleBatchSize, torch.hann_window(global_consts.tripleBatchSize), onesided = True, return_complex = True)
    audioSample.excitation = audioSample.excitation.transpose(0, 1)[:length]
    import matplotlib.pyplot as plt
    plt.plot(audioSample.waveform)
    plt.plot(globalHarmFunction)
    plt.plot(audioSample.waveform - globalHarmFunction)
    plt.show()
    return audioSample
