#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
from math import floor, ceil
import ctypes

import C_Bridge
import global_consts
from Backend.DataHandler.AudioSample import AudioSample
from Backend.ESPER.SpecCalcComponents import finalizeSpectra, separateVoicedUnvoiced, lowRangeSmooth, highRangeSmooth, averageSpectra

def calculateSpectra(audioSample:AudioSample, useVariance:bool = True) -> None:
    """Method for calculating spectral data based on the previously set attributes filterWidth, voicedFilter and unvoicedIterations.
        
    Arguments:
        audioSample: instance of AudioSample to pull data from and to write results to

        useVariance: whether to use variance statistics to reduce outliers in the sample
            
    Returns:
        None
            
    This function separates voiced and unvoiced excitation, and calculates the spectral envelope for the AudioSample object.
    The voiced/unvoiced separation is performed by resampling the signal into a per-utterance representation, and computing a running mean.
    This way, the voiced signal is obtained, and the unvoiced signal is calculated as the difference between the voiced signal and the input.
    The result is an STFT sequence of the unvoiced signal, and a sequence harmonics of the voiced signal.
    Subsequently, the spectral envelope of the signal is calculated, and the unvoiced excitation is obtained as the unvoiced signal divided by this spectral envelope.
    The spectral calculation uses two methods for low and high frequencies respectively:
    -an adaptation of the True Envelope Estimator
    -fourier space running mean smoothing
    The two results are then combined, forming the final spectrum.
    """


    batches = ceil(audioSample.waveform.size()[0] / global_consts.batchSize)
    audioSample.excitation = torch.zeros([2 * batches * (global_consts.halfTripleBatchSize + 1)], dtype = torch.float)
    audioSample.specharm = torch.zeros([batches, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3], dtype = torch.float)
    audioSample.avgSpecharm = torch.zeros([int(global_consts.nHarmonics / 2) + global_consts.halfTripleBatchSize + 2], dtype = torch.float)
    cSample = C_Bridge.makeCSample(audioSample, useVariance)
    C_Bridge.esper.specCalc(cSample, global_consts.config)
    audioSample.excitation = torch.complex(audioSample.excitation[:batches * (global_consts.halfTripleBatchSize + 1)], audioSample.excitation[batches * (global_consts.halfTripleBatchSize + 1):])
    audioSample.excitation = audioSample.excitation.reshape((batches, global_consts.halfTripleBatchSize + 1))

def calculateSpectra(audioSample:AudioSample, useVariance:bool = True) -> None:
    length = floor(audioSample.waveform.size()[0] / global_consts.batchSize)
    audioSample.excitation = torch.empty((length, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64, device = audioSample.waveform.device)
    audioSample.specharm = torch.empty((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device = audioSample.waveform.device)
    signalsAbs = torch.stft(audioSample.waveform, global_consts.tripleBatchSize, global_consts.batchSize, global_consts.tripleBatchSize, torch.hann_window(global_consts.tripleBatchSize, device = audioSample.waveform.device), return_complex = True)
    signalsAbs = torch.sqrt(signalsAbs.transpose(0, 1)[:audioSample.excitation.size()[0]].abs())
    lowSpectra = lowRangeSmooth(audioSample, signalsAbs)
    highSpectra = highRangeSmooth(audioSample, signalsAbs)
    audioSample = finalizeSpectra(audioSample, lowSpectra, highSpectra)
    audioSample = separateVoicedUnvoiced(audioSample)
    audioSample = averageSpectra(audioSample, useVariance)