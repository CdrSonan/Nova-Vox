#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
from math import floor

import C_Bridge
import global_consts
from Backend.DataHandler.AudioSample import AudioSample
from Backend.ESPER.PitchCalculator import calculatePitch

def calculateSpectra(audioSample:AudioSample, useVariance:bool = True, allow_oop:bool = True) -> None:
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


    batches = floor(audioSample.waveform.size()[0] / global_consts.batchSize) + 1
    #audioSample.waveform = torch.sin(torch.linspace(0, 2 * 3.14159265358979323846 * audioSample.waveform.size()[0] / 200, audioSample.waveform.size()[0], device = audioSample.waveform.device)) + 0.95
    audioSample.specharm = torch.zeros([batches, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3], dtype = torch.float)
    audioSample.avgSpecharm = torch.zeros([int(global_consts.nHarmonics / 2) + global_consts.halfTripleBatchSize + 2], dtype = torch.float)
    cSample = C_Bridge.makeCSample(audioSample, useVariance, allow_oop)
    C_Bridge.esper.specCalc(cSample, global_consts.config)

def processWorker(input, output, useVariance, allow_oop):
    while True:
        sample = input.get()
        if sample is None:
            break
        try:
            calculatePitch(sample)
            calculateSpectra(sample, useVariance, allow_oop)
        except Exception as e:
            print(e)
            output.put(None)
            continue
        output.put(sample)

def asyncProcess(useVariance, allow_oop):
    inputQueue = torch.multiprocessing.Queue()
    outputQueue = torch.multiprocessing.Queue()
    processes = []
    for _ in range(torch.multiprocessing.cpu_count()):
        p = torch.multiprocessing.Process(target = processWorker, args = (inputQueue, outputQueue, useVariance, allow_oop))
        p.start()
        processes.append(p)
    return inputQueue, outputQueue, processes
