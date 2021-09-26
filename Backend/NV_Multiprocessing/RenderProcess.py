import math
import torch
import copy
import global_consts
import Backend.Resampler.Resamplers as rs
from Backend.DataHandler.VocalSegment import VocalSegment
def renderProcess(statusControl, voicebankList, aiParamStackList, inputList, outputList, rerenderFlag):
    settings = {}
    with open("settings.ini", 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            settings[line[0]] = line[1]
    if settings["intermediateOutputs"] == "enabled":
        interOutput = True
    elif settings["intermediateOutputs"] == "disabled":
        interOutput = False
    else:
        print("could not read intermediate output setting. Intermediate outputs have been disabled by default.")
        interOutput = False
    if settings["accelerator"] == "CPU":
        device_rs = torch.device("cpu")
        device_ai = torch.device("cpu")
    elif settings["accelerator"] == "Hybrid":
        device_rs = torch.device("cpu")
        device_ai = torch.device("cuda")
    elif settings["accelerator"] == "GPU":
        device_rs = torch.device("cuda")
        device_ai = torch.device("cuda")
    else:
        print("could not read accelerator setting. Accelerator has been set to CPU by default.")
        device_rs = torch.device("cpu")
        device_ai = torch.device("cpu")
    window = torch.hann_window(global_consts.tripleBatchSize)
    while True:
        for i in range(len(statusControl)):
            internalStatusControl = copy.copy(statusControl[i])
            internalInputs = copy.copy(inputList[i])
            internalOutputs = copy.copy(outputList[i])

            voicebank = voicebankList[i]
            aiParamStack = aiParamStackList[i]

            length = internalInputs.pitch.size()[0]

            spectrum = torch.zeros((length, global_consts.halfTripleBatchSize + 1), device = device_rs)
            processedSpectrum = torch.zeros((length, global_consts.halfTripleBatchSize + 1), device = device_rs)
            excitation = torch.zeros((length, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64, device = device_rs)
            voicedExcitation = torch.zeros(length * global_consts.batchSize, device = device_rs)

            previousSpectrum = None
            previousExcitation = None
            previousVoicedExcitation = None
            currentSpectrum = None
            currentExcitation = None
            currentVoicedExcitation = None
            nextSpectrum = None
            nextExcitation = None
            nextVoicedExcitation = None
            
            aiActive = False
            #reverse iterator to set internalStatusControl.ai based on internalStatusControl.rs
            #reset recurrent AI Tensors
            for j in range(len(internalStatusControl.ai) + 1):
                if j > 0:
                    if (aiActive == False) & (internalStatusControl.ai[j - 1].item() == 0):
                        aiActive = True
                    if aiActive:
                        #execute AI code
                        processedSpectrum[internalInputs.borders[3 * (j - 1)]:internalInputs.borders[3 * (j - 1) + 3]] = spectrum[internalInputs.borders[3 * (j - 1)]:internalInputs.borders[3 * (j - 1) + 3]]
                        if internalInputs.endCaps[j - 1]:
                            processedSpectrum[internalInputs.borders[3 * (j - 1) + 3]:internalInputs.borders[3 * (j - 1) + 5]] = spectrum[internalInputs.borders[3 * (j - 1) + 3]:internalInputs.borders[3 * (j - 1) + 5]]
                        internalStatusControl.ai[j - 1] = 1
                        outputList[i].status[j - 1] = 4

                previousSpectrum = currentSpectrum
                previousExcitation = currentExcitation
                previousVoicedExcitation = currentVoicedExcitation
                currentSpectrum = nextSpectrum
                currentExcitation = nextExcitation
                currentVoicedExcitation = nextVoicedExcitation
                nextSpectrum = None
                nextExcitation = None
                nextVoicedExcitation = None
                if j < len(internalStatusControl.ai):
                    if internalStatusControl.rs[j].item() == 0:
                        if (internalInputs.startCaps[j] == False) and (previousSpectrum == None):
                            section = VocalSegment(internalInputs, voicebank, j - 1, device_rs)
                            previousSpectrum = rs.getSpectrum(section, device_rs)
                            previousExcitation = rs.getExcitation(section, device_rs)
                            previousVoicedExcitation = rs.getVoicedExcitation(section, device_rs)
                        if currentSpectrum == None:
                            section = VocalSegment(internalInputs, voicebank, j, device_rs)
                            currentSpectrum = rs.getSpectrum(section, device_rs)
                            currentExcitation = rs.getExcitation(section, device_rs)
                            currentVoicedExcitation = rs.getVoicedExcitation(section, device_rs)
                        if (internalInputs.endCaps[j] == False) and (nextSpectrum == None):
                            section = VocalSegment(internalInputs, voicebank, j + 1, device_rs)
                            nextSpectrum = rs.getSpectrum(section, device_rs)
                            nextExcitation = rs.getExcitation(section, device_rs)
                            nextVoicedExcitation = rs.getVoicedExcitation(section, device_rs)

                        outputList[i].status[j] = 1

                        voicedExcitation[internalInputs.borders[3 * j]*global_consts.batchSize:internalInputs.borders[3 * j + 5]*global_consts.batchSize] = currentVoicedExcitation
                        if internalInputs.startCaps[j]:
                            windowStart = internalInputs.borders[3 * j]
                            windowStartEx = internalInputs.borders[3 * j]
                        else:
                            voicedExcitation[internalInputs.borders[3 * j]*global_consts.batchSize:internalInputs.borders[3 * j + 2]*global_consts.batchSize] += previousVoicedExcitation[(internalInputs.borders[3 * j]-internalInputs.borders[3 * j + 2])*global_consts.batchSize:]
                            windowStart = internalInputs.borders[3 * j + 2]
                            windowStartEx = internalInputs.borders[3 * j + 1]
                            excitation[internalInputs.borders[3 * j]:windowStartEx] = previousExcitation[internalInputs.borders[3 * j] - windowStartEx:]
                            for k in range(internalInputs.borders[3 * j], internalInputs.borders[3 * j + 2]):
                                spectrum[k] = voicebank.crfAi.processData(previousSpectrum[-1], currentSpectrum[0], (k - internalInputs.borders[3 * j]) / (internalInputs.borders[3 * j + 2] - internalInputs.borders[3 * j]))
                        if internalInputs.endCaps[j]:
                            windowEnd = internalInputs.borders[3 * j + 5]
                            windowEndEx = internalInputs.borders[3 * j + 5]
                        else:
                            voicedExcitation[internalInputs.borders[3 * j + 3]*global_consts.batchSize:internalInputs.borders[3 * j + 5]*global_consts.batchSize] += nextVoicedExcitation[0:(internalInputs.borders[3 * j + 5]-internalInputs.borders[3 * j + 3])*global_consts.batchSize]
                            windowEnd = internalInputs.borders[3 * j + 3]
                            windowEndEx = internalInputs.borders[3 * j + 4]
                            excitation[windowEndEx:internalInputs.borders[3 * j + 5]] = nextExcitation[0:internalInputs.borders[3 * j + 5] - windowEndEx]
                            for k in range(internalInputs.borders[3 * j + 3], internalInputs.borders[3 * j + 5]):
                                spectrum[k] = voicebank.crfAi.processData(currentSpectrum[-1], nextSpectrum[0], (k - internalInputs.borders[3 * j + 3]) / (internalInputs.borders[3 * j + 5] - internalInputs.borders[3 * j + 3]))
                        spectrum[windowStart:windowEnd] = currentSpectrum
                        excitation[windowStartEx:windowEndEx] = currentExcitation

                        outputList[i].status[j] = 2
                        
                        for k in range(internalInputs.borders[3 * j], internalInputs.borders[3 * j + 5]):
                            pitchBorder = math.ceil(global_consts.tripleBatchSize / internalInputs.pitch[k])
                            fourierPitchShift = math.ceil(global_consts.tripleBatchSize / voicebank.phonemeDict[internalInputs.phonemes[j]].pitch) - pitchBorder
                            shiftedSpectrum = torch.roll(spectrum[k], fourierPitchShift)
                            slope = torch.zeros(global_consts.halfTripleBatchSize + 1)
                            slope[pitchBorder:pitchBorder + global_consts.pitchShiftSpectralRolloff] = torch.linspace(0, 1, global_consts.pitchShiftSpectralRolloff)
                            slope[pitchBorder + global_consts.pitchShiftSpectralRolloff:] = 1
                            spectrum[k] = (slope * spectrum[k]) + ((1 - slope) * shiftedSpectrum)

                        internalStatusControl.ai[j] = 0
                        internalStatusControl.rs[j] = 1
                        outputList[i].status[j] = 3
                        
                if ((j > 0) & interOutput):
                    if aiActive:
                        voicedSignal = torch.stft(voicedExcitation[0:internalInputs.borders[3 * (j - 1) + 5]*global_consts.batchSize], global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
                        #unvoicedSignal = torch.stft(excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        
                        breathiness = internalInputs.breathiness[0:internalInputs.borders[3 * (j - 1) + 5]]
                        breathinessCompensation = torch.sum(torch.abs(voicedSignal), 0)[0:-1] / torch.sum(torch.abs(excitation[0:internalInputs.borders[3 * (j - 1) + 5]]), 1) * global_consts.breCompPremul
                        breathinessUnvoiced = 1. + breathiness * breathinessCompensation * torch.gt(breathiness, 0) + breathiness * torch.logical_not(torch.gt(breathiness, 0))
                        breathinessVoiced = 1. - (breathiness * torch.gt(breathiness, 0))
                        voicedSignal = voicedSignal[:, 0:-1] * torch.transpose(processedSpectrum[0:internalInputs.borders[3 * (j - 1) + 5]], 0, 1) * breathinessVoiced
                        excitationSignal = torch.transpose(excitation[0:internalInputs.borders[3 * (j - 1) + 5]] * processedSpectrum[0:internalInputs.borders[3 * (j - 1) + 5]], 0, 1) * breathinessUnvoiced

                        internalOutputs.waveform[0:internalInputs.borders[3 * (j - 1) + 5]*global_consts.batchSize] = torch.istft(voicedSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided=True, length = internalInputs.borders[3 * (j - 1) + 5] * global_consts.batchSize).to(device = torch.device("cpu"))
                        excitationSignal = torch.istft(excitationSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided=True, length = internalInputs.borders[3 * (j - 1) + 5] * global_consts.batchSize)
                        internalOutputs.waveform[0:internalInputs.borders[3 * (j - 1) + 5]*global_consts.batchSize] += excitationSignal.to(device = torch.device("cpu"))

                        outputList[i].waveform[0:internalInputs.borders[3 * (j - 1) + 5]*global_consts.batchSize] = internalOutputs.waveform[0:internalInputs.borders[3 * (j - 1) + 5]*global_consts.batchSize]
                        outputList[i].status[j - 1] = 5
                        if internalInputs.endCaps[j - 1] == True:
                            aiActive = False
                            #reset recurrent AI Tensors

                if rerenderFlag.is_set():
                    break
            else:
                continue
            break
        print("rendering finished!")
        rerenderFlag.wait()
        rerenderFlag.clear()






"""
class Synthesizer:
    def __init__(self, sampleRate):
        self.sampleRate = sampleRate
        self.returnSignal = torch.tensor([], dtype = float)
        
    def synthesize(self, breathiness, spectrum, excitation, voicedExcitation):
        Window = torch.hann_window(global_consts.tripleBatchSize)
        
        self.returnSignal = torch.stft(voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        #unvoicedSignal = torch.stft(excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        
        breathinessCompensation = torch.sum(torch.abs(self.returnSignal), 0) / torch.sum(torch.abs(excitation), 0) * global_consts.breCompPremul
        breathinessUnvoiced = 1. + breathiness * breathinessCompensation[0:-1] * torch.gt(breathiness, 0) + breathiness * torch.logical_not(torch.gt(breathiness, 0))
        breathinessVoiced = 1. - (breathiness * torch.gt(breathiness, 0))
        self.returnSignal = self.returnSignal[:, 0:-1] * torch.transpose(spectrum, 0, 1) * breathinessVoiced
        excitation = excitation[:, 0:-1] * torch.transpose(spectrum, 0, 1) * breathinessUnvoiced

        self.returnSignal = torch.istft(self.returnSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided=True)
        excitation = torch.istft(excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided=True)
        self.returnSignal += excitation

        del Window
    """
