from lib2to3.pgen2.literals import evalString
import math
import torch
import global_consts
import logging
import Backend.Resampler.Resamplers as rs
from copy import copy
from Backend.DataHandler.VocalSegment import VocalSegment
from Backend.Param_Components.AiParams import AiParamStack
from Backend.VB_Components.SpecCrfAi import LiteSpecCrfAi
from Backend.VB_Components.Voicebank import LiteVoicebank
from Backend.NV_Multiprocessing.Interface import SequenceStatusControl, StatusChange
from MiddleLayer.IniParser import readSettings

import matplotlib.pyplot as plt

def renderProcess(statusControl, voicebankList, aiParamStackList, inputList, rerenderFlag, connection):
    def posToSegment(index, pos1, pos2):
        pos1Out = None
        for i in range(int(len(inputList[index].borders) / 3)):
            if inputList[index].borders[3 * i + 2] > pos1:
                pos1Out = max(i - 1, 0)
                break
        if pos1Out == None or pos1Out == int(len(inputList[index].borders) / 3) - 1:
            return (0, 0)
        pos2Out = int(len(inputList[index].borders) / 3) - 1
        for i in range(pos1Out, int(len(inputList[index].borders) / 3)):
            if inputList[index].borders[3 * i] > pos2:
                pos2Out = max(i, 0)
                break
        return (pos1Out, pos2Out)
    def trimSequence(index, position, delta):
        phonemes = inputList[index].phonemes
        offsets = inputList[index].offsets
        repetititionSpacing = inputList[index].repetititionSpacing
        borders = inputList[index].borders
        startCaps = inputList[index].startCaps
        endCaps = inputList[index].endCaps
        if delta > 0:
            phonemes = phonemes[0:position] + ["_0"] * delta + phonemes[position:]
            offsets = torch.cat([offsets[0:position], torch.zeros([delta,]), offsets[position:]], 0)
            repetititionSpacing = torch.cat([repetititionSpacing[0:position], torch.full([delta,], 0.5), repetititionSpacing[position:]], 0)
            borders = borders[0:3 * position] + [0] * (3 * delta) + borders[3 * position:]
            startCaps = startCaps[0:position] + [False] * delta + startCaps[position:]
            endCaps = endCaps[0:position] + [False] * delta + endCaps[position:]
            if position == 0:
                startCaps[0] = True
                if len(startCaps) > position + delta:
                    startCaps[position + delta] = False
            if position + delta >= len(endCaps) - 1:
                endCaps[position] = False
                endCaps[-1] = True
            internalStatusControl.rs = torch.cat([internalStatusControl.rs[0:position], torch.zeros([delta,]), internalStatusControl.rs[position:]], 0)
            internalStatusControl.ai = torch.cat([internalStatusControl.ai[0:position], torch.zeros([delta,]), internalStatusControl.ai[position:]], 0)
        elif delta < 0:
            phonemes = phonemes[0:position] + phonemes[position - delta:]
            offsets = torch.cat([offsets[0:position], offsets[position - delta:]], 0)
            repetititionSpacing = torch.cat([repetititionSpacing[0:position], repetititionSpacing[position - delta:]], 0)
            borders = borders[0:3 * position] + borders[3 * position - 3 * delta:]
            startCaps = startCaps[0:position] + startCaps[position - delta:]
            endCaps = endCaps[0:position] + endCaps[position - delta:]
            if len(startCaps) > 0:
                if position == 0:
                    startCaps[0] = True
                if position >= len(endCaps) - 1:
                    endCaps[-1] = True
            internalStatusControl.rs = torch.cat([internalStatusControl.rs[0:position], internalStatusControl.rs[position - delta:]], 0)
            internalStatusControl.ai = torch.cat([internalStatusControl.ai[0:position], internalStatusControl.ai[position - delta:]], 0)
        inputList[index].phonemes = phonemes
        inputList[index].offsets = offsets
        inputList[index].repetititionSpacing = repetititionSpacing
        inputList[index].borders = borders
        inputList[index].startCaps = startCaps
        inputList[index].endCaps = endCaps

    def updateFromMain():
        change = connection.recv()
        #print("change", change.type, change.data2, change.final)
        if change.type == "terminate":
            return True
        elif change.type == "addTrack":
            statusControl.append(SequenceStatusControl(change.data2))
            voicebankList.append(LiteVoicebank(change.data1))
            aiParamStackList.append(change.data3)
            inputList.append(change.data2)
        elif change.type == "removeTrack":
            del statusControl[change.data1]
            del voicebankList[change.data1]
            del aiParamStackList[change.data1]
            del inputList[change.data1]
            connection.send(StatusChange(None, None, None, None))
        elif change.type == "duplicateTrack":
            statusControl.append(copy(statusControl[change.data1]))
            voicebankList.append(copy(voicebankList[change.data1]))
            aiParamStackList.append(copy(aiParamStackList[change.data1]))
            inputList.append(copy(inputList[change.data1]))
        elif change.type == "changeVB":
            del voicebankList[change.data1]
            voicebankList.insert(change.data1, LiteVoicebank(change.data2))
            statusControl[change.data1].rs *= 0
            statusControl[change.data1].ai *= 0
        elif change.type == "addParam":
            aiParamStackList[change.data1].addParam(change.data2)
            statusControl[change.data1].ai *= 0
        elif change.type == "removeParam":
            aiParamStackList[change.data1].removeParam(change.data2)
            statusControl[change.data1].ai *= 0
        elif change.type == "enableParam":
            aiParamStackList[change.data1].enableParam(change.data2)#add named params
            statusControl[change.data1].ai *= 0
        elif change.type == "disableParam":
            aiParamStackList[change.data1].disableParam(change.data2)#add named params
            statusControl[change.data1].ai *= 0
        elif change.type == "moveParam":
            aiParamStackList[change.data1].insert(change.data3, aiParamStackList[change.data1].pop(change.data2))
            statusControl[change.data1].ai *= 0
        elif change.type == "changeInput":
            if change.data2 in ["phonemes", "offsets", "repetititionSpacing"]:
                eval("inputList[change.data1]." + change.data2)[change.data3:change.data3 + len(change.data4)] = change.data4
                statusControl[change.data1].rs[change.data3:change.data3 + len(change.data4)] *= 0
                statusControl[change.data1].ai[change.data3:change.data3 + len(change.data4)] *= 0
            elif change.data2 == "borders":
                for i in range(len(change.data4)):
                    change.data4[i] = int(change.data4[i])
                inputList[change.data1].borders[change.data3:change.data3 + len(change.data4)] = change.data4
                statusControl[change.data1].rs[math.floor(change.data3 / 3):math.floor((change.data3 + len(change.data4)) / 3)] *= 0
                statusControl[change.data1].ai[math.floor(change.data3):math.floor((change.data3 + len(change.data4)) / 3)] *= 0
            elif change.data2 in ["pitch", "steadiness", "breathiness"]:
                eval("inputList[change.data1]." + change.data2)[change.data3:change.data3 + len(change.data4)] = change.data4
                positions = posToSegment(change.data1, change.data3, change.data3 + len(change.data4))
                statusControl[change.data1].rs[positions[0]:positions[1]] *= 0
                statusControl[change.data1].ai[positions[0]:positions[1]] *= 0
            else:
                positions = posToSegment(change.data1, change.data3, change.data3 + len(change.data4))
                inputList[change.data1].aiParamInputs[change.data2][change.data3:change.data3 + len(change.data4)] = change.data4
                statusControl[change.data1].ai[positions[0]:positions[1]] *= 0
        elif change.type == "offset":
            trimSequence(change.data1, change.data2, change.data3)
        if connection.poll() or (change.final == False):
            return updateFromMain()
        return False
    logging.info("render process started, reading settings")
    settings = readSettings()
    if settings["lowSpecMode"] == "enabled":
        interOutput = False
    elif settings["lowSpecMode"] == "disabled":
        interOutput = True
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

    window = torch.hann_window(global_consts.tripleBatchSize, device = device_rs)
    while True:
        logging.info("starting new rendering iteration")
        for i in range(len(statusControl)):
            logging.info("starting new sequence rendering iteration, copying data from main process")
            internalStatusControl = statusControl[i]
            internalInputs = inputList[i]

            voicebank = voicebankList[i]
            voicebank.crfAi = LiteSpecCrfAi(voicebank.crfAi, device_ai)
            aiParamStack = aiParamStackList[i]

            length = internalInputs.pitch.size()[0]

            logging.info("setting up local data structures")
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
                logging.info("starting new segment rendering iteration")
                if j > 0:
                    if (aiActive == False) & (internalStatusControl.ai[j - 1].item() == 0):
                        aiActive = True
                    if aiActive:
                        logging.info("applying AI params to spectrum of sample " + str(j - 1) + ", sequence " + str(i))
                        #execute AI code
                        processedSpectrum[internalInputs.borders[3 * (j - 1)]:internalInputs.borders[3 * (j - 1) + 3]] = torch.square(spectrum[internalInputs.borders[3 * (j - 1)]:internalInputs.borders[3 * (j - 1) + 3]])
                        if internalInputs.endCaps[j - 1]:
                            processedSpectrum[internalInputs.borders[3 * (j - 1) + 3]:internalInputs.borders[3 * (j - 1) + 5]] = torch.square(spectrum[internalInputs.borders[3 * (j - 1) + 3]:internalInputs.borders[3 * (j - 1) + 5]])
                        internalStatusControl.ai[j - 1] = 1
                        connection.send(StatusChange(i, j - 1, 4))

                logging.info("shifting internal data backwards")
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
                        logging.info("calling resamplers for sample " + str(j) + ", sequence " + str(i))
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

                        connection.send(StatusChange(i, j, 1))

                        logging.info("performing pitch shift of sample " + str(j) + ", sequence " + str(i))
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
                                spectrum[k] = voicebank.crfAi.processData(previousSpectrum[-2].to(device = device_ai), previousSpectrum[-1].to(device = device_ai), currentSpectrum[0].to(device = device_ai), currentSpectrum[1].to(device = device_ai), (k - internalInputs.borders[3 * j]) / (internalInputs.borders[3 * j + 2] - internalInputs.borders[3 * j]))
                        
                        if internalInputs.endCaps[j]:
                            windowEnd = internalInputs.borders[3 * j + 5]
                            windowEndEx = internalInputs.borders[3 * j + 5]
                        else:
                            voicedExcitation[internalInputs.borders[3 * j + 3]*global_consts.batchSize:internalInputs.borders[3 * j + 5]*global_consts.batchSize] += nextVoicedExcitation[0:(internalInputs.borders[3 * j + 5]-internalInputs.borders[3 * j + 3])*global_consts.batchSize]
                            windowEnd = internalInputs.borders[3 * j + 3]
                            windowEndEx = internalInputs.borders[3 * j + 4]
                            excitation[windowEndEx:internalInputs.borders[3 * j + 5]] = nextExcitation[0:internalInputs.borders[3 * j + 5] - windowEndEx]
                            for k in range(internalInputs.borders[3 * j + 3], internalInputs.borders[3 * j + 5]):
                                spectrum[k] = voicebank.crfAi.processData(currentSpectrum[-2].to(device = device_ai), currentSpectrum[-1].to(device = device_ai), nextSpectrum[0].to(device = device_ai), nextSpectrum[1].to(device = device_ai), (k - internalInputs.borders[3 * j + 3]) / (internalInputs.borders[3 * j + 5] - internalInputs.borders[3 * j + 3]))
                        
                        #implement crfai skipping if transition was already calculated in the previous frame
                        
                        spectrum[windowStart:windowEnd] = currentSpectrum
                        excitation[windowStartEx:windowEndEx] = currentExcitation

                        connection.send(StatusChange(i, j, 2))
                        
                        logging.info("applying partial pitch shift to spectrum of sample " + str(j) + ", sequence " + str(i))
                        if internalInputs.phonemes[j] != "_autopause":
                            for k in range(internalInputs.borders[3 * j], internalInputs.borders[3 * j + 5]):
                                pitchBorder = math.ceil(global_consts.tripleBatchSize / internalInputs.pitch[k])
                                fourierPitchShift = math.ceil(global_consts.tripleBatchSize / voicebank.phonemeDict[internalInputs.phonemes[j]].pitch) - pitchBorder
                                shiftedSpectrum = torch.roll(spectrum[k], fourierPitchShift)
                                slope = torch.zeros(global_consts.halfTripleBatchSize + 1, device = device_rs)
                                slope[pitchBorder:pitchBorder + global_consts.pitchShiftSpectralRolloff] = torch.linspace(0, 1, global_consts.pitchShiftSpectralRolloff)
                                slope[pitchBorder + global_consts.pitchShiftSpectralRolloff:] = 1
                                spectrum[k] = (slope * spectrum[k]) + ((1 - slope) * shiftedSpectrum)
                        
                        internalStatusControl.ai[j] = 0
                        internalStatusControl.rs[j] = 1
                        connection.send(StatusChange(i, j, 3))
                        
                if ((j > 0) & interOutput) or (j == len(internalStatusControl.ai)):
                    logging.info("performing final rendering up to sample " + str(j - 1) + ", sequence " + str(i))
                    if aiActive:
                        voicedSignal = torch.stft(voicedExcitation[0:internalInputs.borders[3 * (j - 1) + 5]*global_consts.batchSize], global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
        
                        breathiness = internalInputs.breathiness[0:internalInputs.borders[3 * (j - 1) + 5]].to(device = device_rs)
                        breathinessCompensation = torch.sum(torch.abs(voicedSignal), 0)[0:-1] / torch.maximum(torch.sum(torch.abs(excitation[0:internalInputs.borders[3 * (j - 1) + 5]]), 1), torch.tensor([0.0001], device = device_rs)) * global_consts.breCompPremul
                        breathinessUnvoiced = 1. + breathiness * breathinessCompensation * torch.gt(breathiness, 0) + breathiness * torch.logical_not(torch.gt(breathiness, 0))
                        breathinessVoiced = 1. - (breathiness * torch.gt(breathiness, 0))
                        #voicedSignal = torch.ones_like(voicedSignal)
                        #excitation = torch.ones_like(excitation)
                        voicedSignal = voicedSignal[:, 0:-1] * torch.transpose(processedSpectrum[0:internalInputs.borders[3 * (j - 1) + 5]], 0, 1) * breathinessVoiced
                        excitationSignal = torch.transpose(excitation[0:internalInputs.borders[3 * (j - 1) + 5]] * processedSpectrum[0:internalInputs.borders[3 * (j - 1) + 5]], 0, 1) * breathinessUnvoiced
                        #voicedSignal = voicedSignal[:, 0:-1]
                        #excitationSignal = torch.transpose(excitation[0:internalInputs.borders[3 * (j - 1) + 5]], 0, 1)

                        waveform = torch.istft(voicedSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided=True, length = internalInputs.borders[3 * (j - 1) + 5] * global_consts.batchSize).to(device = torch.device("cpu"))
                        excitationSignal = torch.istft(excitationSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided=True, length = internalInputs.borders[3 * (j - 1) + 5] * global_consts.batchSize)
                        waveform += excitationSignal.to(device = torch.device("cpu"))
                        connection.send(StatusChange(i, 0, waveform.detach(), True))
                        connection.send(StatusChange(i, j - 1, 5))
                        if internalInputs.endCaps[j - 1] == True:
                            aiActive = False
                            #reset recurrent AI Tensors

                if (j > 0) & (interOutput == False):
                    connection.send(StatusChange(i, j - 1, 5))

                if rerenderFlag.is_set():
                    break
            else:
                continue
            break
        logging.info("rendering process finished for all sequences, waiting for render semaphore")
        print("rendering finished!")
        rerenderFlag.wait()
        rerenderFlag.clear()
        if connection.poll():
            if updateFromMain():
                break
        






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
