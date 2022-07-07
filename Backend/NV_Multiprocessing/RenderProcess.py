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
from Backend.NV_Multiprocessing.Caching import DenseCache, SparseCache
from Backend.NV_Multiprocessing.Update import trimSequence, posToSegment
from MiddleLayer.IniParser import readSettings

def renderProcess(statusControl, voicebankList, aiParamStackList, inputList, rerenderFlag, connection, remoteConnection):
    def updateFromMain(change, lastZero):
        if change.type == "terminate":
            return True
        elif change.type == "addTrack":
            statusControl.append(SequenceStatusControl(change.data[2]))
            voicebankList.append(LiteVoicebank(change.data[1]))
            aiParamStackList.append(change.data[3])
            inputList.append(change.data[2])
            if settings["cachingMode"] == "best rendering speed":
                length = inputList[-1].pitch.size()[0]
                spectrumCache.append(DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs))
                processedSpectrumCache.append(DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs))
                excitationCache.append(DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs, torch.complex64))
                voicedExcitationCache.append(DenseCache((length * global_consts.batchSize,), device_rs))
        elif change.type == "removeTrack":
            del statusControl[change.data[1]]
            del voicebankList[change.data[1]]
            del aiParamStackList[change.data[1]]
            del inputList[change.data[1]]
            if settings["cachingMode"] == "best rendering speed":
                del spectrumCache[change.data[1]]
                del processedSpectrumCache[change.data[1]]
                del excitationCache[change.data[1]]
                del voicedExcitationCache[change.data[1]]
            remoteConnection.put(StatusChange(None, None, None, None))
        elif change.type == "duplicateTrack":
            statusControl.append(copy(statusControl[change.data[1]]))
            voicebankList.append(copy(voicebankList[change.data[1]]))
            aiParamStackList.append(copy(aiParamStackList[change.data[1]]))
            inputList.append(copy(inputList[change.data[1]]))
            if settings["cachingMode"] == "best rendering speed":
                spectrumCache.append(copy(spectrumCache[change.data[1]]))
                processedSpectrumCache.append(copy(processedSpectrumCache[change.data[1]]))
                excitationCache.append(copy(excitationCache[change.data[1]]))
                voicedExcitationCache.append(copy(voicedExcitationCache[change.data[1]]))
        elif change.type == "changeVB":
            del voicebankList[change.data[1]]
            voicebankList.insert(change.data[1], LiteVoicebank(change.data[2]))
            statusControl[change.data[1]].rs *= 0
            statusControl[change.data[1]].ai *= 0
        elif change.type == "addParam":
            aiParamStackList[change.data[1]].addParam(change.data[2])
            statusControl[change.data[1]].ai *= 0
        elif change.type == "removeParam":
            aiParamStackList[change.data[1]].removeParam(change.data[2])
            statusControl[change.data[1]].ai *= 0
        elif change.type == "enableParam":
            if change.data[2] == "breathiness":
                inputList[change.data[1]].useBreathiness = True
                statusControl[change.data[1]].rs *= 0
            elif change.data[2] == "steadiness":
                inputList[change.data[1]].useSteadiness = True
                statusControl[change.data[1]].rs *= 0
            elif change.data[2] == "vibrato speed":
                inputList[change.data[1]].useVibratoSpeed = True
                statusControl[change.data[1]].rs *= 0
            elif change.data[2] == "vibrato strength":
                inputList[change.data[1]].useVibratoStrength = True
                statusControl[change.data[1]].rs *= 0
            else:
                aiParamStackList[change.data[1]].enableParam(change.data[2])
            statusControl[change.data[1]].ai *= 0
        elif change.type == "disableParam":
            if change.data[2] == "breathiness":
                inputList[change.data[1]].useBreathiness = False
                statusControl[change.data[1]].rs *= 0
            elif change.data[2] == "steadiness":
                inputList[change.data[1]].useSteadiness = False
                statusControl[change.data[1]].rs *= 0
            elif change.data[2] == "vibrato speed":
                inputList[change.data[1]].useVibratoSpeed = False
                statusControl[change.data[1]].rs *= 0
            elif change.data[2] == "vibrato strength":
                inputList[change.data[1]].useVibratoStrength = False
                statusControl[change.data[1]].rs *= 0
            else:
                aiParamStackList[change.data[1]].disableParam(change.data[2])
            statusControl[change.data[1]].ai *= 0
        elif change.type == "moveParam":
            aiParamStackList[change.data[1]].insert(change.data[3], aiParamStackList[change.data[1]].pop(change.data[2]))
            statusControl[change.data[1]].ai *= 0
        elif change.type == "changeInput":
            if change.data[2] in ["phonemes", "offsets", "repetititionSpacing"]:
                if change.data[2] == "phonemes":
                    for j in range(len(change.data[4])):
                        if inputList[change.data[1]].phonemes[change.data[3] + j] == "_autopause":
                            inputList[change.data[1]].startCaps[change.data[3] + j] = False
                            inputList[change.data[1]].endCaps[change.data[3] + j] = False
                            if change.data[3] + j + 1 < len(inputList[change.data[1]].startCaps):
                                inputList[change.data[1]].startCaps[change.data[3] + j + 1] = False
                            if change.data[3] + j > 0:
                                inputList[change.data[1]].endCaps[change.data[3] + j - 1] = False
                eval("inputList[change.data[1]]." + change.data[2])[change.data[3]:change.data[3] + len(change.data[4])] = change.data[4]
                statusControl[change.data[1]].rs[change.data[3]:change.data[3] + len(change.data[4])] *= 0
                statusControl[change.data[1]].ai[change.data[3]:change.data[3] + len(change.data[4])] *= 0
                if change.data[2] == "phonemes":
                    for j in range(len(change.data[4])):
                        if change.data[4][j] == "_autopause":
                            inputList[change.data[1]].startCaps[change.data[3] + j] = True
                            inputList[change.data[1]].endCaps[change.data[3] + j] = True
                            if change.data[3] + j + 1 < len(inputList[change.data[1]].startCaps):
                                inputList[change.data[1]].startCaps[change.data[3] + j + 1] = True
                            if change.data[3] + j > 0:
                                inputList[change.data[1]].endCaps[change.data[3] + j - 1] = True
                        if j + 1 == len(change.data[4]):
                            inputList[change.data[1]].endCaps[change.data[3] + j] = True
            elif change.data[2] == "borders":
                start = inputList[change.data[1]].borders[change.data[3]] * global_consts.batchSize
                end = inputList[change.data[1]].borders[change.data[3] + len(change.data[4]) - 1] * global_consts.batchSize
                if lastZero == None or lastZero != [change.data[1], start, end - start]:
                    lastZero = [change.data[1], start, end - start]
                    remoteConnection.put(StatusChange(change.data[1], start, end - start, "zeroAudio"))
                for i in range(len(change.data[4])):
                    change.data[4][i] = int(change.data[4][i])
                inputList[change.data[1]].borders[change.data[3]:change.data[3] + len(change.data[4])] = change.data[4]
                statusControl[change.data[1]].rs[math.floor(change.data[3] / 3):math.floor((change.data[3] + len(change.data[4])) / 3)] *= 0
                statusControl[change.data[1]].ai[math.floor(change.data[3] / 3):math.floor((change.data[3] + len(change.data[4])) / 3)] *= 0
            elif change.data[2] in ["pitch", "steadiness", "breathiness"]:
                eval("inputList[change.data[1]]." + change.data[2])[change.data[3]:change.data[3] + len(change.data[4])] = change.data[4]
                positions = posToSegment(change.data[1], change.data[3], change.data[3] + len(change.data[4]), inputList)
                statusControl[change.data[1]].rs[positions[0]:positions[1]] *= 0
                statusControl[change.data[1]].ai[positions[0]:positions[1]] *= 0
            else:
                positions = posToSegment(change.data[1], change.data[3], change.data[3] + len(change.data[4]), inputList)
                inputList[change.data[1]].aiParamInputs[change.data[2]][change.data[3]:change.data[3] + len(change.data[4])] = change.data[4]
                statusControl[change.data[1]].ai[positions[0]:positions[1]] *= 0
        elif change.type == "offset":
            inputList, internalStatusControl = trimSequence(change.data[1], change.data[2], change.data[3], inputList, internalStatusControl)
        if change.final == False:
            return updateFromMain(connection.get(), lastZero)
        else:
            try:
                return updateFromMain(connection.get_nowait(), lastZero)
            except:
                return False

    #reading settings, setting device and interOutput properties accordingly
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

    #setting up caching and other required data that is independent of each individual rendering iteration
    window = torch.hann_window(global_consts.tripleBatchSize, device = device_rs)
    lastZero = None
    if settings["cachingMode"] == "best rendering speed":
        spectrumCache = []
        processedSpectrumCache = []
        excitationCache = []
        voicedExcitationCache = []
        for i in range(len(statusControl)):
            length = inputList[i].pitch.size()[0]
            spectrumCache.append(torch.zeros((length, global_consts.halfTripleBatchSize + 1), device = device_rs))
            processedSpectrumCache.append(torch.zeros((length, global_consts.halfTripleBatchSize + 1), device = device_rs))
            excitationCache.append(torch.zeros((length, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64, device = device_rs))
            voicedExcitationCache.append(torch.zeros(length * global_consts.batchSize, device = device_rs))

    #main loop consisting of rendering iterations, and updates when required
    while True:
        logging.info("starting new rendering iteration")

        #iterate through individual VocalSequence objects
        for i in range(len(statusControl)):

            #set up data structures specific to rendering iteration
            logging.info("starting new sequence rendering iteration, copying data from main process")
            internalStatusControl = statusControl[i]
            internalInputs = inputList[i]

            voicebank = voicebankList[i]
            voicebank.crfAi = LiteSpecCrfAi(voicebank.crfAi, device_ai)
            aiParamStack = aiParamStackList[i]

            length = internalInputs.pitch.size()[0]

            logging.info("setting up local data structures")

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
            if settings["cachingMode"] == "best rendering speed":
                spectrum = spectrumCache[i]
                processedSpectrum = processedSpectrumCache[i]
                excitation = excitationCache[i]
                voicedExcitation = voicedExcitationCache[i]
                indicator = -1
            elif settings["cachingMode"] == "save RAM":
                spectrum = SparseCache((length, global_consts.halfTripleBatchSize + 1), device_rs)
                processedSpectrum = SparseCache((length, global_consts.halfTripleBatchSize + 1), device_rs)
                excitation = SparseCache((length, global_consts.halfTripleBatchSize + 1), device_rs, torch.complex64)
                voicedExcitation = SparseCache((length * global_consts.batchSize,), device_rs)
                indicator = 0
            else:
                spectrum = DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs)
                processedSpectrum = DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs)
                excitation = DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs, torch.complex64)
                voicedExcitation = DenseCache((length * global_consts.batchSize,), device_rs)
                indicator = 0
            firstPoint = None

            #go through internalStatusControl lists and set all parts required to render a pause-to-pause section to the value of indicator.
            #when caching is available, this causes them to be loaded from cache, otherwise they are re-computed.
            for k in range(1, len(internalStatusControl.rs)):
                if internalStatusControl.rs[k] == 0 and internalStatusControl.rs[k - 1] > 0:
                    for j in range(k):
                        if internalInputs.phonemes[k - j] == "_autopause":
                            break
                        internalStatusControl.rs[k - j] = indicator
                        internalStatusControl.ai[k - j] = indicator
                    for j in range(len(internalStatusControl.ai) - k):
                        if internalInputs.phonemes[k + j] == "_autopause":
                            break
                        internalStatusControl.ai[k + j] = indicator
            firstPoint = len(internalStatusControl.ai) - 1
            lastPoint = len(internalStatusControl.ai) - 1
            for k in range(len(internalStatusControl.ai)):
                if internalStatusControl.ai[k] <= 0:
                    firstPoint = k
                    break
            for k in range(len(internalStatusControl.ai)):
                if internalStatusControl.ai[len(internalStatusControl.ai) - k - 1] <= 0:
                    lastPoint = len(internalStatusControl.ai) - k - 1
                    break
            #TODO: reset recurrent AI Tensors
            #iterate through segments in VocalSequence
            for j in range(len(internalStatusControl.ai) + 1):
                #apply Node tree... eventually
                logging.info("starting new segment rendering iteration")
                if j > 0:
                    if (aiActive == False) & (internalStatusControl.ai[j - 1].item() == 0):
                        aiActive = True
                    if aiActive:
                        logging.info("applying AI params to spectrum of sample " + str(j - 1) + ", sequence " + str(i))
                        #execute AI code
                        processedSpectrum.write(torch.square(spectrum.read(internalInputs.borders[3 * (j - 1)], internalInputs.borders[3 * (j - 1) + 3])), internalInputs.borders[3 * (j - 1)], internalInputs.borders[3 * (j - 1) + 3])
                        if internalInputs.endCaps[j - 1]:
                            processedSpectrum.write(torch.square(spectrum.read(internalInputs.borders[3 * (j - 1) + 3], internalInputs.borders[3 * (j - 1) + 5])), internalInputs.borders[3 * (j - 1) + 3], internalInputs.borders[3 * (j - 1) + 5])
                        internalStatusControl.ai[j - 1] = 1
                        remoteConnection.put(StatusChange(i, j - 1, 4))

                #move data to prepare for next sample
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

                #perform resampling if required
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
                        remoteConnection.put(StatusChange(i, j, 1))

                        #calculate CrfAi transitions as required
                        logging.info("performing pitch shift of sample " + str(j) + ", sequence " + str(i))
                        voicedExcitation.write(currentVoicedExcitation, internalInputs.borders[3 * j]*global_consts.nHarmonics, internalInputs.borders[3 * j + 5]*global_consts.nHarmonics)
                        if internalInputs.startCaps[j]:
                            windowStart = internalInputs.borders[3 * j]
                            windowStartEx = internalInputs.borders[3 * j]
                        else:
                            voicedExcitation.write(voicedExcitation.read(internalInputs.borders[3*j]*global_consts.batchSize, internalInputs.borders[3*j+2]*global_consts.batchSize)+previousVoicedExcitation[(internalInputs.borders[3*j]-internalInputs.borders[3*j+2])*global_consts.batchSize:], internalInputs.borders[3*j]*global_consts.batchSize, internalInputs.borders[3*j+2]*global_consts.batchSize)
                            windowStart = internalInputs.borders[3 * j + 2]
                            windowStartEx = internalInputs.borders[3 * j + 1]
                            excitation.write(previousExcitation[internalInputs.borders[3 * j] - windowStartEx:], internalInputs.borders[3 * j], windowStartEx)
                            for k in range(internalInputs.borders[3 * j], internalInputs.borders[3 * j + 2]):
                                if previousSpectrum.size()[0] == 1:
                                    specA = previousSpectrum[-1]
                                else:
                                    specA = previousSpectrum[-2]
                                if currentSpectrum.size()[0] == 1:
                                    specB = currentSpectrum[0]
                                else:
                                    specB = currentSpectrum[1]
                                spectrum.write(voicebank.crfAi.processData(specA.to(device = device_ai), previousSpectrum[-1].to(device = device_ai), currentSpectrum[0].to(device = device_ai), specB.to(device = device_ai), (k - internalInputs.borders[3 * j]) / (internalInputs.borders[3 * j + 2] - internalInputs.borders[3 * j])), k)
                        
                        if internalInputs.endCaps[j]:
                            windowEnd = internalInputs.borders[3 * j + 5]
                            windowEndEx = internalInputs.borders[3 * j + 5]
                        else:
                            voicedExcitation.write(voicedExcitation.read(internalInputs.borders[3*j+3]*global_consts.batchSize, internalInputs.borders[3*j+5]*global_consts.batchSize)+nextVoicedExcitation[0:(internalInputs.borders[3*j+5]-internalInputs.borders[3*j+3])*global_consts.batchSize], internalInputs.borders[3*j+3]*global_consts.batchSize, internalInputs.borders[3*j+5]*global_consts.batchSize)
                            windowEnd = internalInputs.borders[3 * j + 3]
                            windowEndEx = internalInputs.borders[3 * j + 4]
                            excitation.write(nextExcitation[0:internalInputs.borders[3 * j + 5] - windowEndEx], windowEndEx, internalInputs.borders[3 * j + 5])
                            for k in range(internalInputs.borders[3 * j + 3], internalInputs.borders[3 * j + 5]):
                                if currentSpectrum.size()[0] == 1:
                                    specA = currentSpectrum[-1]
                                else:
                                    specA = currentSpectrum[-2]
                                if nextSpectrum.size()[0] == 1:
                                    specB = nextSpectrum[0]
                                else:
                                    specB = nextSpectrum[1]
                                spectrum.write(voicebank.crfAi.processData(specA.to(device = device_ai), currentSpectrum[-1].to(device = device_ai), nextSpectrum[0].to(device = device_ai), specB.to(device = device_ai), (k - internalInputs.borders[3 * j + 3]) / (internalInputs.borders[3 * j + 5] - internalInputs.borders[3 * j + 3])), k)
                        
                        #TODO: implement crfai skipping if transition was already calculated in the previous frame
                        
                        #istft of voiced excitation + pitch shift

                        #write remaining spectral data to cache
                        spectrum.write(currentSpectrum, windowStart, windowEnd)
                        excitation.write(currentExcitation, windowStartEx, windowEndEx)

                        remoteConnection.put(StatusChange(i, j, 2))
                        
                        #apply pitch shift to spectrum
                        logging.info("applying partial pitch shift to spectrum of sample " + str(j) + ", sequence " + str(i))
                        if internalInputs.phonemes[j] != "_autopause":
                            for k in range(internalInputs.borders[3 * j], internalInputs.borders[3 * j + 5]):
                                pitchBorder = math.ceil(global_consts.tripleBatchSize / internalInputs.pitch[k])
                                fourierPitchShift = math.ceil(global_consts.tripleBatchSize / voicebank.phonemeDict[internalInputs.phonemes[j]].pitch) - pitchBorder
                                shiftedSpectrum = torch.roll(spectrum.read(k), fourierPitchShift)
                                slope = torch.zeros(global_consts.halfTripleBatchSize + 1, device = device_rs)
                                slope[pitchBorder:pitchBorder + global_consts.pitchShiftSpectralRolloff] = torch.linspace(0, 1, global_consts.pitchShiftSpectralRolloff)
                                slope[pitchBorder + global_consts.pitchShiftSpectralRolloff:] = 1
                                spectrum.write((slope * spectrum.read(k)) + ((1 - slope) * shiftedSpectrum), k)
                        
                        internalStatusControl.ai[j] = 0
                        internalStatusControl.rs[j] = 1
                        remoteConnection.put(StatusChange(i, j, 3))
                        
                #final rendering and istft of pause-to-pause segment
                if ((j > 0) & interOutput) or (j == lastPoint):
                    logging.info("performing final rendering up to sample " + str(j - 1) + ", sequence " + str(i))
                    if aiActive:
                        startPoint = internalInputs.borders[3 * firstPoint]
                        voicedSignal = torch.stft(voicedExcitation.read(startPoint*global_consts.batchSize, internalInputs.borders[3 * (j - 1) + 5]*global_consts.batchSize), global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
        
                        if internalInputs.useBreathiness:
                            breathiness = internalInputs.breathiness[startPoint:internalInputs.borders[3 * (j - 1) + 5]].to(device = device_rs)
                        else:
                            breathiness = torch.zeros([internalInputs.borders[3 * (j - 1) + 5] - startPoint,], device = device_rs)
                        breathinessCompensation = torch.sum(torch.abs(voicedSignal), 0)[0:-1] / torch.maximum(torch.sum(torch.abs(excitation.read(startPoint, internalInputs.borders[3 * (j - 1) + 5])), 1), torch.tensor([0.0001], device = device_rs)) * global_consts.breCompPremul
                        breathinessUnvoiced = 1. + breathiness * breathinessCompensation * torch.gt(breathiness, 0) + breathiness * torch.logical_not(torch.gt(breathiness, 0))
                        breathinessVoiced = 1. - (breathiness * torch.gt(breathiness, 0))

                        voicedSignal = voicedSignal[:, 0:-1] * torch.transpose(processedSpectrum.read(startPoint, internalInputs.borders[3 * (j - 1) + 5]), 0, 1) * breathinessVoiced
                        excitationSignal = torch.transpose(excitation.read(startPoint, internalInputs.borders[3 * (j - 1) + 5]) * processedSpectrum.read(startPoint, internalInputs.borders[3 * (j - 1) + 5]), 0, 1) * breathinessUnvoiced

                        waveform = torch.istft(voicedSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided=True, length = internalInputs.borders[3 * (j - 1) + 5] * global_consts.batchSize).to(device = torch.device("cpu"))
                        excitationSignal = torch.istft(excitationSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided=True, length = internalInputs.borders[3 * (j - 1) + 5] * global_consts.batchSize)
                        waveform += excitationSignal.to(device = torch.device("cpu"))
                        lastZero = None
                        remoteConnection.put(StatusChange(i, startPoint*global_consts.batchSize, waveform.detach(), "updateAudio"))
                        remoteConnection.put(StatusChange(i, j - 1, 5))
                        if internalInputs.endCaps[j - 1] == True:
                            aiActive = False
                            #TODO: reset recurrent AI Tensors

                if (j > 0) & (interOutput == False):
                    remoteConnection.put(StatusChange(i, j - 1, 5))

                #update cache with new data, if cache is used
                if settings["cachingMode"] == "best rendering speed":
                    spectrumCache[i] = spectrum
                    processedSpectrumCache[i] = processedSpectrum
                    excitationCache[i] = excitation
                    voicedExcitationCache[i] = voicedExcitation

                #update or wait as required
                if rerenderFlag.is_set():
                    break
            else:
                continue
            break
        logging.info("rendering process finished for all sequences, waiting for render semaphore")
        rerenderFlag.wait()
        rerenderFlag.clear()
        try:
            c = connection.get_nowait()
        except:
            c = None
        if c != None:
            if updateFromMain(c, lastZero):
                break
