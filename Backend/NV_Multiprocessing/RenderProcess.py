# Copyright 2022, 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

def renderProcess(statusControlIn, voicebankListIn, nodeGraphListIn, inputListIn, connectionIn, remoteConnectionIn):

    import math
    import torch
    import global_consts
    import logging
    from traceback import print_exc
    import Backend.Resampler.Resamplers as rs
    from copy import copy
    from Backend.DataHandler.VocalSegment import VocalSegment
    from Backend.VB_Components.Voicebank import LiteVoicebank
    from Backend.NV_Multiprocessing.Interface import SequenceStatusControl, StatusChange
    from Backend.NV_Multiprocessing.Caching import DenseCache, SparseCache
    from Backend.NV_Multiprocessing.Update import trimSequence, posToSegment
    from Backend.Util import ensureTensorLength
    from MiddleLayer.IniParser import readSettings
    from Backend.Resampler.CubicSplineInter import interp
    from Backend.Resampler.PhaseShift import phaseShift

    global statusControl, voicebankList, nodeGraphList, inputList, connection, remoteConnection, internalStatusControl

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

    #setting up initial data structures
    statusControl = statusControlIn
    voicebankList = []
    for vbPath in voicebankListIn:
        voicebankList.append(LiteVoicebank(vbPath, device = device_ai))
    nodeGraphList = nodeGraphListIn#TODO: add node unwrapping
    inputList = inputListIn
    connection = connectionIn
    remoteConnection = remoteConnectionIn

    def updateFromMain(change, lastZero):
        global statusControl, voicebankList, nodeGraphList, inputList, connection, remoteConnection, internalStatusControl
        if change.type == "terminate":
            return True
        elif change.type == "addTrack":
            statusControl.append(SequenceStatusControl(change.data[2]))
            voicebankList.append(LiteVoicebank(change.data[0], device = device_ai))
            nodeGraphList.append(change.data[1])
            inputList.append(change.data[2])
            if settings["cachingMode"] == "best rendering speed":
                length = inputList[-1].pitch.size()[0]
                spectrumCache.append(DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs))
                processedSpectrumCache.append(DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs))
                excitationCache.append(DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs, torch.complex64))
                pitchCache.append(DenseCache((length,), device_rs))
        elif change.type == "removeTrack":
            del statusControl[change.data[0]]
            del voicebankList[change.data[0]]
            del nodeGraphList[change.data[0]]
            del inputList[change.data[0]]
            if settings["cachingMode"] == "best rendering speed":
                del spectrumCache[change.data[0]]
                del processedSpectrumCache[change.data[0]]
                del excitationCache[change.data[0]]
                del pitchCache[change.data[0]]
            remoteConnection.put(StatusChange(None, None, None, "deletion"))
        elif change.type == "duplicateTrack":
            statusControl.append(copy(statusControl[change.data[0]]))
            voicebankList.append(copy(voicebankList[change.data[0]]))
            nodeGraphList.append(copy(nodeGraphList[change.data[0]]))
            inputList.append(copy(inputList[change.data[0]]))
            if settings["cachingMode"] == "best rendering speed":
                spectrumCache.append(copy(spectrumCache[change.data[0]]))
                processedSpectrumCache.append(copy(processedSpectrumCache[change.data[0]]))
                excitationCache.append(copy(excitationCache[change.data[0]]))
                pitchCache.append(copy(pitchCache[change.data[0]]))
        elif change.type == "changeVB":
            del voicebankList[change.data[0]]
            voicebankList.insert(change.data[0], LiteVoicebank(change.data[1], device = device_ai))
            statusControl[change.data[0]].rs *= 0
            statusControl[change.data[0]].ai *= 0
        elif change.type == "nodeUpdate":
            nodeGraphList[change.data[0]].update(change.data[1])
            statusControl[change.data[0]].ai *= 0
        elif change.type == "enableParam":
            if change.data[1] == "breathiness":
                inputList[change.data[0]].useBreathiness = True
                statusControl[change.data[0]].rs *= 0
            elif change.data[1] == "steadiness":
                inputList[change.data[0]].useSteadiness = True
                statusControl[change.data[0]].rs *= 0
            elif change.data[1] == "vibrato speed":
                inputList[change.data[0]].useVibratoSpeed = True
                statusControl[change.data[0]].rs *= 0
            elif change.data[1] == "vibrato strength":
                inputList[change.data[0]].useVibratoStrength = True
                statusControl[change.data[0]].rs *= 0
            elif change.data[1] == "AI balance":
                inputList[change.data[0]].useAIBalance = True
                statusControl[change.data[0]].rs *= 0
            else:
                nodeGraphList[change.data[1]].enableParam(change.data[2])
            statusControl[change.data[1]].ai *= 0
        elif change.type == "disableParam":
            if change.data[1] == "breathiness":
                inputList[change.data[0]].useBreathiness = False
                statusControl[change.data[0]].rs *= 0
            elif change.data[1] == "steadiness":
                inputList[change.data[0]].useSteadiness = False
                statusControl[change.data[0]].rs *= 0
            elif change.data[1] == "vibrato speed":
                inputList[change.data[0]].useVibratoSpeed = False
                statusControl[change.data[0]].rs *= 0
            elif change.data[1] == "vibrato strength":
                inputList[change.data[0]].useVibratoStrength = False
                statusControl[change.data[0]].rs *= 0
            elif change.data[1] == "AI balance":
                inputList[change.data[0]].useAIBalance = False
                statusControl[change.data[0]].rs *= 0
            else:
                nodeGraphList[change.data[0]].disableParam(change.data[1])
            statusControl[change.data[0]].ai *= 0
        elif change.type == "changeInput":
            if change.data[1] in ["phonemes", "offsets", "repetititionSpacing"]:
                eval("inputList[change.data[0]]." + change.data[1])[change.data[2]:change.data[2] + len(change.data[3])] = change.data[3]
                statusControl[change.data[0]].rs[change.data[2]:change.data[2] + len(change.data[3])] *= 0
                statusControl[change.data[0]].ai[change.data[2]:change.data[2] + len(change.data[3])] *= 0
                if change.data[1] == "phonemes":
                    for j in range(len(change.data[3])):
                        if change.data[3][j] == "_autopause":
                            if change.data[2] + j + 1 < len(inputList[change.data[0]].startCaps):
                                inputList[change.data[0]].startCaps[change.data[2] + j + 1] = True
                            if change.data[2] + j > 0:
                                inputList[change.data[0]].endCaps[change.data[2] + j - 1] = True
                        else:
                            if change.data[2] + j + 1 < len(inputList[change.data[0]].startCaps):
                                inputList[change.data[0]].startCaps[change.data[2] + j + 1] = False
                            if change.data[2] + j > 0:
                                inputList[change.data[0]].endCaps[change.data[2] + j - 1] = False
                    if change.data[2] + len(change.data[3]) == len(inputList[change.data[0]].startCaps) > 0:
                        inputList[change.data[0]].endCaps[-1] = True
                    if change.data[2] == 0 and len(inputList[change.data[0]].startCaps) > 0:
                        inputList[change.data[0]].startCaps[0] = True
            elif change.data[1] == "borders":
                print("RENDERER: recv border change", change.data[2], change.data[3], inputList[change.data[0]].borders)
                start = inputList[change.data[0]].borders[change.data[2]] * global_consts.batchSize
                end = inputList[change.data[0]].borders[change.data[2] + len(change.data[3]) - 1] * global_consts.batchSize
                if lastZero == None or lastZero != [change.data[0], start, end - start]:
                    lastZero = [change.data[0], start, end - start]
                    remoteConnection.put(StatusChange(change.data[0], start, end - start, "zeroAudio"))
                for i in range(len(change.data[3])):
                    change.data[3][i] = int(change.data[3][i])
                inputList[change.data[0]].borders[change.data[2]:change.data[2] + len(change.data[3])] = change.data[3]
                start = max(math.floor(change.data[2] / 3) - 1, 0)
                end = math.floor((change.data[2] + len(change.data[3]) - 1) / 3) + 1
                statusControl[change.data[0]].rs[start:end] *= 0
                statusControl[change.data[0]].ai[start:end] *= 0
            elif change.data[1] in ["pitch", "steadiness", "breathiness", "aiBalance", "vibratoSpeed", "vibratoStrength"]:
                eval("inputList[change.data[0]]." + change.data[1])[change.data[2]:change.data[2] + len(change.data[3])] = change.data[3]
                positions = posToSegment(change.data[0], change.data[2], change.data[2] + len(change.data[3]), inputList)
                statusControl[change.data[0]].rs[positions[0]:positions[1]] *= 0
                statusControl[change.data[0]].ai[positions[0]:positions[1]] *= 0
            else:
                positions = posToSegment(change.data[0], change.data[2], change.data[2] + len(change.data[3]), inputList)
                inputList[change.data[0]].aiParamInputs[change.data[1]][change.data[2]:change.data[2] + len(change.data[3])] = change.data[3]
                statusControl[change.data[0]].ai[positions[0]:positions[1]] *= 0
        elif change.type == "offset":
            inputList, internalStatusControl = trimSequence(change.data[0], change.data[1], change.data[2], inputList, internalStatusControl)
        elif change.type == "changeLength":
            inputList[change.data[0]].length = change.data[1]
            inputList[change.data[0]].pitch = ensureTensorLength(inputList[change.data[0]].pitch, change.data[1], -1.)
            inputList[change.data[0]].steadiness = ensureTensorLength(inputList[change.data[0]].steadiness, change.data[1], 0.)
            inputList[change.data[0]].breathiness = ensureTensorLength(inputList[change.data[0]].breathiness, change.data[1], 0.)
            inputList[change.data[0]].aiBalance = ensureTensorLength(inputList[change.data[0]].aiBalance, change.data[1], 0.)
            inputList[change.data[0]].vibratoSpeed = ensureTensorLength(inputList[change.data[0]].vibratoSpeed, change.data[1], 0.)
            inputList[change.data[0]].vibratoStrength = ensureTensorLength(inputList[change.data[0]].vibratoStrength, change.data[1], 0.)

        if change.final == False:
            return updateFromMain(connection.get(), lastZero)
        else:
            try:
                return updateFromMain(connection.get_nowait(), lastZero)
            except:
                return False

    def pitchAdjust(spectrumInput, j, k, internalInputs, voicebank, previousShift, pitchOffset, device):
        if internalInputs.phonemes[j] == "_autopause":
            return spectrumInput, 0.
        steadiness = torch.pow(1. - internalInputs.steadiness[k], 2)
        targetPitch = global_consts.tripleBatchSize / (internalInputs.pitch[k] + pitchOffset * steadiness)
        nativePitch = global_consts.tripleBatchSize / (voicebank.phonemeDict[internalInputs.phonemes[j]][0].pitch + pitchOffset * steadiness)
        inputSpectrum = spectrumInput[global_consts.nHarmonics + 2:]
        phaseDifference = global_consts.batchSize / internalInputs.pitch[k].to(torch.float64)
        harmonics = spectrumInput[:int(global_consts.nHarmonics / 2) + 1]
        originSpace = torch.min(torch.linspace(0, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = device), torch.tensor([global_consts.halfTripleBatchSize,], device = device))
        newHarmonics = harmonics / interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = device), torch.square(inputSpectrum), originSpace)
        targetSpace = torch.min(torch.linspace(0, int(global_consts.nHarmonics / 2) * targetPitch, int(global_consts.nHarmonics / 2) + 1, device = device), torch.tensor([global_consts.halfTripleBatchSize,], device = device))
        newHarmonics *= interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = device), torch.square(inputSpectrum), targetSpace)
        slope = torch.ones([global_consts.pitchShiftSpectralRolloff,], device = device)
        slope[int(global_consts.pitchShiftSpectralRolloff / 2):] = torch.linspace(1, 0, int(global_consts.pitchShiftSpectralRolloff / 2), device = device)
        newHarmonics[:global_consts.pitchShiftSpectralRolloff] *= 1. - slope
        newHarmonics[:global_consts.pitchShiftSpectralRolloff] += slope * harmonics[:global_consts.pitchShiftSpectralRolloff]
        phases = spectrumInput[int(global_consts.nHarmonics / 2) + 1:global_consts.nHarmonics + 2]
        phases = phaseShift(phases, previousShift, device)
        previousShift += phaseDifference * 2 * math.pi
        previousShift = previousShift % (2 * math.pi)
        return torch.cat((newHarmonics, phases, torch.square(inputSpectrum)), 0), previousShift

    #setting up caching and other required data that is independent of each individual rendering iteration
    window = torch.hann_window(global_consts.tripleBatchSize, device = device_rs)
    lastZero = None
    if settings["cachingMode"] == "best rendering speed":
        spectrumCache = []
        processedSpectrumCache = []
        excitationCache = []
        pitchCache = []
        for i in range(len(statusControl)):
            length = inputList[i].pitch.size()[0]
            spectrumCache.append(DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs))
            processedSpectrumCache.append(DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs))
            excitationCache.append(DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs, torch.complex64))
            pitchCache.append(DenseCache((length,), device_rs))

    #main loop consisting of rendering iterations, and updates when required
    while True:
        logging.info("starting new rendering iteration")

        try:
            #iterate through individual VocalSequence objects
            for i, internalStatusControl in enumerate(statusControl):

                #set up data structures specific to rendering iteration
                logging.info("starting new sequence rendering iteration, copying data from main process")
                internalInputs = inputList[i]

                voicebank = voicebankList[i]
                voicebank.ai.device = device_ai
                nodeGraph = nodeGraphList[i]

                length = internalInputs.pitch.size()[0]

                logging.info("setting up local data structures")

                previousSpectrum = None
                previousExcitation = None
                previousPitch = None
                currentSpectrum = None
                currentExcitation = None
                currentPitch = None
                nextSpectrum = None
                nextExcitation = None
                nextPitch = None
                
                aiActive = False
                if settings["cachingMode"] == "best rendering speed":
                    spectrum = spectrumCache[i]
                    processedSpectrum = processedSpectrumCache[i]
                    excitation = excitationCache[i]
                    pitch = pitchCache[i]
                    indicator = -1
                elif settings["cachingMode"] == "save RAM":
                    spectrum = SparseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs)
                    processedSpectrum = SparseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs)
                    excitation = SparseCache((length, global_consts.halfTripleBatchSize + 1), device_rs, torch.complex64)
                    pitch = SparseCache((length,), device_rs)
                    indicator = 0
                else:
                    spectrum = DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs)
                    processedSpectrum = DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs)
                    excitation = DenseCache((length, global_consts.halfTripleBatchSize + 1), device_rs, torch.complex64)
                    pitch = DenseCache((length,), device_rs)
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
                for k, flag in enumerate(internalStatusControl.ai):
                    if flag <= 0:
                        firstPoint = k
                        break
                for k in range(len(internalStatusControl.ai)):
                    if internalStatusControl.ai[len(internalStatusControl.ai) - k - 1] <= 0:
                        lastPoint = len(internalStatusControl.ai) - k - 1
                        break

                voicebank.ai.reset()
                previousShift = 0.
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
                            processedSpectrum.write(spectrum.read(internalInputs.borders[3 * (j - 1)], internalInputs.borders[3 * (j - 1) + 3]), internalInputs.borders[3 * (j - 1)], internalInputs.borders[3 * (j - 1) + 3])
                            if internalInputs.endCaps[j - 1]:
                                processedSpectrum.write(spectrum.read(internalInputs.borders[3 * (j - 1) + 3], internalInputs.borders[3 * (j - 1) + 5]), internalInputs.borders[3 * (j - 1) + 3], internalInputs.borders[3 * (j - 1) + 5])
                            internalStatusControl.ai[j - 1] = 1
                            remoteConnection.put(StatusChange(i, j - 1, 4))

                    #move data to prepare for next sample
                    logging.info("shifting internal data backwards")
                    previousSpectrum = currentSpectrum
                    previousExcitation = currentExcitation
                    previousPitch = currentPitch
                    currentSpectrum = nextSpectrum
                    currentExcitation = nextExcitation
                    currentPitch = nextPitch
                    nextSpectrum = None
                    nextExcitation = None
                    nextPitch = None

                    #perform resampling if required
                    if j < len(internalStatusControl.ai):
                        if internalStatusControl.rs[j].item() == 0:
                            logging.info("calling resamplers for sample " + str(j) + ", sequence " + str(i))
                            if (internalInputs.startCaps[j] == False) and (previousSpectrum == None):
                                section = VocalSegment(internalInputs, voicebank, j - 1, device_rs)
                                previousSpectrum = rs.getSpecharm(section, device_rs)
                                previousExcitation = rs.getExcitation(section, device_rs)
                                previousPitch = rs.getPitch(section, device_rs)
                            if currentSpectrum == None:
                                section = VocalSegment(internalInputs, voicebank, j, device_rs)
                                currentSpectrum = rs.getSpecharm(section, device_rs)
                                currentExcitation = rs.getExcitation(section, device_rs)
                                currentPitch = rs.getPitch(section, device_rs)
                            if (internalInputs.endCaps[j] == False) and (nextSpectrum == None):
                                section = VocalSegment(internalInputs, voicebank, j + 1, device_rs)
                                nextSpectrum = rs.getSpecharm(section, device_rs)
                                nextExcitation = rs.getExcitation(section, device_rs)
                                nextPitch = rs.getPitch(section, device_rs)
                            remoteConnection.put(StatusChange(i, j, 1))

                            #calculate CrfAi transitions as required
                            logging.info("performing pitch shift of sample " + str(j) + ", sequence " + str(i))
                            if internalInputs.startCaps[j]:
                                windowStart = internalInputs.borders[3 * j]
                                windowStartEx = internalInputs.borders[3 * j]
                            else:
                                windowStart = internalInputs.borders[3 * j + 2]
                                windowStartEx = internalInputs.borders[3 * j + 1]
                                excitation.write(previousExcitation[internalInputs.borders[3 * j] - windowStartEx:], internalInputs.borders[3 * j], windowStartEx)
                                if previousSpectrum.size()[0] == 1:
                                    specA = previousSpectrum[-1]
                                else:
                                    specA = previousSpectrum[-2]
                                if currentSpectrum.size()[0] == 1:
                                    specB = currentSpectrum[0]
                                else:
                                    specB = currentSpectrum[1]
                                aiSpec = voicebank.ai.interpolate(
                                    specA.to(device = device_ai),
                                    previousSpectrum[-1].to(device = device_ai),
                                    currentSpectrum[0].to(device = device_ai),
                                    specB.to(device = device_ai),
                                    internalInputs.borders[3 * j + 2] - internalInputs.borders[3 * j],
                                    internalInputs.pitch[internalInputs.borders[3 * j]:internalInputs.borders[3 * j + 2]],
                                    internalInputs.borders[3 * j + 1] - internalInputs.borders[3 * j]
                                )
                                aiBalance = internalInputs.aiBalance[internalInputs.borders[3 * j]:internalInputs.borders[3 * j + 2]].unsqueeze(1).to(device = device_rs)
                                aiSpec = (0.5 - 0.5 * aiBalance) * aiSpec[0].to(device = device_rs) + (0.5 + 0.5 * aiBalance) * aiSpec[1].to(device = device_rs)
                                pitchOffset = previousPitch[internalInputs.borders[3 * j] - internalInputs.borders[3 * j + 2]:] + currentPitch[:internalInputs.borders[3 * j + 2] - internalInputs.borders[3 * j]]
                                for k in range(internalInputs.borders[3 * j], internalInputs.borders[3 * j + 2]):
                                    aiSpecOut, previousShift = pitchAdjust(aiSpec[k - internalInputs.borders[3 * j]], j, k, internalInputs, voicebank, previousShift, pitchOffset[k - internalInputs.borders[3 * j]], device_rs)
                                    spectrum.write(aiSpecOut.to(device_rs), k)
                                pitch.write(pitchOffset.to(device_rs), internalInputs.borders[3 * j], internalInputs.borders[3 * j + 2])
                            if internalInputs.endCaps[j]:
                                windowEnd = internalInputs.borders[3 * j + 5]
                                windowEndEx = internalInputs.borders[3 * j + 5]
                            else:
                                windowEnd = internalInputs.borders[3 * j + 3]
                                windowEndEx = internalInputs.borders[3 * j + 4]
                                excitation.write(nextExcitation[0:internalInputs.borders[3 * j + 5] - windowEndEx], windowEndEx, internalInputs.borders[3 * j + 5])
                            aiSpec = torch.roll(currentSpectrum, (-1,), (0,))
                            aiSpec[0] = currentSpectrum[0]
                            aiSpec = torch.squeeze(voicebank.ai.predict(aiSpec.to(device = device_ai))).to(device = device_rs)
                            aiBalance = internalInputs.aiBalance[windowStart:windowEnd].unsqueeze(1).to(device = device_rs)
                            aiSpec = (0.5 - 0.5 * aiBalance) * currentSpectrum + (0.5 + 0.5 * aiBalance) * aiSpec
                            pitchOffset = currentPitch[windowStart - internalInputs.borders[3 * j]:windowEnd - internalInputs.borders[3 * j]]
                            for k in range(currentSpectrum.size()[0]):
                                aiSpecOut, previousShift = pitchAdjust(aiSpec[k], j,  windowStart + k, internalInputs, voicebank, previousShift, pitchOffset[k], device_rs)
                                spectrum.write(aiSpecOut.to(device_rs), windowStart + k)
                            pitch.write(pitchOffset.to(device_rs), windowStart, windowEnd)
                            excitation.write(currentExcitation.to(device_rs), windowStartEx, windowEndEx)

                            if internalInputs.endCaps[j] == False:
                                if currentSpectrum.size()[0] == 1:
                                    specA = currentSpectrum[-1]
                                else:
                                    specA = currentSpectrum[-2]
                                if nextSpectrum.size()[0] == 1:
                                    specB = nextSpectrum[0]
                                else:
                                    specB = nextSpectrum[1]
                                aiSpec = voicebank.ai.interpolate(
                                    specA.to(device = device_ai),
                                    currentSpectrum[-1].to(device = device_ai),
                                    nextSpectrum[0].to(device = device_ai),
                                    specB.to(device = device_ai),
                                    internalInputs.borders[3 * j + 5] - internalInputs.borders[3 * j + 3],
                                    internalInputs.pitch[internalInputs.borders[3 * j + 3]:internalInputs.borders[3 * j + 5]],
                                    internalInputs.borders[3 * j + 4] - internalInputs.borders[3 * j + 3]
                                )
                                aiBalance = internalInputs.aiBalance[internalInputs.borders[3 * j + 3]:internalInputs.borders[3 * j + 5]].unsqueeze(1).to(device = device_rs)
                                aiSpec = (0.5 - 0.5 * aiBalance) * aiSpec[0].to(device = device_rs) + (0.5 + 0.5 * aiBalance) * aiSpec[1].to(device = device_rs)
                                pitchOffset = currentPitch[internalInputs.borders[3 * j + 3] - internalInputs.borders[3 * j + 5]:] + nextPitch[:internalInputs.borders[3 * j + 5] - internalInputs.borders[3 * j + 3]]
                                for k in range(internalInputs.borders[3 * j + 3], internalInputs.borders[3 * j + 5]):
                                    aiSpecOut, previousShift = pitchAdjust(aiSpec[k - internalInputs.borders[3 * j + 3]], j, k, internalInputs, voicebank, previousShift, pitchOffset[k - internalInputs.borders[3 * j + 3]], device_rs)
                                    spectrum.write(aiSpecOut.to(device_rs), k)
                                pitch.write(pitchOffset.to(device_rs), internalInputs.borders[3 * j + 3], internalInputs.borders[3 * j + 5])
                            
                            #TODO: implement crfai skipping if transition was already calculated in the previous frame
                            
                            #istft of voiced excitation + pitch shift

                            #write remaining spectral data to cache
                            #import matplotlib.pyplot as plt
                            #plt.imshow(spectrum.read(0, 1000).detach()) #TODO: remove gradient tracking in cache
                            #plt.show()

                            remoteConnection.put(StatusChange(i, j, 2))

                            #apply pitch shift to spectrum
                            logging.info("applying partial pitch shift to spectrum of sample " + str(j) + ", sequence " + str(i))
                            if internalInputs.phonemes[j] != "_autopause":
                                previousShift = 0.
                            internalStatusControl.ai[j] = 0
                            internalStatusControl.rs[j] = 1
                            remoteConnection.put(StatusChange(i, j, 3))

                    #final rendering and istft of pause-to-pause segment
                    if ((j > 0) & interOutput) or (j == lastPoint):
                        logging.info("performing final rendering up to sample " + str(j - 1) + ", sequence " + str(i))
                        if aiActive:
                            startPoint = internalInputs.borders[3 * firstPoint]
                            abs = processedSpectrum.read(startPoint, internalInputs.borders[3 * (j - 1) + 5])[:, :int(global_consts.nHarmonics / 2) + 1]
                            angle = processedSpectrum.read(startPoint, internalInputs.borders[3 * (j - 1) + 5])[:, int(global_consts.nHarmonics / 2) + 1:global_consts.nHarmonics + 2]
                            pitchOffset = pitch.read(startPoint, internalInputs.borders[3 * (j - 1) + 5])
                            steadiness = torch.pow(1. - internalInputs.steadiness[startPoint:internalInputs.borders[3 * (j - 1) + 5]], 2)
                            harms = torch.polar(abs, angle)
                            #sines = harms.imag
                            #cosines = harms.real
                            #newSines = torch.empty(harms.size()[0], global_consts.halfTripleBatchSize + 1)
                            #newCosines = torch.empty(harms.size()[0], global_consts.halfTripleBatchSize + 1)
                            voicedSignal = torch.empty((internalInputs.borders[3 * (j - 1) + 5] - startPoint, global_consts.halfTripleBatchSize + 1), dtype = torch.complex64, device = device_rs)
                            for k in range(harms.size()[0]):
                                requiredSize = global_consts.tripleBatchSize / (internalInputs.pitch[k + startPoint].item() + pitchOffset[k] * steadiness[k])
                                harmCurve = torch.tile(torch.fft.irfft(harms[k], global_consts.nHarmonics), (math.ceil(requiredSize),))[:int(requiredSize * global_consts.nHarmonics)]
                                voicedSignal[k] = torch.fft.rfft(interp(torch.linspace(0, 1, int(requiredSize * global_consts.nHarmonics), device = device_rs), harmCurve, torch.linspace(0, 1, global_consts.tripleBatchSize, device = device_rs)) * window)
                                #inputFreqs = torch.linspace(0, int(global_consts.nHarmonics / 2), int(global_consts.nHarmonics / 2) + 1).unsqueeze(0).tile(global_consts.halfTripleBatchSize + 1, 1) * requiredSize
                                #outputFreqs = torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1).unsqueeze(1).tile(1, int(global_consts.nHarmonics / 2) + 1)
                                #sineToSine = outputFreqs * torch.sin(2 * math.pi * inputFreqs) / (torch.pow(inputFreqs, 2) - torch.pow(outputFreqs, 2)) / math.pi
                                #sineToSine[0, 0] = 1.
                                #cosineToCosine = inputFreqs * torch.sin(2 * math.pi * inputFreqs) / (torch.pow(inputFreqs, 2) - torch.pow(outputFreqs, 2)) / math.pi
                                #cosineToCosine[0, 0] = 1.
                                #sineToCosine = (-inputFreqs * torch.cos(2 * math.pi * inputFreqs) + inputFreqs) / (torch.pow(inputFreqs, 2) - torch.pow(outputFreqs, 2)) / math.pi
                                #sineToCosine[0, 0] = 1.
                                #cosineToSine = (outputFreqs * torch.cos(2 * math.pi * inputFreqs) - outputFreqs) / (torch.pow(inputFreqs, 2) - torch.pow(outputFreqs, 2)) / math.pi
                                #cosineToSine[0, 0] = 1.
                                #newSines[k] = torch.matmul(sineToSine, sines[k]) + torch.matmul(cosineToSine, cosines[k])
                                #newCosines[k] = torch.matmul(cosineToCosine, cosines[k]) + torch.matmul(sineToCosine, sines[k])
                            #voicedSignal = torch.complex(newCosines, newSines)
                            if internalInputs.useBreathiness:
                                breathiness = internalInputs.breathiness[startPoint:internalInputs.borders[3 * (j - 1) + 5]].to(device = device_rs)
                            else:
                                breathiness = torch.zeros([internalInputs.borders[3 * (j - 1) + 5] - startPoint,], device = device_rs)

                            excitationSignal = torch.transpose(excitation.read(startPoint, internalInputs.borders[3 * (j - 1) + 5]) * processedSpectrum.read(startPoint, internalInputs.borders[3 * (j - 1) + 5])[:, global_consts.nHarmonics + 2:], 0, 1)
                            breathinessCompensation = torch.sum(torch.abs(voicedSignal), 1) / torch.maximum(torch.sum(torch.abs(excitation.read(startPoint, internalInputs.borders[3 * (j - 1) + 5])), 1), torch.tensor([0.0001], device = device_rs)) * global_consts.breCompPremul
                            breathinessUnvoiced = 1. + breathiness * breathinessCompensation * torch.gt(breathiness, 0) + breathiness * torch.logical_not(torch.gt(breathiness, 0))
                            breathinessVoiced = 1. - (breathiness * torch.gt(breathiness, 0))
                            voicedSignal *= torch.unsqueeze(breathinessVoiced, 1)
                            excitationSignal *= breathinessUnvoiced
                            voicedSignal = torch.transpose(voicedSignal, 0, 1)
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

                    #update or wait as required
                    try:
                        c = connection.get_nowait()
                    except:
                        c = False
                    if c:
                        updateFromMain(c, lastZero)
                        break
                else:
                    continue
                break
            else:
                logging.info("rendering iteration finished, waiting for new data...")
                print("rendering iteration finished, waiting for new data...")
                c = connection.get()
                if updateFromMain(c, lastZero):
                    break
                
        except Exception as exc:
            print_exc()
            remoteConnection.put(StatusChange(None, None, exc, "error"))
            break
