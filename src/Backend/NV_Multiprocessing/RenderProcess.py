# Copyright 2022, 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

def renderProcess(statusControlIn, voicebankListIn, nodeGraphListIn, inputListIn, connectionIn, remoteConnectionIn):

    import math
    import ctypes
    import torch
    import global_consts
    import logging
    from traceback import print_exc
    import Backend.Resampler.Resamplers as rs
    from copy import deepcopy
    from os import getenv, path
    from Backend.DataHandler.VocalSegment import VocalSegment
    from Backend.DataHandler.VoicebankManager import VoicebankManager
    from Backend.NV_Multiprocessing.Interface import SequenceStatusControl, StatusChange
    from Backend.NV_Multiprocessing.Caching import DenseCache, SparseCache
    from Backend.NV_Multiprocessing.Update import trimSequence, posToSegment, unpackNodes
    from Backend.Node import NodeBaseLib
    from Util import ensureTensorLength
    from MiddleLayer.IniParser import readSettings
    from Backend.Resampler.CubicSplineInter import interp
    from C_Bridge import esper

    global statusControl, voicebankList, nodeGraphList, inputList, connection, remoteConnection, internalStatusControl

    logging.basicConfig(level = logging.INFO)
    logPath = path.join(getenv("APPDATA"), "Nova-Vox", "Logs", "editor_renderer.log")
    logging.basicConfig(format='%(asctime)s:%(process)s:%(levelname)s:%(message)s',
                        filename=logPath,
                        filemode = "w",
                        force = True,
                        level=logging.INFO)

    #reading settings, setting device and interOutput properties accordingly
    logging.info("render process started, reading settings")
    settings = readSettings()
    if settings["lowspecmode"] == "enabled":
        interOutput = False
    elif settings["lowspecmode"] == "disabled":
        interOutput = True
    else:
        print("could not read intermediate output setting. Intermediate outputs have been disabled by default.")
        interOutput = False
    if settings["accelerator"] == "CPU":
        device_rs = torch.device("cpu")
        device_ai = torch.device("cpu")
    elif settings["accelerator"] == "GPU":
        device_rs = torch.device("cpu")
        device_ai = torch.device("cuda")
    else:
        print("could not read accelerator setting. Accelerator has been set to CPU by default.")
        device_rs = torch.device("cpu")
        device_ai = torch.device("cpu")

    #setting up initial data structures
    statusControl = statusControlIn
    voicebankManager = VoicebankManager()
    voicebankList = []
    for vbPath in voicebankListIn:
        voicebankList.append(voicebankManager.getVoicebank(vbPath, device_ai))
    nodeGraphList = nodeGraphListIn
    inputList = inputListIn
    connection = connectionIn
    remoteConnection = remoteConnectionIn

    def updateFromMain(change):
        global statusControl, voicebankList, nodeGraphList, inputList, connection, remoteConnection, internalStatusControl
        if change.type == "terminate":
            return True
        elif change.type == "addTrack":
            statusControl.append(SequenceStatusControl(change.data[2]))
            voicebankList.append(voicebankManager.getVoicebank(change.data[0], device_ai))
            nodeGraphList.append(unpackNodes(*change.data[1]))
            inputList.append(change.data[2])
            if settings["cachingmode"] == "best rendering speed":
                length = inputList[-1].pitch.size()[0]
                spectrumCache.append(DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs))
                processedSpectrumCache.append(DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs))
                pitchCache.append(DenseCache((length,), device_rs))
                processedPitchCache.append(DenseCache((length,), device_rs))
        elif change.type == "removeTrack":
            del statusControl[change.data[0]]
            del voicebankList[change.data[0]]
            voicebankManager.clean()
            del nodeGraphList[change.data[0]]
            del inputList[change.data[0]]
            if settings["cachingmode"] == "best rendering speed":
                del spectrumCache[change.data[0]]
                del processedSpectrumCache[change.data[0]]
                del pitchCache[change.data[0]]
                del processedPitchCache[change.data[0]]
            remoteConnection.put(StatusChange(None, None, None, "deletion"))
        elif change.type == "duplicateTrack":
            statusControl.append(statusControl[change.data[0]].duplicate())
            voicebankList.append(voicebankList[change.data[0]])
            nodeGraphList.append(deepcopy(nodeGraphList[change.data[0]]))
            inputList.append(inputList[change.data[0]].duplicate())
            if settings["cachingmode"] == "best rendering speed":
                spectrumCache.append(spectrumCache[change.data[0]].duplicate())
                processedSpectrumCache.append(processedSpectrumCache[change.data[0]].duplicate())
                pitchCache.append(pitchCache[change.data[0]].duplicate())
                processedPitchCache.append(processedPitchCache[change.data[0]].duplicate())
        elif change.type == "changeVB":
            del voicebankList[change.data[0]]
            voicebankList.insert(change.data[0], voicebankManager.getVoicebank(change.data[1], device_ai))
            voicebankManager.clean()
            statusControl[change.data[0]].rs *= 0
            statusControl[change.data[0]].ai *= 0
        elif change.type == "changeTrackSettings":
            if change.data[1] == "unvoicedShift":
                inputList[change.data[0]].unvoicedShift = change.data[2]
                statusControl[change.data[0]].rs *= 0
            else:
                print("unknown track setting change: " + change.data[1])
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
            elif change.data[1] == "gender factor":
                inputList[change.data[0]].useGenderFactor = True
                statusControl[change.data[0]].rs *= 0
            else:
                nodeGraphList[change.data[0]][1][change.data[1]].enabled = True
            statusControl[change.data[0]].ai *= 0
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
            elif change.data[1] == "gender factor":
                inputList[change.data[0]].useGenderFactor = False
                statusControl[change.data[0]].rs *= 0
            else:
                nodeGraphList[change.data[0]][1][change.data[1]].enabled = True
            statusControl[change.data[0]].ai *= 0
        elif change.type == "changeInput":
            if change.data[1] in ["phonemes", "offsets", "repetititionSpacing"]:
                eval("inputList[change.data[0]]." + change.data[1])[change.data[2]:change.data[2] + len(change.data[3])] = change.data[3]
                statusControl[change.data[0]].rs[change.data[2]:change.data[2] + len(change.data[3])] *= 0
                statusControl[change.data[0]].ai[change.data[2]:change.data[2] + len(change.data[3])] *= 0
            elif change.data[1] == "borders":
                start = inputList[change.data[0]].borders[change.data[2]] * global_consts.batchSize
                end = inputList[change.data[0]].borders[change.data[2] + len(change.data[3]) - 1] * global_consts.batchSize
                remoteConnection.put(StatusChange(change.data[0], start, end - start, "zeroAudio"))
                for i in range(len(change.data[3])):
                    change.data[3][i] = int(change.data[3][i])
                inputList[change.data[0]].borders[change.data[2]:change.data[2] + len(change.data[3])] = change.data[3]
                start = max(math.floor(change.data[2] / 3) - 1, 0)
                end = math.floor((change.data[2] + len(change.data[3]) - 1) / 3) + 1
                statusControl[change.data[0]].rs[start:end] *= 0
                statusControl[change.data[0]].ai[start:end] *= 0
            elif change.data[1] in ["pitch", "steadiness", "breathiness", "aiBalance", "genderFactor", "vibratoSpeed", "vibratoStrength"]:
                eval("inputList[change.data[0]]." + change.data[1])[change.data[2]:change.data[2] + len(change.data[3])] = change.data[3]
                positions = posToSegment(change.data[0], change.data[2], change.data[2] + len(change.data[3]), inputList)
                statusControl[change.data[0]].rs[positions[0]:positions[1]] *= 0
                statusControl[change.data[0]].ai[positions[0]:positions[1]] *= 0
            else:
                positions = posToSegment(change.data[0], change.data[2], change.data[2] + len(change.data[3]), inputList)
                nodeGraphList[change.data[0]][1][change.data[1]]._value[change.data[2]:change.data[2] + len(change.data[3])] = change.data[3]
                statusControl[change.data[0]].ai[positions[0]:positions[1]] *= 0
        elif change.type == "offset":
            inputList, statusControl = trimSequence(change.data[0], change.data[1], change.data[2], inputList, statusControl)
            remoteConnection.put(StatusChange(change.data[0], None, None, "offsetApplied"))
        elif change.type == "changeLength":
            inputList[change.data[0]].length = change.data[1]
            inputList[change.data[0]].pitch = ensureTensorLength(inputList[change.data[0]].pitch, change.data[1], -1.)
            inputList[change.data[0]].steadiness = ensureTensorLength(inputList[change.data[0]].steadiness, change.data[1], 0.)
            inputList[change.data[0]].breathiness = ensureTensorLength(inputList[change.data[0]].breathiness, change.data[1], 0.)
            inputList[change.data[0]].aiBalance = ensureTensorLength(inputList[change.data[0]].aiBalance, change.data[1], 0.)
            inputList[change.data[0]].genderFactor = ensureTensorLength(inputList[change.data[0]].genderFactor, change.data[1], 0.)
            inputList[change.data[0]].vibratoSpeed = ensureTensorLength(inputList[change.data[0]].vibratoSpeed, change.data[1], 0.)
            inputList[change.data[0]].vibratoStrength = ensureTensorLength(inputList[change.data[0]].vibratoStrength, change.data[1], 0.)
            for key in nodeGraphList[change.data[0]][1].keys():
                nodeGraphList[change.data[0]][1][key]._value = ensureTensorLength(inputList[change.data[0]][1][key]._value, change.data[1], 0.)
        elif change.type == "changeNodegraph":
            nodeGraphList[change.data[0]] = unpackNodes(*change.data[1])
            statusControl[change.data[0]].ai *= 0

        if change.final == False:
            return updateFromMain(connection.get())
        else:
            try:
                return updateFromMain(connection.get_nowait())
            except:
                return False
    
    def pitchAdjust(specharm, j, start, end, internalInputs, voicebank, pitchOffset):
        specharm_cpy = specharm.clone().contiguous()
        if internalInputs.phonemes[j] in ("_autopause", "pau"):
            return specharm_cpy
        steadiness = torch.pow(1. - internalInputs.steadiness[start:end], 2)
        srcPitch = voicebank.phonemeDict.fetch(internalInputs.phonemes[j], True)[0].pitch + pitchOffset * steadiness
        tgtPitch = internalInputs.pitch[start:end] + pitchOffset * steadiness
        srcPitch = torch.max(srcPitch, torch.tensor([10.,])).contiguous()
        tgtPitch = torch.max(tgtPitch, torch.tensor([10.,])).contiguous()
        genderFactor = internalInputs.genderFactor[start:end]
        tgtPitch = torch.where(genderFactor > 0., tgtPitch * (1. + genderFactor), tgtPitch / (1. - genderFactor))
        breathiness = internalInputs.breathiness[start:end].to(torch.float32).contiguous()
        formantShift = torch.tensor([internalInputs.unvoicedShift,]).contiguous()
        specharm_ptr = ctypes.cast(specharm_cpy.data_ptr(), ctypes.POINTER(ctypes.c_float))
        srcPitch_ptr = ctypes.cast(srcPitch.data_ptr(), ctypes.POINTER(ctypes.c_float))
        tgtPitch_ptr = ctypes.cast(tgtPitch.data_ptr(), ctypes.POINTER(ctypes.c_float))
        formantShift_ptr = ctypes.cast(formantShift.data_ptr(), ctypes.POINTER(ctypes.c_float))
        breathiness_ptr = ctypes.cast(breathiness.data_ptr(), ctypes.POINTER(ctypes.c_float))
        esper.pitchShift(specharm_ptr, srcPitch_ptr, tgtPitch_ptr, formantShift_ptr, breathiness_ptr, end - start, global_consts.config)
        return specharm_cpy
    
    def processNodegraph(earlyBorders, spectrum, pitch, internalInputs, j, nodeGraph, nodeInputs, nodeParams, nodeParamData, nodeOutput):
        if earlyBorders:
            start = internalInputs.borders[3 * j]
            end = internalInputs.borders[3 * j + 3]
        else:
            start = internalInputs.borders[3 * j + 3]
            end = internalInputs.borders[3 * j + 5]
        specPart = spectrum.read(start, end)
        pitchPart = pitch.read(start, end)
        if nodeOutput == None:
            return specPart, pitchPart
        audio = torch.cat((specPart, internalInputs.pitch[start:end].unsqueeze(1) + pitchPart.unsqueeze(1)), 1)
        output = torch.zeros_like(audio)
        length = audio.size()[0]
        for k in range(length):
            if earlyBorders and j > 0:
                if k < internalInputs.borders[3 * j + 1] - start:
                    fadeIn = 0.5 * k / (internalInputs.borders[3 * j + 1] - start)
                elif k < end - start:
                    fadeIn = 0.5 + 0.5 * (k - internalInputs.borders[3 * j + 1] + start) / (end - internalInputs.borders[3 * j + 1])
                else:
                    fadeIn = 1.
            else:
                fadeIn = None
            for input in nodeInputs:
                input.audio = audio[k]
                input.phoneme = (internalInputs.phonemes[j - 1], internalInputs.phonemes[j], fadeIn) if fadeIn != None else (internalInputs.phonemes[j], internalInputs.phonemes[j], 0.5) 
                input.pitch = internalInputs.pitch[start + k] + pitchPart[k]
                input.transition = 0. if fadeIn == None else fadeIn
                input.breathiness = internalInputs.breathiness[start + k]
                input.steadiness = internalInputs.steadiness[start + k]
                input.AIBalance = internalInputs.aiBalance[start + k]
                input.genderFactor = internalInputs.genderFactor[start + k]
                input.loopOffset = internalInputs.offsets[j - 1] * (1. - fadeIn) + internalInputs.offsets[j] * fadeIn if fadeIn != None else internalInputs.offsets[j]
                input.loopOverlap = internalInputs.repetititionSpacing[j - 1] * (1. - fadeIn) + internalInputs.repetititionSpacing[j] * fadeIn if fadeIn != None else internalInputs.repetititionSpacing[j]
                input.vibratoStrength = internalInputs.vibratoStrength[start + k]
                input.vibratoSpeed = internalInputs.vibratoSpeed[start + k]
            for param in nodeParams:
                param.curve = nodeParamData[param.auxData["name"]]._value[start + k]
            nodeOutput.calculate()
            output[k] = nodeOutput.audio
            for node in nodeGraph[0]:
                node.isUpdated = False
        return output[:-1], output[-1]
            
    
    def finalRender(specharm:torch.Tensor, pitch:torch.Tensor, length:int, device:torch.device) -> torch.Tensor:
        import matplotlib.pyplot as plt
        plt.imshow(torch.log(specharm[:, :] + 0.001).cpu())
        #plt.imshow(specharm[:, :].cpu())
        plt.show()
        
        renderTarget = torch.zeros([length * global_consts.batchSize,], device = device)
        specharm = specharm.contiguous()
        specharm_ptr = ctypes.cast(specharm.data_ptr(), ctypes.POINTER(ctypes.c_float))
        pitch = pitch.contiguous()
        pitch_ptr = ctypes.cast(pitch.data_ptr(), ctypes.POINTER(ctypes.c_float))
        renderTarget = renderTarget.contiguous()
        renderTarget_ptr = ctypes.cast(renderTarget.data_ptr(), ctypes.POINTER(ctypes.c_float))
        phase = torch.tensor([0.], device = device)
        phase_ptr = ctypes.cast(phase.data_ptr(), ctypes.POINTER(ctypes.c_float))
        esper.render(specharm_ptr, pitch_ptr, phase_ptr, renderTarget_ptr, length, global_consts.config)
        return renderTarget

    #setting up caching and other required data that is independent of each individual rendering iteration
    if settings["cachingmode"] == "best rendering speed":
        spectrumCache = []
        processedSpectrumCache = []
        pitchCache = []
        processedPitchCache = []
        for i in range(len(statusControl)):
            length = inputList[i].pitch.size()[0]
            spectrumCache.append(DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs))
            processedSpectrumCache.append(DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs))
            pitchCache.append(DenseCache((length,), device_rs))
            processedPitchCache.append(DenseCache((length,), device_rs))

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
                nodeInputs = []
                nodeParamData = nodeGraph[1]
                nodeParams = []
                nodeOutput = None
                for node in nodeGraph[0]:
                    if isinstance(node, NodeBaseLib.InputNode):
                        nodeInputs.append(node)
                    elif isinstance(node, NodeBaseLib.CurveInputNode):
                        nodeParams.append(node)
                    elif isinstance(node, NodeBaseLib.OutputNode):
                        nodeOutput = node

                length = internalInputs.pitch.size()[0]

                logging.info("setting up local data structures")

                previousSpectrum = None
                previousPitch = None
                currentSpectrum = None
                currentPitch = None
                nextSpectrum = None
                nextPitch = None
                
                if settings["cachingmode"] == "best rendering speed":
                    spectrum = spectrumCache[i]
                    processedSpectrum = processedSpectrumCache[i]
                    pitch = pitchCache[i]
                    processedPitch = processedPitchCache[i]
                    indicator = 1
                elif settings["cachingmode"] == "save RAM":
                    spectrum = SparseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs)
                    processedSpectrum = SparseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs)
                    pitch = SparseCache((length,), device_rs)
                    processedPitch = SparseCache((length,), device_rs)
                    indicator = 0
                else:
                    spectrum = DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs)
                    processedSpectrum = DenseCache((length, global_consts.nHarmonics + global_consts.halfTripleBatchSize + 3), device_rs)
                    pitch = DenseCache((length,), device_rs)
                    processedPitch = DenseCache((length,), device_rs)
                    indicator = 0
                firstPoint = None

                #go through internalStatusControl lists and set all parts required to render a pause-to-pause section to the value of indicator.
                #when caching is available, this causes them to be loaded from cache, otherwise they are re-computed.
                
                for k in range(len(internalStatusControl.ai)):
                    if internalInputs.phonemes[k] in ("_autopause", "pau"):
                        internalStatusControl.rs[k] = 1
                        internalStatusControl.ai[k] = 1
                aiActive = False
                for k in range(len(internalStatusControl.ai)):
                    if internalStatusControl.ai[k] == 0:
                        aiActive = True
                    if internalInputs.phonemes[k] in ("_autopause", "pau"):
                        aiActive = False
                    if aiActive:
                        internalStatusControl.ai[k] = 0
                aiActive = False
                for k in range(len(internalStatusControl.ai)):
                    if internalStatusControl.ai[len(internalStatusControl.ai) - k - 1] == 0:
                        aiActive = True
                    if internalInputs.phonemes[len(internalStatusControl.ai) - k - 1] in ("_autopause", "pau"):
                        aiActive = False
                    if aiActive:
                        internalStatusControl.ai[len(internalStatusControl.ai) - k - 1] = 0
                aiActive = False
                for k in range(len(internalStatusControl.ai)):
                    if internalStatusControl.ai[k] == 0:
                        internalStatusControl.rs[k] = indicator

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
                            nodeOutputs = processNodegraph(True, spectrum, pitch, internalInputs, j - 1, nodeGraph, nodeInputs, nodeParams, nodeParamData, nodeOutput)
                            processedSpectrum.write(nodeOutputs[0], internalInputs.borders[3 * (j - 1)], internalInputs.borders[3 * (j - 1) + 3])
                            processedPitch.write(nodeOutputs[1], internalInputs.borders[3 * (j - 1)], internalInputs.borders[3 * (j - 1) + 3])
                            if (j == len(internalStatusControl.ai) or internalInputs.phonemes[j] in ("pau", "_autopause")):
                                nodeOutputs = processNodegraph(False, spectrum, pitch, internalInputs, j - 1, nodeGraph, nodeInputs, nodeParams, nodeParamData, nodeOutput)
                                processedSpectrum.write(nodeOutputs[0], internalInputs.borders[3 * (j - 1) + 3], internalInputs.borders[3 * (j - 1) + 5])
                                processedPitch.write(nodeOutputs[1], internalInputs.borders[3 * (j - 1) + 3], internalInputs.borders[3 * (j - 1) + 5])
                            internalStatusControl.ai[j - 1] = 1
                        remoteConnection.put(StatusChange(i, j - 1, 4))

                    #move data to prepare for next sample
                    logging.info("shifting internal data backwards")
                    previousSpectrum = currentSpectrum
                    previousPitch = currentPitch
                    currentSpectrum = nextSpectrum
                    currentPitch = nextPitch
                    nextSpectrum = None
                    nextPitch = None

                    #perform resampling if required
                    if j < len(internalStatusControl.ai):
                        if internalStatusControl.rs[j].item() == 0:
                            logging.info("calling resamplers for sample " + str(j) + ", sequence " + str(i))
                            if j > 0 and internalInputs.phonemes[j - 1] not in ("pau", "_autopause") and (previousSpectrum == None):
                                section = VocalSegment(internalInputs, voicebank, j - 1, device_rs)
                                previousSpectrum = rs.getSpecharm(section, device_rs)
                                previousPitch = rs.getPitch(section, device_rs)
                            if currentSpectrum == None:
                                section = VocalSegment(internalInputs, voicebank, j, device_rs)
                                currentSpectrum = rs.getSpecharm(section, device_rs)
                                currentPitch = rs.getPitch(section, device_rs)
                            if j + 1 < len(internalStatusControl.ai) and internalInputs.phonemes[j + 1] not in ("pau", "_autopause") and (nextSpectrum == None):
                                section = VocalSegment(internalInputs, voicebank, j + 1, device_rs)
                                nextSpectrum = rs.getSpecharm(section, device_rs)
                                nextPitch = rs.getPitch(section, device_rs)
                            remoteConnection.put(StatusChange(i, j, 1))

                            #calculate CrfAi transitions as required
                            logging.info("performing pitch shift of sample " + str(j) + ", sequence " + str(i))
                            previousExpression = internalInputs.phonemes[j].split("_")
                            if len(previousExpression) > 1:
                                previousExpression = previousExpression[-1]
                            else:
                                expression = ""
                            expression = internalInputs.phonemes[j].split("_")
                            if len(expression) > 1:
                                expression = expression[-1]
                            else:
                                expression = ""
                            nextExpression = internalInputs.phonemes[j].split("_")
                            if len(nextExpression) > 1:
                                nextExpression = nextExpression[-1]
                            else:
                                nextExpression = ""
                            if j == 0 or internalInputs.phonemes[j - 1] in ("pau", "_autopause"):
                                windowStart = internalInputs.borders[3 * j]
                            else:
                                windowStart = internalInputs.borders[3 * j + 2]
                                if previousSpectrum.size()[0] < 3:
                                    specA = previousSpectrum[-1]
                                    specB = previousSpectrum[-1]
                                else:
                                    specA = previousSpectrum[-3]
                                    specB = previousSpectrum[-2]
                                if currentSpectrum.size()[0] < 3:
                                    specC = currentSpectrum[0]
                                    specD = currentSpectrum[0]
                                else:
                                    specC = currentSpectrum[0]
                                    specD = currentSpectrum[1]
                                aiSpec = voicebank.ai.interpolate(
                                    specA.to(device = device_ai),
                                    specB.to(device = device_ai),
                                    specC.to(device = device_ai),
                                    specD.to(device = device_ai),
                                    voicebank.phonemeDict.fetch(internalInputs.phonemes[j - 1], True)[0].embedding,
                                    voicebank.phonemeDict.fetch(internalInputs.phonemes[j], True)[0].embedding,
                                    previousExpression,
                                    expression,
                                    internalInputs.borders[3 * j + 2] - internalInputs.borders[3 * j] + 1,
                                    internalInputs.pitch[internalInputs.borders[3 * j] - 1:internalInputs.borders[3 * j + 2]],
                                    internalInputs.borders[3 * j + 1] - internalInputs.borders[3 * j]
                                )
                                aiBalance = internalInputs.aiBalance[internalInputs.borders[3 * j] - 1:internalInputs.borders[3 * j + 2]].unsqueeze(1).to(device = device_rs)
                                aiSpec = (0.5 - 0.5 * aiBalance) * aiSpec[0].to(device = device_rs) + (0.5 + 0.5 * aiBalance) * aiSpec[1].to(device = device_rs)
                                pitchOffset = previousPitch[internalInputs.borders[3 * j] - internalInputs.borders[3 * j + 2]:] + currentPitch[:internalInputs.borders[3 * j + 2] - internalInputs.borders[3 * j]]
                                aiSpecOut = pitchAdjust(aiSpec, j, internalInputs.borders[3 * j], internalInputs.borders[3 * j + 2], internalInputs, voicebank, pitchOffset)
                                spectrum.write(aiSpecOut.to(device_rs), internalInputs.borders[3 * j] - 1, internalInputs.borders[3 * j + 2])
                                pitch.write(pitchOffset.to(device_rs), internalInputs.borders[3 * j], internalInputs.borders[3 * j + 2])
                            if j + 1 == len(internalStatusControl.ai) or internalInputs.phonemes[j + 1] in ("pau", "_autopause"):
                                windowEnd = internalInputs.borders[3 * j + 5]
                            else:
                                windowEnd = internalInputs.borders[3 * j + 3]
                            aiSpec = torch.squeeze(voicebank.ai.refine(currentSpectrum.to(device = device_ai), expression)).to(device = device_rs)
                            aiBalance = internalInputs.aiBalance[windowStart:windowEnd].unsqueeze(1).to(device = device_rs)
                            aiSpec = (0.5 - 0.5 * aiBalance) * currentSpectrum + (0.5 + 0.5 * aiBalance) * aiSpec
                            pitchOffset = currentPitch[windowStart - internalInputs.borders[3 * j]:windowEnd - internalInputs.borders[3 * j]]
                            aiSpecOut = pitchAdjust(aiSpec, j, windowStart, windowEnd, internalInputs, voicebank, pitchOffset)
                            spectrum.write(aiSpecOut.to(device_rs), windowStart, windowEnd)
                            pitch.write(pitchOffset.to(device_rs), windowStart, windowEnd)

                            if (j + 1 == len(internalStatusControl.ai) or internalInputs.phonemes[j + 1] in ("pau", "_autopause")) == False:
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
                                    voicebank.phonemeDict.fetch(internalInputs.phonemes[j], True)[0].embedding,
                                    voicebank.phonemeDict.fetch(internalInputs.phonemes[j + 1], True)[0].embedding,
                                    expression,
                                    nextExpression,
                                    internalInputs.borders[3 * j + 5] - internalInputs.borders[3 * j + 3] + 1,
                                    internalInputs.pitch[internalInputs.borders[3 * j + 3] - 1:internalInputs.borders[3 * j + 5]],
                                    internalInputs.borders[3 * j + 4] - internalInputs.borders[3 * j + 3]
                                )
                                aiBalance = internalInputs.aiBalance[internalInputs.borders[3 * j + 3] - 1:internalInputs.borders[3 * j + 5]].unsqueeze(1).to(device = device_rs)
                                aiSpec = (0.5 - 0.5 * aiBalance) * aiSpec[0].to(device = device_rs) + (0.5 + 0.5 * aiBalance) * aiSpec[1].to(device = device_rs)
                                pitchOffset = currentPitch[internalInputs.borders[3 * j + 3] - internalInputs.borders[3 * j + 5]:] + nextPitch[:internalInputs.borders[3 * j + 5] - internalInputs.borders[3 * j + 3]]
                                aiSpecOut = pitchAdjust(aiSpec, j, internalInputs.borders[3 * j + 3], internalInputs.borders[3 * j + 5], internalInputs, voicebank, pitchOffset)
                                spectrum.write(aiSpecOut.to(device_rs), internalInputs.borders[3 * j + 3] - 1, internalInputs.borders[3 * j + 5])
                                pitch.write(pitchOffset.to(device_rs), internalInputs.borders[3 * j + 3], internalInputs.borders[3 * j + 5])
                            
                            #TODO: implement crfai skipping if transition was already calculated in the previous frame
                            
                            #istft of voiced excitation + pitch shift

                            remoteConnection.put(StatusChange(i, j, 2))

                            #apply pitch shift to spectrum
                            logging.info("applying partial pitch shift to spectrum of sample " + str(j) + ", sequence " + str(i))
                            if internalInputs.phonemes[j] not in ("_autopause", "pau"):
                                internalStatusControl.ai[j] = 0
                            internalStatusControl.rs[j] = 1
                            remoteConnection.put(StatusChange(i, j, 3))

                    #final rendering and istft of pause-to-pause segment
                    if (j > 0) & (interOutput or (j - 1 == lastPoint)):
                        logging.info("performing final rendering up to sample " + str(j - 1) + ", sequence " + str(i))
                        if aiActive:
                            startPoint = internalInputs.borders[3 * firstPoint]
                            endPoint = internalInputs.borders[3 * lastPoint + 5]
                            
                            if internalInputs.useBreathiness:
                                breathiness = internalInputs.breathiness[startPoint:endPoint].to(device = device_rs).to(torch.float32).contiguous()
                            else:
                                breathiness = torch.zeros([endPoint - startPoint,], device = device_rs).contiguous()
                            
                            steadiness = torch.pow(1. - internalInputs.steadiness[startPoint:endPoint], 2)
                            pitchOffset = processedPitch.read(startPoint, endPoint)
                            pitchOffset = internalInputs.pitch[startPoint:endPoint] + pitchOffset * steadiness
                            
                            specharm = processedSpectrum.read(startPoint, endPoint).clone().contiguous()
                            specharm_ptr = ctypes.cast(specharm.data_ptr(), ctypes.POINTER(ctypes.c_float))
                            breathiness_ptr = ctypes.cast(breathiness.data_ptr(), ctypes.POINTER(ctypes.c_float))
                            esper.applyBreathiness(specharm_ptr, breathiness_ptr, endPoint - startPoint, global_consts.config)
                            output = finalRender(specharm, pitchOffset, endPoint - startPoint, device_rs)
                            
                            remoteConnection.put(StatusChange(i, startPoint*global_consts.batchSize, output, "updateAudio"))
                        remoteConnection.put(StatusChange(i, j - 1, 5))
                    if j > 0 and (j == len(internalStatusControl.ai) or internalInputs.phonemes[j] in ("pau", "_autopause")):
                        aiActive = False
                        #TODO: reset recurrent AI Tensors

                    if (j > 0) & (interOutput == False):
                        remoteConnection.put(StatusChange(i, j - 1, 5))

                    #update cache with new data, if cache is used
                    if settings["cachingmode"] == "best rendering speed":
                        spectrumCache[i] = spectrum
                        processedSpectrumCache[i] = processedSpectrum
                        pitchCache[i] = pitch
                        processedPitchCache[i] = processedPitch

                    #update or wait as required
                    try:
                        c = connection.get_nowait()
                    except:
                        c = False
                    if c:
                        updateFromMain(c)
                        break
                else:
                    continue
                break
            else:
                logging.info("rendering iteration finished, waiting for new data...")
                print("rendering iteration finished, waiting for new data...")
                c = connection.get()
                if updateFromMain(c):
                    break
                
        except Exception as exc:
            print_exc()
            remoteConnection.put(StatusChange(None, None, exc, "error"))
            break
