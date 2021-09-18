from Backend.DataHandler.VocalSegment import VocalSegment
import torch
import copy
import global_consts
import Resampler.Resamplers as rs
def renderProcess(statusControl, inputList, outputList):
    while True:
        for i in len(statusControl):
            internalStatusControl = copy.copy(statusControl[i])
            internalInputs = copy.copy(inputList[i])
            internalOutputs = copy.copy(outputList[i])

            length = internalStatusControl.size()[0]

            spectrum = torch.zeros((length, global_consts.halfTripleBatchSize + 1))
            processedSpectrum = torch.zeros((length, global_consts.halfTripleBatchSize + 1))
            excitation = torch.zeros(length * global_consts.batchSize)
            voicedExcitation = torch.zeros(length * global_consts.batchSize)
            
            aiActive = False
            #reset recurrent AI Tensors
            for j in len(internalStatusControl.ai):
                if j > 0:
                    if (aiActive == False) & internalStatusControl.ai[j - 1] == 1:
                        aiActive = True

                    #execute AI code
                    processedSpectrum[:] = spectrum[:]
                    #execute AI code
                
                previousSpectrum = currentSpectrum
                previousExcitation = currentExcitation
                previousVoicedExcitation = currentVoicedExcitation
                currentSpectrum = nextSpectrum
                currentExcitation = nextExcitation
                currentVoicedExcitation = nextVoicedExcitation
                nextSpectrum = None
                nextExcitation = None
                nextVoicedExcitation = None
                if internalStatusControl.rs[j] == 1:
                    if (internalInputs.startCaps[j] == False) and (previousSpectrum == None):
                        section = VocalSegment()
                        previousSpectrum = rs.getSpectrum(section)
                        previousExcitation = rs.getExcitation(section)
                        previousVoicedExcitation = rs.getVoicedExcitation(section)
                    if currentSpectrum == None:
                        section = VocalSegment()
                        currentSpectrum = rs.getSpectrum(section)
                        currentExcitation = rs.getExcitation(section)
                        currentVoicedExcitation = rs.getVoicedExcitation(section)
                    if (internalInputs.startCaps[j] == False) and (previousSpectrum == None):
                        section = VocalSegment()
                        nextSpectrum = rs.getSpectrum(section)
                        nextExcitation = rs.getExcitation(section)
                        nextVoicedExcitation = rs.getVoicedExcitation(section)

                #resampler
                spectrum[:]
                excitation[:]
                voicedExcitation[:]

                if j > 0:
                    if aiActive:
                        pass
                        #synthesizer

                if (aiActive == True) & internalInputs.endCaps[j - 1] == True:
                        aiActive = False
                        #reset recurrent AI Tensors