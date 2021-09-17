import Resampler.Resamplers as rs

def resamplerProcess(statusControl, inputList, firstInterList):

    previousExcitation = None
    previousVoicedExcitation = None
    previousSpectrum = None
    currentExcitation = None
    currentVoicedExcitation = None
    currentSpectrum = None
    nextExcitation = None
    nextVoicedExcitation = None
    nextSpectrum = None

    previousTrack = None
    previousSegment = None

    internalStatusControl = []

    for i in statusControl:
        internalStatusControl.append(dict())
        for j in i.identities:
            internalStatusControl[i][j] = statusControl.resamplerVersions[i]
    while True:
        for i in statusControl:
            for j in i.resamplerVersions.size():
                if internalStatusControl[i][i.identities[j]] < i.resamplerVersions[j]:
                    currentTrack = i
                    currentSegment = j
                    break
            else:
                continue
            break

        if (currentTrack == previousTrack) & (currentSegment == previousSegment +1):
            previousExcitation = currentExcitation
            previousVoicedExcitation = currentVoicedExcitation
            previousSpectrum = currentSpectrum
            currentExcitation = nextExcitation
            currentVoicedExcitation = nextVoicedExcitation
            currentSpectrum = nextSpectrum
            nextExcitation = rs.getExcitation()
            nextVoicedExcitation = rs.getVoicedExcitation()
            nextSpectrum = rs.getSpectrum()
        elif (currentTrack == previousTrack) & (currentSegment == previousSegment +2):
            previousExcitation = nextExcitation
            previousVoicedExcitation = nextVoicedExcitation
            previousSpectrum = nextSpectrum
            currentExcitation = rs.getExcitation()
            currentVoicedExcitation = rs.getVoicedExcitation()
            currentSpectrum = rs.getSpectrum()
            nextExcitation = rs.getExcitation()
            nextVoicedExcitation = rs.getVoicedExcitation()
            nextSpectrum = rs.getSpectrum()
        else:
            previousExcitation = rs.getExcitation()
            previousVoicedExcitation = rs.getVoicedExcitation()
            previousSpectrum = rs.getSpectrum()
            currentExcitation = rs.getExcitation()
            currentVoicedExcitation = rs.getVoicedExcitation()
            currentSpectrum = rs.getSpectrum()
            nextExcitation = rs.getExcitation()
            nextVoicedExcitation = rs.getVoicedExcitation()
            nextSpectrum = rs.getSpectrum()