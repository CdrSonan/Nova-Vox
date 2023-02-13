# Copyright 2022, 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from MiddleLayer.DataHandlers import Track, Note
import global_consts

def rescaleFromReference(note:Note, reference:tuple, borders:list) -> list:
    borders = [i - reference[0] for i in borders]
    scaling = note.length / reference[1]
    borders = [i * scaling for i in borders]
    borders = [i + note.xPos for i in borders]
    return borders

def recalculateBorders(index:int, track:Track, referenceLength:int = None) -> tuple:
    phonemes = track.phonemes[track.notes[index].phonemeStart:track.notes[index].phonemeEnd]
    if len(phonemes) == 0:
        return (track.notes[index].phonemeStart, track.notes[index].phonemeEnd)
    print("before", track.borders)
    if phonemes[0] == "_autopause":
        phonemes = phonemes[1:]
        autopause = track.notes[index].xPos - track.notes[index - 1].xPos - track.notes[index - 1].length
    else:
        autopause = False
    phonemeLengths = [track.phonemeLengths[i] if i in track.phonemeLengths else None for i in phonemes]
    if len(phonemeLengths) > 0 and phonemeLengths[0]:
        preutterance = phonemeLengths[0]
        phonemes = phonemes[1:]
        phonemeLengths = phonemeLengths[1:]
    else:
        preutterance = None
    length = track.notes[index].length
    if index < len(track.notes) - 1:
        length = min(length, track.notes[index + 1].xPos - track.notes[index].xPos)
    if sum([i for i in phonemeLengths if i]) >= length:
        compression = True
    else:
        compression = False
        
    effectiveStart = track.notes[index].phonemeStart
    if autopause:
        effectiveStart += 1
    if preutterance:
        effectiveStart += 1

    if preutterance:
        if autopause:
            preutterance = min(preutterance, (1. - global_consts.refTransitionFrac) * autopause)
        transitionLength = min(preutterance * global_consts.refTransitionFrac, global_consts.refTransitionLength)
        preutterance = [transitionLength, preutterance - 2 * transitionLength, transitionLength]

    if index == 0:
        segmentLengths = [global_consts.refTransitionLength,]
    else:
        segmentLengths = []
    for i in phonemeLengths:
        if i:
            transitionLength = min(i * global_consts.refTransitionFrac, global_consts.refTransitionLength)
            segmentLengths.extend([transitionLength, i - 2 * transitionLength, transitionLength])
        else:
            segmentLengths.extend([global_consts.refTransitionLength, None, global_consts.refTransitionLength])
    #if index == len(track.notes) - 1:
        #segmentLengths.append(global_consts.refTransitionLength)
    segmentLengths.append(global_consts.refTransitionLength)

    """if referenceLength and not compression:
        if index == 0:
            offset = track.borders[0] - track.notes[index].xPos
        else:
            offset = track.borders[(track.notes[index].phonemeStart) * 3 + 1] - track.notes[index].xPos
        if preutterance:
            borders = [track.borders[track.notes[index].phonemeStart * 3 + 1],
                       track.borders[track.notes[index].phonemeStart * 3 + 2],
                       track.borders[track.notes[index].phonemeStart * 3 + 3],
                       track.borders[track.notes[index].phonemeStart * 3 + 4]]
            borders = rescaleFromReference(track.notes[index], (track.notes[index].xPos, referenceLength), borders)
            preutterance = [borders[1] - borders[0],
                            borders[2] - borders[1],
                            borders[3] - borders[2]]
        for i, phonemeLength in enumerate(phonemeLengths):
            borders = [track.borders[(effectiveStart + i) * 3 + 1],
                       track.borders[(effectiveStart + i) * 3 + 2],
                       track.borders[(effectiveStart + i) * 3 + 3],
                       track.borders[(effectiveStart + i) * 3 + 4]]
            borders = rescaleFromReference(track.notes[index], (track.notes[index].xPos, referenceLength), borders)
            if phonemeLength:
                segmentLengths[3 * i + 2] = borders[2] - borders[1]
            segmentLengths[3 * i + 1] = borders[1] - borders[0]
            segmentLengths[3 * i + 3] = borders[3] - borders[2]
    else:
        offset = 0"""
    offset = 0
    dropinLength = (length - sum([i for i in segmentLengths if i])) / max(sum([1 for i in segmentLengths if not i]), 1)
    dropinLength = max(dropinLength, global_consts.refPhonemeLength)
    segmentLengths = [i if i else dropinLength for i in segmentLengths]
    compression = length / sum(segmentLengths)
    segmentLengths = [i * compression for i in segmentLengths]
    
    if autopause:
        if preutterance:
            preuttrComp = autopause / sum(preutterance)
            if preuttrComp < 1:
                preutterance = [i * preuttrComp for i in preutterance]
        autopause = min(autopause * global_consts.refTransitionFrac, global_consts.refTransitionLength)
    
    startPos = track.notes[index].xPos + offset * compression
    segmentLengths.insert(0, startPos)
    effectiveStart = effectiveStart * 3
    if index > 0:
        effectiveStart += 1
    end = effectiveStart + len(segmentLengths)
    for i in range(1, len(segmentLengths)):
        segmentLengths[i] += segmentLengths[i - 1]
    track.borders[effectiveStart:end] = segmentLengths

    if preutterance:
        preutterance[2] = track.borders[effectiveStart] - preutterance[2]
        preutterance[1] = preutterance[2] - preutterance[1]
        preutterance[0] = preutterance[1] - preutterance[0]
        track.borders[effectiveStart - 3:effectiveStart] = preutterance
        effectiveStart -= 3
    if autopause:
        start = recalculateBorders(index - 1, track)[0]
        track.borders[effectiveStart - 1] = track.borders[effectiveStart] - autopause
        track.borders[effectiveStart - 2] = track.borders[effectiveStart - 3] + autopause
    else:
        start = track.notes[index].phonemeStart
    print("after", track.borders, effectiveStart)
    return start, end
