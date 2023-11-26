# Copyright 2022, 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from copy import copy

from MiddleLayer.DataHandlers import Note, NoteContext
import global_consts

def calculateBorders(note:Note, context:NoteContext) -> None:
    phonemes = copy(note.phonemes)
        #handle delegate logic
    phonemes = [i for i in phonemes if i in note.track.phonemeLengths.keys()]
    if context.preutterance:
        phonemes.pop(0)
    phonemeLengths = [note.track.phonemeLengths[i] for i in phonemes]
    availableSpace = context.end - context.start
    reservedSpace = sum([i for i in phonemeLengths if i])
    numDropinPhonemes = sum([1 for i in phonemeLengths if not i])
    if len(phonemes) > 0 and availableSpace >= reservedSpace + numDropinPhonemes * global_consts.refPhonemeLength:
        dropinLength = (availableSpace - reservedSpace) / numDropinPhonemes
    else:
        dropinLength = global_consts.refPhonemeLength
    phonemeLengths = [i if i else dropinLength for i in phonemeLengths]
    if len(phonemes) > 0:
        compression = availableSpace / sum(phonemeLengths)
        phonemeLengths = [i * compression for i in phonemeLengths]
    mainBorders = [context.start,]
    for i in phonemeLengths:
        mainBorders.append(mainBorders[-1] + i)
    leadingBorders = []
    for i in range(1, len(mainBorders)):
        leadingBorders.append(mainBorders[i] - min(global_consts.refTransitionLength, (mainBorders[i] - mainBorders[i - 1]) * global_consts.refTransitionFrac))
    trailingBorders = []
    for i in range(len(mainBorders) - 1):
        trailingBorders.append(mainBorders[i] + min(global_consts.refTransitionLength, (mainBorders[i + 1] - mainBorders[i]) * global_consts.refTransitionFrac))
    if context.preutterance:
        mainBorders.insert(0, context.start - context.preutterance)
        leadingBorders.insert(0, context.start - min(global_consts.refTransitionLength, context.preutterance * global_consts.refTransitionFrac))
        trailingBorders.insert(0, context.start - context.preutterance + min(global_consts.refTransitionLength, context.preutterance * global_consts.refTransitionFrac))
    if context.trailingAutopause:
        mainBorders.append(context.trailingAutopause)
        leadingBorders.append(context.trailingAutopause - min(global_consts.refTransitionLength, (context.trailingAutopause - context.end) * global_consts.refTransitionFrac))
        trailingBorders.append(context.end + min(global_consts.refTransitionLength, (context.trailingAutopause - context.end) * global_consts.refTransitionFrac))
    borders = []
    for i in range(len(mainBorders) - 1):
        borders.append(mainBorders[i])
        borders.append(trailingBorders[i])
        borders.append(leadingBorders[i])
    note.borders = borders
    print("borders:", note.borders, context.start, context.end)
    if note.track.notes.index(note) == 0:
        if len(note.phonemes) > 0:
            note.track.borders.wrappingBorders[0] = max(note.borders[0] - global_consts.refTransitionLength, 0)
        else:
            note.track.borders.wrappingBorders[0] = max(note.xPos - global_consts.refTransitionLength, 0)
    if note.track.notes.index(note) == len(note.track.notes) - 1:
        note.track.borders.wrappingBorders[1] = note.xPos + note.length
        note.track.borders.wrappingBorders[2] = note.xPos + note.length + global_consts.refTransitionLength
