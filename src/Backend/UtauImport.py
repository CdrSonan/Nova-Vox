#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from Backend.DataHandler.UtauSample import UtauSample
from os import path
import csv
from copy import copy
from global_consts import controlPhonemes, consonantEndOffset

def createKey(phoneme:str, expression:str):
    if expression == "":
        return phoneme
    else:
        return "_".join([phoneme, expression])

def fetchSamples(filename:str, properties:list, otoPath:str, prefix:str, postfix:str, expr:str, phonFile:str, convFile:str = None, forceJIS:bool = True) -> list:
    """Function for fetching Nova-Vox samples based on a line from an oto.ini file and phoneme list.

    Parameters:
        filename: The name of the referenced .wav file / the first item in the oto.ini line

        properties: The rest of the oto.ini line in list format

        otoPath: The filepath to the oto.ini file. Used for finding the .wav files in the same directory.

        prefix: Prefix to strip from each oto.ini entry (if possible) before converting to Nova-Vox samples

        postfix: Postfix to strip from each oto.ini entry (if possible) before converting to Nova-Vox samples

        phonFile: path to the file specifying a list of phonemes the Voicebank should have in Nova-Vox, and their corresponding phoneme types

        convFile: optional file used for mapping arbitrary parts of UTAU sample names to sequences of phonemes specified in the phonFile. Used for converting Hiragana to Romaji.

        forceJIS: Boolean Flag indicating whether the oto.ini file should be decoded using the shift-JIS codec, instead of the system locale default codec.

    Returns:
        List of UtauSample objects, each representing a transition or phoneme sample from the .wav file

    This function always adds the Nova-Vox breath phonemes and reserved control phonemes to the list specified in phonFile. When parsing the oto.ini file, all samples containing
    a reserved control phoneme are discarded. (Unless they are remapped to a different key in the convFile)"""

    if forceJIS:
        encoding = "shift_jis"
    else:
        encoding = None
    phonemes = []
    types = []
    reader = csv.reader(open(phonFile, encoding = encoding), delimiter = " ")
    for row in reader:
        phonemes.append(row[0])
        types.append(row[1])
    for i in controlPhonemes:
        phonemes.append(i[0])
        types.append(i[1])

    delimiters = [" ", "_", "+"]
    alias = properties[0]
    if alias.startswith(prefix) and prefix != "":
        alias = alias[len(prefix):]
    if alias.endswith(postfix) and postfix != "":
        alias = alias[0:-len(postfix)]
    offset = max(float(properties[1]), 0)
    fixed = max(float(properties[2]), 0)
    blank = float(properties[3])
    preuttr = max(float(properties[4]), 0)
    overlap = max(float(properties[5]), 0)
    
    if alias == "":
        alias = path.splitext(path.split(filename)[1])[0]
    aliasCopy = copy(alias)
    if convFile != None:
        reader = csv.reader(open(convFile, encoding = "utf_8"), delimiter = ",")
        for row in reader:
            aliasCopy = aliasCopy.replace(row[0], row[1])
    #obtain sequence of phonemes and of phoneme types for the current oto.ini line
    sequence = []
    typeSequence = []
    while len(aliasCopy) > 0:
        if aliasCopy[0] == "-":
            aliasCopy = aliasCopy[1:]
        if aliasCopy[0] in delimiters:
            aliasCopy = aliasCopy[1:]
        for i in range(len(phonemes)):
            p = phonemes[i]
            f = aliasCopy.find(p)
            if f == 0:
                #handle numeric postfixes for phoneme variants
                if len(aliasCopy) > len(p):
                    if aliasCopy[len(p)] in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                        p += aliasCopy[len(p)]
                sequence.append(p)
                typeSequence.append(types[i])
                aliasCopy = aliasCopy[len(p):]
                break
        else:
            raise LookupError("filename/alias " + alias + "/" + aliasCopy + " contains one or several phonemes not in the specified phoneme list")
    #fetch samples based on the obtained sequences
    output = []
    filepath = path.join(otoPath, filename)
    intermediate = offset + preuttr + min((fixed - preuttr) / 2, consonantEndOffset)
    output.append(UtauSample(filepath, 2, None, 0, None, 0, 0, 0, 0, 0))
    if len(sequence) == 1:
        sharedKey = createKey(sequence[0], expr)
        if typeSequence[0] == "V":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False, 0x0000ff00)
            embedPart = 0x0000ff00
        elif typeSequence[0] == "C":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, False, 0x00000000)
            embedPart = 0x00000000
        elif typeSequence[0] == "T":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, True, 0x0000ffff)
            embedPart = 0x0000ffff
        elif typeSequence[0] == "P":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, True, 0x000000ff)
            embedPart = 0x000000ff
        output.append(sample)
        sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap, embedding = (0xf000ff00, embedPart))
        output.append(sample)
    elif len(sequence) == 2:
        sharedKey = createKey(sequence[1], expr)
        if typeSequence[1] == "V":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False, 0x0000ff00)
            embedPart = 0x0000ff00
        elif typeSequence[1] == "C":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, False, 0x00000000)
            embedPart = 0x00000000
        elif typeSequence[1] == "T":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, True, 0x0000ffff)
            embedPart = 0x0000ffff
        elif typeSequence[1] == "P":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, True, 0x000000ff)
            embedPart = 0x000000ff
        output.append(sample)
        sharedKey = createKey(sequence[0], expr)
        if typeSequence[0] == "V":
            sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap, embedding = (0x0000ff00, embedPart))
            output.append(sample)
        elif typeSequence[0] == "C":
            sample = UtauSample(filepath, 0, sharedKey, offset, intermediate, offset, fixed, blank, preuttr, overlap, False, False, 0x00000000)
            output.append(sample)
            sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, embedding = (0x00000000, embedPart))
            output.append(sample)
        elif typeSequence[0] == "T":
            sample = UtauSample(filepath, 0, sharedKey, offset, intermediate, offset, fixed, blank, preuttr, overlap, True, True, 0x0000ffff)
            output.append(sample)
            sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, embedding = (0x0000ffff, embedPart))
            output.append(sample)
        elif typeSequence[0] == "P":
            sample = UtauSample(filepath, 0, sharedKey, offset, intermediate, offset, fixed, blank, preuttr, overlap, False, True, 0x000000ff)
            output.append(sample)
            sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, embedding = (0x000000ff, embedPart))
            output.append(sample)
    elif len(sequence) == 3:
        sharedKey = createKey(sequence[1], expr)
        if typeSequence[1] == "V":
            sample = UtauSample(filepath, 0, sharedKey, offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap, True, False, 0x0000ff00)
            embedPart = 0x0000ff00
        elif typeSequence[1] == "C":
            sample = UtauSample(filepath, 0, sharedKey, offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap, False, False, 0x00000000)
            embedPart = 0x00000000
        elif typeSequence[1] == "T":
            sample = UtauSample(filepath, 0, sharedKey, offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap, True, True, 0x0000ffff)
            embedPart = 0x0000ffff
        elif typeSequence[1] == "P":
            sample = UtauSample(filepath, 0, sharedKey, offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap, False, True, 0x000000ff)
            embedPart = 0x000000ff
        output.append(sample)
        sharedKey = createKey(sequence[2], expr)
        if typeSequence[2] == "V":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False, 0x0000ff00)
            embedPart2 = 0x0000ff00
        elif typeSequence[2] == "C":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, False, 0x00000000)
            embedPart2 = 0x00000000
        elif typeSequence[2] == "T":
            sample = UtauSample(filepath, 0, sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, True, 0x0000ffff)
            embedPart2 = 0x0000ffff
        elif typeSequence[2] == "P":
            sample = UtauSample(filepath, 0,sharedKey, offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, True, 0x000000ff)
            embedPart2 = 0x000000ff
        output.append(sample)
        sample = UtauSample(filepath, 1, None, offset, offset + (overlap + preuttr) / 2, offset, fixed, blank, preuttr, overlap, embedding = (0xf000ff00, embedPart))
        output.append(sample)
        sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, embedding = (embedPart, embedPart2))
        output.append(sample)
    return output
