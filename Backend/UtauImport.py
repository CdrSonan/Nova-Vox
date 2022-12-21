#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from Backend.DataHandler.UtauSample import UtauSample
from os import path
import csv
from copy import copy
from global_consts import controlPhonemes, consonantEndOffset

def fetchSamples(filename:str, properties:list, otoPath:str, prefix:str, postfix:str, phonFile:str, convFile:str = None, forceJIS:bool = True) -> list:
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
                if len(aliasCopy) > len(p):
                    if aliasCopy[len(p)] in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                        p += aliasCopy[len(p)]
                sequence.append(p)
                typeSequence.append(types[i])
                aliasCopy = aliasCopy[len(p):]
                break
        else:
            raise LookupError("filename/alias " + alias + "/" + aliasCopy + " contains one or several phonemes not in the specified phoneme list")
        
    output = []
    filepath = path.join(otoPath, filename)
    intermediate = offset + preuttr + min((fixed - preuttr) / 2, consonantEndOffset)
    output.append(UtauSample(filepath, 2, None, 0, None, 0, 0, 0, 0, 0))
    if len(sequence) == 1:
        sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap)
        output.append(sample)
        if typeSequence[0] == "V":
            sample = UtauSample(filepath, 1, None, offset + fixed, None, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 0, sequence[0], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
        elif typeSequence[0] == "C":
            sample = UtauSample(filepath, 0, sequence[0], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, False)
        elif typeSequence[0] == "T":
            sample = UtauSample(filepath, 0, sequence[0], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, True)
        elif typeSequence[0] == "P":
            sample = UtauSample(filepath, 0, sequence[0], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, True)
        output.append(sample)
    elif len(sequence) == 2:
        if typeSequence[0] == "V":
            sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
        elif typeSequence[0] == "C":
            sample = UtauSample(filepath, 0, sequence[0], offset, intermediate, offset, fixed, blank, preuttr, overlap, False, False)
            output.append(sample)
            sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
        elif typeSequence[0] == "T":
            sample = UtauSample(filepath, 0, sequence[0], offset, intermediate, offset, fixed, blank, preuttr, overlap, True, True)
            output.append(sample)
            sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
        elif typeSequence[0] == "P":
            sample = UtauSample(filepath, 0, sequence[0], offset, intermediate, offset, fixed, blank, preuttr, overlap, False, True)
            output.append(sample)
            sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
        if typeSequence[1] == "V":
            sample = UtauSample(filepath, 1, None, offset + fixed, None, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
        elif typeSequence[1] == "C":
            sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, False)
        elif typeSequence[1] == "T":
            sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, True)
        elif typeSequence[1] == "P":
            sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, True)
        output.append(sample)
    elif len(sequence) == 3:
        sample = UtauSample(filepath, 1, None, offset, offset + (overlap + preuttr) / 2, offset, fixed, blank, preuttr, overlap)
        output.append(sample)
        if typeSequence[1] == "V":
            sample = UtauSample(filepath, 1, None, offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 0, sequence[1], offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap, True, False)
        elif typeSequence[1] == "C":
            sample = UtauSample(filepath, 0, sequence[1], offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap, False, False)
        elif typeSequence[1] == "T":
            sample = UtauSample(filepath, 0, sequence[1], offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap, True, True)
        elif typeSequence[1] == "P":
            sample = UtauSample(filepath, 0, sequence[1], offset + (overlap + preuttr) / 2, intermediate, offset, fixed, blank, preuttr, overlap, False, True)
        output.append(sample)
        sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
        output.append(sample)
        if typeSequence[2] == "V":
            sample = UtauSample(filepath, 1, None, offset + fixed, None, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
        elif typeSequence[2] == "C":
            sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, False)
        elif typeSequence[2] == "T":
            sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, True)
        elif typeSequence[2] == "P":
            sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, True)
        output.append(sample)
    return output
