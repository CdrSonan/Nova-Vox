from Backend.DataHandler.UtauSample import UtauSample
from os import path
import csv
from copy import copy
from global_consts import controlPhonemes

def fetchSamples(filename:str, properties:list, otoPath:str, prefix:str, postfix:str, phonFile:str, convFile:str = None) -> list:
    """Function for fetching Nova-Vox samples based on a line from an oto.ini file and phoneme list.

    Parameters:
        filename: The name of the referenced .wav file / the first item in the oto.ini line

        properties: The rest of the oto.ini line in list format

        otoPath: The filepath to the oto.ini file. Used for finding the .wav files in the same directory.

        prefix: Prefix to strip from each oto.ini entry (if possible) before converting to Nova-Vox samples

        postfix: Postfix to strip from each oto.ini entry (if possible) before converting to Nova-Vox samples

        phonFile: path to the file specifying a list of phonemes the Voicebank should have in Nova-Vox, and their corresponding phoneme types

        convFile: optional file used for mapping arbitrary parts of UTAU sample names to sequences of phonemes specified in the phonFile. Used for converting Hiragana to Romaji.

    Returns:
        List of UtauSample objects, each representing a transition or phoneme sample from the .wav file

    This function always adds the Nova-Vox breath phonemes and reserved control phonemes to the list specified in phonFile. When parsing the oto.ini file, all samples containing
    a reserved control phoneme are discarded. (Unless they are remapped to a different key in the convFile)"""


    phonemes = []
    types = []
    reader = csv.reader(open(phonFile), delimiter = " ")
    for row in reader:
        phonemes.append(row[0])
        types.append(row[1])
    for i in controlPhonemes:
        phonemes.append(i[0])
        types.append(i[1])

    delimiters = [" ", "_", "+"]
    alias = properties[0]
    if alias.startswith(prefix):
        alias = alias[len(prefix):]
    if alias.endswith(postfix):
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
        reader = csv.reader(open(convFile), delimiter = ",")
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
    #TODO: adjust to new phoneme types
    intermediate = max(-2 * (fixed - preuttr) + offset + fixed, (fixed / 2) + offset)
    if len(sequence) == 1:
        if typeSequence[0] == "V":
            sample = UtauSample(filepath, 0, sequence[0], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
            sample.end -= blank
            output.append(sample)
        elif typeSequence[0] == "C":
            sample = UtauSample(filepath, 0, sequence[0], offset, None, offset, fixed, blank, preuttr, overlap, False, True)
            sample.end -= blank
            output.append(sample)
        elif typeSequence[0] == "c":
            sample = UtauSample(filepath, 0, sequence[0], offset, None, offset, fixed, blank, preuttr, overlap, False, False)
            sample.end -= blank
            output.append(sample)
    elif len(sequence) == 2:
        if typeSequence[0] == "V":
            if typeSequence[1] == "V":
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "C":
                sample = UtauSample(filepath, 0, sequence[0], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, True, False)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, True)
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "c":
                sample = UtauSample(filepath, 0, sequence[0], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, True, False)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, False)
                sample.end -= blank
                output.append(sample)
        elif typeSequence[0] == "C" :
            if typeSequence[1] == "V":
                sample = UtauSample(filepath, 0, sequence[0], offset, intermediate, offset, fixed, blank, preuttr, overlap, False, True)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "C" or typeSequence[1] == "c":
                print("skipped CC sample")
        elif typeSequence[0] == "c":
            if typeSequence[1] == "V":
                sample = UtauSample(filepath, 0, sequence[0], offset, intermediate, offset, fixed, blank, preuttr, overlap, False, False)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
                sample.end -= blank
                output.append(sample)
    elif len(sequence) == 3:
        if typeSequence[0] == "V":
            if typeSequence[1] == "C":
                sample = UtauSample(filepath, 1, None, offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], intermediate, offset + overlap, offset, fixed, blank, preuttr, overlap, False, True)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "c":
                sample = UtauSample(filepath, 1, None, offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], intermediate, offset + overlap, offset, fixed, blank, preuttr, overlap, False, False)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, True, False)
                sample.end -= blank
                output.append(sample)
        elif typeSequence[0] == "C" or typeSequence[0] == "c":
            if typeSequence[2] == "C":
                sample = UtauSample(filepath, 0, sequence[1], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, True, False)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, True)
                sample.end -= blank
                output.append(sample)
            elif typeSequence[2] == "c":
                sample = UtauSample(filepath, 0, sequence[1], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, True, False)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, False, False)
                sample.end -= blank
                output.append(sample)
    return output
