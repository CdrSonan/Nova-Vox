from Backend.DataHandler.UtauSample import UtauSample
from os import path
from math import max

def replaceKana(input):
    input.replace() #todo
    return input

def fetchSamples(filename, properties, phonemes, types, otoPath):
    alias = properties[0]
    offset = properties[1]
    fixed = properties[2]
    blank = properties[3]
    preuttr = properties[4]
    overlap = properties[5]
    
    if alias == "":
        alias = path.splitext(path.split(filename)[1])[0]
        aliasCopy = replaceKana(alias)
    sequence = []
    typeSequence = []
    delimiters = [" ", "_"]
    while len(aliasCopy) > 0:
        for i in range(len(phonemes)):
            p = phonemes[i]
            f = aliasCopy.find(p)
            if f == 0:
                sequence.append(p)
                typeSequence.append(types[i])
                break
        else:
            raise LookupError("filename/alias " + alias + " contains one or several phonemes not in the specified phoneme list")
        if aliasCopy[0] in delimiters:
            aliasCopy = aliasCopy[1:]
        
    output = []

    filepath = path.join(otoPath, filename)

    intermediate = max(-2 * (fixed - preuttr) + offset + fixed, (fixed / 2) + offset)

    if len(sequence) == 1:
        if typeSequence[0] == "V":
            sample = UtauSample(filepath, 0, sequence[0], offset + fixed, None, offset, fixed, blank, preuttr, overlap)
            sample.end -= blank
            output.append(sample)
        elif typeSequence[0] == "C":
            sample = UtauSample(filepath, 0, sequence[0], offset, None, offset, fixed, blank, preuttr, overlap)
            sample.end -= blank
            output.append(sample)
    elif len(sequence) == 2:
        if typeSequence[0] == "V":
            if typeSequence[1] == "V":
                #sample = UtauSample(filepath, 0, sequence[0], offset + fixed, None, offset, fixed, blank, preuttr, overlap)
                #sample.end -= blank
                #output.append(sample)
                pass
            elif typeSequence[1] == "C":
                sample = UtauSample(filepath, 0, sequence[0], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap)
                sample.end -= blank
                output.append(sample)
        elif typeSequence[0] == "C":
            if typeSequence[1] == "V":
                sample = UtauSample(filepath, 0, sequence[0], offset, intermediate, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap)
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "C":
                #sample = UtauSample(filepath, 0, sequence[0], offset, None, offset, fixed, blank, preuttr, overlap)
                #sample.end -= blank
                #output.append(sample)
                pass
    elif len(sequence) == 3:
        if typeSequence[0] == "V":
            sample = UtauSample(filepath, 1, None, offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 0, sequence[1], intermediate, offset + overlap, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap)
            sample.end -= blank
            output.append(sample)
        elif typeSequence[0] == "C":
            sample = UtauSample(filepath, 0, sequence[1], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap)
            output.append(sample)
            sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap)
            sample.end -= blank
            output.append(sample)

    return output

