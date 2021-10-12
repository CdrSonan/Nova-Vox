from Backend.DataHandler.UtauSample import UtauSample
from os import path

def replaceKana(input):
    input.replace("あ", "a")
    input.replace("い", "i")
    input.replace("う", "u")
    input.replace("え", "e")
    input.replace("お", "o")
    input.replace("か", "ka")
    input.replace("き", "ki")
    input.replace("く", "ku")
    input.replace("け", "ke")
    input.replace("こ", "ko")
    input.replace("が", "ga")
    input.replace("ぎ", "gi")
    input.replace("ぐ", "gu")
    input.replace("げ", "ge")
    input.replace("ご", "go")
    input.replace("さ", "sa")
    input.replace("し", "shi")
    input.replace("す", "su")
    input.replace("せ", "se")
    input.replace("そ", "so")
    input.replace("ざ", "za")
    input.replace("じ", "ji")
    input.replace("ず", "zu")
    input.replace("ぜ", "ze")
    input.replace("ぞ", "zo")
    input.replace("た", "ta")
    input.replace("ち", "chi")
    input.replace("つ", "tsu")
    input.replace("て", "te")
    input.replace("と", "to")
    input.replace("だ", "da")
    input.replace("ぢ", "ji")
    input.replace("づ", "zu")
    input.replace("で", "de")
    input.replace("ど", "do")
    input.replace("な", "na")
    input.replace("に", "ni")
    input.replace("ぬ", "nu")
    input.replace("ね", "ne")
    input.replace("の", "no")
    input.replace("は", "ha")
    input.replace("ひ", "hi")
    input.replace("ふ", "fu")
    input.replace("へ", "he")
    input.replace("ほ", "ho")
    input.replace("ば", "ba")
    input.replace("び", "bi")
    input.replace("ぶ", "bu")
    input.replace("べ", "be")
    input.replace("ぼ", "bo")
    input.replace("ぱ", "pa")
    input.replace("ぴ", "pi")
    input.replace("ぷ", "pu")
    input.replace("ぺ", "pe")
    input.replace("ぽ", "po")
    input.replace("ま", "ma")
    input.replace("み", "mi")
    input.replace("む", "mu")
    input.replace("め", "me")
    input.replace("も", "mo")
    input.replace("や", "ya")
    input.replace("ゆ", "yu")
    input.replace("よ", "yo")
    input.replace("ら", "ra")
    input.replace("り", "ri")
    input.replace("る", "ru")
    input.replace("れ", "re")
    input.replace("ろ", "ro")
    input.replace("わ", "wa")
    input.replace("を", "wo")
    input.replace("ん", "n")
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

