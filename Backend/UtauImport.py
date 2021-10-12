from Backend.DataHandler.UtauSample import UtauSample
from os import path

def replaceKana(input):
    print("in", input)
    input = input.replace("きゃ", "kya")
    input = input.replace("きゅ", "kyu")
    input = input.replace("きょ", "kyo")
    input = input.replace("ぎゃ", "gya")
    input = input.replace("ぎゅ", "gyu")
    input = input.replace("ぎょ", "gyo")
    input = input.replace("しゃ", "shya")
    input = input.replace("しゅ", "shyu")
    input = input.replace("しょ", "shyo")
    input = input.replace("じゃ", "jya")
    input = input.replace("じゅ", "jyu")
    input = input.replace("じょ", "jyo")
    input = input.replace("とぅ", "to")
    input = input.replace("あ", "a")
    input = input.replace("ぁ", "a")
    input = input.replace("い", "i")
    input = input.replace("ぃ", "i")
    input = input.replace("う", "u")
    input = input.replace("え", "e")
    input = input.replace("ぇ", "e")
    input = input.replace("お", "o")
    input = input.replace("ぉ", "o")
    input = input.replace("か", "ka")
    input = input.replace("き", "ki")
    input = input.replace("く", "ku")
    input = input.replace("け", "ke")
    input = input.replace("こ", "ko")
    input = input.replace("が", "ga")
    input = input.replace("ぎ", "gi")
    input = input.replace("ぐ", "gu")
    input = input.replace("げ", "ge")
    input = input.replace("ご", "go")
    input = input.replace("さ", "sa")
    input = input.replace("し", "shi")
    input = input.replace("す", "su")
    input = input.replace("せ", "se")
    input = input.replace("そ", "so")
    input = input.replace("ざ", "za")
    input = input.replace("じ", "ji")
    input = input.replace("ず", "zu")
    input = input.replace("ぜ", "ze")
    input = input.replace("ぞ", "zo")
    input = input.replace("た", "ta")
    input = input.replace("ち", "chi")
    input = input.replace("つ", "tsu")
    input = input.replace("て", "te")
    input = input.replace("と", "to")
    input = input.replace("だ", "da")
    input = input.replace("ぢ", "ji")
    input = input.replace("づ", "zu")
    input = input.replace("で", "de")
    input = input.replace("ど", "do")
    input = input.replace("な", "na")
    input = input.replace("に", "ni")
    input = input.replace("ぬ", "nu")
    input = input.replace("ね", "ne")
    input = input.replace("の", "no")
    input = input.replace("は", "ha")
    input = input.replace("ひ", "hi")
    input = input.replace("ふ", "fu")
    input = input.replace("へ", "he")
    input = input.replace("ほ", "ho")
    input = input.replace("ば", "ba")
    input = input.replace("び", "bi")
    input = input.replace("ぶ", "bu")
    input = input.replace("べ", "be")
    input = input.replace("ぼ", "bo")
    input = input.replace("ぱ", "pa")
    input = input.replace("ぴ", "pi")
    input = input.replace("ぷ", "pu")
    input = input.replace("ぺ", "pe")
    input = input.replace("ぽ", "po")
    input = input.replace("ま", "ma")
    input = input.replace("み", "mi")
    input = input.replace("む", "mu")
    input = input.replace("め", "me")
    input = input.replace("も", "mo")
    input = input.replace("や", "ya")
    input = input.replace("ゆ", "yu")
    input = input.replace("よ", "yo")
    input = input.replace("ら", "ra")
    input = input.replace("り", "ri")
    input = input.replace("る", "ru")
    input = input.replace("れ", "re")
    input = input.replace("ろ", "ro")
    input = input.replace("わ", "wa")
    input = input.replace("を", "wo")
    input = input.replace("ん", "n")
    input = input.replace("ヴ", "wu")
    print("out", input)
    return input

def fetchSamples(filename, properties, phonemes, types, otoPath):
    alias = properties[0]
    offset = float(properties[1])
    fixed = float(properties[2])
    blank = float(properties[3])
    preuttr = float(properties[4])
    overlap = float(properties[5])
    
    if alias == "":
        alias = path.splitext(path.split(filename)[1])[0]
    aliasCopy = replaceKana(alias)
    sequence = []
    typeSequence = []
    delimiters = [" ", "_", "+"]
    while len(aliasCopy) > 0:
        if aliasCopy[0] in delimiters:
            aliasCopy = aliasCopy[1:]
        for i in range(len(phonemes)):
            p = phonemes[i]
            f = aliasCopy.find(p)
            if f == 0:
                sequence.append(p)
                typeSequence.append(types[i])
                aliasCopy = aliasCopy[len(p):]
                break
        else:
            raise LookupError("filename/alias " + alias + "/" + aliasCopy + " contains one or several phonemes not in the specified phoneme list")
        
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

