from Backend.DataHandler.UtauSample import UtauSample
from os import path

def replaceKana(input):
    """Helper function that replaces Hiragana in a string with their Romaji phoneme equivalents. !!!TO BE REPLACED!!!"""

    input = input.replace("きゃ", "kya")
    input = input.replace("きゅ", "kyu")
    input = input.replace("きょ", "kyo")
    input = input.replace("きぇ", "kye")
    input = input.replace("にゃ", "Nya")
    input = input.replace("にゅ", "Nyu")
    input = input.replace("にょ", "Nyo")
    input = input.replace("にぇ", "Nye")
    input = input.replace("みゃ", "mya")
    input = input.replace("みゅ", "myu")
    input = input.replace("みょ", "myo")
    input = input.replace("みぇ", "mye")
    input = input.replace("りゃ", "rya")
    input = input.replace("りゅ", "ryu")
    input = input.replace("りょ", "ryo")
    input = input.replace("りぇ", "rye")
    input = input.replace("びゃ", "bya")
    input = input.replace("びゅ", "byu")
    input = input.replace("びょ", "byo")
    input = input.replace("びぇ", "bye")
    input = input.replace("ぴゃ", "pya")
    input = input.replace("ぴゅ", "pyu")
    input = input.replace("ぴょ", "pyo")
    input = input.replace("ぴぇ", "pye")
    input = input.replace("ぎゃ", "gya")
    input = input.replace("ぎゅ", "gyu")
    input = input.replace("ぎょ", "gyo")
    input = input.replace("ぎぇ", "gye")
    input = input.replace("しゃ", "shya")
    input = input.replace("しゅ", "shyu")
    input = input.replace("しょ", "shyo")
    input = input.replace("しぇ", "shye")
    input = input.replace("じゃ", "jya")
    input = input.replace("じゅ", "jyu")
    input = input.replace("じょ", "jyo")
    input = input.replace("じぇ", "jye")
    input = input.replace("とぅ", "tu")
    input = input.replace("てぃ", "ti")
    input = input.replace("いぇ", "ie")
    input = input.replace("うぃ", "ui")
    input = input.replace("うぉ", "uo")
    input = input.replace("うぇ", "ue")
    input = input.replace("ヴぁ", "wa")
    input = input.replace("ヴぃ", "wi")
    input = input.replace("ヴぇ", "we")
    input = input.replace("ヴぉ", "wo")
    input = input.replace("くぁ", "ka")
    input = input.replace("くぃ", "ki")
    input = input.replace("くぇ", "ke")
    input = input.replace("くぉ", "ko")
    input = input.replace("ぐぁ", "ga")
    input = input.replace("ぐぃ", "gi")
    input = input.replace("ぐぇ", "ge")
    input = input.replace("ぐぉ", "go")
    input = input.replace("すぁ", "sa")
    input = input.replace("すぃ", "si")
    input = input.replace("すぇ", "se")
    input = input.replace("すぉ", "so")
    input = input.replace("ずぁ", "za")
    input = input.replace("ずぃ", "zi")
    input = input.replace("ずぇ", "ze")
    input = input.replace("ずぉ", "zo")
    input = input.replace("でぃ", "di")
    input = input.replace("でゃ", "dya")
    input = input.replace("でょ", "dyo")
    input = input.replace("でゅ", "dyu")
    input = input.replace("てぃ", "ti")
    input = input.replace("てゃ", "tya")
    input = input.replace("てょ", "tyo")
    input = input.replace("てゅ", "tyu")
    input = input.replace("どぅ", "do")
    input = input.replace("ちぇ", "che")
    input = input.replace("ちゃ", "cha")
    input = input.replace("ちょ", "cho")
    input = input.replace("ちゅ", "chu")
    input = input.replace("つぁ", "tsa")
    input = input.replace("つぃ", "tsi")
    input = input.replace("つぇ", "tse")
    input = input.replace("つぉ", "tso")
    input = input.replace("ひぇ", "he")
    input = input.replace("ひゃ", "hya")
    input = input.replace("ひょ", "hyo")
    input = input.replace("ひゅ", "hyu")
    input = input.replace("ふぁ", "fa")
    input = input.replace("ふぃ", "fi")
    input = input.replace("ふぇ", "fe")
    input = input.replace("ふぉ", "fo")

    input = input.replace("あ", "a")
    input = input.replace("い", "i")
    input = input.replace("う", "u")
    input = input.replace("え", "e")
    input = input.replace("お", "o")
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
    input = input.replace("な", "Na")
    input = input.replace("に", "Ni")
    input = input.replace("ぬ", "Nu")
    input = input.replace("ね", "Ne")
    input = input.replace("の", "No")
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
    input = input.replace("吸息", "=")
    input = input.replace("息", "<")
    input = input.replace("吸", ">")
    return input

def fetchSamples(filename, properties, phonemes, types, otoPath):
    """Function for fetching Nova-Vox samples based on a line from an oto.ini file and phoneme list.

    Parameters:
        filename: The name of the referenced .wav file / the first item in the oto.ini line

        properties: THe rest of the oto.ini line in list format

        phonemes: The phoneme list used for classifying samples in list format

        types: The types of the phonemes in the phoneme list, also in list format. Each item can be either C (consonant) or V (vowel).

        otoPath: The filepath to the oto.ini file. Used for finding the .wav files in the same directory.

    Returns: List of UtauSample objects, each representing a transition or phoneme sample from the .wav file
    """

    alias = properties[0]
    offset = max(float(properties[1]), 0)
    fixed = max(float(properties[2]), 0)
    blank = float(properties[3])
    preuttr = max(float(properties[4]), 0)
    overlap = max(float(properties[5]), 0)
    
    if alias == "":
        alias = path.splitext(path.split(filename)[1])[0]
    aliasCopy = replaceKana(alias)
    sequence = []
    typeSequence = []
    delimiters = [" ", "_", "+"]
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

    intermediate = max(-2 * (fixed - preuttr) + offset + fixed, (fixed / 2) + offset)
    if len(sequence) == 1:
        if typeSequence[0] == "V":
            sample = UtauSample(filepath, 0, sequence[0], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "V")
            sample.end -= blank
            output.append(sample)
        elif typeSequence[0] == "C" or typeSequence[0] == "c":
            sample = UtauSample(filepath, 0, sequence[0], offset, None, offset, fixed, blank, preuttr, overlap, "C")
            sample.end -= blank
            output.append(sample)
    elif len(sequence) == 2:
        if typeSequence[0] == "V":
            if typeSequence[1] == "V":
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, "VV")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "V")
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "C":
                sample = UtauSample(filepath, 0, sequence[0], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, "V")
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, "VC")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "C")
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "c":
                sample = UtauSample(filepath, 0, sequence[0], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, "V")
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, "Vc")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "c")
                sample.end -= blank
                output.append(sample)
        elif typeSequence[0] == "C" :
            if typeSequence[1] == "V":
                sample = UtauSample(filepath, 0, sequence[0], offset, intermediate, offset, fixed, blank, preuttr, overlap, "C")
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, "CV")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "V")
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "C" or typeSequence[1] == "c":
                print("skipped CC sample")
        elif typeSequence[0] == "c":
            if typeSequence[1] == "V":
                sample = UtauSample(filepath, 0, sequence[0], offset, intermediate, offset, fixed, blank, preuttr, overlap, "c")
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, "cV")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "V")
                sample.end -= blank
                output.append(sample)
    elif len(sequence) == 3:
        if typeSequence[0] == "V":
            if typeSequence[1] == "C":
                sample = UtauSample(filepath, 1, None, offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, "VC")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], intermediate, offset + overlap, offset, fixed, blank, preuttr, overlap, "C")
                output.append(sample)
                sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap, "CV")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "V")
                sample.end -= blank
                output.append(sample)
            elif typeSequence[1] == "c":
                sample = UtauSample(filepath, 1, None, offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, "Vc")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[1], intermediate, offset + overlap, offset, fixed, blank, preuttr, overlap, "c")
                output.append(sample)
                sample = UtauSample(filepath, 1, None, offset + overlap, offset + fixed, offset, fixed, blank, preuttr, overlap, "cV")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "V")
                sample.end -= blank
                output.append(sample)
        elif typeSequence[0] == "C" or typeSequence[0] == "c":
            if typeSequence[2] == "C":
                sample = UtauSample(filepath, 0, sequence[1], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, "V")
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, "VC")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "C")
                sample.end -= blank
                output.append(sample)
            elif typeSequence[2] == "c":
                sample = UtauSample(filepath, 0, sequence[1], offset + overlap, intermediate, offset, fixed, blank, preuttr, overlap, "V")
                output.append(sample)
                sample = UtauSample(filepath, 1, None, intermediate, offset + fixed, offset, fixed, blank, preuttr, overlap, "Vc")
                output.append(sample)
                sample = UtauSample(filepath, 0, sequence[2], offset + fixed, None, offset, fixed, blank, preuttr, overlap, "c")
                sample.end -= blank
                output.append(sample)
    return output

