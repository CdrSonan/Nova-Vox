class VocalSequence:
    """temporary class for combining several VocalSegments into a sequence. Currently no acceleration structure"""
    def __init__(self, length, borders, phonemes, startCaps, endCaps, offsets, repetititionSpacing, pitch, steadiness, breathiness, vibratoSpeed, vibratoStrength, useBreathiness, useSteadiness, useVibratoSpeed, useVibratoStrength):
        self.length = length
        self.phonemeLength = len(phonemes)
        self.borders = borders
        self.phonemes = phonemes
        self.startCaps = startCaps
        self.endCaps = endCaps
        if self.phonemeLength > 0:
            startCaps[0] = True
            endCaps[-1] = True
        self.offsets = offsets
        self.repetititionSpacing = repetititionSpacing
        self.pitch = pitch
        self.steadiness = steadiness
        self.breathiness = breathiness
        self.vibratoSpeed = vibratoSpeed
        self.vibratoStrength = vibratoStrength
        self.useBreathiness = useBreathiness
        self.useSteadiness = useSteadiness
        self.useVibratoSpeed = useVibratoSpeed
        self.useVibratoStrength = useVibratoStrength

        self.parameters = []