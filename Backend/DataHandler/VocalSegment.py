class VocalSegment:
    """Class representing the segment covered by a single phoneme within a VocalSequence.
    
    Attributes:
        start1-3, end1-3: timing borders of the segment
        
        startCap, endCap: Whether there is a transition from the previous, and to the next phoneme
        
        phonemeKey: The key of the phoneme of the sequence
        
        vb: The Voicebank to use data from
        
        offset: The offset applied to the audio before sampling. Non-zero values discard the beginning of the audio.
        
        repetititionSpacing: The amout of overlap applied when looping the sample
        
        pitch: relevant part of the pitch parameter curve
        
        steadiness: relevant part of the steadiness parameter curve
        
        breathiness: relevant part of the breathiness parameter curve
        
    Methods:
        phaseShift: helper function for phase shifting audio by a certain phase at a certain pitch
        
        loopSamplerVoicedExcitation: helper function for looping the voiced excitation signal
        
        loopSamplerSpectrum: helper function for looping time sequences of spectra
        
        getSpectrum: samples the time sequence of spectra for the segment
        
        getExcitation: samples the unvoiced excitation signal of the segment
        
        getVoicedExcitation: samples the voiced excitation signal of the segment"""


    def __init__(self, inputs, vb, index):
        self.start1 = inputs.borders[3*index]
        self.start2 = inputs.borders[3*index+1]
        self.start3 = inputs.borders[3*index+2]
        self.end1 = inputs.borders[3*index+3]
        self.end2 = inputs.borders[3*index+4]
        self.end3 = inputs.borders[3*index+5]
        self.startCap = inputs.startCaps[index]
        self.endCap = inputs.endCaps[index]
        self.phonemeKey = inputs.phonemes[index]
        self.vb = vb
        self.offset = inputs.offsets[index]
        self.repetititionSpacing = inputs.repetititionSpacing[index]
        self.pitch = inputs.pitch[self.start1:self.end3]
        self.steadiness = inputs.steadiness[self.start1:self.end3]

    def __init__(self, start1, start2, start3, end1, end2, end3, startCap, endCap, phonemeKey, vb, offset, repetititionSpacing, pitch, steadiness):
        self.start1 = start1
        self.start2 = start2
        self.start3 = start3
        self.end1 = end1
        self.end2 = end2
        self.end3 = end3
        self.startCap = startCap
        self.endCap = endCap
        self.phonemeKey = phonemeKey
        self.vb = vb
        self.offset = offset
        self.repetititionSpacing = repetititionSpacing
        self.pitch = pitch
        self.steadiness = steadiness
