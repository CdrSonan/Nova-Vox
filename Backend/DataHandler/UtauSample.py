from Backend.AudioSample import AudioSample
from global_consts import sampleRate
from os import path

class UtauSample:
    """class representing an audio sample loaded from UTAU.
    
    Methods:
        __init__: initialises the object based on both UTAU and Nova-Vox sample properties
        
        updateHandle: updates the handle of the sample, which is used to represent it in the devkit UI
        
        convert: returns a Nova-Vox compatible AudioSample object of the sample"""

    def __init__(self, filepath, _type, key, start, end, offset, fixed, blank, preuttr, overlap):
        """initialises the object based on both UTAU and Nova-Vox sample properties.
        
        Arguments:
            filepath: the path to the .wav file of the sample

            _type: the type of the sample. Can be either 0 for phoneme or 1 for transition

            key: the kay of the phoneme. Expected to be None for transition samples.

            start: the start of the sample in Nova-Vox in ms relative to the start of the .wav file

            end: the end of the sample in Nova-Vox in ms relative to the start of the .wav file

            offset: the offset property of the sample in UTAU

            fixed: the consonant property of the sample in UTAU

            blank: the cutoff property of the sample in UTAU

            preuttr: the pre-utterance property of the sample in UTAU

            overlap: the overlap property of the sample in UTAU
            
        Returns:
            None"""

        self.audioSample = AudioSample(filepath)
        self._type = _type
        if self._type == 0:
            self.key = key
        else:
            self.key = None
        self.start = start
        if end == None:
            if blank >= 0:
                self.end = self.audioSample.waveform.size()[0] * 1000 / sampleRate
            else:
                self.end = offset
        else:
            self.end = end
        self.handle = path.split(self.audioSample.filepath)[1] + ", " + str(self.start) + " - " + str(self.end)

        self.offset = offset
        self.fixed = fixed
        self.blank = blank
        self.preuttr = preuttr
        self.overlap = overlap

    def updateHandle(self):
        """updates the handle of the sample, which is used to represent it in the devkit UI, to reflect changed sample properties"""

        self.handle = path.split(self.audioSample.filepath)[1] + ", " + str(self.start) + " - " + str(self.end)

    def convert(self):
        """returns a Nova-Vox compatible AudioSample object of the sample.
        
        The waveform is trimmed to the area between start and end, and all UTAU-specific data is discarded"""

        start = int(self.start * sampleRate / 1000)
        end = int(self.end * sampleRate / 1000)
        audioSample = AudioSample(self.audioSample.filepath)
        audioSample.waveform = audioSample.waveform[start:end]
        return audioSample