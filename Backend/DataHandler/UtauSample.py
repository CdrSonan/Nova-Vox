from Backend.AudioSample import AudioSample
from global_consts import sampleRate

class UtauSample:
    def __init__(self, filepath, type, key, start, end, offset, fixed, blank, preuttr, overlap):
        self.audioSample = AudioSample(filepath)
        self.type = type
        if self.type == 0:
            self.key = key
        else:
            self.key = None
        self.start = start
        self.end = end
        self.handle = filepath + ", " + str(start) + " - " + str(end)

        self.offset = offset
        self.fixed = fixed
        self.blank = blank
        self.preuttr = preuttr
        self.overlap = overlap

    def updateHandle(self):
        self.handle = self.audioSample.filepath + ", " + str(self.start) + " - " + str(self.end)

    def convert(self):
        start = int(self.start * sampleRate / 1000)
        end = int(self.end * sampleRate / 1000)
        self.audioSample.waveform = self.audioSample.waveform[start:end]
        return self.audioSample