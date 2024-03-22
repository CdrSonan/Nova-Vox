#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from Backend.DataHandler.AudioSample import AudioSample, AISample
from global_consts import sampleRate
from os import path

class UtauSample():
    """class representing an audio sample loaded from UTAU.
    
    Methods:
        __init__: initialises the object based on both UTAU and Nova-Vox sample properties
        
        updateHandle: updates the handle of the sample, which is used to represent it in the devkit UI
        
        convert: returns a Nova-Vox compatible AudioSample or AISample object of the sample"""


    def __init__(self, filepath:str, _type:int, key:str, start:float, end:float, offset:float, fixed:float, blank:float, preuttr:float, overlap:float, pitch:float, isVoiced:bool = True, isPlosive:bool = False, embedding:int = 0) -> None:
        """initialises the object based on both UTAU and Nova-Vox sample properties.
        
        Arguments:
            filepath: the path to the .wav file of the sample

            _type: the type of the sample. Can be either 0 for phoneme, 1 for transition or 2 fo sequence

            key: the kay of the phoneme. Expected to be None for transition samples.

            start: the start of the sample in Nova-Vox in ms relative to the start of the .wav file

            end: the end of the sample in Nova-Vox in ms relative to the start of the .wav file

            offset: the offset property of the sample in UTAU

            fixed: the consonant property of the sample in UTAU

            blank: the cutoff property of the sample in UTAU

            preuttr: the pre-utterance property of the sample in UTAU

            overlap: the overlap property of the sample in UTAU

            isVoiced: flag indicating whether the sample is voiced (voicedExcitation is muted for unvoiced samples during ESPER processing)

            isPlosive: flag indicating whether the sample is considered a plosive sound (if possible, plosives retain their original length after border creation during synthesis)
            
        Returns:
            None"""
            

        self.audioSample = AISample(filepath, _type == 1)
        self._type = _type
        self.key = key
        self.start = start
        if end == None:
            if blank >= 0:
                timesize = self.audioSample.waveform.size()[0] * 1000 / sampleRate
                self.end = timesize - blank
            else:
                self.end = offset - blank
        else:
            self.end = end
        self.handle = path.split(self.audioSample.filepath)[1] + ", " + str(self.start) + " - " + str(self.end)
        self.offset = offset
        self.fixed = fixed
        self.blank = blank
        self.preuttr = preuttr
        self.overlap = overlap
        self.audioSample.isVoiced = isVoiced
        self.audioSample.isPlosive = isPlosive
        self.audioSample.embedding = embedding
        self.audioSample.expectedPitch = pitch
        self.audioSample.key = self.key

    def updateHandle(self) -> None:
        """updates the handle of the sample, which is used to represent it in the devkit UI, to reflect changed sample properties"""

        self.handle = path.split(self.audioSample.filepath)[1] + ", " + str(self.start) + " - " + str(self.end)

    def convert(self, ai:bool = False) -> AudioSample or AISample:
        """returns a Nova-Vox compatible AudioSample object of the sample.
        The waveform is trimmed to the area between start and end, and all UTAU-specific data is discarded. the ai flag indicates whether an AISample instance is produced instead of an AudioSample instance."""

        start = int(self.start * sampleRate / 1000)
        end = int(self.end * sampleRate / 1000)
        self.audioSample.waveform = self.audioSample.waveform[start:end]
        if ai:
            return self.audioSample
        else:
            return self.audioSample.convert(False)
