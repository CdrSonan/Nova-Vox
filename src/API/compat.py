#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from torch import linspace
from Backend.Resampler.CubicSplineInter import interp
from Util import convertFormat
import API.Ops

def getPerNoteData(arrayFormat:str = "list", useNominalTimings:bool = False) -> list:
    """Returns the data of the current project in a per-note format using common Python data structures compatible with other engines.
    
    Arguments:
        arrayFormat {str} -- The format of the returned arrays. Can be "list", "numpy" or "torch".
        
        useNominalTimings {bool} -- Whether to use the nominal timings of the notes instead of the actual timing markers.
        
    Returns:
        list -- A list of tracks. Each track is a dictionary with the following keys:
            volume {float} -- The volume of the track. Between 0 and 1.2.
            voicebank {str} -- The path to the voicebank used by the track.
            mixinVoicebank {str} -- The path to the voicebank used for the mix-in feature by the track.
            notes {list} -- A list of notes. Each note is a dictionary with the following keys:
                start {int} -- The start time of the note in engine ticks. Each tick equals 4ms.
                end {int} -- The end time of the note in engine ticks. Each tick equals 4ms.
                pitch {int} -- The MIDI pitch of the note in semitones, lowered by one octave.
                phonemes {list} -- A list of the phonemes the note contains. Each phoneme is a string.
                pitchCurve {array} -- The pitch curve of the note. The format depends on the arrayFormat argument.
                breathiness {array} -- The breathiness curve of the note. The format depends on the arrayFormat argument.
                steadiness {array} -- The steadiness curve of the note. The format depends on the arrayFormat argument.
                aiBalance {array} -- The AI balance curve of the note. The format depends on the arrayFormat argument.
                loopOverlap {array} -- The loop overlap of each phoneme of the note. The format depends on the arrayFormat argument. Contains one number per phoneme.
                loopOffset {array} -- The loop offset of each phoneme of the note. The format depends on the arrayFormat argument. Contains one number per phoneme.
                vibratoSpeed {array} -- The vibrato speed curve of the note. The format depends on the arrayFormat argument.
                vibratoStrength {array} -- The vibrato strength curve of the note. The format depends on the arrayFormat argument.
        Unless otherwise specified above, all arrays contain one element per engine tick, or 4ms, and all elements are between -1 and 1 (inclusive)."""
    
    global middleLayer
    from UI.editor.Main import middleLayer
    tracks = []
    for track in middleLayer.trackList:
        notes = []
        for index, note in enumerate(track.notes):
            if useNominalTimings:
                start = note.xPos
                end = note.xPos + note.length
            else:
                if index == 0:
                    start = track.borders[0]
                    start_p = 0
                else:
                    start = track.borders[track.phonemeIndices[index - 1] * 3 + 1]
                    start_p = track.phonemeIndices[index - 1]
                end = track.borders[track.phonemeIndices[index] * 3 + 1]
                end_p = track.phonemeIndices[index]
            if note.autopause:
                phonemes = [*note.phonemes, "_autopause"]
            else:
                phonemes = note.phonemes
            notes.append({"start": start,
                          "end": end,
                          "pitch": note.yPos,
                          "phonemes": phonemes,
                          "pitchCurve": convertFormat(track.pitch[start:end], arrayFormat),
                          "breathiness": convertFormat(track.breathiness[start:end], arrayFormat),
                          "steadiness": convertFormat(track.steadiness[start:end], arrayFormat),
                          "aiBalance": convertFormat(track.aiBalance[start:end], arrayFormat),
                          "genderFactor": convertFormat(track.genderFactor[start:end], arrayFormat),
                          "loopOverlap": convertFormat(track.loopOverlap[start_p:end_p], arrayFormat),
                          "loopOffset": convertFormat(track.loopOffset[start_p:end_p], arrayFormat),
                          "vibratoSpeed": convertFormat(track.vibratoSpeed[start:end], arrayFormat),
                          "vibratoStrength": convertFormat(track.vibratoStrength[start:end], arrayFormat)})
        tracks.append({"notes": notes,
                       "volume": track.volume,
                       "voicebank": track.vbPath,
                       "mixinVoicebank": track.mixinVb})
    return tracks

def importFromPerNoteData(tracks:list, wipe:bool = False, useNominalTimings:bool = False) -> None:
    """imports the data from a per-note data structure into the current project. See getPerNoteData for the expected format.
    The array format of the data is automatically detected, and all arrays (except loopOverlap and loopOffset) can contain an arbitrary number of data points >= 2, which are interpolated as required.
    If wipe is True, the current project will be cleared first.
    If useNominalTimings is True, the nominal timings of the notes will be used for array/curve alignment instead of the actual timing markers."""
    
    global middleLayer
    from UI.editor.Main import middleLayer
    if wipe:
        while len(middleLayer.trackList) > 0:
            API.Ops.DeleteTrack(0)()
    for track in tracks:
        API.Ops.ImportVoicebank(track["voicebank"])()
        middleLayer.trackList[-1].volume = track["volume"]
        #TODO: add mixin vb hook once implemented
        for index, note in enumerate(track["notes"]):
            API.Ops.ChangeTrack(len(middleLayer.trackList) - 1)()
            API.Ops.AddNote(len(middleLayer.trackList[-1].notes), note["start"], note["pitch"])()
            API.Ops.ChangeNoteLength(len(middleLayer.trackList[-1].notes) - 1, note["start"], note["end"] - note["start"])()
            middleLayer.trackList[-1].notes[-1].phonemeMode = True
            API.Ops.ChangeLyrics(len(middleLayer.trackList[-1].notes) - 1, "".join(note["phonemes"]))()
            if index == 0:
                start = middleLayer.trackList[-1].borders[0]
                start_p = 0
            else:
                start = middleLayer.trackList[-1].borders[middleLayer.trackList[-1].phonemeIndices[index - 1] * 3 + 1]
                start_p = middleLayer.trackList[-1].phonemeIndices[index - 1]
            end = middleLayer.trackList[-1].borders[middleLayer.trackList[-1].phonemeIndices[index] * 3 + 1]
            end_p = middleLayer.trackList[-1].phonemeIndices[index]
            middleLayer.trackList[-1].loopOverlap[start_p:end_p] = convertFormat(note["loopOverlap"], "torch")
            middleLayer.trackList[-1].loopOffset[start_p:end_p] = convertFormat(note["loopOffset"], "torch")
            def interpolation(source, target):
                target[start:end] = interp(linspace(0, 1, source.size()[0]), source, linspace(0, 1, end - start))
            interpolation(convertFormat(note["pitchCurve"], "torch"),  middleLayer.trackList[-1].pitch[start:end])
            interpolation(convertFormat(note["breathiness"], "torch"), middleLayer.trackList[-1].breathiness[start:end])
            interpolation(convertFormat(note["steadiness"], "torch"), middleLayer.trackList[-1].steadiness[start:end])
            interpolation(convertFormat(note["aiBalance"], "torch"), middleLayer.trackList[-1].aiBalance[start:end])
            interpolation(convertFormat(note["genderFactor"], "torch"), middleLayer.trackList[-1].genderFactor[start:end])
            interpolation(convertFormat(note["vibratoSpeed"], "torch"), middleLayer.trackList[-1].vibratoSpeed[start:end])
            interpolation(convertFormat(note["vibratoStrength"], "torch"), middleLayer.trackList[-1].vibratoStrength[start:end])
    middleLayer.validate()
