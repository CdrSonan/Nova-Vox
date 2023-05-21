#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from Util import convertToFormat

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
                loopOverlap {array} -- The loop overlap of each phoneme of the note. The format depends on the arrayFormat argument.
                loopOffset {array} -- The loop offset of each phoneme of the note. The format depends on the arrayFormat argument.
                vibratoSpeed {array} -- The vibrato speed curve of the note. The format depends on the arrayFormat argument.
                vibratoStrength {array} -- The vibrato strength curve of the note. The format depends on the arrayFormat argument.
        Unless otherwise specified above, all arrays contain one element per engine tick, or 4ms, and are between -1 and 1 (inclusive)."""
    
    global middleLayer
    from UI.code.editor.Main import middleLayer
    tracks = []
    for track in middleLayer.trackList:
        notes = []
        for note in track.notes:
            if useNominalTimings:
                start = note.xPos
                end = note.xPos + note.length
            else:
                start = track.borders[note.phonemeStart * 3 + 1]
                end = track.borders[note.phonemeEnd * 3 + 1]
            notes.append({"start": start,
                          "end": end,
                          "pitch": note.yPos,
                          "phonemes": track.phonemes[note.phonemeStart:note.phonemeEnd],
                          "pitchCurve": convertToFormat(track.pitch[start:end], arrayFormat),
                          "breathiness": convertToFormat(track.breathiness[start:end], arrayFormat),
                          "steadiness": convertToFormat(track.steadiness[start:end], arrayFormat),
                          "aiBalance": convertToFormat(track.aiBalance[start:end], arrayFormat),
                          "loopOverlap": convertToFormat(track.loopOverlap[note.phonemeStart:note.phonemeEnd], arrayFormat),
                          "loopOffset": convertToFormat(track.loopOffset[note.phonemeStart:note.phonemeEnd], arrayFormat),
                          "vibratoSpeed": convertToFormat(track.vibratoSpeed[start:end], arrayFormat),
                          "vibratoStrength": convertToFormat(track.vibratoStrength[start:end], arrayFormat)})
        tracks.append({"notes": notes,
                       "volume": track.volume,
                       "voicebank": track.vbPath,
                       "mixinVoicebank": track.mixinVb})
    return tracks
            