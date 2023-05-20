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
        
        useNominalTimings {bool} -- Whether to use the nominal timings of the notes instead of the actual timing markers."""
    
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
            