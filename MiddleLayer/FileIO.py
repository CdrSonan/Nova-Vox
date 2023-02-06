#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
from MiddleLayer.MiddleLayer import MiddleLayer
from MiddleLayer.DataHandlers import Note

def validateTrackData(trackData:dict) -> dict:
    """validates the data of a track after loading it from a file"""

    return trackData

def saveNVX(path:str, middleLayer:MiddleLayer) -> None:
    """backend function for saving a .nvx file"""

    tracks = []
    for track in middleLayer.trackList:
        track.validate()
        notes = []
        for note in track.notes:
            notes.append({
                "length": note.length,
                "xPos": note.xPos,
                "yPos": note.yPos,
                "phonemeMode": note.phonemeMode,
                "content": note.content,
                "phonemeStart": note.phonemeStart,
                "phonemeEnd": note.phonemeEnd
            })
        tracks.append({
            "volume": track.volume,
            "vbPath": track.vbPath,
            "notes": notes,
            "phonemes": track.phonemes,
            "pitch": track.pitch,
            "basePitch": track.basePitch,
            "breathiness": track.breathiness,
            "steadiness": track.steadiness,
            "aiBalance": track.aiBalance,
            "loopOverlap": track.loopOverlap,
            "loopOffset": track.loopOffset,
            "vibratoSpeed": track.vibratoSpeed,
            "vibratoStrength": track.vibratoStrength,
            "usePitch": track.usePitch,
            "useBreathiness": track.useBreathiness,
            "useSteadiness": track.useSteadiness,
            "useAIBalance": track.useAIBalance,
            "useVibratoSpeed": track.useVIbratoSpeed,
            "useVibratoStrength": track.useVibratoStrength,
            "pauseThreshold": track.pauseThreshold,
            "mixinVB": track.mixinVB,
            "nodegraph": None,
            "borders": track.borders,
            "length": track.length
        })
    data = {
        "tracks": tracks,
        "audioBuffer": middleLayer.audioBuffer
    }
    torch.save(data, path)

def loadNVX(path:str, middleLayer:MiddleLayer) -> None:
    """backend function for loading a .nvx file"""

    data = torch.load(path, map_location = torch.device("cpu"))
    tracks = data["tracks"]
    for i in range(len(middleLayer.trackList)):
        middleLayer.deleteTrack(0)
    for trackData in tracks:
        track = validateTrackData(trackData)
        vbData = torch.load(track.vbPath, map_location = torch.device("cpu"))["metadata"]
        middleLayer.importVoicebankNoSubmit(track.vbPath, vbData.name, vbData.image)
        middleLayer.trackList[-1].volume = track["volume"]
        for note in track["notes"]:
            middleLayer.trackList[-1].notes.append(Note(note.xPos, note.yPos, note.phonemeStart, note.phonemeEnd))
            middleLayer.trackList[-1].notes[-1].length = note.length
            middleLayer.trackList[-1].notes[-1].phonemeMode = note.phonemeMode
            middleLayer.trackList[-1].notes[-1].content = note.content
        middleLayer.trackList[-1].phonemes = track["phonemes"]
        middleLayer.trackList[-1].pitch = track["pitch"]
        middleLayer.trackList[-1].basePitch = track["basePitch"]
        middleLayer.trackList[-1].breathiness = track["breathiness"]
        middleLayer.trackList[-1].steadiness = track["steadiness"]
        middleLayer.trackList[-1].aiBalance = track["aiBalance"]
        middleLayer.trackList[-1].loopOverlap = track["loopOverlap"]
        middleLayer.trackList[-1].loopOffset = track["loopOffset"]
        middleLayer.trackList[-1].vibratoSpeed = track["vibratoSpeed"]
        middleLayer.trackList[-1].vibratoStrength = track["vibratoStrength"]
        middleLayer.trackList[-1].usePitch = track["usePitch"]
        middleLayer.trackList[-1].useBreathiness = track["useBreathiness"]
        middleLayer.trackList[-1].useSteadiness = track["useSteadiness"]
        middleLayer.trackList[-1].useAIBalance = track["useAIBalance"]
        middleLayer.trackList[-1].useVibratoSpeed = track["useVibratoSpeed"]
        middleLayer.trackList[-1].useVibratoStrength = track["useVibratoStrength"]
        middleLayer.trackList[-1].nodegraph = track["nodegraph"]
        middleLayer.trackList[-1].borders = track["borders"]
        middleLayer.trackList[-1].length = track["length"]
        middleLayer.trackList[-1].mixinVB = track["mixinVB"]
        middleLayer.trackList[-1].pauseThreshold = track["pauseThreshold"]
    middleLayer.validate()
