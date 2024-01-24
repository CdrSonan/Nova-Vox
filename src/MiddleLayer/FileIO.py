#Copyright 2023, 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch

from API.Addon import override

@override
def validateTrackData(trackData:dict) -> dict:
    """validates the data of a track after loading it from a file"""

    return trackData

@override
def saveNVX(path:str, middleLayer) -> None:
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
                "pronuncIndex": note.pronuncIndex,
                "phonemes": note.phonemes,
                "autopause": note.autopause,
                "borders": note.borders,
                "carryOver": note.carryOver,
                "loopOverlap": note.loopOverlap,
                "loopOffset": note.loopOffset,
            })
        tracks.append({
            "volume": track.volume,
            "vbPath": track.vbPath,
            "notes": notes,
            "pitch": track.pitch,
            "basePitch": track.basePitch,
            "breathiness": track.breathiness,
            "steadiness": track.steadiness,
            "aiBalance": track.aiBalance,
            "vibratoSpeed": track.vibratoSpeed,
            "vibratoStrength": track.vibratoStrength,
            "usePitch": track.usePitch,
            "useBreathiness": track.useBreathiness,
            "useSteadiness": track.useSteadiness,
            "useAIBalance": track.useAIBalance,
            "useVibratoSpeed": track.useVibratoSpeed,
            "useVibratoStrength": track.useVibratoStrength,
            "pauseThreshold": track.pauseThreshold,
            "mixinVB": track.mixinVB,
            "nodegraph": track.nodegraph,
            "length": track.length,
            "wrappingBorders": track.borders.wrappingBorders
        })
    data = {
        "tracks": tracks,
        "audioBuffer": middleLayer.audioBuffer
    }
    torch.save(data, path)
