#Copyright 2023, 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import torch
import h5py


from API.Addon import override

@override
def saveNVX(path:str, middleLayer) -> None:
    """backend function for saving a .nvx file"""

    str_dtype = h5py.string_dtype(encoding = "utf-8", length = None)
    tracks = []
    with h5py.File(path, "w") as f:
        for i, track in enumerate(middleLayer.trackList):
            track.validate()
            f.create_group(str(i))
            group = f[str(i)]
            group.create_dataset("lengths", data = [note.length for note in track.notes], dtype = "int32")
            group.create_dataset("xPos", data = [note.xPos for note in track.notes], dtype = "int64")
            group.create_dataset("yPos", data = [note.yPos for note in track.notes], dtype = "int32")
            group.create_dataset("phonemeMode", data = [note.phonemeMode for note in track.notes], dtype = "bool")
            group.create_dataset("content", data = [note.content.encode("utf-8") for note in track.notes], dtype = str_dtype)
            group.create_dataset("pronuncIndex", data = [-1 if note.pronuncIndex is None else note.pronuncIndex for note in track.notes], dtype = "int16")
            group.create_dataset("phonemes", data = [" ".join(note.phonemes).encode("utf-8") for note in track.notes], dtype = str_dtype)
            group.create_dataset("autopause", data = [note.autopause for note in track.notes], dtype = "bool")
            group.create_dataset("carryOver", data = [note.carryOver for note in track.notes], dtype = "bool")
            group.create_dataset("loopOverlap", data = track.loopOverlap, dtype = "float32")
            group.create_dataset("loopOffset", data = track.loopOffset, dtype = "float32")
            group.create_dataset("pitch", data = track.pitch, dtype = "float32")
            group.create_dataset("basePitch", data = track.basePitch, dtype = "float32")
            group.create_dataset("breathiness", data = track.breathiness, dtype = "float32")
            group.create_dataset("steadiness", data = track.steadiness, dtype = "float32")
            group.create_dataset("aiBalance", data = track.aiBalance, dtype = "float32")
            group.create_dataset("vibratoSpeed", data = track.vibratoSpeed, dtype = "float32")
            group.create_dataset("vibratoStrength", data = track.vibratoStrength, dtype = "float32")
            group.create_dataset("borders", data = track.borders[:], dtype = "int64")
            group.create_group("nodegraph")
            group.attrs["volume"] = track.volume
            group.attrs["vbPath"] = track.vbPath
            group.attrs["usePitch"] = track.usePitch
            group.attrs["useBreathiness"] = track.useBreathiness
            group.attrs["useSteadiness"] = track.useSteadiness
            group.attrs["useAIBalance"] = track.useAIBalance
            group.attrs["useVibratoSpeed"] = track.useVibratoSpeed
            group.attrs["useVibratoStrength"] = track.useVibratoStrength
            group.attrs["pauseThreshold"] = track.pauseThreshold
            if track.mixinVB is None:
                group.attrs["mixinVB"] = ""
            else:
                group.attrs["mixinVB"] = track.mixinVB
            group.attrs["length"] = track.length
        return
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
