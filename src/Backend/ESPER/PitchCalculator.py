#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import math
import torch
import C_Bridge
from Backend.DataHandler.AudioSample import AudioSample
import global_consts

def calculatePitch(audioSample:AudioSample) -> None:
    """Fallback method for calculating pitch data for an AudioSample object based on the previously set attributes expectedPitch and searchRange.
    
    Arguments:
        audioSample: The AudioSample object the operation is to be performed on
    
    Returns:
        None
    
    This method for pitch calculation uses 0-transitions to determine the borders between vocal chord vibrations. The algorithm searches for such transitions around expectedPitch (should be a value in Hz),
    with the range around it being defined by searchRange (should be a value between 0 and 1), which is interpreted as a percentage of the wavelength of expectedPitch.
    The function fills the pitchDeltas and pitch properties. Compared to the non-legacy version, it can be applied to smaller search ranges without the risk of failure, but suffers from a worse
    signal-to-noise ratio."""
    
    batches = math.floor(audioSample.waveform.size()[0] / global_consts.batchSize) + 1
    audioSample.pitchDeltas = torch.zeros([batches,], dtype = torch.int)
    audioSample.pitchMarkers = torch.zeros([audioSample.waveform.size()[0],], dtype = torch.int)
    audioSample.pitchMarkerValidity = torch.zeros([audioSample.waveform.size()[0],], dtype = torch.int8)
    audioSample.excitation = torch.zeros([2 * batches * (global_consts.halfTripleBatchSize + 1)], dtype = torch.float)
    cSample = C_Bridge.makeCSample(audioSample, False, False)
    C_Bridge.esper.pitchCalcFallback(cSample, global_consts.config)
    audioSample.pitchDeltas = audioSample.pitchDeltas[audioSample.pitchDeltas.nonzero()].flatten()
    audioSample.pitchMarkers = audioSample.pitchMarkers[audioSample.pitchMarkers.nonzero()].flatten()
    audioSample.pitch = torch.median(audioSample.pitchDeltas)
