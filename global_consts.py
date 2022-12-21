#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#data batching
sampleRate = 48000
tickRate = 250
batchSize = 192
tripleBatchSize = 576
halfTripleBatchSize = 288

#analysis spectral processing
filterBSMult = 2
DIOBias = 0.4
DIOBias2 = 0.2
DIOTolerance = 0.2
DIOLastWinTolerance = 0.9
filterTEEMult = 32
filterHRSSMult = 4
nFormants = 50
nHarmonics = 128
ampContThreshold = 10

#synthesis spectral processing
spectralRolloff1 = 144
spectralRolloff2 = 192
pitchShiftSpectralRolloff = 4
breCompPremul = 0.6
crfBorderAbs = 7
crfBorderRel = 0.1

#audio pipeline
audioBufferSize = 2048

#pitch determination
pitchTransitionLength = 25
pitchDipWidth = 40
pitchDipHeight = 0.2

#editor UI
octaves = 4

#self-identification
language = "en"
version = "Alpha 5.0"

#devkit defaults
defaultExpectedPitch = 249.
defaultSearchRange = 0.55
defaultVoicedThrh = 0.5
defaultSpecWidth = 2
defaultSpecDepth = 30
defaultTempWidth = 2
defaultTempDepth = 10

#oto.ini parser
consonantEndOffset = 15

#control phoneme list
controlPhonemes = [
    ("-", "C"),
    ("=", "C"),
    (">", "C"),
    ("<", "C"),
    ("+", "R"),
    ("_0", "R"),
    ("_X", "R"),
    ("_autopause", "R")
]
