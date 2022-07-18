# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 08:47:26 2021

@author: CdrSonan
"""

#data batching
sampleRate = 48000
tickRate = 250
batchSize = 192
tripleBatchSize = 576
halfTripleBatchSize = 288

#analysis spectral processing
filterBSMult = 3
DIOBias = 0.4
DIOTolerance = 2.
DIOLastWinTolerance = 0.9
filterTEEMult = 32
filterHRSSMult = 4
nFormants = 50
nHarmonics = 128

#synthesis spectral processing
spectralRolloff1 = 144
spectralRolloff2 = 192
pitchShiftSpectralRolloff = 10
breCompPremul = 0.4

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
defaultVoicedFilter = 10
defaultUnvoicedIterations = 20

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