# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 08:47:26 2021

@author: CdrSonan
"""

sampleRate = 48000
tickRate = 250
batchSize = 192
tripleBatchSize = 576
halfTripleBatchSize = 288

filterBSMult = 4
filterTEEMult = 8

pitchShiftSpectralRolloff = 10

spectralRolloff1 = 144
spectralRolloff2 = 192

breCompPremul = 0.4

audioBufferSize = 2048

pitchTransitionLength = 25
pitchDipWidth = 40
pitchDipHeight = 0.2

octaves = 4

language = "en"
version = "Alpha 2.0"