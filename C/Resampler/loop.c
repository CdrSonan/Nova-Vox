// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/Resampler/loop.h"

#include <malloc.h>
#include "C/util.h"
#include "C/interpolation.h"

void loopSamplerSpecharm(float* input, int length, float* output, int targetLength, float spacing, engineCfg config) {
    int effSpacing = ceildiv(spacing * length,  2);
    int requiredInstances = targetLength / (length - effSpacing);
    int lastWin = targetLength - requiredInstances * (length - effSpacing);
    if (targetLength <= length) {
        for (int i = 0; i < targetLength * config.frameSize; i++) {
            *(output + i) = *(input + i);
        }
    } else {
        float* buffer = (float*) malloc((length - effSpacing) * config.frameSize * sizeof(float)); //allocate buffer
        for (int i = 0; i < (length - effSpacing) * config.frameSize; i++) { //add first window to output and fill buffer
            *(output + i) = *(input + i);
            *(buffer + i) = *(input + i);
        }
        for (int i = 0; i < effSpacing; i++) { //modify start of buffer to include transition
            for (int j = 0; j < config.halfHarmonics; j++) {
                *(buffer + i * config.frameSize + j) *= (float)(i + 1) / (float)(effSpacing + 1);
                *(buffer + i * config.frameSize + j) += *(input + (length - effSpacing + i) * config.frameSize + j) * (1. - ((float)(i + 1) / (float)(effSpacing + 1)));
            }
            phaseInterp_inplace(buffer + i * config.frameSize + config.halfHarmonics, input + (length - effSpacing + i) * config.frameSize + config.halfHarmonics, config.halfHarmonics, (float)(i + 1) / (float)(effSpacing + 1));
            for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++) {
                *(buffer + i * config.frameSize + j) *= (float)(i + 1) / (float)(effSpacing + 1);
                *(buffer + i * config.frameSize + j) += *(input + (length - effSpacing + i) * config.frameSize + j) * (1. - ((float)(i + 1) / (float)(effSpacing + 1)));
            }
        }
        for (int i = 1; i < requiredInstances * config.frameSize; i++) { //add mid windows from buffer to output
            for (int j = 0; j < (length - effSpacing) * config.frameSize; j++) {
                *(output + i * (length - effSpacing) * config.frameSize + j) = *(buffer + j);
            }
        }
        if (lastWin > effSpacing) { //add final window, which has a length below the buffer size. Use buffer data if the window is still long enough to require a transition, otherwise fall back to input data
            for (int i = 0; i < lastWin * config.frameSize; i++) {
                *(output + requiredInstances * (length - effSpacing) + i) = *(buffer + i);
            }
        } else {
            for (int i = 0; i < lastWin * config.frameSize; i++) {
                *(output + requiredInstances * (length - effSpacing) + i) = *(input + length - effSpacing + i);
            }
        }
        free(buffer);
    }
}

void loopSamplerPitch(int* input, int length, float* output, int targetLength, float spacing) {
    int effSpacing = ceildiv(spacing * length,  2);
    int requiredInstances = targetLength / (length - effSpacing);
    int lastWin = targetLength - requiredInstances * (length - effSpacing);
    if (targetLength <= length) {
        for (int i = 0; i < targetLength; i++) {
            *(output + i) = (float)*(input + i);
        }
    } else {
        float* buffer = (float*) malloc((length - effSpacing) * sizeof(float)); //allocate buffer
        for (int i = 0; i < (length - effSpacing); i++) { //add first window to output and fill buffer
            *(output + i) = (float)*(input + i);
            *(buffer + i) = (float)*(input + i);
        }
        for (int i = 0; i < effSpacing; i++) { //modify start of buffer to include transition
            *(buffer + i) *= (float)(i + 1) / (float)(effSpacing + 1);
            *(buffer + i) += (float)*(input + length - effSpacing + i) * (1. - ((float)(i + 1) / (float)(effSpacing + 1)));
        }
        for (int i = 1; i < requiredInstances; i++) { //add mid windows from buffer to output
            for (int j = 0; j < length; j++) {
                *(output + i * (length - effSpacing) + j) = *(buffer + j);
            }
        }
        if (lastWin > effSpacing) { //add final window, which has a length below the buffer size. Use buffer data if the window is still long enough to require a transition, otherwise fall back to input data
            for (int i = 0; i < lastWin; i++) {
                *(output + requiredInstances * (length - effSpacing) + i) = *(buffer + i);
            }
        } else {
            for (int i = 0; i < lastWin; i++) {
                *(output + requiredInstances * (length - effSpacing) + i) = *(input + length - effSpacing + i);
            }
        }
        free(buffer);
    }
}
