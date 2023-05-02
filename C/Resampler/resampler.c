// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/Resampler/resampler.h"

#include <malloc.h>
#include <math.h>
#include "C/util.h"
#include "C/Resampler/loop.h"

__declspec(dllexport) void __cdecl resampleSpecharm(float* avgSpecharm, float* specharm, int length, float* steadiness, float spacing, short flags, float* output, segmentTiming timings, engineCfg config) {
    float* buffer = (float*) malloc(timings.windowEnd * config.frameSize * sizeof(float));
    loopSamplerSpecharm(specharm, length, buffer, timings.windowEnd, spacing, config);
    for (int i = timings.windowStart; i < timings.windowEnd; i++) {
        for (int j = 0; j < config.halfHarmonics; j++) {
            *(buffer + i * config.frameSize + j) *= powf(1. - *(steadiness + i), 2.);
            *(buffer + i * config.frameSize + j) += *(avgSpecharm + j);
        }
        for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++) {
            *(buffer + i * config.frameSize + j) *= powf(1. - *(steadiness + i), 2.);
            *(buffer + i * config.frameSize + j) += *(avgSpecharm - config.halfHarmonics + j);
        }
    }
    if (flags % 2 == 1) {
        float factor = -log2f((float)(timings.start2 - timings.start1) / (float)(timings.start3 - timings.start1));
        for (int i = timings.windowStart; i < timings.windowStart + timings.start3 - timings.start1; i++) {
            for (int j = 0; j < config.halfHarmonics; j++) {
                *(buffer + i * config.frameSize + j) *= powf((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
            }
            for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++) {
                *(buffer + i * config.frameSize + j) *= powf((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
            }
        }
    }
    if (flags % 4 == 1) {
        float factor = -log2f((float)(timings.end3 - timings.end2) / (float)(timings.end3 - timings.end1));
        for (int i = timings.windowEnd + timings.end1 - timings.end3; i < timings.windowEnd; i++) {
            for (int j = 0; j < config.halfHarmonics; j++) {
                *(buffer + i * config.frameSize + j) *= powf(1. - ((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1)), factor);
            }
            for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++) {
                *(buffer + i * config.frameSize + j) *= powf(1. - ((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1)), factor);
            }
        }
    }
    for (int i = 0; i < timings.windowEnd - timings.windowStart; i++) {
        *(output + i) = *(buffer + timings.windowStart + i);
    }
    free(buffer);
}

__declspec(dllexport) void __cdecl resamplePitch(int* pitchDeltas, int length, float* pitch, float spacing, short flags, float* output, segmentTiming timings, engineCfg config) {
    int maxPitch = 0;
    for (int i = 0; i < length; i++) {
        if (*(pitchDeltas + i) > maxPitch) {
            maxPitch = *(pitchDeltas + i);
        }
    }
    float minPitch = 0.;
    for (int i = 0; i < timings.end3 - timings.start1; i++) {
        if (*(pitch + i) < minPitch) {
            minPitch = *(pitch + i);
        }
    }
    int requiredSize = ceildiv(maxPitch, (int)minPitch * (timings.end3 - timings.start1) * config.batchSize);
    float* buffer = (float*) malloc((timings.end3 - timings.start1) * sizeof(float));
    loopSamplerPitch(pitchDeltas, length, buffer, requiredSize, spacing);
    for (int i = 0; i < requiredSize; i++) {
        *(buffer + i) += *(pitch + i);
    }
    if (flags % 2 == 0) {
        float factor = -log2f((float)(timings.start2 - timings.start1) / (float)(timings.start3 - timings.start1));
        for (int i = 0; i < timings.start3 - timings.start1; i++) {
            *(buffer + i) *= powf((float)(i + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
        }
    }
    if (flags % 4 == 0) {
        float factor = -log2f((float)(timings.end3 - timings.end2) / (float)(timings.end3 - timings.end1));
        for (int i = timings.end1 - timings.start1; i < timings.end3 - timings.start1; i++) {
            *(buffer + i) *= powf(1. - ((float)(i - timings.end1 + timings.start1 + 1) / (float)(timings.end3 - timings.end1 + 1)), factor);
        }
    }
    free(buffer);
}
