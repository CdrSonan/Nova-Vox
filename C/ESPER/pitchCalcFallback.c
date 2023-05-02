// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/ESPER/pitchCalcFallback.h"

#include <malloc.h>
#include <math.h>
#include "C/util.h"

__declspec(dllexport) void __cdecl pitchCalcFallback(cSample sample, engineCfg config) {
    unsigned int batchSize = (int)((1. + sample.config.searchRange) * (float)config.sampleRate / (float)sample.config.expectedPitch);
    unsigned int lowerLimit = (int)((1. - sample.config.searchRange) * (float)config.sampleRate / (float)sample.config.expectedPitch);
    unsigned int batchStart = 0;
    unsigned int* zeroTransitions = (unsigned int*) malloc(batchSize * sizeof(unsigned int));
    unsigned int numZeroTransitions;
    double error;
    double newError;
    unsigned int delta;
    float bias;
    unsigned int offset;
    unsigned int* intermediateBuffer = (unsigned int*) malloc(ceildiv(sample.config.length, lowerLimit) * sizeof(unsigned int));
    unsigned int intermBufferLen = 0;
    while (batchStart + batchSize <= sample.config.length - batchSize) {
        numZeroTransitions = 0;
        for (int i = batchStart + lowerLimit; i < batchStart + batchSize; i++) {
            if ((*(sample.waveform + i - 1) < 0) && (*(sample.waveform + i) > 0)) {
                *(zeroTransitions + numZeroTransitions) = i;
                numZeroTransitions++;
            }
        }
        error = -1;
        delta = config.sampleRate / sample.config.expectedPitch;
        for (int i = 0; i < numZeroTransitions; i++) {
            offset = *(zeroTransitions + i);
            bias = fabsf(offset - batchStart - (float)config.sampleRate / (float)sample.config.expectedPitch);
            newError = 0;
            for (int j = 0; j < batchSize; j++) {
                newError += powf(*(sample.waveform + batchStart + j) - *(sample.waveform + offset + j), 2.) * bias;
            }
            if ((error > newError) || (error == -1)) {
                delta = i - batchStart;
                error = newError;
            }
        }
        *(intermediateBuffer + intermBufferLen) = delta;
        intermBufferLen++;
        batchStart += delta;
    }
    free(zeroTransitions);
    unsigned int cursor = 0;
    unsigned int cursor2 = 0;
    sample.config.pitchLength = 0;
    for (int i = 0; i < intermBufferLen; i++) {
        sample.config.pitchLength += *(intermediateBuffer + i);
    }
    sample.config.pitchLength /= config.batchSize;
    sample.pitchDeltas = (int*) realloc(sample.pitchDeltas, sample.config.pitchLength * sizeof(int));
    sample.config.pitch = 0;
    for (int i = 0; i < sample.config.pitchLength; i++) {
        while(cursor2 >= *(intermediateBuffer + cursor)) {
            if (cursor < intermBufferLen - 1) {
                cursor++;
            }
            cursor2 -= *(intermediateBuffer + cursor);
        }
        cursor2 += config.batchSize;
        *(sample.pitchDeltas + i) = *(intermediateBuffer + cursor);
        sample.config.pitch += *(intermediateBuffer + cursor);
    }
    free(intermediateBuffer);
    sample.config.pitch /= sample.config.pitchLength;
}
