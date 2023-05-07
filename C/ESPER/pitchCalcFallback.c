// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/ESPER/pitchCalcFallback.h"

#include <malloc.h>
#include <math.h>
#include "C/util.h"

//fallback function for calculating the approximate time-dependent pitch of a sample.
//Used when the Torchaudio implementation fails, likely due to a too narrow search range setting or the sample being too short.
__declspec(dllexport) void __cdecl pitchCalcFallback(cSample sample, engineCfg config) {
    //limits for autocorrelation search
    unsigned int batchSize = (int)((1. + sample.config.searchRange) * (float)config.sampleRate / (float)sample.config.expectedPitch);
    unsigned int lowerLimit = (int)((1. - sample.config.searchRange) * (float)config.sampleRate / (float)sample.config.expectedPitch);
    unsigned int batchStart = 0;
    //oversized array for holding all zeroTransitions within a batch
    unsigned int* zeroTransitions = (unsigned int*) malloc(batchSize * sizeof(unsigned int));
    unsigned int numZeroTransitions;
    double error;
    double newError;
    unsigned int delta;
    float bias;
    unsigned int offset;
    //buffer for storing pitch in pitch-synchronous format
    unsigned int* intermediateBuffer = (unsigned int*) malloc(ceildiv(sample.config.length, lowerLimit) * sizeof(unsigned int));
    unsigned int intermBufferLen = 0;
    //run until sample is fully processed
    while (batchStart + batchSize <= sample.config.length - batchSize) {
        //get all zero-transitions in the current batch
        numZeroTransitions = 0;
        for (int i = batchStart + lowerLimit; i < batchStart + batchSize; i++) {
            if ((*(sample.waveform + i - 1) < 0) && (*(sample.waveform + i) > 0)) {
                *(zeroTransitions + numZeroTransitions) = i;
                numZeroTransitions++;
            }
        }
        //calculate autocorrelation error for each zeroTransition; load the value of the best candidate into delta
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
        //add best candidate to buffer and use it as starting point for the next batch
        *(intermediateBuffer + intermBufferLen) = delta;
        intermBufferLen++;
        batchStart += delta;
    }
    free(zeroTransitions);
    unsigned int cursor = 0;
    unsigned int cursor2 = 0;
    //calculate "virtual" pitch-based sample length
    sample.config.pitchLength = 0;
    for (int i = 0; i < intermBufferLen; i++) {
        sample.config.pitchLength += *(intermediateBuffer + i);
    }
    sample.config.pitchLength /= config.batchSize;
    //TODO: get rid of pointer reallocation, since it can break PyTorch memory management!!!
    sample.pitchDeltas = (int*) realloc(sample.pitchDeltas, sample.config.pitchLength * sizeof(int));
    //transform from pitch-synchronous to time-synchronous space and fill average pitch field
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
