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

//C implementation of the ESPER specharm resampler. Respects loop spacing setting and start/end fading flags.
__declspec(dllexport) void __cdecl resampleSpecharm(float* avgSpecharm, float* specharm, int length, float* steadiness, float spacing, int startCap, int endCap, float* output, segmentTiming timings, engineCfg config)
{
    for (int i = 0; i < length * config.frameSize; i++) {if (*(specharm + i) != *(specharm + i)) printf("NaN in!!!!!\n");}
    printf("flags: %i, %i\n", startCap, endCap);
    float* buffer = (float*) malloc(timings.windowEnd * config.frameSize * sizeof(float));
    //loop specharm
    loopSamplerSpecharm(specharm, length, buffer, timings.windowEnd, spacing, config);
    for (int i = 0; i < timings.windowEnd * config.frameSize; i++)
    {
        if (isnan(*(buffer + i)))
        {
            printf("loop NaN\n");
            //break;
        }
        if (isinf(*(buffer + i)))
        {
            printf("loop inf\n");
            //break;
        }
    }
    //scale specharm according to steadiness setting and add it to average
    for (int i = 0; i < timings.windowEnd; i++)
    {
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            *(buffer + i * config.frameSize + j) *= powf(1. - *(steadiness + i), 2.);
            *(buffer + i * config.frameSize + j) += *(avgSpecharm + j);
        }
        for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++)
        {
            *(buffer + i * config.frameSize + j) *= powf(1. - *(steadiness + i), 2.);
            *(buffer + i * config.frameSize + j) += *(avgSpecharm - config.halfHarmonics + j);
        }
    }
    //fade in sample if required
    if (0)//startCap == 0)
    {
        printf("fade in\n");
        float factor = -log2f((float)(timings.start2 - timings.start1) / (float)(timings.start3 - timings.start1));
        for (int i = timings.windowStart; i < timings.windowStart + timings.start3 - timings.start1; i++)
        {
            for (int j = 0; j < config.halfHarmonics; j++)
            {
                *(buffer + i * config.frameSize + j) *= powf((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
            }
            for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++)
            {
                *(buffer + i * config.frameSize + j) *= powf((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
            }
        }
    }
    //fade out sample if required
    if (0)//endCap == 0)
    {
        printf("fade out\n");
        float factor = -log2f((float)(timings.end3 - timings.end2) / (float)(timings.end3 - timings.end1));
        for (int i = timings.windowEnd + timings.end1 - timings.end3; i < timings.windowEnd; i++)
        {
            for (int j = 0; j < config.halfHarmonics; j++)
            {
                *(buffer + i * config.frameSize + j) *= powf(1. - ((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1)), factor);
            }
            for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++)
            {
                *(buffer + i * config.frameSize + j) *= powf(1. - ((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1)), factor);
            }
        }
    }
    //fill output buffer
    for (int i = 0; i < (timings.windowEnd - timings.windowStart) * config.frameSize; i++)
    {
        *(output + i) = *(buffer + timings.windowStart * config.frameSize + i);
    }
    free(buffer);
    for (int i = 0; i < (timings.windowEnd - timings.windowStart) * config.frameSize; i++)
    {
        if (isnan(*(buffer + i)))
        {
            printf("out NaN\n");
            break;
        }
        if (isinf(*(buffer + i)))
        {
            printf("out inf\n");
            break;
        }
    }
}

//C implementation of the ESPER pitch resampler. Respects loop spacing setting and start/end fading flags.
__declspec(dllexport) void __cdecl resamplePitch(short* pitchDeltas, int length, float pitch, float spacing, int startCap, int endCap, float* output, int requiredSize, segmentTiming timings)
{
    //loop pitch
    loopSamplerPitch(pitchDeltas, length, output, requiredSize, spacing);
    //load data into output buffer
    for (int i = 0; i < requiredSize; i++)
    {
        *(output + i) -= pitch;
    }
    //fade in if required
    if (startCap == 0)
    {
        float factor = -log2f((float)(timings.start2 - timings.start1) / (float)(timings.start3 - timings.start1));
        for (int i = 0; i < timings.start3 - timings.start1; i++)
        {
            *(output + i) *= powf((float)(i + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
        }
    }
    //fade out if required
    if (endCap == 0)
    {
        float factor = -log2f((float)(timings.end3 - timings.end2) / (float)(timings.end3 - timings.end1));
        for (int i = requiredSize - timings.end3 + timings.end1; i < requiredSize; i++)
        {
            *(output + i) *= powf(1. - ((float)(i - (requiredSize - timings.end3 + timings.end1) + 1) / (float)(timings.end3 - timings.end1 + 1)), factor);
        }
    }
}
