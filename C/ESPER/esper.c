// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/ESPER/esper.h"

#include "lib/fftw/fftw3.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include "C/util.h"
#include "C/fft.h"
#include "C/ESPER/components.h"

__declspec(dllexport) void __cdecl specCalc(cSample sample, engineCfg config) {
    sample.config.batches = ceildiv(sample.config.length, config.batchSize);
    fftwf_complex* buffer = stft(sample.waveform, sample.config.length, config);
    float* signalsAbs = (float*) malloc(sample.config.batches * (config.halfTripleBatchSize + 1) * sizeof(float));
    for (int i = 0; i < sample.config.batches * (config.halfTripleBatchSize + 1); i++) {
        *(signalsAbs + i) = sqrtf(cpxAbsf(*(buffer + i)));
    }
    free(buffer);
    float* lowSpectra = lowRangeSmooth(sample, signalsAbs, config);
    float* highSpectra = highRangeSmooth(sample, signalsAbs, config);
    finalizeSpectra(sample, lowSpectra, highSpectra, config);
    free(lowSpectra);
    free(highSpectra);
    separateVoicedUnvoiced(sample, config);
    averageSpectra(sample, config);
}
