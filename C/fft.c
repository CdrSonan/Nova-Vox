// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/fft.h"

#include "lib/fftw/fftw3.h"
#include <malloc.h>
#include <math.h>
#include "C/util.h"

fftwf_complex* fft(fftwf_complex* input, int length) {
    fftwf_complex* output = (fftwf_complex*) malloc(length * sizeof(fftwf_complex));
    fftwf_plan plan = fftwf_plan_dft_1d(length, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return output;
}

fftwf_complex* ifft(fftwf_complex* input, int length) {
    fftwf_complex* output = (fftwf_complex*) malloc(length * sizeof(fftwf_complex));
    fftwf_plan plan = fftwf_plan_dft_1d(length, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return output;
}

fftwf_complex* rfft(float* input, int length) {
    fftwf_complex* output = (fftwf_complex*) malloc((ceildiv(length, 2) + 1) * sizeof(fftwf_complex));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(length, input, output, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return output;
}

void rfft_inpl(float* input, int length, fftwf_complex* output) {
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(length, input, output, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

float* irfft(fftwf_complex* input, int length) {
    float* output = (float*) malloc(length * sizeof(float));
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(length, input, output, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return output;
}

void irfft_inpl(fftwf_complex* input, int length, float* output) {
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(length, input, output, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

fftwf_complex* stft(float* input, int length, engineCfg config) {
    int batches = ceildiv(length, config.batchSize);
    int rightpad = batches * config.batchSize - length + config.batchSize;
    // extended input buffer aligned with batch size
    float* in = (float*) malloc((config.batchSize + length + rightpad) * sizeof(float));
    // fill input buffer, extend will data with reflection padding on both sides
    for (int i = 0; i < config.batchSize; i++) {
        *(in + i) = *(input + config.batchSize - i);
    }
    for (int i = 0; i < length; i++) {
        *(in + config.batchSize + i) = *(input + i);
    }
    for (int i = 0; i < rightpad; i++) {
        *(in + config.batchSize + length + i) = *(input + length - 2 - i);
    }
    // allocate output buffer of desired size
    fftwf_complex* out = (fftwf_complex*) malloc(batches * (config.halfTripleBatchSize + 1) * sizeof(fftwf_complex));
    // fft setup
    for (int i = 0; i < batches; i++) {
        // allocation within loop for future openMP support
        float* buffer = (float*) malloc((config.tripleBatchSize) * sizeof(float));
        for (int j = 0; j < config.tripleBatchSize; j++) {
            // apply hanning window to data and load the result into buffer
            *(buffer + j) = *(in + i * config.batchSize + j) * pow(sin(pi * j / (config.tripleBatchSize - 1)), 2);
        }
        fftwf_plan plan = fftwf_plan_dft_r2c_1d(config.tripleBatchSize, buffer, out + i * (config.halfTripleBatchSize + 1), FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        free(buffer);
    }
    free(in);
    return out;
}

void stft_inpl(float* input, int length, engineCfg config, float* output) {
    int batches = ceildiv(length, config.batchSize);
    fftwf_complex* buffer = stft(input, length, config);
    for (int i = 0; i < batches * (config.halfTripleBatchSize + 1); i++) {
        *(output + i) = (*(buffer + i))[0] * 2 / 3;
        *(output + batches * (config.halfTripleBatchSize + 1) + i) = (*(buffer + i))[1];
    }
    free(buffer);
}

float* istft(fftwf_complex* input, int batches, int targetLength, engineCfg config) {
    // fft setup
    // extended input buffer aligned with batch size
    float* mainBuffer = (float*) malloc(config.batchSize * (batches + 2) * sizeof(float));
    for (int i = 0; i < config.batchSize * (batches + 2); i++) {
        *(mainBuffer + i) = 0.;
    }
    // smaller buffer for individual fft result
    float* buffer = (float*) malloc(config.tripleBatchSize * sizeof(float));
    for (int i = 0; i < batches; i++) {
        // perform ffts
        fftwf_plan plan = fftwf_plan_dft_c2r_1d(config.tripleBatchSize, input + i * (config.halfTripleBatchSize + 1), buffer, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        for (int j = 0; j < config.tripleBatchSize; j++) {
            // fill result into main buffer with overlap
            *(mainBuffer + i * config.batchSize + j) += *(buffer + j);
        }
    }
    free(buffer);
    // rescale waveform at the edges of the main buffer, to account for missing contributions of out-of-bounds windows
    for (int i = 0; i < config.batchSize; i++) {
        *(mainBuffer + i) *= 1 + pow(cos(pi * (i / config.tripleBatchSize + 2 / 3)), 2);
        *(mainBuffer + config.batchSize * (batches + 2) - 1 - i) *= 1 + pow(cos(pi * i / (config.tripleBatchSize - 1)), 2);
    }
    // allocate output buffer and transfer relevant data into it
    float* output = (float*) malloc(targetLength * sizeof(float));
    for (int i = 0; i < targetLength; i++) {
        *(output + i) = *(mainBuffer + config.batchSize + i) * 2 / 3;
    }
    free(mainBuffer);
    return output;
}

float* istft_hann(fftwf_complex* input, int batches, int targetLength, engineCfg config) {
    // fft setup
    // extended input buffer aligned with batch size
    float* mainBuffer = (float*) malloc(config.batchSize * (batches + 2) * sizeof(float));
    for (int i = 0; i < config.batchSize * (batches + 2); i++) {
        *(mainBuffer + i) = 0.;
    }
    // smaller buffer for individual fft result
    float* buffer = (float*) malloc(config.tripleBatchSize * sizeof(float));
    for (int i = 0; i < batches; i++) {
        // perform ffts
        fftwf_plan plan = fftwf_plan_dft_c2r_1d(config.tripleBatchSize, input + i * (config.halfTripleBatchSize + 1), buffer, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        for (int j = 0; j < config.tripleBatchSize; j++) {
            // fill result into main buffer with overlap
            *(mainBuffer + i * config.batchSize + j) += *(buffer + j)* pow(sin(pi * j / (config.tripleBatchSize - 1)), 2);
        }
    }
    free(buffer);
    // rescale waveform at the edges of the main buffer, to account for missing contributions of out-of-bounds windows
    for (int i = 0; i < config.batchSize; i++) {
        *(mainBuffer + i) *= 1 + pow(cos(pi * (i / config.tripleBatchSize + 2 / 3)), 2);
        *(mainBuffer + config.batchSize * (batches + 2) - 1 - i) *= 1 + pow(cos(pi * i / (config.tripleBatchSize - 1)), 2);
    }
    // allocate output buffer and transfer relevant data into it
    float* output = (float*) malloc(targetLength * sizeof(float));
    for (int i = 0; i < targetLength; i++) {
        *(output + i) = *(mainBuffer + config.halfTripleBatchSize + i) * 2 / 3;
    }
    free(mainBuffer);
    return output;
}
