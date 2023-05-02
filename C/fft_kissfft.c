// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/fft.h"

#include <math.h>
#include "C/util.h"
#include "lib/kissfft/kiss_fft.h"
#include "lib/kissfft/kiss_fftr.h"

#include "lib/fftw/fftw3.h"
#include "lib/fftw/fftw3.h"

cpxfloat* fft(cpxfloat* input, int length) {
    kiss_fft_cfg internalcfg = kiss_fft_alloc(length, 0, NULL, NULL);
    kiss_fft_cpx* output = (kiss_fft_cpx*) malloc(length * sizeof(kiss_fft_cpx));
    kiss_fft(internalcfg, input, output);
    kiss_fft_free(internalcfg);
    return output;
}

cpxfloat* ifft(cpxfloat* input, int length) {
    kiss_fft_cfg internalcfg = kiss_fft_alloc(length, 0, NULL, NULL);
    kiss_fft_cpx* output = (kiss_fft_cpx*) malloc(length * sizeof(kiss_fft_cpx));
    kiss_fft(internalcfg, input, output);
    kiss_fft_free(internalcfg);
    return output;
}

cpxfloat* rfft(float* input, int length) {
    kiss_fftr_cfg internalcfg = kiss_fftr_alloc(length, 0, NULL, NULL);
    kiss_fft_cpx* output = (kiss_fft_cpx*) malloc((length / 2 + 1) * sizeof(kiss_fft_cpx));
    kiss_fftr(internalcfg, input, output);
    kiss_fft_free(internalcfg);
    return output;
}

void rfft_inpl(float* input, int length, cpxfloat* output) {
    kiss_fftr_cfg internalcfg = kiss_fftr_alloc(length, 0, NULL, NULL);
    kiss_fftr(internalcfg, input, output);
    kiss_fft_free(internalcfg);
}

float* irfft(cpxfloat* input, int length) {
    kiss_fftr_cfg internalcfg = kiss_fftr_alloc(length, 1, NULL, NULL);
    float* output = (float*) malloc((length - 1) * 2 * sizeof(float));
    kiss_fftri(internalcfg, input, output);
    kiss_fft_free(internalcfg);
    return output;
}

void irfft_inpl(cpxfloat* input, int length, float* output) {
    kiss_fftr_cfg internalcfg = kiss_fftr_alloc(length, 1, NULL, NULL);
    kiss_fftri(internalcfg, input, output);
    kiss_fft_free(internalcfg);
}

cpxfloat* stft(float* input, int length, engineCfg config) {
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
    cpxfloat* out = (cpxfloat*) malloc(batches * (config.halfTripleBatchSize + 1) * sizeof(cpxfloat));
    // fft setup
    kiss_fftr_cfg internalcfg = kiss_fftr_alloc(config.tripleBatchSize, 0, NULL, NULL);
    for (int i = 0; i < batches; i++) {
        // allocation within loop for future openMP support
        float* buffer = (float*) malloc((config.tripleBatchSize) * sizeof(float));
        for (int j = 0; j < config.tripleBatchSize; j++) {
            // apply hanning window to data and load the result into buffer
            *(buffer + j) = *(in + i * config.batchSize + j) * pow(cos(pi * j / config.tripleBatchSize), 2);
        }
        // perform ffts
        kiss_fftr(internalcfg, buffer, out + i * (config.halfTripleBatchSize + 1));
        free(buffer);
    }
    free(in);
    kiss_fft_free(internalcfg);
    return out;
}

void stft_inpl(float* input, int length, engineCfg config, float* output) {
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
    // fft setup
    kiss_fftr_cfg internalcfg = kiss_fftr_alloc(config.tripleBatchSize, 0, NULL, NULL);
    cpxfloat* buffer = (cpxfloat*) malloc(batches * (config.halfTripleBatchSize + 1) * sizeof(cpxfloat));
    for (int i = 0; i < batches; i++) {
        // allocation within loop for future openMP support
        float* inputBuffer = (float*) malloc((config.tripleBatchSize) * sizeof(float));
        for (int j = 0; j < config.tripleBatchSize; j++) {
            // apply hanning window to data and load the result into buffer
            *(inputBuffer + j) = *(in + i * config.batchSize + j) * pow(cos(pi * j / config.tripleBatchSize), 2);
        }
        // perform ffts
        kiss_fftr(internalcfg, buffer, buffer + i * (config.halfTripleBatchSize + 1));
        free(inputBuffer);
    }
    free(in);
    kiss_fft_free(internalcfg);
    for (int i = 0; i < batches * (config.halfTripleBatchSize + 1); i++) {
        *(output + i) = (*(buffer + i)).r;
        *(output + batches * (config.halfTripleBatchSize + 1) + i) = (*(buffer + i)).i;
    }
    free(buffer);
}

float* istft(cpxfloat* input, int batches, int targetLength, engineCfg config) {
    // fft setup
    kiss_fftr_cfg internalcfg = kiss_fftr_alloc(config.tripleBatchSize, 1, NULL, NULL);
    // extended input buffer aligned with batch size
    float* mainBuffer = (float*) malloc(config.batchSize * (batches + 2) * sizeof(float));
    for (int i = 0; i < config.batchSize * (batches + 2); i++) {
        *(mainBuffer + i) = 0.;
    }
    // smaller buffer for individual fft result
    float* buffer = (float*) malloc((config.tripleBatchSize) * sizeof(float));
    for (int i = 0; i < batches; i++) {
        // perform ffts
        kiss_fftri(internalcfg, input + i * (config.halfTripleBatchSize + 1), buffer);
        for (int j = 0; j < config.tripleBatchSize; j++) {
            // fill result into main buffer with overlap
            *(mainBuffer + i * config.batchSize + j) += *(buffer + j);
        }
    }
    kiss_fft_free(internalcfg);
    free(buffer);
    // rescale waveform at the edges of the main buffer, to account for missing contributions of out-of-bounds windows
    for (int i = 0; i < config.batchSize; i++) {
        *(mainBuffer + i) *= 1 + pow(cos(pi * (i / config.tripleBatchSize + 2 / 3)), 2);
        *(mainBuffer + config.batchSize * (batches + 2) - 1 - i) *= 1 + pow(cos(pi * i / config.tripleBatchSize), 2);
    }
    // allocate output buffer and transfer relevant data into it
    float* output = (float*) malloc(targetLength * sizeof(float));
    for (int i = 0; i < targetLength; i++) {
        *(output + i) = *(mainBuffer + config.batchSize + i) * 2 / 3;
    }
    free(mainBuffer);
    return output;
}

float* istft_hann(cpxfloat* input, int batches, int targetLength, engineCfg config) {
    // fft setup
    kiss_fftr_cfg internalcfg = kiss_fftr_alloc(config.tripleBatchSize, 1, NULL, NULL);
    // extended input buffer aligned with batch size
    float* mainBuffer = (float*) malloc(config.batchSize * (batches + 2) * sizeof(float));
    for (int i = 0; i < config.batchSize * (batches + 2); i++) {
        *(mainBuffer + i) = 0.;
    }
    // smaller buffer for individual fft result
    float* buffer = (float*) malloc((config.tripleBatchSize) * sizeof(float));
    for (int i = 0; i < batches; i++) {
        // perform ffts
        kiss_fftri(internalcfg, input + i * (config.halfTripleBatchSize + 1), buffer);
        for (int j = 0; j < config.tripleBatchSize; j++) {
            // fill result into main buffer with overlap
            *(mainBuffer + i * config.batchSize + j) += *(buffer + j) * pow(sin(pi * (j + 1) / (config.tripleBatchSize + 1)), 2);
        }
    }
    kiss_fft_free(internalcfg);
    free(buffer);
    // rescale waveform at the edges of the main buffer, to account for missing contributions of out-of-bounds windows
    for (int i = 0; i < config.batchSize; i++) {
        *(mainBuffer + i) *= 1 + pow(cos(pi * (i / config.tripleBatchSize + 2 / 3)), 2);
        *(mainBuffer + config.batchSize * (batches + 2) - 1 - i) *= 1 + pow(cos(pi * i / config.tripleBatchSize), 2);
    }
    // allocate output buffer and transfer relevant data into it
    float* output = (float*) malloc(targetLength * sizeof(float));
    for (int i = 0; i < targetLength; i++) {
        *(output + i) = *(mainBuffer + config.batchSize + i) * 2 / 3;
    }
    free(mainBuffer);
    return output;
}
