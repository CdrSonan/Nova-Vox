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

//wrapper for a 1d fft
fftwf_complex* fft(fftwf_complex* input, int length)
{
    fftwf_complex* output = (fftwf_complex*) malloc(length * sizeof(fftwf_complex));
    fftwf_plan plan = fftwf_plan_dft_1d(length, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return output;
}

//wrapper for an inverse 1d fft
fftwf_complex* ifft(fftwf_complex* input, int length)
{
    fftwf_complex* output = (fftwf_complex*) malloc(length * sizeof(fftwf_complex));
    fftwf_plan plan = fftwf_plan_dft_1d(length, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return output;
}

//wrapper for a real-valued 1d fft. Output has size length/2 + 1
fftwf_complex* rfft(float* input, int length)
{
    fftwf_complex* output = (fftwf_complex*) malloc((ceildiv(length, 2) + 1) * sizeof(fftwf_complex));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(length, input, output, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return output;
}

//wrapper for a real-valued fft, where the result is written into a predetermined, rather than newly allocated location
void rfft_inpl(float* input, int length, fftwf_complex* output)
{
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(length, input, output, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

//wrapper for an inverse, real-valued fft. length refers to the length of the recovered signal, not the length of the input buffer.
float* irfft(fftwf_complex* input, int length)
{
    float* output = (float*) malloc(length * sizeof(float));
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(length, input, output, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return output;
}

//wrapper for an inverse, real-valued fft. The result is written into already allocated storage given by pointer.
//length refers to the length of the recovered signal, not the length of the input buffer.
void irfft_inpl(fftwf_complex* input, int length, float* output)
{
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(length, input, output, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

//short-time fft of real-valued data with Hanning windowing.
//The required batch size parameters are wrapped into the config argument.
//The i-th window of the transform is centered at i * config.batchSize, and reflection padding is applied to the signal as needed to accomplish this.
fftwf_complex* stft(float* input, int length, engineCfg config)
{
    int batches = ceildiv(length, config.batchSize);
    int rightpad = batches * config.batchSize - length + config.batchSize;
    // extended input buffer aligned with batch size
    float* in = (float*) malloc((config.batchSize + length + rightpad) * sizeof(float));
    // fill input buffer, extend will data with reflection padding on both sides
    for (int i = 0; i < config.batchSize; i++)
    {
        *(in + i) = *(input + config.batchSize - i);
    }
    for (int i = 0; i < length; i++)
    {
        *(in + config.batchSize + i) = *(input + i);
    }
    for (int i = 0; i < rightpad; i++)
    {
        *(in + config.batchSize + length + i) = *(input + length - 2 - i);
    }
    // allocate output buffer of desired size
    fftwf_complex* out = (fftwf_complex*) malloc(batches * (config.halfTripleBatchSize + 1) * sizeof(fftwf_complex));
    // fft setup
    for (int i = 0; i < batches; i++)
    {
        // allocation within loop for future openMP support
        float* buffer = (float*) malloc((config.tripleBatchSize) * sizeof(float));
        for (int j = 0; j < config.tripleBatchSize; j++)
        {
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

//short-time fft of real-valued data with Hanning windowing, where the output is written to a previously allocated float array by separating real and imaginary parts.
//The required batch size parameters are wrapped into the config argument, while the normalization factor is hard coded into the function, rather than inferred.
//The i-th window of the transform is centered at i * config.batchSize, and reflection padding is applied to the signal as needed to accomplish this.
void stft_inpl(float* input, int length, engineCfg config, float* output)
{
    int batches = ceildiv(length, config.batchSize);
    fftwf_complex* buffer = stft(input, length, config);
    for (int i = 0; i < batches * (config.halfTripleBatchSize + 1); i++)
    {
        *(output + i) = (*(buffer + i))[0];
        *(output + batches * (config.halfTripleBatchSize + 1) + i) = (*(buffer + i))[1];
    }
    free(buffer);
}

//inverse short-time fft for real-valued data. Assumes the input was created using Hanning windows, and is padded for centered windows.
//The required batch size parameters are wrapped into the config argument, while the normalization factor is hard coded into the function, rather than inferred.
float* istft(fftwf_complex* input, int batches, int targetLength, engineCfg config)
{
    // fft setup
    // extended input buffer aligned with batch size
    float* mainBuffer = (float*) malloc(config.batchSize * (batches + 2) * sizeof(float));
    for (int i = 0; i < config.batchSize * (batches + 2); i++)
    {
        *(mainBuffer + i) = 0.;
    }
    // smaller buffer for individual fft result
    float* buffer = (float*) malloc(config.tripleBatchSize * sizeof(float));
    for (int i = 0; i < batches; i++)
    {
        // perform ffts
        fftwf_plan plan = fftwf_plan_dft_c2r_1d(config.tripleBatchSize, input + i * (config.halfTripleBatchSize + 1), buffer, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        for (int j = 0; j < config.tripleBatchSize; j++)
        {
            // fill result into main buffer with overlap
            *(mainBuffer + i * config.batchSize + j) += *(buffer + j);
        }
    }
    free(buffer);
    // allocate output buffer and transfer relevant data into it
    float* output = (float*) malloc(targetLength * sizeof(float));
    for (int i = 0; i < targetLength; i++)
    {
        *(output + i) = *(mainBuffer + config.halfTripleBatchSize + i) * 2 / 3;
    }
    free(mainBuffer);
    return output;
}

//inverse short-time fft for real-valued data. Smoothes the result by applying Hanning windows to each input batch.
//The required batch size parameters are wrapped into the config argument, while the normalization factor is hard coded into the function, rather than inferred.
float* istft_hann(fftwf_complex* input, int batches, int targetLength, engineCfg config)
{
    // fft setup
    // extended input buffer aligned with batch size
    float* mainBuffer = (float*) malloc(config.batchSize * (batches + 2) * sizeof(float));
    for (int i = 0; i < config.batchSize * (batches + 2); i++)
    {
        *(mainBuffer + i) = 0.;
    }
    // smaller buffer for individual fft result
    float* buffer = (float*) malloc(config.tripleBatchSize * sizeof(float));
    for (int i = 0; i < batches; i++)
    {
        // perform ffts
        fftwf_plan plan = fftwf_plan_dft_c2r_1d(config.tripleBatchSize, input + i * (config.halfTripleBatchSize + 1), buffer, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        for (int j = 0; j < config.tripleBatchSize; j++)
        {
            // fill result into main buffer with overlap
            *(mainBuffer + i * config.batchSize + j) += *(buffer + j)* pow(sin(pi * j / (config.tripleBatchSize - 1)), 2);
        }
    }
    free(buffer);
    // allocate output buffer and transfer relevant data into it
    float* output = (float*) malloc(targetLength * sizeof(float));
    for (int i = 0; i < targetLength; i++)
    {
        *(output + i) = *(mainBuffer + config.halfTripleBatchSize + i) * 2 / 3;
    }
    free(mainBuffer);
    return output;
}
