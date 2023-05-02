// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "C/util.h"
#include "lib/fftw/fftw3.h"

fftwf_complex* fft(fftwf_complex* input, int length);

fftwf_complex* ifft(fftwf_complex* input, int length);

fftwf_complex* rfft(float* input, int length);

void rfft_inpl(float* input, int length, fftwf_complex* output);

float* irfft(fftwf_complex* input, int length);

void irfft_inpl(fftwf_complex* input, int length, float* output);

fftwf_complex* stft(float* input, int length, engineCfg config);

void stft_inpl(float* input, int length, engineCfg config, float* output);

float* istft(fftwf_complex* input, int batches, int targetLength, engineCfg config);

float* istft_hann(fftwf_complex* input, int batches, int targetLength, engineCfg config);
