// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/util.h"

#include "lib/fftw/fftw3.h"
#include <malloc.h>
#include <math.h>

void dynIntArray_init(dynIntArray* array) {
    (*array).content = (int*) malloc(sizeof(int));
    (*array).length = 0;
    (*array).maxlen = 1;
}

void dynIntArray_dealloc(dynIntArray* array) {
    free((*array).content);
    (*array).length = 0;
    (*array).maxlen = 0;
}

void dynIntArray_append(dynIntArray* array, int value) {
    if ((*array).length == (*array).maxlen) {
        (*array).content = (int*) realloc((*array).content, 2 * (*array).maxlen * sizeof(int));
        (*array).maxlen *= 2;
    }
    *((*array).content + (*array).length) = value;
    (*array).length++;
}

int ceildiv(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

unsigned int findIndex(int* markers, unsigned int markerLength, int position) {
    int low = -1;
    int high = markerLength;
    int mid;
    while (low + 1 < high) {
        mid = (low + high) / 2;
        if (position > *(markers + mid)) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return (unsigned int)high;
}

unsigned int findIndex_double(double* markers, unsigned int markerLength, int position) {
    int low = -1;
    int high = markerLength;
    int mid;
    while (low + 1 < high) {
        mid = (low + high) / 2;
        if ((double)position > *(markers + mid)) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return (unsigned int)high;
}

float cpxAbsf(fftwf_complex input) {
    return sqrtf(powf(input[0], 2) + powf(input[1], 2));
}

float cpxArgf(fftwf_complex input) {
    return atan2f(input[0], input[1]);
}

float pi = 3.1415926535897932384626433;
