// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#include "C/interpolation.h"

#include <stdlib.h>
#include <math.h>
#include "C/util.h"

float* hPoly(float* input, int length) {
    float* temp = (float*)malloc(4 * length * sizeof(float));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < length; j++) {
            *(temp + j + i * length) = pow(*(input + j), i);
        }
    }
    float matrix[4][4] = { {1, 0, -3, 2}, {0, 1, -2, 1}, {0, 0, 3, -2}, {0, 0, -1, 1} };
    float* output = (float*)malloc(4 * length * sizeof(float));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < length; j++) {
            *(output + j + i * length) = 0;
            for (int k = 0; k < 4; k++) {
                *(output + j + i * length) += *(temp + j + k * length) * matrix[i][k];
            }
        }
    }
    free(temp);
    return output;
}

float* interp(float* x, float* y, float* xs, int len, int lenxs) {
    //fill old space derivative array
    float* m = (float*) malloc(len * sizeof(float));
    for (int i = 0; i < (len - 1); i++) {
        *(m + i + 1) = (*(y + i + 1) - *(y + i)) / (*(x + i + 1) - *(x + i));
    }
    *m = *(m + 1);
    for (int i = 1; i < (len - 1); i++) {
        *(m + i) = (*(m + i) + *(m + i + 1)) / 2.;
    }
    //get correct indices for xs
    float* idxs = (float*) malloc(lenxs * sizeof(float));
    int i = 0; //iterator for xs
    int j = 0; //iterator for x
    while ((j < len - 1) && (i < lenxs)) { //since both x and xs are sorted, iterating linearly is O(n), compared to O(n log n) for repeated binary search
        if (*(xs + i) > *(x + j + 1)) {
            j++;
        } else {
            *(idxs + i) = j;
            i++;
        }
    }
    while (i < lenxs) { //all remaining points are behind the last element of x
        *(idxs + i) = j;
        i++;
    }
    //compute new space derivatives and hermite polynomial base
    float* dx = (float*) malloc(lenxs * sizeof(float));
    float* hh = (float*) malloc(lenxs * sizeof(float));
    int offset;
    for (int i = 0; i < lenxs; i++) {
        offset = *(idxs + i);
        *(dx + i) = *(x + 1 + offset) - *(x + offset);
        *(hh + i) = (*(xs + i) - *(x + offset)) / *(dx + i);
    }
    float* h = hPoly(hh, lenxs);
    free(hh);
    //compute final data
    float* ys = (float*) malloc(lenxs * sizeof(float));
    for (int i = 0; i < lenxs; i++) {
        offset = *(idxs + i);
        *(ys + i) = (*(h + i) * *(y + offset)) + (*(h + i + lenxs) * *(m + offset) * *(dx + i)) + (*(h + i + 2 * lenxs) * *(y + offset + 1)) + (*(h + i + 3 * lenxs) * *(m + offset + 1) * *(dx + i));
    }
    free(m);
    free(idxs);
    free(dx);
    free(h);
    return ys;
}

float* extrap(float* x, float* y, float* xs, int len, int lenxs) {
    float largeY = *(y + len - 1) + (*(y + len - 1) - *(y + len - 2)) * (*(xs + lenxs - 1) - *(x + len - 1)) / (*(x + len - 1) - *(x + len - 2));
    float smallY = *y - (*(y + 1) - *y) * (*x - *xs) / (*(x + 1) - *x);
    float* xnew;
    float* ynew;
    int freeNew = 1;
    if ((*xs < *x) && (*(xs + lenxs - 1) > *(x + len - 1))) {
        xnew = (float*) malloc((len + 2) * sizeof(float));
        ynew = (float*) malloc((len + 2) * sizeof(float));
        for (int i = 0; i < len; i++) {
            *(xnew + i + 1) = *(x + i);
            *(ynew + i + 1) = *(y + i);
        }
        *xnew = *xs;
        *ynew = smallY;
        *(xnew + len + 1) = *(xs + lenxs - 1);
        *(ynew + len + 1) = largeY;
    } else {
        if (*xs < *x) {
            xnew = (float*) malloc((len + 1) * sizeof(float));
            ynew = (float*) malloc((len + 1) * sizeof(float));
            for (int i = 0; i < len; i++) {
                *(xnew + i + 1) = *(x + i);
                *(ynew + i + 1) = *(y + i);
            }
            *xnew = *xs;
            *ynew = smallY;
        } else if (*(xs + lenxs - 1) > *(x + len - 1)) {
            xnew = (float*) malloc((len + 1) * sizeof(float));
            ynew = (float*) malloc((len + 1) * sizeof(float));
            for (int i = 0; i < len; i++) {
                *(xnew + i) = *(x + i);
                *(ynew + i) = *(y + i);
            }
            *(xnew + len) = *(xs + lenxs - 1);
            *(ynew + len) = largeY;
        } else {
            xnew = x;
            ynew = y;
            freeNew = 0;
        }
    }
    float* ys = (float*) malloc(lenxs * sizeof(float));
    ys = interp(xnew, ynew, xs, len + 1, lenxs);
    if (freeNew == 1) {
        free(xnew);
        free(ynew);
    }
    return ys;
}

void phaseInterp_inplace(float* phasesA, float* phasesB, int len, float factor) {
    float* bufferA = (float*)malloc(len * sizeof(float));
    float* bufferB = (float*)malloc(len * sizeof(float));
    for (int i = 0; i < len; i++) {
        *(bufferA + i) = *(phasesB + i) - *(phasesA + i);
        *(bufferB + i) = *(bufferA + i) - (2 * pi);
        if (fabsf(*(bufferA + i)) >= fabsf(*(bufferB + i))) {
            *(phasesA + i) = fmodf(*(phasesA + i) + *(bufferA + i) * factor, 2 * pi);
        }
        else {
            *(phasesA + i) = fmodf(*(phasesA + i) + *(bufferB + i) * factor, 2 * pi);
        }
    }
    free(bufferA);
    free(bufferB);
}
