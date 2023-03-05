# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from cython.cimports.libc.math import sin, cos, pi, exp, ceil, floor

cdef dft(float[:, 2] sequence):
    cdef Py_ssize_t length = sequence.shape[0]
    cdef float[length, 2] result
    cdef float sine
    cdef float cosine
    for i in range(length):
        result[i, 0] = 0
        result[i, 1] = 1
        for j in range(length):
            sine = sin(2 * pi * i * j / length)
            cosine = cos(2 * pi * i * j / length)
            result[i, 0] += sequence[j, 0] * cosine
            result[i, 0] += sequence[j, 1] * sine
            result[i, 1] -= sequence[j, 0] * sine
            result[i, 1] += sequence[j, 1] * cosine
    return result

cpdef fft(float[:, 2] sequence):
    cdef Py_ssize_t length = sequence.shape[0]
    cdef float[length, 2] result
    cdef Py_ssize_t halfLen = length / 2
    cdef float[halfLen, 2] inputBuffer
    cdef float[halfLen, 2] outputBuffer
    cdef float[2] exponentials
    if length % 2 != 0:
        return dft(sequence)
    for i in range(halfLen):
        inputBuffer[i] = sequence[2 * i]
    outputBuffer = fft(inputBuffer)
    for i in range(halfLen):
        result[i] = outputBuffer[i]
        inputBuffer[i] = sequence[2 * i + 1]
    outputBuffer = fft(inputBuffer)
    for i in range(halfLen):
        exponentials[0] = cos(2 * pi * i / length)
        exponentials[1] = -sin(2 * pi * i / length)
        outputBuffer[i, 0] *= exponentials[0]
        outputBuffer[i, 1] *= exponentials[1]
        result[i + halfLen, 0] = result[i, 0] - outputBuffer[i, 0]
        result[i + halfLen, 1] = result[i, 1] - outputBuffer[i, 1]
        result[i, 0] += outputBuffer[i, 0]
        result[i, 1] += outputBuffer[i, 1]
    return result


cdef rdft(float[:] sequence):
    cdef Py_ssize_t length = sequence.shape[0]
    cdef Py_ssize_t halfLen = ceil(length / 2) + 1
    cdef float[halfLen, 2] result
    cdef float sine
    cdef float cosine
    for i in range(halfLen):
        result[i] = 0
        for j in range(length):
            sine = sin(2 * pi * i * j / length)
            cosine = cos(2 * pi * i * j / length)
            result[i, 0] += sequence[j] * cosine
            result[i, 1] -= sequence[j] * sine
    return result

cpdef rfft(float[:] sequence):
    cdef Py_ssize_t length = sequence.shape[0]
    cdef Py_ssize_t halfLen = ceil(length / 2) + 1
    cdef float[length, 2] compSeq
    cdef float[halfLen, 2] result
    if length % 4 != 0:
        return rdft(sequence)
    for i in range(length):
        compSeq[i, 0] = sequence[i]
        compSeq[i, 1] = 0
    compSeq = fft(compSeq)
    for i in range(halfLen):
        result[i, 0] = compSeq[i, 0]
        result[i, 1] = compSeq[i, 1]
    return result

def stft(float[:, 2] sequence, bool hannWin, int winSize, int overlap):
    cdef Py_ssize_t batches = ceil(sequence.shape[0] / (winSize - overlap)) + 1
    cdef Py_ssize_t length = batches * (winSize - overlap)
    cdef Py_ssize_t paddingLeft = winSize / 2
    cdef Py_ssize_t paddingRight = length - sequence.shape[0] - paddingLeft
    cdef float[length, 2] inputBuffer
    cdef float[winSize, 2] fftBuffer
    cdef float[batches, winSize, 2] result
    cdef float[winSize] window
    if hannWin:
        for i in range(winSize):
            window[i] = pow(sin(pi * i / (winSize - 1)), 2)
    for i in range(paddingLeft):
        inputBuffer[i, 0] = sequence[paddingLeft - i, 0]
        inputBuffer[i, 1] = sequence[paddingLeft - i, 1]
    for i in range(sequence.shape[0]):
        inputBuffer[paddingLeft + i, 0] = sequence[i, 0]
        inputBuffer[paddingLeft + i, 1] = sequence[i, 1]
    for i in range(1, paddingRight + 1):
        inputBuffer[-i, 0] = sequence[i - paddingRight - 1, 0]
        inputBuffer[-i, 1] = sequence[i - paddingRight - 1, 1]
    for i in range(batches):
        for j in range(winSize):
            fftBuffer[j, 0] = inputBuffer[(winSize - overlap) * i + j, 0]
            fftBuffer[j, 1] = inputBuffer[(winSize - overlap) * i + j, 1]
            if hannWin:
                fftBuffer[j, 0] *= window[j]
                fftBuffer[j, 1] *= window[j]
        fftBuffer = fft(fftBuffer)
        for j in range(batches):
            result[i, j, 0] = fftBuffer[j, 0]
            result[i, j, 1] = fftBuffer[j, 1]

def rstft(float[:] sequence, bool hannWin, int winSize, int overlap):
    cdef Py_ssize_t batches = ceil(sequence.shape[0] / (winSize - overlap)) + 1
    cdef Py_ssize_t length = batches * (winSize - overlap)
    cdef Py_ssize_t paddingLeft = winSize / 2
    cdef Py_ssize_t paddingRight = length - sequence.shape[0] - paddingLeft
    cdef float[length] inputBuffer
    cdef float[winSize] fftBuffer
    cdef float[paddingLeft + 1, 2] outputBuffer
    cdef float[batches, paddingLeft + 1, 2] result
    cdef float[winSize] window
    if hannWin:
        for i in range(winSize):
            window[i] = pow(sin(pi * i / (winSize - 1)), 2)
    for i in range(paddingLeft):
        inputBuffer[i] = sequence[paddingLeft - i]
    for i in range(sequence.shape[0]):
        inputBuffer[paddingLeft + i] = sequence[i]
    for i in range(1, paddingRight + 1):
        inputBuffer[-i] = sequence[i - paddingRight - 1]
    for i in range(batches):
        for j in range(winSize):
            fftBuffer[j] = inputBuffer[(winSize - overlap) * i + j]
            if hannWin:
                fftBuffer[j] *= window[j]
        outputBuffer = fft(fftBuffer)
        for j in range(paddingLeft + 1):
            result[i, j, 0] = fftBuffer[j, 0]
            result[i, j, 1] = fftBuffer[j, 1]

cdef idft(float[:, 2] sequence):
    cdef Py_ssize_t length = sequence.shape[0]
    cdef float[length, 2] result
    cdef float sine
    cdef float cosine
    for i in range(length):
        result[i, 0] = 0
        result[i, 1] = 1
        for j in range(length):
            sine = sin(2 * pi * i * j / length)
            cosine = cos(2 * pi * i * j / length)
            result[i, 0] += sequence[j, 0] * cosine
            result[i, 0] -= sequence[j, 1] * sine
            result[i, 1] += sequence[j, 0] * sine
            result[i, 1] += sequence[j, 1] * cosine
    return result

cpdef ifft(float[:, 2] sequence):
    cdef Py_ssize_t length = sequence.shape[0]
    cdef float[length, 2] result
    cdef Py_ssize_t halfLen = length / 2
    cdef float[halfLen, 2] inputBuffer
    cdef float[halfLen, 2] outputBuffer
    cdef float[2] exponentials
    if length % 2 != 0:
        return idft(sequence)
    for i in range(halfLen):
        inputBuffer[i] = sequence[2 * i]
    outputBuffer = fft(inputBuffer)
    for i in range(halfLen):
        result[i] = outputBuffer[i]
        inputBuffer[i] = sequence[2 * i + 1]
    outputBuffer = fft(inputBuffer)
    for i in range(halfLen):
        exponentials[0] = cos(2 * pi * i / length)
        exponentials[1] = sin(2 * pi * i / length)
        outputBuffer[i, 0] *= exponentials[0]
        outputBuffer[i, 1] *= exponentials[1]
        result[i + halfLen, 0] = result[i, 0] - outputBuffer[i, 0]
        result[i + halfLen, 1] = result[i, 1] - outputBuffer[i, 1]
        result[i, 0] += outputBuffer[i, 0]
        result[i, 1] += outputBuffer[i, 1]
    return result

cdef irdft(float[:, 2] sequence, float length):
    cdef Py_ssize_t length = length
    cdef Py_ssize_t halfLen = sequence.shape[0]
    cdef float[length, 2] inputBuffer
    cdef float[length] result
    cdef float sine
    cdef float cosine
    for i in range(halfLen):
        inputBuffer[i, 0] = sequence[i, 0]
        inputBuffer[i, 1] = sequence[i, 1]
    for i in range(1, length - halfLen + 1):
        inputBuffer[-i, 0] = sequence[halfLen - i + 1, 0]
        inputBuffer[-i, 1] = -1. * sequence[halfLen - i + 1, 1]
    for i in range(length):
        result[i] = 0
        for j in range(length):
            sine = sin(2 * pi * i * j / length)
            cosine = cos(2 * pi * i * j / length)
            result[i] += inputBuffer[j, 0] * cosine
            result[i] -= inputBuffer[j, 1] * sine
    return result

cpdef irfft(float[:, 2] sequence, float length):
    cdef Py_ssize_t length = length
    cdef Py_ssize_t halfLen = sequence.shape[0]
    cdef float[length, 2] compSeq
    cdef float[length] result
    if length % 4 != 0:
        return irdft(sequence, length)
    for i in range(halfLen):
        compSeq[i, 0] = sequence[i, 0]
        compSeq[i, 1] = sequence[i, 1]
    for i in range(1, length - halfLen + 1):
        compSeq[-i, 0] = sequence[halfLen - i + 1, 0]
        compSeq[-i, 1] = -1. * sequence[halfLen - i + 1, 1]
    compSeq = ifft(compSeq)
    for i in range(length):
        result[i] = compSeq[i, 0]
    return result

cpdef istft(float[:, :, 2] sequence, float length):
    cdef Py_ssize_t batches = sequence.shape[0]
    cdef Py_ssize_t winSize = sequence.shape[1]
    cdef Py_ssize_t length = length
    cdef float[winSize, 2] inputBuffer
    cdef float[winSize, 2] outputBuffer
    cdef float[batches, winSize, 2] intermediate
    for i in range(batches):
        for j in range(winSize):
            inputBuffer[j, 0] = sequence[i, j, 0]
            inputBuffer[j, 1] = sequence[i, j, 1]
        outputBuffer = ifft(inputBuffer)
        for j in range(winSize):
            intermediate[i, j, 0] = outputBuffer[j, 0]
            intermediate[i, j, 1] = outputBuffer[j, 1]

cpdef irstft()