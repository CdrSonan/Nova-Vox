# Copyright 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.25)
project(Nova-Vox_C_components VERSION 0.7.6 LANGUAGES C)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#set(KISSFFT_DATATYPE "float" CACHE STRING "Principal datatype of kissfft: double, float (default), int16_t, int32_t, simd")
#set(KISSFFT_OPENMP OFF CACHE BOOL "Build kissfft with OpenMP support")
#set(KISSFFT_PKGCONFIG ON CACHE BOOL "Build pkg-config files")
#set(KISSFFT_STATIC ON CACHE BOOL "Build kissfft as static (ON) or shared library (OFF)")
#set(KISSFFT_TEST OFF CACHE BOOL "Build kissfft tests")
#set(KISSFFT_TOOLS OFF CACHE BOOL "Build kissfft command-line tools")
#set(KISSFFT_USE_ALLOCA OFF CACHE BOOL "Use alloca instead of malloc")

add_library(esper SHARED
    C/ESPER/esper.c
    C/ESPER/pitchCalcFallback.c
    C/ESPER/components.c
    C/fft.c
    C/interpolation.c
    C/util.c)
target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/")
set_property(TARGET esper PROPERTY C_STANDARD 11)
target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/lib/fftw/libfftw3f-3.lib")

add_library(resampler SHARED
    C/Resampler/resampler.c
    C/Resampler/loop.c
    C/interpolation.c
    C/util.c)
target_include_directories(resampler PRIVATE "${PROJECT_SOURCE_DIR}/")
set_property(TARGET resampler PROPERTY C_STANDARD 11)

set_target_properties(esper resampler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")
