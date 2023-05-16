// Copyright 2023 Contributors to the Nova-Vox project

// This file is part of Nova-Vox.
// Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
// Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "C/util.h"

__declspec(dllexport) void __cdecl resampleSpecharm(float* avgSpecharm, float* specharm, int length, float* steadiness, float spacing, int startCap, int endCap, float* output, segmentTiming timings, engineCfg config);

__declspec(dllexport) void __cdecl resamplePitch(short* pitchDeltas, int length, float pitch, float spacing, int startCap, int endCap, float* output, int requiredSize, segmentTiming timings);
