#pragma once
#include "./../Core/macros.h"
#include <cstddef>

ACMB_CUDA_NAMESPACE_BEGIN

// Returns true if Cuda is present on this GPU
bool isCudaAvailable();

// Returns available GPU memory in bytes
size_t getCudaAvailableMemory();

ACMB_CUDA_NAMESPACE_END
