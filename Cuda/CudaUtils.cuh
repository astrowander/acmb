#pragma once
#include "./../Core/macros.h"
#include "cuda_runtime.h"
ACMB_CUDA_NAMESPACE_BEGIN

template<typename T>
__device__ inline T Clamp( T x, T l, T u )
{
    return ( ( x < l ) ? l :
             ( ( x > u ) ? u : x ) );
}

template<typename T>
__device__ inline T Max()
{
    return T{};
}

template<>
__device__ inline uint8_t Max()
{
    return 255;
}

template<>
__device__ inline uint16_t Max()
{
    return 65535;
}

ACMB_CUDA_NAMESPACE_END
