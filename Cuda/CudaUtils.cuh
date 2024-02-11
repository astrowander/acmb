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
__device__ inline T Max( T a, T b)
{
    return ((a > b) ? a : b);
}

template<typename T>
__device__ inline T MaxValue()
{
    return T{};
}

template<>
__device__ inline uint8_t MaxValue()
{
    return 255;
}

template<>
__device__ inline uint16_t MaxValue()
{
    return 65535;
}

__device__ inline float QuadraticInterpolation( float t, float t0, float t1, float t2 )
{
    auto a = ( t0 + t2 ) / 2 - t1;
    auto b = -( 3 * t0 + t2 ) / 2 + 2 * t1;
    //auto c = t0;
    auto res = a * t * t + b * t + t0;
    return res;
}

ACMB_CUDA_NAMESPACE_END
