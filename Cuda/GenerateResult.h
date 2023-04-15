#pragma once
#include "./../Core/enums.h"
#include <cstdint>

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class Bitmap;

ACMB_NAMESPACE_END

ACMB_CUDA_NAMESPACE_BEGIN

template<typename ChannelType>
void GeneratingResultKernel( const float* pMeans, ChannelType* pOutput, const size_t size );

ACMB_CUDA_NAMESPACE_END
