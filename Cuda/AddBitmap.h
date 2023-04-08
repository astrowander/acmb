#pragma once
#include "./../Core/macros.h"
#include <cstdint>

ACMB_CUDA_NAMESPACE_BEGIN

template<typename ChannelType>
void AddBitmapHelper( ChannelType* pPixels, float* pMeans, float* pDevs, uint16_t* pCounts, size_t size );

ACMB_CUDA_NAMESPACE_END