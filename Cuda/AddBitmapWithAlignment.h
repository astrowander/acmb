#pragma once
#include "./../Registrator/BaseStacker.h"

ACMB_NAMESPACE_BEGIN
class IBitmap;
ACMB_NAMESPACE_END

ACMB_CUDA_NAMESPACE_BEGIN

template<typename ChannelType>
void AddBitmapWithAlignmentHelper( const ChannelType* pixels, const uint32_t width, const uint32_t height, const uint32_t channelCount,
                                   const BaseStacker::Grid& grid,
                                   float* pMeans, float* pDevs, uint16_t* pCounts );

ACMB_CUDA_NAMESPACE_END
