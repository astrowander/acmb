#pragma once
#include "CudaBasic.h"

ACMB_CUDA_NAMESPACE_BEGIN

using Triangle = double2[3];

struct TransAffine
{
    double sx = 1.0;
    double shy = 0.0;
    double shx = 0.0;
    double sy = 1.0;
    double tx = 0.0;
    double ty = 0.0;
};

struct TriangleTransformPair
{
    Triangle triangle;
    TransAffine transform;
};

using GridCell = DynamicArray<TriangleTransformPair>;
using Grid = DynamicArray<TriangleTransformPair*>;

template<typename ChannelType>
void AddBitmapWithAlignmentKernel( const ChannelType* pixels, const uint32_t width, const uint32_t height, const uint32_t channelCount,
                              const TriangleTransformPair* grid, const uint32_t* cellOffsets, const size_t gridWidth, const size_t gridPixelSize,
                              float* pMeans, float* pDevs, uint16_t* pCounts );

ACMB_CUDA_NAMESPACE_END
