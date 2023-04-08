#pragma once
#include "CudaBasic.h"
#include "CudaUtils.cuh"

ACMB_CUDA_NAMESPACE_BEGIN

using Triangle = double2[3];

struct TransAffine
{
    static const TransAffine identity;
    double sx, shy, shx, sy, tx, ty;

    __device__ TransAffine() :
        sx( 1.0 ), shy( 0.0 ), shx( 0.0 ), sy( 1.0 ), tx( 0.0 ), ty( 0.0 )
    {
    }
};

struct TriangleTransformPair
{
    Triangle triangle;
    TransAffine transform;
};

using GridCell = DynamicArray<TriangleTransformPair>;
using Grid = DynamicArray<TriangleTransformPair*>;

ACMB_CUDA_NAMESPACE_END
