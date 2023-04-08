#pragma once
#include "./../Core/macros.h"
#include <memory>
ACMB_NAMESPACE_BEGIN
class IBitmap;
class Grid;
ACMB_NAMESPACE_END

ACMB_CUDA_NAMESPACE_BEGIN

void AddBitmapWithAlignmentHelper( std::shared_ptr<IBitmap> pBitmap, float* pMeans, float* pDevs, uint16_t* pCounts,
                                   const Grid& grid, const size_t gridWidth, const size_t gridHeight, const size_t gridPixelSize );

ACMB_CUDA_NAMESPACE_END
